# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial, partialmethod
from typing import List, Optional

import torch
import torch.nn as nn

from rnapro.openfold_local.model.primitives import Attention, LayerNorm, Linear
from rnapro.openfold_local.utils.chunk_utils import chunk_layer
from rnapro.openfold_local.utils.tensor_utils import permute_final_dims


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False)

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                triangle_attention=triangle_attention,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    @torch.jit.ignore
    def _inplace_row_chunk(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        triangle_attention: str = "torch",
    ) -> None:
        """In-place row-wise triangle attention with fused residual addition.

        Processes rows of the pair matrix in small chunks, writing
        ``z[row_chunk] += attn_output`` directly. Avoids allocating a
        full [I, J, C_in] output buffer.

        Args:
            z: Original pair tensor to accumulate into (may be transposed).
                [*, I, J, C_in] -- modified **in-place**.
            x: LayerNorm(z) -- the normalised input for QKV projection.
                [*, I, J, C_in]
            biases: [mask_bias, triangle_bias] precomputed from x/mask.
            chunk_size: Number of rows per chunk.
            triangle_attention: Backend selection string.
        """
        mask_bias, triangle_bias = biases
        I = x.shape[-3]

        for i_start in range(0, I, chunk_size):
            i_end = min(i_start + chunk_size, I)

            # Slice rows for this chunk -- x is read-only, slicing is a view
            x_chunk = x[..., i_start:i_end, :, :]  # [*, chunk, J, C_in]

            # Slice row-dependent biases
            # mask_bias: [*, I, 1, 1, J] -> chunk along I
            mb_chunk = mask_bias[..., i_start:i_end, :, :, :]
            # triangle_bias: [*, 1, H, I, J] -> shared across rows (no slice needed)
            chunk_biases = [mb_chunk, triangle_bias]

            # Run standard attention on the small chunk
            out_chunk = self.mha(
                q_x=x_chunk,
                kv_x=x_chunk,
                biases=chunk_biases,
                triangle_attention=triangle_attention,
            )

            # Fused residual: write directly into z
            z[..., i_start:i_end, :, :] += out_chunk
            del out_chunk, x_chunk, mb_chunk

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        _z_for_inplace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
            _z_for_inplace:
                When provided, the attention output is accumulated
                directly into this tensor via ``z += attn(x)``, and the
                return value is None.  The caller must NOT add the result
                to z again.  Only used when ``inplace_safe=True``.
        Returns:
            [*, I, J, C_in] output tensor, or None if _z_for_inplace is used.
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if _z_for_inplace is not None and chunk_size is not None:
            # In-place row-wise path: accumulate directly into z
            # z is already in the correct orientation (transposed if needed
            # by the caller before passing here).
            z_target = _z_for_inplace
            if not self.starting:
                z_target = z_target.transpose(-2, -3)
            self._inplace_row_chunk(
                z_target, x, biases, chunk_size, triangle_attention,
            )
            if not self.starting:
                x = z_target.transpose(-2, -3)
            return None
        elif chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                triangle_attention=triangle_attention,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
