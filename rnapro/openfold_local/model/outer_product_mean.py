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

"""
2024-11-13 Modification:

- The 'bias' term has been removed in several linear layers.
- Use inhouse Layernorm in OuterProductMean to replace nn.Layernorm.

Protenix Team
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from rnapro.openfold_local.model.primitives import LayerNorm, Linear
from rnapro.openfold_local.utils.chunk_utils import chunk_layer
from rnapro.openfold_local.utils.precision_utils import is_fp16_enabled


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    @torch.jit.ignore
    def _inplace_chunk(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        norm: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
    ) -> None:
        """In-place row-wise OPM with fused residual addition.

        Processes row-chunks of the outer product and writes
        ``z[row_chunk] += opm_output / norm`` directly, avoiding a full
        [N_res, N_res, C_z] output buffer.

        Args:
            a: [*, N_res, N_seq, C] projected MSA (transposed).
            b: [*, N_res, N_seq, C] projected MSA (transposed).
            norm: [*, N_res, N_res, 1] normalisation denominator.
            z: [*, N_res, N_res, C_z] pair tensor -- modified **in-place**.
            chunk_size: Number of rows per chunk.
        """
        N_res = a.shape[-3]
        for i_start in range(0, N_res, chunk_size):
            i_end = min(i_start + chunk_size, N_res)
            a_chunk = a[..., i_start:i_end, :, :]  # [*, chunk, N_seq, C]
            # b is full -- needed for the einsum contraction over N_res
            outer_chunk = self._opm(a_chunk, b)  # [*, chunk, N_res, C_z]
            outer_chunk /= norm[..., i_start:i_end, :, :]
            z[..., i_start:i_end, :, :] += outer_chunk
            del outer_chunk, a_chunk

    def _forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _z_for_inplace: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            _z_for_inplace:
                When provided, accumulates the OPM output directly into
                this tensor via ``z += opm_result``. Returns None.
        Returns:
            [*, N_res, N_res, C_z] pair embedding update, or None if in-place.
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        if _z_for_inplace is not None and chunk_size is not None:
            self._inplace_chunk(a, b, norm, _z_for_inplace, chunk_size)
            return None

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, C_z]
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm

        return outer

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _z_for_inplace: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if is_fp16_enabled():
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward(
                    m.float(), mask, chunk_size, inplace_safe,
                    _z_for_inplace=_z_for_inplace,
                )
        else:
            return self._forward(
                m, mask, chunk_size, inplace_safe,
                _z_for_inplace=_z_for_inplace,
            )
