# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 AlQuraishi Laboratory
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

import yaml
import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from torch.nn.parameter import Parameter
from functools import partialmethod
from typing import Union, List


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


def recursive_linear_init(m, scale_factor):
    for child_name, child in m.named_modules():
        if "gate" not in child_name:
            custom_weight_init(child, scale_factor)


def custom_weight_init(m, scale_factor):
    if isinstance(m, nn.Linear):
        d_model = (
            m.in_features
        )  # Set d_model to the input dimension of the linear layer
        upper = 1.0 / (d_model**0.5) * scale_factor
        lower = -1.0 / (d_model**0.5) * scale_factor
        torch.nn.init.uniform_(m.weight, lower, upper)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class TransitionLayer(nn.Module):
    def __init__(self, input_dim, n=4):
        super(TransitionLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_a = nn.Linear(input_dim, n * input_dim, bias=False)
        self.linear_b = nn.Linear(input_dim, n * input_dim, bias=False)
        self.linear_out = nn.Linear(n * input_dim, input_dim, bias=False)

    def forward(self, x):
        # Step 1: Apply LayerNorm
        x = self.layer_norm(x)

        # Step 2: Compute a and b using LinearNoBias (implemented with Linear and bias=False)
        a = self.linear_a(x)
        b = self.linear_b(x)

        # Step 3: Element-wise multiplication of swish(a) and b
        swish_a = a * torch.sigmoid(a)  # Swish activation directly in forward
        x = swish_a * b

        # Step 4: Pass through another LinearNoBias layer
        x = self.linear_out(x)

        return x


def init_weights(m):
    if m is not None and isinstance(m, nn.Linear):
        pass


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.gamma=torch.tensor(32.0)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn + mask  # this is actually the bias

        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e9"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        if src_mask is not None:
            src_mask = src_mask.clone().unsqueeze(-1).long()
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.float()
            # src_mask=src_mask.unsqueeze(-1)#.float()
            attn_mask = (
                torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1).long()
            )
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        return q, attn


class ConvTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        pairwise_dimension,
        use_triangular_attention,
        dim_msa,
        dropout=0.1,
        k=3,
    ):
        super(ConvTransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(
            d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout
        )

        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        # self.conv=nn.Conv1d(d_model,d_model,k,padding=k//2)

        self.triangle_update_out = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="outgoing"
        )
        self.triangle_update_in = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="ingoing"
        )

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="row"
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="col"
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=d_model, dim_msa=dim_msa, pairwise_dim=pairwise_dimension
        )

        # self.sequence_transititon=TransitionLayer(d_model)
        # self.pair_transition=TransitionLayer(pairwise_dimension)
        self.sequence_transititon = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension),
        )

    def forward(self, input):

        src, pairwise_features, src_mask, return_aw = input
        # src_mask=None
        # return_aw=False
        # use_gradient_checkpoint = False

        pairwise_bias = self.pairwise2heads(
            self.pairwise_norm(pairwise_features)
        ).permute(0, 3, 1, 2)

        # self attention
        res = src
        src, attention_weights = self.self_attn(
            src, src, src, mask=pairwise_bias, src_mask=src_mask
        )
        src = res + self.dropout1(src)
        src = self.norm1(src)

        # sequence transition
        res = src
        src = self.sequence_transititon(src)
        src = res + self.dropout2(src)
        src = self.norm2(src)

        # pair track ops
        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask)
        )
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask)
        )
        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask)
            )
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask)
            )
        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)
        if return_aw:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(Outer_Product_Mean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep, pair_rep=None):
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc -> bijcd", seq_rep, seq_rep)
        outer_product = rearrange(outer_product, "b i j c d -> b i j (c d)")
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product


class relpos(nn.Module):

    def __init__(self, dim=64):
        super(relpos, self).__init__()
        self.linear = nn.Linear(33, dim)

    def forward(self, src):
        L = src.shape[1]
        res_id = torch.arange(L).to(src.device).unsqueeze(0)
        device = res_id.device
        bin_values = torch.arange(-16, 17, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(16, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k i d, ... k j d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask=None):
        src_mask = src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))

        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class RibonanzaNet(nn.Module):

    # def __init__(self, ntoken=5, nclass=1, ninp=512, nhead=8, nlayers=9, kmers=9, dropout=0):
    def __init__(self, config):

        super(RibonanzaNet, self).__init__()
        self.config = config
        nhid = config.ninp * 4
        self._tied_weights_keys = (
            []
        )  # avoids AttributeError: 'RibonanzaNet' object has no attribute '_tied_weights_keys'
        self.transformer_encoder = []
        # print(f"constructing {config.nlayers} ConvTransformerEncoderLayers")
        for i in range(config.nlayers):
            if i != config.nlayers - 1:
                k = config.k
            else:
                k = 1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=nhid,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dim_msa=config.dim_msa,
                    dropout=config.dropout,
                    k=k,
                )
            )

        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        for i, layer in enumerate(self.transformer_encoder):
            scale_factor = 1 / (i + 1) ** 0.5
            # scale_factor=i+1
            # scale_factor=0
            recursive_linear_init(layer, scale_factor)

        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)

        recursive_linear_init(self.decoder, scale_factor)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=config.ninp,
            dim_msa=config.dim_msa,
            pairwise_dim=config.pairwise_dimension,
        )
        self.pos_encoder = relpos(config.pairwise_dimension)
        self.use_gradient_checkpoint = False

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs

        return custom_forward

    def forward(self, src, src_mask=None, return_aw=False):
        B, L = src.shape
        src = src
        src = self.encoder(src).reshape(B, L, -1)

        pairwise_features = self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pos_encoder(src)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            src, pairwise_features = layer(
                [src, pairwise_features, src_mask, return_aw]
            )

        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0

        if return_aw:
            return output, attention_weights
        else:
            return output

    def get_embeddings(self, src, src_mask=None, return_aw=False,
                       layer_weights=None):
        """Get weighted embeddings from all transformer layers.

        Args:
            src: Input token IDs [B, L]
            src_mask: Source mask
            return_aw: Whether to return attention weights
            layer_weights: Optional [N_layers] tensor of pre-softmaxed weights.
                If provided, computes weighted sum incrementally to avoid
                stacking all 48 layers (saves ~5.6GB VRAM at L=500).
                If None, falls back to stacking all layers (original behavior).

        Returns:
            If layer_weights is None:
                (all_sequence_features [N, B, L, D], all_pairwise_features [N, B, L, L, D])
            If layer_weights is provided:
                (weighted_seq_features [B, L, D], weighted_pair_features [B, L, L, D])
        """
        B, L = src.shape
        src = src
        src = self.encoder(src).reshape(B, L, -1)

        if self.use_gradient_checkpoint:
            pairwise_features = checkpoint.checkpoint(
                self.custom(self.outer_product_mean), src
            )
            pairwise_features = pairwise_features + self.pos_encoder(src)
        else:
            pairwise_features = self.outer_product_mean(src)
            pairwise_features = pairwise_features + self.pos_encoder(src)

        if layer_weights is not None:
            # Incremental aggregation: accumulate weighted sum layer-by-layer
            # instead of stacking all 48 layers into [48, B, L, L, 128]
            weights = layer_weights.softmax(0)
            seq_accum = torch.zeros_like(src)
            pair_accum = torch.zeros_like(pairwise_features)
            for i, layer in enumerate(self.transformer_encoder):
                src, pairwise_features = checkpoint.checkpoint(
                    self.custom(layer),
                    [src, pairwise_features, src_mask, return_aw],
                    use_reentrant=False,
                )
                w = weights[i]
                seq_accum = seq_accum + w * src
                pair_accum = pair_accum + w * pairwise_features
            return seq_accum, pair_accum
        else:
            # Original behavior: stack all layers (high memory)
            all_sequence_features = []
            all_pairwise_features = []
            for i, layer in enumerate(self.transformer_encoder):
                src, pairwise_features = checkpoint.checkpoint(
                    self.custom(layer),
                    [src, pairwise_features, src_mask, return_aw],
                    use_reentrant=False,
                )
                all_sequence_features.append(src)
                all_pairwise_features.append(pairwise_features)
            all_sequence_features = torch.stack(all_sequence_features, 0)
            all_pairwise_features = torch.stack(all_pairwise_features, 0)
            return all_sequence_features, all_pairwise_features


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise="row"):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z, src_mask):
        """
        how to do masking
        for row tri attention:
        attention matrix is brijh, where b is batch, r is row, h is head
        so mask should be b()ijh, i.e. take self attention mask and unsqueeze(1,-1)
        add negative inf to matrix before softmax

        for col tri attention
        attention matrix is bijlh, so take self attention mask and unsqueeze(3,-1)

        take src_mask and spawn pairwise mask, and unsqueeze accordingly
        """

        # spwan pair mask
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))

        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(
            lambda x: rearrange(x, "b i j (h d)->b i j h d", h=self.n_heads), (q, k, v)
        )
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** 0.5
        if wise == "row":
            eq_attn = "brihd,brjhd->brijh"
            eq_multi = "brijh,brjhd->brihd"
            b = rearrange(b, "b i j (r h)->b r i j h", r=1)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, "b i j->b 1 i j 1")
        elif wise == "col":
            eq_attn = "bilhd,bjlhd->bijlh"
            eq_multi = "bijlh,bjlhd->bilhd"
            b = rearrange(b, "b i j (l h)->b i j l h", l=1)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, "b i j->b i j 1 1")
        else:
            raise ValueError("wise should be col or row!")
        logits = torch.einsum(eq_attn, q, k) / scale + b
        logits = logits.masked_fill(attn_mask == -1, float("-1e-9"))
        attn = logits.softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, "b i j h d-> b i j (h d)")
        z_ = self.to_out(out)
        return z_


class GatedSequenceFeatureInjector(nn.Module):
    def __init__(self, c_s_new: int, c_s: int, gate_type="channel"):
        super().__init__()
        self.proj = nn.Linear(c_s_new, c_s)  # project LM feature to match s dim

        if gate_type == "channel":
            self.gate_param = nn.Parameter(torch.zeros(c_s))  # one gate per channel
        elif gate_type == "scalar":
            self.gate_param = nn.Parameter(torch.tensor(0.0))  # one global gate
        else:
            raise ValueError("gate_type must be 'channel' or 'scalar'")

        self.gate_type = gate_type

    def forward(self, s: torch.Tensor, new_seq_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [N_seq, N_res, C_s] — original sequence representation
            new_seq_feature: [N_res, C_s_new] — new sequence feature to inject

        Returns:
            updated_s: [N_seq, N_res, C_s]
        """
        new_proj = self.proj(new_seq_feature)  # [N_res, C_s]

        if self.gate_type == "channel":
            gate = torch.sigmoid(self.gate_param).view(1, -1)  # [1,1,C_s]
        else:
            gate = torch.sigmoid(self.gate_param)  # scalar

        # Broadcast new feature across N_seq dimension
        gated_feature = gate * new_proj  # [1, N_res, C_s]

        return s + gated_feature


class GatedPairwiseFeatureInjector(nn.Module):
    def __init__(self, c_pair: int, c_z: int, gate_type="channel"):
        super().__init__()
        self.proj = nn.Linear(c_pair, c_z)  # project pair_feature to match z dim

        if gate_type == "channel":
            self.gate_param = nn.Parameter(torch.zeros(c_z))  # one gate per channel
        elif gate_type == "scalar":
            self.gate_param = nn.Parameter(torch.tensor(0.0))  # one global gate
        else:
            raise ValueError("gate_type must be 'channel' or 'scalar'")

        self.gate_type = gate_type

    def forward(self, z: torch.Tensor, pair_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N_res, N_res, C_z] — original pair representation
            pair_feature: [N_res, N_res, C_pair] — new feature to inject

        Returns:
            updated_z: [N_res, N_res, C_z]
        """
        pair_proj = self.proj(pair_feature)  # [N_res, N_res, C_z]

        if self.gate_type == "channel":
            gate = torch.sigmoid(self.gate_param).view(1, 1, -1)  # [1,1,C_z]
        else:  # scalar gate
            gate = torch.sigmoid(self.gate_param)  # scalar
        gated_feature = gate * pair_proj

        return z + gated_feature


if __name__ == "__main__":
    from Functions import *  # noqa

    config = load_config_from_yaml("configs/pairwise.yaml")
    model = RibonanzaNet(config).cuda()
    x = torch.ones(4, 128).long().cuda()
    mask = torch.ones(4, 128).long().cuda()
    mask[:, 120:] = 0
    print(model(x, src_mask=mask).shape)
