import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from bnnp.nn import FusedLinear, ReLU2, RMSNorm, RoPE1d


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, activation=ReLU2):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.up_proj = FusedLinear(dim, mlp_dim)
        self.act = activation()
        self.down_proj = FusedLinear(mlp_dim, dim, zero_init=True)

    def forward(self, x):
        return x + self.down_proj(self.act(self.up_proj(self.norm(x))))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, head_dim: int, rope: RoPE1d | None):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.nheads = dim // head_dim

        self.norm = RMSNorm(dim, affine=False)
        self.qkv_weight = nn.Parameter(torch.randn(3, dim, dim) / dim**0.5 / 2)
        self.qk_norm = RMSNorm(head_dim, affine=False)
        self.rope = rope
        self.o_proj = FusedLinear(dim, dim, zero_init=True)

    def forward(
        self,
        input_NTD,
        rotations: tuple[Tensor, Tensor] | None = None,
        block_mask: BlockMask | None = None,
    ):
        N, T, _ = input_NTD.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_NTD)
        q_NThd, k_NThd, v_NThd = (
            (self.norm(input_NTD) @ qkv_weight.t())
            .view(N, T, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_NThd, k_NThd = self.qk_norm(q_NThd), self.qk_norm(k_NThd)
        if rotations is None:
            assert self.rope is None
        else:
            q_NThd = self.rope(q_NThd, *rotations)
            k_NThd = self.rope(k_NThd, *rotations)
        if block_mask is None:
            x_NThd = F.scaled_dot_product_attention(
                q_NThd.transpose(1, 2),
                k_NThd.transpose(1, 2),
                v_NThd.transpose(1, 2),
            ).transpose(1, 2)
        else:
            x_NThd = flex_attention(
                q_NThd.transpose(1, 2),
                k_NThd.transpose(1, 2),
                v_NThd.transpose(1, 2),
                block_mask=block_mask,
            ).transpose(1, 2)
        return input_NTD + self.o_proj(x_NThd.flatten(2, 3))

    def predict(
        self,
        input_ND,
        rotations: tuple[Tensor, Tensor] | None,
        past_kv: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        N, _ = input_ND.shape
        qkv_weight = self.qkv_weight.flatten(0, 1).type_as(input_ND)
        q_Nhd, k_Nhd, v_Nhd = (
            (self.norm(input_ND) @ qkv_weight.t())
            .view(N, 3 * self.nheads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q_Nhd, k_Nhd = self.qk_norm(q_Nhd), self.qk_norm(k_Nhd)
        if self.rope is not None:
            q_Nhd = self.rope(q_Nhd, *rotations)
            k_Nhd = self.rope(k_Nhd, *rotations)
        if past_kv is not None:
            past_k_NhTd, past_v_NhTd = past_kv
            k_NhTd = torch.cat([past_k_NhTd, k_Nhd.unsqueeze(2)], dim=2)
            v_NhTd = torch.cat([past_v_NhTd, v_Nhd.unsqueeze(2)], dim=2)
        else:
            k_NhTd = k_Nhd.unsqueeze(2)
            v_NhTd = v_Nhd.unsqueeze(2)
        x_Nh1d = F.scaled_dot_product_attention(q_Nhd.unsqueeze(2), k_NhTd, v_NhTd)
        x_ND = x_Nh1d.flatten(1, 3)
        return input_ND + self.o_proj(x_ND), (k_NhTd, v_NhTd)


class Decoder(nn.Module):
    """A baseline causal transformer decoder with RoPE and sliding window attention."""

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        head_dim: int,
        depth: int,
        window_size: int,
        use_rope: bool,
        rotary_min_freq: float,
        rotary_max_freq: float,
    ):
        super().__init__()
        self.window_size = window_size

        self.rope = (
            RoPE1d(
                head_dim=head_dim,
                min_freq=rotary_min_freq,
                max_freq=rotary_max_freq,
                p_zero_freqs=0.0,
            )
            if use_rope
            else None
        )

        self.cos = nn.Buffer()
        self.sin = nn.Buffer()
        self.block_mask = None

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleDict()
            block["attn"] = SelfAttention(
                dim=dim, mlp_dim=mlp_dim, head_dim=head_dim, rope=self.rope
            )
            block["mlp"] = MLPBlock(dim=dim, mlp_dim=mlp_dim)
            self.blocks.append(block)

    def precompute_caches(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.rope is not None:
            cos, sin = self.rope.precompute_rotations(
                torch.arange(seq_len, device=device)
            )
            self.cos = cos.to(dtype)
            self.sin = sin.to(dtype)

        def sliding_window_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx < kv_idx + self.window_size)

        self.block_mask = create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
        )

    def forward(self, x_NTD: Tensor):
        """Training forward (fixed seq_len, no kv_cache)"""
        assert self.block_mask is not None, "Must call precompute_caches before forward"
        rotations = (
            (self.cos.type_as(x_NTD), self.sin.type_as(x_NTD))
            if self.rope is not None
            else None
        )
        for block in self.blocks:
            x_NTD = block["attn"](
                x_NTD, rotations=rotations, block_mask=self.block_mask
            )
            x_NTD = block["mlp"](x_NTD)
        return x_NTD

    def predict(self, x_ND, cache: tuple[int, list[tuple[Tensor, Tensor]]] | None):
        """Inference (one input token, optional kv_cache). Returns new kv_cache."""
        if cache is None:
            t = 0
            past_key_values = [None for _ in range(len(self.blocks))]
        else:
            t, past_key_values = cache

        if self.rope is not None:
            cos, sin = self.rope.precompute_rotations(
                torch.tensor([t], device=x_ND.device)
            )
            rotations = (cos.type_as(x_ND), sin.type_as(x_ND))
        else:
            rotations = None

        new_key_values = []
        for block, past_kv in zip(self.blocks, past_key_values):
            x_ND, (new_k, new_v) = block["attn"].predict(
                x_ND, rotations=rotations, past_kv=past_kv
            )
            start = max(0, new_k.size(2) - self.window_size + 1)
            new_key_values.append((new_k[:, :, start:], new_v[:, :, start:]))

            x_ND = block["mlp"](x_ND)

        return x_ND, (t + 1, new_key_values)
