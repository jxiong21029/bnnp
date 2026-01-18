import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mpparam(*size, gain=0.5, device=None, dtype=None, group=None):
    """Initialize a weight which maps RMS norm 1 inputs to RMS norm `gain` inputs."""
    param = nn.Parameter(
        torch.randn(*size, device=device, dtype=dtype) * (gain / math.sqrt(size[-1]))
    )
    if group is not None:
        param._group = group  # ty: ignore
    return param


class ReLU2(nn.Module):
    def forward(self, x: Tensor):
        return F.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine=False, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        if affine:
            self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
            self.weight._group = "scalar"  # ty: ignore
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor):
        assert x.size(-1) == self.dim, (
            f"expected x.size(-1) == {self.dim} but got {x.size(-1)=}"
        )
        return F.rms_norm(
            x.float(),
            (x.size(-1),),
            None if self.weight is None else (self.weight + 1),
        ).type_as(x)


class FusedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: bool = False,
        zero_init: bool = False,
        gain: float = 0.5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = mpparam(
            out_features, in_features, gain=gain, device=device, dtype=dtype
        )

        if scale:
            self.scale = nn.Parameter(
                torch.ones(out_features, device=device, dtype=dtype)
            )
            if zero_init:
                self.scale.data.zero_()
        else:
            self.register_parameter("scale", None)
            if zero_init:
                self.weight.data.zero_()

    def forward(self, x: Tensor):
        if self.scale is not None:
            # Fused per-channel scaling
            weight = self.weight * self.scale.unsqueeze(1)
        else:
            weight = self.weight

        return x @ weight.type_as(x).t()


class Embedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, gain=0.5, device=None, dtype=None
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.weight._group = "embed"  # ty: ignore
        self.weight.data.mul_(gain)


class Output(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, pad_to: int | None = None, device=None, dtype=None
    ):
        super().__init__()
        if pad_to is not None:
            padded_out_dim = (out_dim + pad_to - 1) // pad_to * pad_to
        else:
            padded_out_dim = out_dim
        self.out_dim = out_dim

        self.norm = RMSNorm(dim, device=device, dtype=dtype)
        self.linear = FusedLinear(
            dim, padded_out_dim, zero_init=True, device=device, dtype=dtype
        )
        self.linear.weight._group = "low_rank"

    def forward(self, x):
        return self.linear(self.norm(x))[..., : self.out_dim]
