import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mpparam(*size, gain=0.5, device=None, dtype=None):
    """Initialize a weight which maps RMS norm 1 inputs to RMS norm `gain` inputs."""
    return nn.Parameter(
        torch.randn(*size, device=device, dtype=dtype) * (gain / math.sqrt(size[-1]))
    )


class ReLU2(nn.Module):
    def forward(self, x: Tensor):
        return F.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine=False, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        if affine:
            self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor):
        assert x.size(-1) == self.dim, f"expected x.size(-1) == {self.dim} but got {x.size(-1)=}"
        return F.rms_norm(
            x.float(), (x.size(-1),), weight=None if self.weight is None else (self.weight + 1)
        ).type_as(x)


class Linear(nn.Module):
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

        self.weight = mpparam(out_features, in_features, gain=gain, device=device, dtype=dtype)

        if scale:
            self.scale = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
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
    def __init__(self, num_embeddings: int, embedding_dim: int, gain=0.5, device=None, dtype=None):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.weight.data.mul_(gain / embedding_dim**0.5)

    def forward(self, input: Tensor):
        output = super().forward(input)
        return output * (self.embedding_dim**0.5)
