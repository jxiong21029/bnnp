import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.relu(x).square()


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, activation=ReLU2):
        super().__init__()
        self.mlp = nn.Sequential(
            RMSNorm(dim, affine=False),
            FusedLinear(dim, mlp_dim),
            activation(),
            FusedLinear(mlp_dim, dim, zero_init=True, scale=True),
        )

    def forward(self, x):
        return x + self.mlp(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine=False):
        super().__init__()
        self.dim = dim
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.dim, (
            f"expected x.size(-1) == {self.dim} but got {x.size(-1)=}"
        )
        return F.rms_norm(x.float(), (x.size(-1),), self.weight).type_as(x)


class FusedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale: bool = False,
        zero_init: bool = False,
        gain: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features) * gain
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        if scale:
            self.scale = nn.Parameter(torch.ones(out_features))
            if zero_init:
                self.scale.data.zero_()
        else:
            self.register_parameter("scale", None)
            if zero_init:
                self.weight.data.zero_()

    def forward(self, x: torch.Tensor):
        if self.scale is not None:
            # Fused per-channel scaling
            weight = self.weight * self.scale.unsqueeze(1)
        else:
            weight = self.weight

        out = x @ weight.type_as(x).t()

        if self.bias is not None:
            out = out + self.bias.type_as(out)
        return out


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, gain=0.5, dtype=None):
        super().__init__(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=dtype
        )
        self.weight._is_embed = True
        self.weight.data.mul_(gain)


class Output(nn.Module):
    def __init__(self, dim: int, out_dim: int, pad_to: int | None = None):
        super().__init__()
        pad_to = pad_to if pad_to is not None else out_dim
        padded_out_dim = (out_dim + pad_to - 1) // pad_to * pad_to
        self.out_dim = out_dim

        self.norm = RMSNorm(dim)
        self.linear = FusedLinear(dim, padded_out_dim, zero_init=True)
        self.linear.weight._is_output = True

    def forward(self, x):
        return self.linear(self.norm(x))[..., : self.out_dim]
