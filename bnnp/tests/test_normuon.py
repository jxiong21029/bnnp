import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnnp.muon import Muon
from bnnp.nn import FusedLinear, RMSNorm
from bnnp.normuon import NorMuon


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.in_norm = RMSNorm(dim, affine=False)
        self.up_proj = FusedLinear(dim, mlp_dim)
        self.down_proj = FusedLinear(mlp_dim, dim)

    def forward(self, inputs):
        x = self.in_norm(inputs)
        x = self.up_proj(x)
        x = F.relu(x).square()
        x = self.down_proj(x)
        return inputs + x


def main():
    mlp = MLP(128, 512)
    mlp_orig = copy.deepcopy(mlp)
    out = FusedLinear(128, 128)
    out.weight.requires_grad_(False)

    print(f"Note: 1/sqrt(128)=={1 / 128**0.5:.6f}, 1/sqrt(512)=={1 / 512**0.5:.6f}")

    for lr_scaling in ("rms", "mup", "moonlight"):
        for optim_cls in (Muon, NorMuon):
            optim = optim_cls(
                mlp.parameters(),
                lr=0.1,
                lr_scaling=lr_scaling,
                ns_steps=25,
                weight_decay=0.0,
            )
            x = torch.randn(512, 128)
            out(mlp(x)).square().mean().backward()
            optim.step()
            print(
                f"{lr_scaling=}, {optim_cls.__name__}, "
                f"up_proj delta RMS: {(mlp.up_proj.weight.data - mlp_orig.up_proj.weight.data).square().mean().sqrt():.6f}, "
                f"down proj delta RMS: {(mlp.down_proj.weight.data - mlp_orig.down_proj.weight.data).square().mean().sqrt():.6f}"
            )

            mlp = copy.deepcopy(mlp_orig)


if __name__ == "__main__":
    main()
