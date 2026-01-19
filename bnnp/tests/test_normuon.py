import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from bnnp.distmuon.normuon import DistNorMuon
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


def train_normuon():
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


def train_normuon_vs_distnormuon():
    try:

        class MyModel(nn.Module):
            def __init__(self, dim: int, mlp_dim: int, depth: int):
                super().__init__()
                self.blocks = nn.ModuleList()
                for _ in range(depth):
                    self.blocks.append(
                        nn.Sequential(
                            RMSNorm(dim),
                            FusedLinear(dim, mlp_dim),
                            nn.SiLU(),
                            FusedLinear(mlp_dim, dim),
                        )
                    )
                self.out_head = FusedLinear(dim, dim)

            def forward(self, x: Tensor):
                for block in self.blocks:
                    x = x + block(x)
                return self.out_head(x)

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)
        torch.distributed.init_process_group(backend="nccl", device_id=device)

        B, D = 64, 128
        raw_model = MyModel(dim=D, mlp_dim=2 * D, depth=2).to(device)
        raw_copy = copy.deepcopy(raw_model)
        init_state = copy.deepcopy(raw_model.state_dict())
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

        results = []
        for i in range(2):
            raw_model.load_state_dict(init_state)
            if i == 0:
                optim = NorMuon(
                    model.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.01
                )
            else:
                optim = DistNorMuon(
                    model.parameters(),
                    distributed_mesh=model.process_group,
                    lr=0.01,
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                )
                assert optim._device_rank == local_rank == rank
                assert optim._world_size == world_size

            rng = torch.Generator(device)
            rng.manual_seed(42 + rank)
            losses = []
            for step in (
                pbar := tqdm.trange(32, ncols=88, desc=optim.__class__.__name__)
            ):
                x = torch.randn(B, D, device=device, generator=rng)
                y = (
                    -0.7 * x
                    + 0.2
                    + 0.3 * torch.randn(B, D, device=device, generator=rng)
                )
                loss = (model(x) - y).pow(2).mean()
                loss.backward()
                losses.append(loss.detach().clone())
                optim.step()
                optim.zero_grad(set_to_none=True)
                pbar.set_postfix(dict(loss=loss.item()))

                if step == 1:
                    print(
                        f"Up proj 1 delta RMS: {(raw_model.blocks[0][1].weight.data - raw_copy.blocks[0][1].weight.data).square().mean().sqrt():.6f}, "
                        f"Down proj 1 delta RMS: {(raw_model.blocks[0][3].weight.data - raw_copy.blocks[0][3].weight.data).square().mean().sqrt():.6f}, "
                        f"Up proj 2 delta RMS: {(raw_model.blocks[1][1].weight.data - raw_copy.blocks[1][1].weight.data).square().mean().sqrt():.6f}, "
                        f"Down proj 2 delta RMS: {(raw_model.blocks[1][3].weight.data - raw_copy.blocks[1][3].weight.data).square().mean().sqrt():.6f}, "
                    )

            results.append(torch.stack(losses))
            del optim, loss, rng

        print((results[0] - results[1]).abs().mean())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_normuon_vs_distnormuon()
