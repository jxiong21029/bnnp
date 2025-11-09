import copy
import os

import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from bnnp import DistMuon, Muon
from bnnp.nn import FusedLinear, RMSNorm


def test_muon():
    model = FusedLinear(32, 64)
    optim = Muon(model.parameters(), lr=0.1)

    before = copy.deepcopy(model.weight.data)

    model(torch.randn(32)).mean().backward()
    assert model.weight.grad.std() > 0

    optim.step()
    assert not torch.allclose(model.weight.data, before)


def test_distmuon():
    model = FusedLinear(32, 64)
    optim = DistMuon(model.parameters(), lr=0.1)

    before = copy.deepcopy(model.weight.data)

    model(torch.randn(32)).mean().backward()
    assert model.weight.grad.std() > 0

    optim.step()
    assert not torch.allclose(model.weight.data, before)


def train_muon_vs_distmuon():
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
        assert world_size > 1

        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)
        torch.distributed.init_process_group(backend="nccl", device_id=device)

        B, D = 64, 128
        raw_model = MyModel(dim=D, mlp_dim=2 * D, depth=2).to(device)
        init_state = copy.deepcopy(raw_model.state_dict())
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

        results = []
        for i in range(2):
            raw_model.load_state_dict(init_state)
            if i == 0:
                optim = Muon(model.parameters(), lr=0.01, mu=0.9, weight_decay=0.01)
            else:
                optim = DistMuon(
                    model.parameters(),
                    distributed_mesh=model.process_group,
                    lr=0.01,
                    mu=0.9,
                    weight_decay=0.01,
                )
                assert optim._device_rank == local_rank == rank
                assert optim._world_size == world_size

            rng = torch.Generator(device)
            rng.manual_seed(42 + rank)
            losses = []
            for step in (pbar := tqdm.trange(32)):
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

            results.append(torch.stack(losses))
            del optim, loss, rng

        print((results[0] - results[1]).abs().mean())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_muon_vs_distmuon()
