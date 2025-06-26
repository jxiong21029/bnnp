import copy

import torch

from bnnp import Muon
from bnnp.nn import FusedLinear


def test_muon():
    model = FusedLinear(32, 64)
    optim = Muon(model.parameters(), lr=0.1)

    before = copy.deepcopy(model.weight.data)

    model(torch.randn(32)).mean().backward()
    assert model.weight.grad.std() > 0

    optim.step()
    assert not torch.allclose(model.weight.data, before)
