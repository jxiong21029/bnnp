import torch
import torch.nn as nn

from bnnp.nn import mpparam


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mpparam(out_features, in_features)

    def forward(self, x):
        return x @ self.weight.T


def test_mpparam():
    linear = CustomLinear(4, 8)
    assert next(linear.parameters()).shape == (8, 4)

    linear(torch.randn(4)).mean().backward()
    assert linear.weight.grad is not None
    assert linear.weight.grad.pow(2).sum() > 0
