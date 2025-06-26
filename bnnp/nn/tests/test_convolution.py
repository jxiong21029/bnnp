import torch

from bnnp.nn import Conv2d, Conv3d


def test_conv2d():
    conv = Conv2d(3, 64, kernel_size=3, padding="same")
    assert conv.weight.shape == (64, 3 * 9)

    output = conv(torch.randn(4, 3, 16, 16))
    assert output.shape == (4, 64, 16, 16)
    assert output.std() >= 0


def test_conv3d():
    conv = Conv3d(3, 64, kernel_size=3, padding="same")
    assert conv.weight.shape == (64, 3 * 27)

    output = conv(torch.randn(4, 3, 8, 16, 16))
    assert output.shape == (4, 64, 8, 16, 16)
    assert output.std() >= 0
