import einops
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from bnnp.nn.patchify import extract_patches, grid_sample_batched


def grid_sample_batched_ref(input_NCHW: Tensor, grid_NLhw2: Tensor) -> Tensor:
    out = F.grid_sample(
        einops.repeat(input_NCHW, "N C H W -> (N L) C H W", L=grid_NLhw2.size(1)),
        grid_NLhw2.flatten(0, 1),
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return rearrange(out, "(N L) C h w -> N L C h w", N=input_NCHW.size(0))


def test_grid_sample_batched():
    torch.manual_seed(0)

    batch_size = 16
    channels = 3
    image_size = 128
    n_grids = 224
    grid_size = 16

    test_image = torch.randn(batch_size, channels, image_size, image_size)
    test_grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2) - 1

    out_ref = grid_sample_batched_ref(test_image, test_grid)
    out_torch = grid_sample_batched(test_image, test_grid)

    assert torch.allclose(out_ref, out_torch, atol=1e-4)


def test_extract_patches():
    image_NCHW = torch.arange(16).reshape(1, 1, 4, 4).float()
    y0_NL = torch.tensor(-0.5).view(1, 1)
    x0_NL = torch.tensor(0.0).view(1, 1)
    y1_NL = torch.tensor(0.5).view(1, 1)
    x1_NL = torch.tensor(1.0).view(1, 1)

    patch_NLChw = extract_patches(image_NCHW, y0_NL, x0_NL, y1_NL, x1_NL, h=2, w=2)
    patch_hw = patch_NLChw.squeeze((0, 1, 2))

    assert torch.allclose(patch_hw, torch.tensor([[6, 7], [10, 11]]).float())


def test_extract_patches_subpixel():
    image_NCHW = torch.zeros(2, 2)
    image_NCHW[0, 1] = 1.0
    image_NCHW = image_NCHW.reshape(1, 1, 2, 2)

    y0_NL = torch.tensor([[-0.5, -0.5]])
    x0_NL = torch.tensor([[-0.5, 0.0]])
    y1_NL = torch.tensor([[0.5, 0.5]])
    x1_NL = torch.tensor([[0.5, 1.0]])

    patch_NLChw = extract_patches(image_NCHW, y0_NL, x0_NL, y1_NL, x1_NL, h=1, w=1)
    patch_Lhw = patch_NLChw.squeeze((0, 2))

    assert torch.allclose(patch_Lhw[0], torch.tensor([[0.25]]))
    assert torch.allclose(patch_Lhw[1], torch.tensor([[0.5]]))
