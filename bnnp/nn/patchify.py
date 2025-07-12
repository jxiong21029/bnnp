import torch
from einops import rearrange
from torch import Tensor


def grid_sample_batched(input_NCHW: Tensor, grid_NLhw2: Tensor) -> Tensor:
    N, C, H, W = input_NCHW.shape
    _, L, h, w, _ = grid_NLhw2.shape
    x, y = grid_NLhw2.unbind(-1)

    y = y * H / 2 + (H - 1) / 2
    x = x * W / 2 + (W - 1) / 2

    y0 = torch.floor(y)
    x0 = torch.floor(x)
    y1 = y0 + 1.0
    x1 = x0 + 1.0
    wy1 = y - y0
    wy0 = 1 - wy1
    wx1 = x - x0
    wx0 = 1 - wx1
    y0i = y0.clamp(0, H - 1).type(torch.int64)
    y1i = y1.clamp(0, H - 1).type(torch.int64)
    x0i = x0.clamp(0, W - 1).type(torch.int64)
    x1i = x1.clamp(0, W - 1).type(torch.int64)

    w00 = (wy0 * wx0).unsqueeze(1)
    w01 = (wy0 * wx1).unsqueeze(1)
    w10 = (wy1 * wx0).unsqueeze(1)
    w11 = (wy1 * wx1).unsqueeze(1)

    idx00 = y0i * W + x0i
    idx01 = y0i * W + x1i
    idx10 = y1i * W + x0i
    idx11 = y1i * W + x1i

    idx00 = rearrange(idx00, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx01 = rearrange(idx01, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx10 = rearrange(idx10, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    idx11 = rearrange(idx11, "N L h w -> N 1 (L h w)").expand(N, C, -1)
    flat_input = input_NCHW.reshape(N, C, H * W)

    output = flat_input.gather(-1, idx00).reshape(N, C, L, h, w).mul_(w00)
    output.addcmul_(flat_input.gather(-1, idx01).reshape(N, C, L, h, w), w01)
    output.addcmul_(flat_input.gather(-1, idx10).reshape(N, C, L, h, w), w10)
    output.addcmul_(flat_input.gather(-1, idx11).reshape(N, C, L, h, w), w11)
    return output.transpose(1, 2)


def extract_patches(
    images_NCHW: Tensor,
    y0_NL: Tensor,
    x0_NL: Tensor,
    y1_NL: Tensor,
    x1_NL: Tensor,
    h: int,
    w: int,
):
    """Extracts patches from batch of images using area-style intepolation.

    Expects coordinates (y0, x0, y1, x1) normalized to [-1.0, 1.0].

    N: batch size
    L: number of patches per image
    h: resized patch height, in pixels
    w: resized patch width, in pixels

    Returns patches of shape (N, L, C, h, w).
    """
    assert torch.is_floating_point(images_NCHW)
    N, C, H, W = images_NCHW.shape
    _, L = y0_NL.shape
    assert y0_NL.shape == y1_NL.shape == x0_NL.shape == x1_NL.shape == (N, L)
    device = images_NCHW.device

    offsets_y = torch.linspace(0, 1, h + 1, device=device)
    offsets_x = torch.linspace(0, 1, w + 1, device=device)
    ys_NLh = y0_NL[..., None] + (y1_NL - y0_NL)[..., None] * offsets_y
    xs_NLw = x0_NL[..., None] + (x1_NL - x0_NL)[..., None] * offsets_x

    # Multiply by H / (H + 1) or W / (W + 1) for y and x, respectively, in order to
    # convert from "coordinates of the corner in the image (which has size H x W)" to
    # "coordinates of the pixel center in the integral (which has size H+1 x W+1)".
    ys_NLhw = (ys_NLh * H / (H + 1)).view(N, L, h + 1, 1).expand(N, L, h + 1, w + 1)
    xs_NLhw = (xs_NLw * W / (W + 1)).view(N, L, 1, w + 1).expand(N, L, h + 1, w + 1)
    grid_xy_NLhw2 = torch.stack([xs_NLhw, ys_NLhw], dim=-1)

    integral_NCHW = torch.zeros((N, C, H + 1, W + 1), device=images_NCHW.device)
    integral_NCHW[:, :, 1:, 1:] = images_NCHW.cumsum(dim=-1).cumsum(dim=-2)

    integral_samples_NLChw = grid_sample_batched(
        integral_NCHW, grid_xy_NLhw2.float()
    ).reshape(N, L, C, h + 1, w + 1)

    i00 = integral_samples_NLChw[..., :-1, :-1]
    i01 = integral_samples_NLChw[..., :-1, 1:]
    i10 = integral_samples_NLChw[..., 1:, :-1]
    i11 = integral_samples_NLChw[..., 1:, 1:]

    # Box-sum via 4-corner trick, then divide by pixel area
    sums = i11 - i01 - i10 + i00
    area = (y1_NL - y0_NL) * (x1_NL - x0_NL) * (H * W / h / w / 4)
    patches = sums / area.view(N, L, 1, 1, 1)
    return patches.type_as(images_NCHW)
