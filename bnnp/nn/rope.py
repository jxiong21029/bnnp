import torch
import torch.nn as nn
from torch import Tensor


class RoPE1d(nn.Module):
    def __init__(
        self,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float,
    ):
        super().__init__()
        self.n_freqs = head_dim // 2
        omega = min_freq * (max_freq / min_freq) ** torch.linspace(
            0, 1, round(self.n_freqs * (1 - p_zero_freqs))
        )
        omega = torch.cat((torch.zeros(self.n_freqs - len(omega)), omega))
        self.register_buffer("freqs_F", omega)

    def precompute_rotations(self, pos: Tensor):
        """Expects pos with shape either (B, T) or (T,) and returns cos, sin each with shape either (B, T, 1, F) or (T, 1, F)"""
        theta = pos[..., None, None].float() * self.freqs_F
        return torch.cos(theta), torch.sin(theta)

    def forward(self, input_BThd: Tensor, cos: Tensor, sin: Tensor):
        """Expects cos/sin with shapes either (B, T) or (T,)"""
        dtype = input_BThd.dtype
        x_BThF, y_BThF = input_BThd.chunk(2, dim=-1)
        x_out_BThF = x_BThF * cos.to(dtype) - y_BThF * sin.to(dtype)
        y_out_BThF = x_BThF * sin.to(dtype) + y_BThF * cos.to(dtype)
        return torch.cat((x_out_BThF, y_out_BThF), dim=-1)


class GGRoPENd(nn.Module):
    """Golden Gate RoPE for N>=2 dimensions (including the special case for N=2).

    See: https://jerryxio.ng/posts/nd-rope.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        pos_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float,
    ):
        super().__init__()
        self.n_freqs = head_dim // 2  # total number of frequencies (incl. 0)
        omega = min_freq * (max_freq / min_freq) ** torch.linspace(
            0, 1, round(self.n_freqs * (1 - p_zero_freqs))
        )
        omega_F = torch.cat((torch.zeros(self.n_freqs - len(omega)), omega))
        directions_hFP = self.make_directions(n_heads * self.n_freqs, pos_dim).view(
            n_heads, self.n_freqs, pos_dim
        )
        self.register_buffer("freqs_hFP", directions_hFP * omega_F[None, :, None])

    @property
    def pos_dim(self) -> int:
        return self.freqs_hFP.size(-1)

    @staticmethod
    def make_directions(n: int, d: int):
        def _phi(m: int):
            x = 2.0
            for _ in range(20):
                x = (1 + x) ** (1.0 / (m + 1.0))
            return x

        if d == 2:
            angles = torch.arange(n, dtype=torch.float64) * (torch.pi / _phi(1))
            directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
            return directions.float()
        else:
            g = _phi(d)
            alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
            i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
            z = torch.fmod(i * alpha, 1.0)
            directions = torch.erfinv(2.0 * z - 1.0)
            directions = directions / directions.norm(dim=1, keepdim=True)
            return directions.float()

    def precompute_rotations(self, pos: Tensor):
        """Expects pos with shape either (B, T, P) or (T, P)"""
        theta = (pos[..., None, None, :].float() * self.freqs_hFP).sum(dim=-1)
        return torch.cos(theta), torch.sin(theta)

    def forward(self, input_BThd: Tensor, cos: Tensor, sin: Tensor):
        dtype = input_BThd.dtype
        x_BThF, y_BThF = input_BThd.chunk(2, dim=-1)
        x_out_BThF = x_BThF * cos.to(dtype) - y_BThF * sin.to(dtype)
        y_out_BThF = x_BThF * sin.to(dtype) + y_BThF * cos.to(dtype)
        return torch.cat((x_out_BThF, y_out_BThF), dim=-1)
