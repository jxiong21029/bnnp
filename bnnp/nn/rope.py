import torch
import torch.nn as nn
from torch import Tensor


class _RoPEBase(nn.Module):
    _freqs_name: str

    def _apply(self, fn, recurse=True):
        freqs = getattr(self, self._freqs_name)

        def convert(tensor):
            converted = fn(tensor)
            if tensor is freqs and converted.dtype != torch.float32:
                # Move the original values: casting converted back would retain rounding errors.
                if tensor.is_meta:
                    return converted.float()
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(convert, recurse=recurse)


class RoPE1d(_RoPEBase):
    _freqs_name = "freqs_F"

    def __init__(
        self,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float,
        device=None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        self.n_freqs = head_dim // 2
        freqs = min_freq * (max_freq / min_freq) ** torch.linspace(
            0, 1, round(self.n_freqs * (1 - p_zero_freqs)), device=device, dtype=torch.float32
        )
        freqs = torch.cat(
            (torch.zeros(self.n_freqs - len(freqs), device=device, dtype=torch.float32), freqs)
        )
        self.freqs_F = nn.Buffer(freqs)

    def precompute_rotations(self, seq_len: int | None = None, positions: Tensor | None = None):
        """Expects pos with shape either (B, N) or (N,) and returns cos, sin each with shape either (B, N, 1, F) or (N, 1, F)"""
        if seq_len is not None:
            if positions is not None:
                raise ValueError("Must specify exactly one of seq_len and positions")
            positions = torch.arange(seq_len, device=self.freqs_F.device)
        elif positions is None:
            raise ValueError("Must specify exactly one of seq_len and positions")

        theta = positions[..., None, None].float() * self.freqs_F
        return torch.cos(theta), torch.sin(theta)

    def forward(self, input_BThd: Tensor, rotations: tuple[Tensor, Tensor]):
        x_BNhF, y_BNhF = input_BThd.chunk(2, dim=-1)
        cos, sin = rotations
        cos, sin = cos.type_as(input_BThd), sin.type_as(input_BThd)
        x_out_BNhF = x_BNhF * cos - y_BNhF * sin
        y_out_BNhF = x_BNhF * sin + y_BNhF * cos
        return torch.cat((x_out_BNhF, y_out_BNhF), dim=-1)


class RoPENd(_RoPEBase):
    """RoPE for N>=2 dimensions with initialization following https://jerryxio.ng/posts/nd-rope."""

    _freqs_name = "freqs_hFP"

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        pos_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float,
        device=None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        self.n_freqs = head_dim // 2  # total number of frequencies (incl. 0)
        omega = min_freq * (max_freq / min_freq) ** torch.linspace(
            0, 1, round(self.n_freqs * (1 - p_zero_freqs)), device=device, dtype=torch.float32
        )
        omega_F = torch.cat(
            (torch.zeros(self.n_freqs - len(omega), device=device, dtype=torch.float32), omega)
        )
        directions_hFP = self.make_directions(n_heads * self.n_freqs, pos_dim, device=device).view(
            n_heads, self.n_freqs, pos_dim
        )
        self.freqs_hFP = nn.Buffer(directions_hFP * omega_F[None, :, None])

    @property
    def pos_dim(self) -> int:
        return self.freqs_hFP.size(-1)

    @staticmethod
    def make_directions(n: int, d: int, device=None):
        def _phi(m: int):
            x = 2.0
            for _ in range(20):
                x = (1 + x) ** (1.0 / (m + 1.0))
            return x

        if d == 2:
            angles = torch.arange(n, device=device, dtype=torch.float64) * (torch.pi / _phi(1))
            directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
            return directions.float()
        else:
            g = _phi(d)
            alpha = (1.0 / g) ** torch.arange(1, d + 1, device=device, dtype=torch.float64)
            i = torch.arange(1, n + 1, device=device, dtype=torch.float64).unsqueeze(1)
            z = torch.fmod(i * alpha, 1.0)
            directions = torch.erfinv(2.0 * z - 1.0)
            directions = directions / directions.norm(dim=1, keepdim=True)
            return directions.float()

    def precompute_rotations(
        self,
        size: tuple[int, int] | None = None,
        n_extra: int = 0,
        base_hw: tuple[int, int] | None = None,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Precompute cos/sin tables for either the provided positions or for a 2d input of
        a given size, plus an optional prefix of n_extra tokens treated as if they were
        located at (0, 0).
        """
        device = self.freqs_hFP.device

        if positions is None:
            if size is None:
                raise ValueError("Must provide exactly one of size and positions")
            if self.pos_dim != 2:
                raise ValueError("Must pass positions to precompute_rotations for pos_dim != 2")
            h, w = size
            ylim, xlim = (h / w) ** 0.5, (w / h) ** 0.5
            y_H1 = torch.linspace(-ylim, ylim, h, device=device, dtype=torch.float32)[:, None]
            x_1W = torch.linspace(-xlim, xlim, w, device=device, dtype=torch.float32)[None, :]
            y_HW, x_HW = y_H1.expand(h, w), x_1W.expand(h, w)
            pos_NP = torch.stack([x_HW, y_HW], dim=-1).flatten(0, 1)
            pos_NP = torch.cat([pos_NP.new_zeros(n_extra, self.pos_dim), pos_NP], dim=0)
            if base_hw is None:
                freqs = self.freqs_hFP
            else:
                # Smallest frequencies are left unchanged, largest are scaled propto sqrt(h * w)
                base_h, base_w = base_hw
                max_scale = (h * w / base_h / base_w) ** 0.5
                scale_frac = torch.linspace(
                    0, 1, self.freqs_hFP.shape[1], device=device, dtype=torch.float32
                )
                freqs = self.freqs_hFP * (max_scale ** scale_frac[:, None])
        else:
            if size is not None:
                raise ValueError("Must specify exactly one of size and positions")
            pos_NP = torch.cat([positions.new_zeros(n_extra, self.pos_dim), positions], dim=0)
            freqs = self.freqs_hFP
        theta_NhF = (pos_NP[:, None, None, :].float() * freqs).sum(dim=-1)

        return theta_NhF.cos(), theta_NhF.sin()

    def forward(self, input_BThd: Tensor, rotations: tuple[Tensor, Tensor]):
        x_BNhF, y_BNhF = input_BThd.chunk(2, dim=-1)
        cos, sin = rotations  # cos, sin can be any shape that broadcasts with BNhF
        cos, sin = cos.type_as(input_BThd), sin.type_as(input_BThd)
        x_out_BNhF = x_BNhF * cos - y_BNhF * sin
        y_out_BNhF = x_BNhF * sin + y_BNhF * cos
        return torch.cat((x_out_BNhF, y_out_BNhF), dim=-1)
