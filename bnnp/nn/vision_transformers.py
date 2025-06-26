import torch
import torch.nn as nn


class RotaryNd(nn.Module):
    def __init__(self, head_dim: int, pos_dim: int, min_freq: float, max_freq: float):
        super().__init__()
        n_freqs = head_dim // 2
        # TODO: deterministic ND initialization
        freqs = torch.randn(n_freqs, pos_dim)
        freqs = freqs / freqs.pow(2).mean(dim=-1, keepdim=True).sqrt()
        freqs = freqs * (
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs)[:, None]
        )
        self.freqs_FP = nn.Buffer(freqs)

    def forward(self, x_NTD: torch.Tensor, pos_NTP: torch.Tensor) -> torch.Tensor:
        theta_NTF = (self.freqs_FP * pos_NTP[..., None, :]).mean(dim=-1)
        cos_NTF, sin_NTF = torch.cos(theta_NTF), torch.sin(theta_NTF)

        x_NTF, y_NTF = x_NTD.float().chunk(2, dim=-1)
        x_out_NTF = x_NTF * cos_NTF - y_NTF * sin_NTF
        y_out_NTF = x_NTF * sin_NTF + y_NTF * cos_NTF
        return torch.cat([x_out_NTF, y_out_NTF], dim=-1).type_as(x_NTD)
