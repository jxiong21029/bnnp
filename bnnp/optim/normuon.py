from typing import Literal

from .base import _MuonBase


class NorMuon(_MuonBase):
    """Muon with row/column second-moment normalization (https://arxiv.org/abs/2510.05491).

    lr_scaling targets the same update scale as Muon, adjusting for normalization via an
    additional 1 / sqrt(max(fan_out, fan_in)) factor. The normalized update scales are:
        rms: 1 / sqrt(fan_in)
        mup: sqrt(fan_out / fan_in / max(fan_out, fan_in))
        moonlight: 0.2
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        lr_scaling: Literal["rms", "mup", "moonlight"] = "rms",
    ):
        defaults = dict(
            lr=lr,
            algorithm="normuon",
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            lr_scaling=lr_scaling,
        )
        super().__init__(params, defaults)
