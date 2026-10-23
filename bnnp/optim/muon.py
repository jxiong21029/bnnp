from typing import Literal

from .base import _MuonBase


class Muon(_MuonBase):
    """Momentum orthogonalized by Newton-Schulz.

    betas: Muon uses betas[0] for momentum; AdamW groups use both values.

    lr_scaling:
        rms: Keller Jordan's original sqrt(max(1, fan_out / fan_in)) scale
        mup: sqrt(fan_out / fan_in)
        moonlight: 0.2 * sqrt(max(fan_out, fan_in)) scale

    weight decay uses the unscaled learning rate in every mode.
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
            algorithm="muon",
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            lr_scaling=lr_scaling,
        )
        super().__init__(params, defaults)
