from typing import Literal

import torch
from torch.optim.adam import adam

# Newton-schulz coefficients from the Polar Express (https://arxiv.org/abs/2505.16932)
COEFFS = [
    (8.205160, -22.901935, 16.460725),
    (4.066395, -2.861154, 0.518400),
    (3.909595, -2.823352, 0.525037),
    (3.285564, -2.415302, 0.485294),
    (2.277873, -1.619822, 0.398481),
    (1.872576, -1.230704, 0.358516),
    (1.856437, -1.213239, 0.356800),
    (1.856436, -1.213238, 0.356800),
]


def orthogonalize(G: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Computes the semi-orthogonalization of G via Newton-Schulz iteration."""
    assert G.ndim >= 2, "Newton-Schulz expects at least 2D tensor"
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
    for i in range(steps):
        a, b, c = COEFFS[min(i, len(COEFFS) - 1)]
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.mT
    return X.type_as(G)


class NorMuon(torch.optim.Optimizer):
    """See: https://arxiv.org/abs/2510.05491."""

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
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if lr_scaling not in ("rms", "mup", "moonlight"):
            raise ValueError(
                f"Invalid lr_scaling value: {lr_scaling}. Must be 'rms', 'mup', or 'moonlight'."
            )
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

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["algorithm"] == "adamw":
                self.adamw_step(group)
                continue
            if group["algorithm"] != "normuon":
                raise ValueError(
                    f"Unknown algorithm {group['algorithm']}, expected either 'normuon' or 'adamw'"
                )

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            lr_scaling = group["lr_scaling"]

            for param in group["params"]:
                g = param.grad
                if g is None:
                    continue

                state = self.state[param]
                reduce_dim = -1 if param.size(-1) < param.size(-2) else -2
                d_out, d_in = g.size(-2), g.size(-1)
                if "m_buffer" not in state:
                    state["m_buffer"] = torch.zeros_like(g)

                    if reduce_dim == -2:
                        reduced_shape = g.shape[:-2] + (1, d_in)
                    else:
                        reduced_shape = g.shape[:-2] + (d_out, 1)
                    state["v_buffer"] = g.new_zeros(reduced_shape)
                    state["step"] = torch.tensor(
                        0.0, device=g.device, dtype=torch.float32
                    )

                m = state["m_buffer"]
                m.lerp_(g, 1 - betas[0])

                g = g.lerp_(m, betas[0]) if nesterov else m
                g = orthogonalize(g, steps=ns_steps, eps=eps)

                v = state["v_buffer"]
                v.lerp_(g.square().mean(dim=reduce_dim, keepdim=True), 1 - betas[1])
                state["step"] += 1
                correction = 1 - betas[1] ** (state["step"])

                g.div_((v / correction).sqrt_().add_(1e-8))
                if lr_scaling == "rms":
                    g.mul_(d_in**-0.5)
                elif lr_scaling == "mup":
                    g.mul_((d_out / d_in / max(d_in, d_out)) ** 0.5)
                elif lr_scaling != "moonlight":
                    raise ValueError(f"unknown value for lr_scaling: {lr_scaling}")

                g = g.add_(param, alpha=weight_decay)
                param.sub_(g, alpha=lr)

    def adamw_step(self, group):
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        for param in group["params"]:
            if param.grad is None:
                continue

            params.append(param)
            grads.append(param.grad)
            state = self.state[param]
            if "step" not in state:
                state["step"] = torch.tensor(0.0)  # Host on CPU.
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

        beta1, beta2 = group["betas"]
        adam(
            params=params,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=[],
            state_steps=state_steps,
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            decoupled_weight_decay=True,  # AdamW
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )
