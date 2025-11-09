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
    for a, b, c in COEFFS[:steps]:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.mT
        X = X * (G.size(-2) / G.size(-1)) ** 0.5
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    NOTE: This implementation is intended for single-process or DDP training; not
    compatible with FSDP.

    NOTE: This optimizer should not be used for the embedding layer, the final fully
    connected layer, or any {0,1}-D parameters; those should be optimized by a standard
    method (e.g., AdamW).

    See: https://kellerjordan.github.io/posts/muon/, https://arxiv.org/abs/2409.20325
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        mu: float = 0.9,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            algorithm="muon",
            mu=mu,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            if group["algorithm"] == "adamw":
                self.adamw_step(group)
                continue
            if group["algorithm"] != "muon":
                raise ValueError(
                    f"Unknown algorithm {group['algorithm']}, expected either 'muon' or 'adamw'"
                )
            if group["ns_steps"] > len(COEFFS):
                raise ValueError(
                    f"At most {len(COEFFS)} Newton-Schulz steps are supported"
                )

            lr = group["lr"]
            mu = group["mu"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for param in group["params"]:
                g = param.grad
                if g is None:
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mu)
                g = g.lerp_(buf, mu) if nesterov else buf
                g = orthogonalize(g, steps=ns_steps, eps=eps).add_(
                    param, alpha=weight_decay
                )
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

        beta1, beta2 = group["adamw_betas"]
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
