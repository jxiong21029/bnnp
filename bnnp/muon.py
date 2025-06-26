import logging

import torch
import torch.nn as nn

ortho_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)

log = logging.getLogger(__name__)

# See: https://arxiv.org/abs/2505.16932
COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
COEFFS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in COEFFS[:-1]] + COEFFS[-1:]


def orthogonalize(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Computes the semi-orthogonalization of G."""
    assert G.ndim >= 2
    X = G.type(ortho_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01)

    for step in range(steps):
        a, b, c = COEFFS[min(step, len(COEFFS) - 1)]
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X

    if G.size(-2) > G.size(-1):
        # Scale to ensure that the norm of the ROWS of G (i.e. change in output) is 1
        X = X.mT * (G.size(-2) / G.size(-1)) ** 0.5
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    See: https://kellerjordan.github.io/posts/muon/, https://arxiv.org/abs/2409.20325

    NOTE: This optimizer should not be used for the embedding layer, the final fully
    connected layer, or any {0,1}-D parameters; those should be optimized by a standard
    method (e.g., AdamW).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for param in group["params"]:
                g = param.grad
                if g is None:
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = orthogonalize(g, steps=ns_steps).add_(param, alpha=weight_decay)
                param.sub_(g, alpha=lr)


def auto_split_muon_params(
    module: nn.Module, log_level=logging.INFO
) -> tuple[
    list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]
]:
    muon_params = []
    scalar_params = []
    embeds_params = []
    output_params = []
    for name, p in module.named_parameters():
        shape = tuple(p.shape)
        if not p.requires_grad:
            msg = f"{name} {shape} requires_grad=False, skipped"
        elif p.ndim < 2 or (hasattr(p, "_is_scalar") and p._is_scalar):
            msg = f"{name} {shape} (scalar) assigned to AdamW"
            scalar_params.append(p)
        elif hasattr(p, "_is_embed") and p._is_embed:
            msg = f"{name} {shape} (embed) assigned to AdamW"
            embeds_params.append(p)
        elif hasattr(p, "_is_output") and p._is_output:
            msg = f"{name} {shape} (output) assigned to AdamW"
            output_params.append(p)
        else:
            if hasattr(p, "_ortho"):
                raise ValueError(
                    "_ortho is deprecated, use _is_embed or _is_output instead"
                )
            msg = f"{name} {shape} assigned to Muon"
            muon_params.append(p)
        log.log(log_level, msg)

    total_params = sum(
        p.numel() for p in muon_params + scalar_params + embeds_params + output_params
    )
    total_param_tensors = sum(
        len(group)
        for group in (muon_params, scalar_params, embeds_params, output_params)
    )
    log.log(
        log_level,
        "parameter information:\n"
        f"- muon params: {sum(p.numel() for p in muon_params):,} over {len(muon_params):,} tensors\n"
        f"- scalar params: {sum(p.numel() for p in scalar_params):,} over {len(scalar_params):,} tensors\n"
        f"- embeds params: {sum(p.numel() for p in embeds_params):,} over {len(embeds_params):,} tensors\n"
        f"- output params: {sum(p.numel() for p in output_params):,} over {len(output_params):,} tensors\n"
        f"total: {total_params:,} over {total_param_tensors:,} tensors",
    )
    return muon_params, scalar_params, embeds_params, output_params
