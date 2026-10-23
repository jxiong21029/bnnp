from collections import defaultdict

import torch
from torch import Tensor
from torch.optim.adam import adam

# Newton-schulz coefficients from Polar Express (https://arxiv.org/abs/2505.16932) with
# safety factor built-in
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


@torch.compile(fullgraph=True)
def orthogonalize(G: Tensor, steps: int, eps: float) -> Tensor:
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


def create_param_batches(params: list[Tensor], max_numel: int = 2**28):
    groups = defaultdict(list)
    for p in params:
        if p.grad is not None:
            groups[(p.shape, p.dtype, p.grad.dtype, p.device, _layout(p))].append(p)

    for group in groups.values():
        bsz = max(1, max_numel // group[0].numel())
        for i in range(0, len(group), bsz):
            yield group[i : i + bsz]


def _layout(param: Tensor):
    # DTensor parameters with different meshes or placements must not share a batch.
    return (getattr(param, "device_mesh", None), getattr(param, "placements", None))


def normalize_update(
    update: Tensor,
    exp_avg_sqs: list[Tensor],
    steps: list[Tensor],
    beta2: float,
    reduce_dim: int,
    eps: float,
) -> Tensor:
    sq = update.square().mean(dim=reduce_dim, keepdim=True)
    torch._foreach_lerp_(exp_avg_sqs, list(sq.unbind(0)), 1 - beta2)
    v = torch.stack(exp_avg_sqs)
    correction = 1 - beta2 ** torch.stack(steps)
    correction = correction.view(-1, *([1] * (v.ndim - 1)))
    return update.div_((v / correction).sqrt_().add_(eps))


class _MuonBase(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

        for group in self.param_groups:
            algorithm = group["algorithm"]
            if algorithm not in ("adamw", defaults["algorithm"]):
                raise ValueError(
                    f"Unknown algorithm {algorithm}, expected either 'adamw' "
                    f"or '{defaults['algorithm']}'."
                )
            if not 0.0 <= group["lr"]:
                raise ValueError(f"Invalid learning rate: {group['lr']}")
            if not 0.0 <= group["weight_decay"]:
                raise ValueError(f"Invalid weight_decay: {group['weight_decay']}")
            if not 0.0 <= group["eps"]:
                raise ValueError(f"Invalid eps: {group['eps']}")
            betas = group["betas"]
            if len(betas) != 2 or any(not 0.0 <= beta < 1.0 for beta in betas):
                raise ValueError(f"Invalid betas: {betas}")
            if algorithm != "adamw":
                if group["lr_scaling"] not in ("rms", "mup", "moonlight"):
                    raise ValueError(
                        f"Invalid lr_scaling value: {group['lr_scaling']}. "
                        "Must be 'rms', 'mup', or 'moonlight'."
                    )
                if not isinstance(group["ns_steps"], int) or group["ns_steps"] < 1:
                    raise ValueError(f"Invalid ns_steps: {group['ns_steps']}")
                if any(p.ndim < 2 for p in group["params"]):
                    raise ValueError(
                        f"{algorithm} expects at least 2D parameters; use AdamW otherwise"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["algorithm"] == "adamw":
                self.adamw_step(group)
                continue
            for param_batch in create_param_batches(group["params"]):
                self.muon_step_batch(group, param_batch)
        return loss

    def muon_step_batch(self, group, params: list[Tensor]):
        lr = group["lr"]
        normalize = group["algorithm"] == "normuon"
        beta1 = group["betas"][0]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]
        lr_scaling = group["lr_scaling"]

        p0 = params[0]
        d_out, d_in = p0.shape[-2:]
        states = [self._muon_state(p, normalize) for p in params]
        exp_avgs = [self._local_tensor(state["exp_avg"]) for state in states]
        gg = torch.stack([self._local_tensor(self._gradient(p)) for p in params])
        gg_views = list(gg.unbind(0))

        torch._foreach_lerp_(exp_avgs, gg_views, 1 - beta1)
        if nesterov:
            torch._foreach_lerp_(gg_views, exp_avgs, beta1)
        else:
            torch._foreach_copy_(gg_views, exp_avgs)
        if normalize:
            torch._foreach_add_([state["step"] for state in states], 1.0)

        gg = self._compute_update(group, params, states, gg)

        # A semi-orthogonal (d_out, d_in) update has RMS 1 / sqrt(max(d_out, d_in)).
        # Original Muon: https://github.com/KellerJordan/Muon/blob/master/muon.py
        # Moonlight: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        if lr_scaling == "rms":
            scale = max(1, d_out / d_in) ** 0.5
        elif lr_scaling == "mup":
            scale = (d_out / d_in) ** 0.5
        elif lr_scaling == "moonlight":
            scale = 0.2 * max(d_in, d_out) ** 0.5
        else:
            raise ValueError(f"unknown value for lr_scaling: {lr_scaling}")
        if normalize:
            # NorMuon raises the update RMS to roughly 1. Compensate so that the
            # same LR remains comparable to Muon (adaptive updates need not match exactly).
            scale /= max(d_in, d_out) ** 0.5
        gg.mul_(scale)

        # equivalent to p <- p - lr * (update + weight_decay * p)
        update = gg.to(p0.dtype)
        local_params = [self._local_tensor(p) for p in params]
        if weight_decay != 0.0:
            torch._foreach_mul_(local_params, 1 - lr * weight_decay)
        torch._foreach_add_(local_params, list(update.unbind(0)), alpha=-lr)

    def _muon_state(self, param: Tensor, normalize: bool):
        state = self.state[param]
        if state:
            if normalize:
                # Optimizer.load_state_dict preserves the checkpoint device for steps.
                state["step"] = state["step"].to(device=param.device)
            return state
        state["exp_avg"] = torch.zeros_like(param)
        if not normalize:
            return state

        d_out, d_in = param.shape[-2:]
        # Unlike NorMuon paper, we always accumulate 2nd moments along the smaller dimension
        reduce_dim = -1 if d_in < d_out else -2
        if reduce_dim == -2:
            reduced_shape = param.shape[:-2] + (1, d_in)
        else:
            reduced_shape = param.shape[:-2] + (d_out, 1)
        state["step"] = torch.zeros((), device=param.device, dtype=torch.float32)
        state["exp_avg_sq"] = self._new_reduced_state(param, reduced_shape, reduce_dim)
        return state

    def _new_reduced_state(self, param, shape, reduce_dim):
        return torch.zeros(shape, device=param.device, dtype=param.dtype)

    def _compute_update(self, group, params, states, gg):
        gg = orthogonalize(gg, steps=group["ns_steps"], eps=group["eps"])
        return self._normalize_update(group, params, states, gg)

    def _normalize_update(self, group, params, states, gg):
        if group["algorithm"] == "normuon":
            gg = normalize_update(
                gg,
                [state["exp_avg_sq"] for state in states],
                [state["step"] for state in states],
                beta2=group["betas"][1],
                reduce_dim=-1 if params[0].size(-1) < params[0].size(-2) else -2,
                eps=group["eps"],
            )
        return gg

    def _local_tensor(self, tensor):
        return tensor

    def _gradient(self, param):
        return param.grad

    def adamw_step(self, group):
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        for param in group["params"]:
            if param.grad is None:
                continue

            params.append(self._local_tensor(param))
            grads.append(self._local_tensor(self._gradient(param)))
            state = self.state[param]
            if "step" not in state:
                state["step"] = torch.tensor(0.0)  # Host on CPU.
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            exp_avgs.append(self._local_tensor(state["exp_avg"]))
            exp_avg_sqs.append(self._local_tensor(state["exp_avg_sq"]))
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
