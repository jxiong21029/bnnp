import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.optim.adam import adam

_DEFAULT_LOGGER = logging.getLogger(__name__)

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
            groups[(p.shape, p.dtype, p.grad.dtype, p.device)].append(p)

    for group in groups.values():
        bsz = max(1, max_numel // group[0].numel())
        for i in range(0, len(group), bsz):
            yield group[i : i + bsz]


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
            if algorithm == "muon" and not 0.0 <= group["beta"] < 1.0:
                raise ValueError(f"Invalid momentum factor (beta): {group['beta']}")
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
        beta1 = group["betas"][0] if normalize else group["beta"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        eps = group["eps"]
        lr_scaling = group["lr_scaling"]

        p0 = params[0]
        d_out, d_in = p0.shape[-2:]
        # Unlike NorMuon paper, we always accumulate 2nd moments along the smaller dimension
        reduce_dim = -1 if d_in < d_out else -2
        if reduce_dim == -2:
            reduced_shape = p0.shape[:-2] + (1, d_in)
        else:
            reduced_shape = p0.shape[:-2] + (d_out, 1)

        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        steps: list[Tensor] = []
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros(p.shape, device=p.device, dtype=p.dtype)
                if normalize:
                    state["step"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros(reduced_shape, device=p.device, dtype=p.dtype)
            exp_avgs.append(state["exp_avg"])
            if normalize:
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])

        gg = torch.stack([p.grad for p in params])  # type: ignore
        gg_views = list(gg.unbind(0))

        torch._foreach_lerp_(exp_avgs, gg_views, 1 - beta1)
        if nesterov:
            torch._foreach_lerp_(gg_views, exp_avgs, beta1)
        else:
            torch._foreach_copy_(gg_views, exp_avgs)

        gg = orthogonalize(gg, steps=ns_steps, eps=eps)

        if normalize:
            beta2 = group["betas"][1]
            sq = gg.square().mean(dim=reduce_dim, keepdim=True)
            torch._foreach_lerp_(exp_avg_sqs, list(sq.unbind(0)), 1 - beta2)
            torch._foreach_add_(steps, 1.0)

            v = torch.stack(exp_avg_sqs)
            correction = 1 - beta2 ** torch.stack(steps)
            correction = correction.view(-1, *([1] * (v.ndim - 1)))
            gg.div_((v / correction).sqrt_().add_(eps))

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
        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_add_(params, list(update.unbind(0)), alpha=-lr)

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
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
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


class Muon(_MuonBase):
    """Momentum orthogonalized by Newton-Schulz.

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
        beta: float = 0.9,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        lr_scaling: Literal["rms", "mup", "moonlight"] = "rms",
        adamw_betas: tuple[float, float] = (0.9, 0.95),
    ):
        defaults = dict(
            lr=lr,
            algorithm="muon",
            beta=beta,
            betas=adamw_betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            lr_scaling=lr_scaling,
        )
        super().__init__(params, defaults)


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


def _summarize_ranges(nums: list[int]):
    """Converts list of integers to a string of number ranges.

    E.g. [1, 2, 3, 5, 7, 8, 9] -> '1-3,5,7-9'
    """
    if not nums:
        return ""

    nums = sorted(set(nums))
    ranges = []
    start = prev = nums[0]

    for num in nums[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num

    # Add the final range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def summarize_parameters(
    parameters: list[tuple[str, nn.Parameter]],
) -> list[tuple[int, str, nn.Parameter]]:
    """
    Summarizes same-shaped parameters whose names differ only at numeric tokens.

    Grouping repeats to collapse multiple numeric dimensions when they form a complete
    rectangular group.
    """
    summaries: list[tuple[int, list[str | frozenset[int]], nn.Parameter]] = []
    for name, param in parameters:
        tokens = [
            frozenset({int(token)}) if token.isdigit() else token for token in name.split(".")
        ]
        summaries.append((1, tokens, param))

    merged = True
    while merged:
        merged = False
        for left_index, (left_count, left_tokens, left_param) in enumerate(summaries):
            for right_index in range(left_index + 1, len(summaries)):
                right_count, right_tokens, right_param = summaries[right_index]
                if left_param.shape != right_param.shape or len(left_tokens) != len(right_tokens):
                    continue

                differing_index = None
                for token_index, (left_token, right_token) in enumerate(
                    zip(left_tokens, right_tokens)
                ):
                    if left_token == right_token:
                        continue
                    if (
                        differing_index is not None
                        or not isinstance(left_token, frozenset)
                        or not isinstance(right_token, frozenset)
                        or not left_token.isdisjoint(right_token)
                    ):
                        break
                    differing_index = token_index
                else:
                    if differing_index is not None:
                        combined_tokens = left_tokens.copy()
                        left_values = left_tokens[differing_index]
                        right_values = right_tokens[differing_index]
                        assert isinstance(left_values, frozenset)
                        assert isinstance(right_values, frozenset)
                        combined_tokens[differing_index] = left_values | right_values
                        summaries[left_index] = (
                            left_count + right_count,
                            combined_tokens,
                            left_param,
                        )
                        summaries.pop(right_index)
                        merged = True
                        break
            if merged:
                break

    outputs = []
    for count, tokens, param in summaries:
        output_tokens = []
        for token in tokens:
            if isinstance(token, frozenset):
                token_range = _summarize_ranges(list(token))
                output_tokens.append(token_range if len(token) == 1 else f"({token_range})")
            else:
                output_tokens.append(token)
        outputs.append((count, ".".join(output_tokens), param))
    return outputs


def setup_optimizer(
    named_parameters: Iterable[tuple[str, Parameter]],
    variant: Literal["muon", "normuon"],
    base_lr: float,
    lora_lr: float,
    scalar_lr: float,
    base_wd: float,
    betas: tuple[float, float],
    muon_exclusions: list[str] | None = None,
    wd_exclusions: list[str] | None = None,
    strict_exclusions: bool = True,
    logger: logging.Logger | None = None,
) -> Muon | NorMuon:
    """
    variant: matrix optimizer to use, either Muon or NorMuon (with rms lr scaling)
    base_lr: corresponds to the muon learning rate for matrix, O(1 / sqrt(dim)) RMS parameters
    lora_lr: learning rate for AdamW matrix parameters (e.g. input/output projections).
        Matching update RMS under Moonlight's empirical AdamW RMS of ~0.2 gives the
        heuristic lora_lr ~ base_lr / (0.2 * sqrt(fan_in))
    scalar_lr: learning rate used for scalar AdamW parameters (e.g. biases, layer norms)
    base_wd: fractional weight shrinkage per step; AdamW-style decoupled weight decay
        is derived as wd = base_wd / lr (zero-lr groups receive zero decay)
    betas: first and second moments for NorMuon and AdamW; Muon uses only betas[0]
    muon_exclusions: regex patterns (scanned) for 2d parameter keys to exclude from muon
    wd_exclusions: regex patterns (scanned) for parameter keys to use zero weight decay
    strict_exclusions: raise if any exclusion patterns are not matched to any parameter
    """
    if variant not in ("muon", "normuon"):
        raise ValueError(f"Unknown optimizer variant: {variant}. Expected 'muon' or 'normuon'.")
    if logger is None:
        logger = _DEFAULT_LOGGER

    if muon_exclusions is None:
        muon_exclusions = []
    if wd_exclusions is None:
        wd_exclusions = []
    missing_muon_exclusions = set(muon_exclusions)
    missing_wd_exclusions = set(wd_exclusions)

    param_groups = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            matched_muon_exclusions = {excl for excl in muon_exclusions if re.search(excl, name)}
            missing_muon_exclusions.difference_update(matched_muon_exclusions)
            group = "adamw_lora" if matched_muon_exclusions else "muon"
        else:
            group = "adamw_scalar"

        matched_wd_exclusions = {excl for excl in wd_exclusions if re.search(excl, name)}
        missing_wd_exclusions.difference_update(matched_wd_exclusions)
        if matched_wd_exclusions:
            group = group + "_no_wd"
        param_groups.setdefault(group, []).append((name, param))

    if strict_exclusions:
        if missing_muon_exclusions and missing_wd_exclusions:
            raise ValueError(
                f"Did not find muon exclusions: {missing_muon_exclusions}, wd exclusions: {missing_wd_exclusions}"
            )
        elif missing_muon_exclusions:
            raise ValueError(f"Did not find muon exclusions: {missing_muon_exclusions}")
        elif missing_wd_exclusions:
            raise ValueError(f"Did not find wd exclusions: {missing_wd_exclusions}")

    total_params, total_tensors = 0, 0
    parameters = []
    for group, params in param_groups.items():
        if group.startswith("muon"):
            algorithm = variant
            lr = base_lr
        elif group.startswith("adamw_lora"):
            algorithm = "adamw"
            lr = lora_lr
        else:
            assert group.startswith("adamw_scalar")
            algorithm = "adamw"
            lr = scalar_lr
        weight_decay = 0.0 if group.endswith("_no_wd") or lr == 0.0 else base_wd / lr

        n_params = sum(p.numel() for _, p in params)
        total_params += n_params
        total_tensors += len(params)
        logger.info(f"Group {group}: {n_params:,} params over {len(params):,} tensors")

        summary = summarize_parameters(params)
        shape_strings = [
            (f"{count}x " if count > 1 else "") + str(tuple(param.shape))
            for count, _, param in summary
        ]
        maxlen = max(len(s) for s in shape_strings)
        for shape_str, (_, name, _) in zip(shape_strings, summary):
            logger.info(f"    {shape_str:<{maxlen}}  {name}")

        parameters.append(
            dict(
                algorithm=algorithm, params=[p for _, p in params], lr=lr, weight_decay=weight_decay
            )
        )
    logger.info(f"Total: {total_params:,} parameters over {total_tensors:,} tensors")

    if variant == "muon":
        optimizer = Muon(parameters, beta=betas[0], adamw_betas=betas, nesterov=True)
    else:
        optimizer = NorMuon(parameters, betas=betas, nesterov=True)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["initial_wd"] = group["weight_decay"]
    return optimizer
