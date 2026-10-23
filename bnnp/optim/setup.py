import logging
import re
from collections.abc import Iterable
from typing import Literal

import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import Parameter

from .distributed import DistMuon, DistNorMuon
from .muon import Muon
from .normuon import NorMuon

_DEFAULT_LOGGER = logging.getLogger(__name__)


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
    """Summarizes same-shaped parameters whose names differ only numerically.

    For example, when w0 and w1 have the same shape, the input
        [("blocks.0.weight", w0), ("blocks.1.weight", w1)]
    becomes
        [(2, "blocks.(0-1).weight", w0)]
    Each output contains the count, summarized name, and first parameter in the group.
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
    variant: Literal["muon", "normuon", "distmuon", "distnormuon"],
    base_lr: float,
    lora_lr: float,
    scalar_lr: float,
    base_wd: float,
    betas: tuple[float, float],
    muon_exclusions: list[str] | None = None,
    wd_exclusions: list[str] | None = None,
    strict_exclusions: bool = True,
    logger: logging.Logger | None = None,
    *,
    distributed_mesh: DeviceMesh | ProcessGroup | None = None,
) -> Muon | NorMuon | DistMuon | DistNorMuon:
    """
    variant: matrix optimizer to use (with rms lr scaling); prefix with 'dist' for DDP/FSDP2
    base_lr: corresponds to the muon learning rate for matrix, O(1 / sqrt(dim)) RMS parameters
    lora_lr: learning rate for AdamW matrix parameters (e.g. input/output projections).
        Matching update RMS under Moonlight's empirical AdamW RMS of ~0.2 gives the
        heuristic lora_lr ~ base_lr / (0.2 * sqrt(fan_in))
    scalar_lr: learning rate used for non-matrix AdamW parameters (e.g. biases, norm weights)
    base_wd: fractional weight shrinkage per step; AdamW-style decoupled weight decay
        is derived as wd = base_wd / lr (zero-lr groups receive zero decay)
    betas: first and second moments for NorMuon and AdamW; Muon uses only betas[0]
    muon_exclusions: regex patterns (scanned) for 2d parameter keys to exclude from muon
    wd_exclusions: regex patterns (scanned) for parameter keys to use zero weight decay
    strict_exclusions: raise if any exclusion patterns are not matched to any parameter
    distributed_mesh: optional mesh/process group for distributed variants
    """
    optimizer_classes = {
        "muon": Muon,
        "normuon": NorMuon,
        "distmuon": DistMuon,
        "distnormuon": DistNorMuon,
    }
    if variant not in optimizer_classes:
        raise ValueError(
            f"Unknown optimizer variant: {variant}. Expected one of {tuple(optimizer_classes)}."
        )
    if distributed_mesh is not None and not variant.startswith("dist"):
        raise ValueError("distributed_mesh requires a distributed optimizer variant")
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
                f"Did not find muon exclusions: {missing_muon_exclusions}, "
                f"wd exclusions: {missing_wd_exclusions}"
            )
        elif missing_muon_exclusions:
            raise ValueError(f"Did not find muon exclusions: {missing_muon_exclusions}")
        elif missing_wd_exclusions:
            raise ValueError(f"Did not find wd exclusions: {missing_wd_exclusions}")

    summary_rows = {
        group: [
            ((f"{count}x " if count > 1 else "") + str(tuple(param.shape)), name)
            for count, name, param in summarize_parameters(params)
        ]
        for group, params in param_groups.items()
    }
    maxlen = max(
        (len(shape_str) for rows in summary_rows.values() for shape_str, _ in rows), default=0
    )

    total_params, total_tensors = 0, 0
    parameters = []
    for group, params in param_groups.items():
        if group.startswith("muon"):
            algorithm = variant.removeprefix("dist")
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

        for shape_str, name in summary_rows[group]:
            logger.info(f"    {shape_str:<{maxlen}}  {name}")

        parameters.append(
            dict(
                algorithm=algorithm, params=[p for _, p in params], lr=lr, weight_decay=weight_decay
            )
        )
    logger.info(f"Total: {total_params:,} parameters over {total_tensors:,} tensors")

    optimizer_class = optimizer_classes[variant]
    distributed_kwargs = (
        {"distributed_mesh": distributed_mesh} if variant.startswith("dist") else {}
    )
    optimizer = optimizer_class(
        parameters,
        betas=betas,
        nesterov=True,
        **distributed_kwargs,  # type: ignore
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["initial_wd"] = group["weight_decay"]
    return optimizer
