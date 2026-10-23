from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from .base import normalize_update, orthogonalize
from .muon import Muon
from .normuon import NorMuon


def _from_local(local, shape, mesh, placements):
    # Explicit metadata is needed for uneven shards, including empty local shards.
    stride = []
    size = 1
    for dim in reversed(shape):
        stride.append(size)
        size *= max(1, dim)
    return DTensor.from_local(
        local, mesh, placements, shape=torch.Size(shape), stride=tuple(reversed(stride))
    )


def _is_shard(placement):
    # FSDP2 + tensor parallelism can use interleaved shards of the same tensor axis.
    return isinstance(placement, (Shard, _StridedShard))


def _stack_placement(placement, ndim):
    if isinstance(placement, _StridedShard):
        return _StridedShard(placement.dim % ndim + 1, split_factor=placement.split_factor)
    if isinstance(placement, Shard):
        return Shard(placement.dim % ndim + 1)
    return placement


class _DistributedMuon:
    def _configure_distributed(self, distributed_mesh):
        if isinstance(distributed_mesh, DeviceMesh):
            self._process_group = distributed_mesh.get_group(distributed_mesh.ndim - 1)
        elif isinstance(distributed_mesh, ProcessGroup):
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._process_group = dist.group.WORLD if dist.is_initialized() else None
        else:
            raise TypeError("distributed_mesh must be a DeviceMesh, ProcessGroup, or None")
        self._world_size = dist.get_world_size(self._process_group) if self._process_group else 1
        self._rank = dist.get_rank(self._process_group) if self._process_group else 0
        for group in self.param_groups:  # type: ignore
            for param in group["params"]:
                if isinstance(param, DTensor) and any(
                    not (p.is_replicate() or _is_shard(p)) for p in param.placements
                ):
                    raise ValueError("DTensor parameter placements must be Replicate or Shard")

    def _local_tensor(self, tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _gradient(self, param):
        grad = param.grad
        if isinstance(param, DTensor):
            if not isinstance(grad, DTensor):
                raise TypeError("DTensor parameters require DTensor gradients")
            grad = grad.redistribute(param.device_mesh, param.placements)
        return grad

    def _new_reduced_state(self, param, shape, reduce_dim):
        if not isinstance(param, DTensor):
            return super()._new_reduced_state(param, shape, reduce_dim)  # type: ignore
        axis = reduce_dim % param.ndim
        placements = tuple(
            Replicate() if _is_shard(p) and p.dim % param.ndim == axis else p
            for p in param.placements
        )
        local_shape = list(param.to_local().shape)
        local_shape[axis] = 1
        local = torch.zeros(local_shape, device=param.device, dtype=param.dtype)
        return _from_local(local, shape, param.device_mesh, placements)

    def _compute_update(self, group, params, states, gg):
        if isinstance(params[0], DTensor):
            return self._dtensor_update(group, params, states, gg)
        if self._world_size == 1:
            return super()._compute_update(group, params, states, gg)  # type: ignore

        # DDP already synchronized gradients. Share only the orthogonalization work;
        # momentum, adaptive state, and parameter updates stay identical on every rank.
        owned = gg[self._rank :: self._world_size]
        count = (len(params) + self._world_size - 1) // self._world_size
        output = gg.new_zeros((count, *gg.shape[1:]))
        if len(owned):
            output[: len(owned)] = orthogonalize(owned, steps=group["ns_steps"], eps=group["eps"])
        outputs = [torch.empty_like(output) for _ in range(self._world_size)]
        dist.all_gather(outputs, output, group=self._process_group)
        gg = torch.stack(
            [outputs[i % self._world_size][i // self._world_size] for i in range(len(params))]
        )
        return self._normalize_update(group, params, states, gg)  # type: ignore

    def _dtensor_update(self, group, params, states, gg):
        param = params[0]
        mesh = param.device_mesh
        # Stacking parameters adds an axis before all of their existing dimensions.
        placements = tuple(_stack_placement(p, param.ndim) for p in param.placements)
        batch = _from_local(gg, (len(params), *param.shape), mesh, placements)
        # Move matrix shards onto the parameter-batch axis. DTensor handles the
        # all-to-all and uneven splits, giving each owner complete matrices. Keep
        # existing shards of leading batch axes, since those already contain matrices.
        owners = tuple(
            Shard(0) if p.is_replicate() or (_is_shard(p) and p.dim >= batch.ndim - 2) else p
            for p in placements
        )
        owned = batch.redistribute(placements=owners).to_local()
        if owned.numel():
            owned = orthogonalize(owned, steps=group["ns_steps"], eps=group["eps"])

        if group["algorithm"] == "normuon":
            variance = torch.stack([state["exp_avg_sq"] for state in states])
            owned_variance = variance.redistribute(placements=owners).to_local()  # type: ignore
            steps = torch.stack([state["step"] for state in states])
            step_placements = tuple(
                Shard(0) if p.is_shard() and p.dim == 0 else Replicate() for p in owners
            )
            owned_steps = _from_local(steps, steps.shape, mesh, [Replicate()] * mesh.ndim)
            owned_steps = owned_steps.redistribute(placements=step_placements).to_local()
            if len(owned):
                owned = normalize_update(
                    owned,
                    list(owned_variance.unbind(0)),
                    list(owned_steps.unbind(0)),
                    beta2=group["betas"][1],
                    reduce_dim=-1 if param.size(-1) < param.size(-2) else -2,
                    eps=group["eps"],
                )
            variance_update = _from_local(owned_variance, variance.shape, mesh, owners)
            local_variance = variance_update.redistribute(placements=variance.placements).to_local()  # type: ignore
            torch._foreach_copy_(
                [state["exp_avg_sq"].to_local() for state in states],
                list(local_variance.unbind(0)),
            )

        update = _from_local(owned, batch.shape, mesh, owners)
        return update.redistribute(placements=placements).to_local()


class DistMuon(_DistributedMuon, Muon):
    """Muon with distributed orthogonalization for DDP and FSDP2 DTensor parameters.

    distributed_mesh specifies the process group for replicated parameters; it defaults
    to WORLD when initialized. A DeviceMesh uses its last mesh dimension for replicated
    parameters. DTensor parameters always communicate through their own meshes and
    placements.
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
        *,
        distributed_mesh: DeviceMesh | ProcessGroup | None = None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            lr_scaling=lr_scaling,
        )
        self._configure_distributed(distributed_mesh)


class DistNorMuon(_DistributedMuon, NorMuon):
    """NorMuon with the same distributed execution and mesh options as :class:`DistMuon`.

    Momentum, normalization, LR scaling, weight decay, and AdamW fallback match
    :class:`NorMuon`. Reduced second moments retain the parameter's DTensor layout,
    replicating any mesh dimension that shards the reduced row/column axis.
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
        *,
        distributed_mesh: DeviceMesh | ProcessGroup | None = None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            lr_scaling=lr_scaling,
        )
        self._configure_distributed(distributed_mesh)
