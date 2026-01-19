from itertools import chain
from typing import Generator

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

# Reuse Muon's helper functions
from .muon import (
    adamw_update_foreach_async,
    adjust_lr_moonlight,
    adjust_lr_mup,
    adjust_lr_rms,
    muon_update_newton_schulz,
    muon_update_post_orthogonalize,
    muon_update_pre_orthogonalize,
)
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)


class DistNorMuon(Optimizer):
    """Distributed NorMuon optimizer for DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: ProcessGroup for distributed training.
        lr: Base learning rate. For NorMuon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        betas: (beta1, beta2) parameters for both AdamW and NorMuon's adaptive updates.
        weight_decay: Weight decay factor.
        eps: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        lr_scaling: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "rms": Multiply update by sqrt(max(1, d_out / d_in)), for constant
                average update norms, like in Keller Jordan's original implemenetation.
            "mup": Multiply update by sqrt(d_out / d_in), to control the spectral
                norm of the updates.
            "moonlight": Multiply update by max(d_out, d_in) to maintain constant-sized
                elementwise update RMS, similar to AdamW.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    NorMuon optimizer: https://arxiv.org/abs/2510.05491
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: ProcessGroup | None = None,
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        nesterov: bool = False,
        lr_scaling: str = "rms",
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if lr_scaling not in ("rms", "mup", "moonlight"):
            raise ValueError(
                f"Invalid lr_scaling value: {lr_scaling}. Must be 'rms', 'mup', or 'moonlight'."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="normuon",
            step=0,
            eps=eps,
            nesterov=nesterov,
            lr_scaling=lr_scaling,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        normuon_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "normuon":
                normuon_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        normuon_tasks = self._create_normuon_tasks(normuon_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(normuon_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)  # ty: ignore
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
            else:
                assert algo == "normuon"
                reduce_dim = -1 if param.size(-1) < param.size(-2) else -2
                if reduce_dim == -2:
                    reduced_shape = param.shape[:-2] + (1, param.size(-1))
                else:
                    reduced_shape = param.shape[:-2] + (param.size(-2), 1)
                state["variance"] = param.new_zeros(reduced_shape)
        return state

    def _create_normuon_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "normuon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of NorMuon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(p.ndim >= 2 for p in group["params"]), (
                "NorMuon optimizer only supports matrix parameters."
            )

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            normuon_update_args = dict(
                lr=torch.tensor(group["lr"]),
                beta1=torch.tensor(group["beta1"]),
                beta2=torch.tensor(group["beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                eps=torch.tensor(group["eps"]),
                step=torch.tensor(group["step"]),
                nesterov=group["nesterov"],
                lr_scaling=group["lr_scaling"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
            )

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]

                yield AsyncTask(
                    normuon_update_batch_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(gradients, self._world_size),  # ty: ignore
                        M=pad_batch(momentums, self._world_size),
                        V=pad_batch(variances, self._world_size),
                        **normuon_update_args,  # ty: ignore
                    )
                )

    def _create_adamw_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            eps = torch.tensor(group["eps"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,  # ty: ignore
                    eps=eps,  # ty: ignore
                )
            )


def normuon_update_batch_async(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    V: list[Tensor],  # Variance neuron buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Momentum decay
    beta2: Tensor,  # Variance decay
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: Tensor,
    eps: Tensor,
    nesterov: bool,  # Whether to use Nesterov momentum
    lr_scaling: str,  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    process_group: ProcessGroup | None = None,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == len(V)

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=beta1,
        nesterov=nesterov,
    )

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    if len(U) > 1:
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(single_matrix, eps=eps)

        # Allocate empty tensors to receive updates from other devices
        U = [torch.empty_like(u) for u in U]

        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U) == 1
        U[0] = muon_update_newton_schulz(U[0], eps=eps)

    # NorMuon normalization
    U = normuon_normalization(U, V=to_local(V), beta2=beta2, step=step)

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if lr_scaling == "rms":
        adjusted_lr = adjust_lr_rms(lr, X[0].shape)
    elif lr_scaling == "mup":
        adjusted_lr = adjust_lr_mup(lr, X[0].shape)
    elif lr_scaling == "moonlight":
        adjusted_lr = adjust_lr_moonlight(lr, X[0].shape)
    else:
        raise ValueError(f"Unknown lr_scaling value: {lr_scaling}")

    # Compensate for norm adjustment from normuon_normalization
    adjusted_lr = adjusted_lr / (max(U[0].size(-2), U[0].size(-1)) ** 0.5)

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


@torch.compile(fullgraph=True)
def normuon_normalization(
    U: list[Tensor], V: list[Tensor], beta2: Tensor, step: Tensor
):
    """
    NorMuon normalization step after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """

    reduce_dim = -1 if U[0].size(-1) < U[0].size(-2) else -2

    V_dtype = V[0].dtype
    U = [u.to(dtype=V_dtype) for u in U]

    U_sq = torch._foreach_mul(U, U)  # list of u*u, same shapes as U
    reduced_v = [u_sq.mean(dim=reduce_dim, keepdim=True) for u_sq in U_sq]

    torch._foreach_lerp_(V, reduced_v, 1 - beta2)  # Update variance neuron buffer
    denom = torch._foreach_sqrt(V)  # list of sqrt(v)
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, 1e-8)  # denom[i] += 1e-8
    normalized_U = torch._foreach_div(U, denom)  # list of u / denom

    return normalized_U
