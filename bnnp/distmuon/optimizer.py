"""Distributed Muon optimizer implementation for PyTorch FSDP2 and DDP.

Original implementation from:
https://github.com/microsoft/dion/blob/94ae79/dion/muon.py

Modified to use slightly different RMS learning rate scaling, as well as Polar Express
(https://arxiv.org/abs/2505.16932) coefficients for Newton-Schulz orthogonalization.
"""

import math
from itertools import chain
from typing import Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

from .adamw import adamw_update_foreach
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)


class DistMuon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        adamw_betas: Tuple of (beta1, beta2) for AdamW.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        lr_scaling: How to adjust the learning rate for Muon updates ("rms" or "spectral").
            "rms": Multiply update by sqrt(max(1, fan_out / fan_in)), for constant
                average-case RMS norm update sizes, similar to Keller Jordan's original
                implementation.
            "spectral": Multiply update by sqrt(fan_out / fan_in), to control the
                spectral norm (worst-case) update size.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        nesterov: bool = True,
        epsilon: float = 1e-8,
        lr_scaling: Literal["rms", "spectral"] = "rms",
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(adamw_betas) != 2 or adamw_betas[0] < 0.0 or adamw_betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {adamw_betas}")
        if lr_scaling not in ("spectral", "rms"):
            raise ValueError(
                f"Invalid lr_scaling value: {lr_scaling}. Must be 'rms' or 'spectral'."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=adamw_betas[0],
            beta2=adamw_betas[1],
            weight_decay=weight_decay,
            nesterov=nesterov,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
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

        muon_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
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
        return state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == "muon"
            assert all(p.ndim >= 2 for p in group["params"]), (
                "Muon optimizer only supports matrix parameters."
            )

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            muon_update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
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
                print("here0", len(params), params[0].shape, params[0].dtype)

                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "muon") for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # We don't flatten 3D matrices, so we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                    is_batch_sharded = any(
                        p.dim not in matrix_dims for _, p in shard_placements
                    )
                    shard_placements = [
                        (i, p) for i, p in shard_placements if p.dim in matrix_dims
                    ]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that 3+ dimension tensors can have additional sharded batch dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,  # No sharded matrix dim
                                **muon_update_args,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        muon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **muon_update_args,
                        )
                    )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == "adamw"

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
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
                    step=step,
                    epsilon=epsilon,
                )
            )


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    lr_scaling: str,  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert process_group is not None, (
            "process_group must be provided for sharded DTensors"
        )
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert X[0].size(shard_dim) % world_size == 0, (
            f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."
        )

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(single_matrix, epsilon=epsilon)

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U) > 1:
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(single_matrix, epsilon=epsilon)

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
        U[0] = muon_update_newton_schulz(U[0], epsilon=epsilon)

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if lr_scaling == "rms":
        adjusted_lr = adjust_lr_rms(lr, X[0].shape)
    elif lr_scaling == "spectral":
        adjusted_lr = adjust_lr_spectral(lr, X[0].shape)
    else:
        raise ValueError(f"Unknown lr_scaling value: {lr_scaling}")

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(X: Tensor, epsilon: Tensor) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return _newton_schulz_polar_express(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm, in expectation.
    # Similar to Keller Jordan's original implementation.
    fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * max(1, math.sqrt(fan_out / fan_in))
    return adjusted_lr


def adjust_lr_spectral(lr, param_shape):
    # Adjust learning rate to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def _newton_schulz_polar_express(G: Tensor, epsilon: float = 1e-7):
    # Taken from: https://arxiv.org/abs/2505.16932
    polar_express_coeffs = [
        [8.20516041, -22.90193499, 16.46072491],
        [4.06639516, -2.86115409, 0.51839952],
        [3.90959490, -2.82335174, 0.52503698],
        [3.28556402, -2.41530196, 0.48529407],
        [2.27787329, -1.61982177, 0.39848079],
        # [1.87257565, -1.23070426, 0.35851616],
        # [1.85643711, -1.21323928, 0.35679979],
        # [1.85643564, -1.21323768, 0.35679963],
    ]

    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)
    # for step in range(steps):
    #     a, b, c = polar_express_coeffs[min(step, len(polar_express_coeffs) - 1)]
    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.mT
    return X.type_as(G)
