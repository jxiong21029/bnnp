import gc

import torch
import triton
import triton.language as tl


@triton.jit
def _exnorm_fwd_fused(X, R, Y, stride, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)

    X_row = X + row * stride
    R_row = R + row * stride
    Y_row = Y + row * stride

    # -------- 1) Compute mean(residual^2) over last dim --------
    # Accumulate partial sums in a BLOCK_SIZE-wide vector, then reduce.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += r * r

    sum_sq = tl.sum(acc, axis=0)
    mean_sq = sum_sq / D

    # post_scale = (mean_sq + 1).rsqrt()
    post_scale = tl.rsqrt(mean_sq + 1.0)

    # -------- 2) Compute y = post_scale * (x + residual) --------
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)

        y = post_scale * (x + r)

        # tl.store will cast to the dtype of Y (e.g. fp16) if needed,
        # as in Triton's layer-norm tutorial.
        tl.store(Y_row + cols, y, mask=mask)


@triton.jit
def _exnorm_bwd_fused(X, R, DY, DX, DR, stride, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)

    X_row = X + row * stride
    R_row = R + row * stride
    DY_row = DY + row * stride
    DX_row = DX + row * stride
    DR_row = DR + row * stride

    # ---- first pass: recompute s and dot = sum(dy * (x + r)) ----
    acc_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)

        acc_sq += r * r
        acc_dot += dy * (x + r)

    sum_sq = tl.sum(acc_sq, axis=0)
    dot = tl.sum(acc_dot, axis=0)
    mean_sq = sum_sq / D
    s = tl.rsqrt(mean_sq + 1.0)
    s3 = s * s * s
    coeff = -dot * s3 / D  # scalar

    # ---- second pass: write dx, dres ----
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)

        dx = s * dy
        dr = dx + coeff * r

        # Cast back to target dtype if DX/DR are lower precision.
        tl.store(DX_row + cols, dx, mask=mask)
        tl.store(DR_row + cols, dr, mask=mask)


class ExNormFn(torch.autograd.Function):
    """
    Triton implementation of:
        post_scale = residual.pow(2).mean(dim=-1, keepdim=True).add(1).rsqrt()
        return post_scale * (x + residual)
    """

    @staticmethod
    def forward(ctx, x, residual):
        assert x.shape == residual.shape
        assert x.is_cuda and residual.is_cuda
        assert x.dtype == residual.dtype

        x_ND = x.contiguous().view(-1, x.shape[-1])
        r_ND = residual.contiguous().view(-1, residual.shape[-1])
        N, D = x_ND.shape

        y_ND = torch.empty_like(x_ND)

        # Same heuristics as Triton layernorm tutorial for BLOCK_SIZE
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BLOCK_SIZE:
            raise RuntimeError(f"This kernel only supports feature dim <= {BLOCK_SIZE}")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _exnorm_fwd_fused[(N,)](
            x_ND,
            r_ND,
            y_ND,
            x_ND.stride(0),
            D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_ctas=1,
        )

        ctx.save_for_backward(x, residual)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y_ND.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, residual = ctx.saved_tensors
        assert dy.shape == x.shape

        x_ND = x.contiguous().view(-1, x.shape[-1])
        r_ND = residual.contiguous().view(-1, residual.shape[-1])
        dy_ND = dy.contiguous().view(-1, dy.shape[-1])
        N, D = x_ND.shape

        dx_ND = torch.empty_like(x_ND)
        dr_ND = torch.empty_like(r_ND)

        _exnorm_bwd_fused[(N,)](
            x_ND,
            r_ND,
            dy_ND,
            dx_ND,
            dr_ND,
            x_ND.stride(0),
            D,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
            num_ctas=1,
        )

        dx = dx_ND.view_as(x)
        dr = dr_ND.view_as(residual)
        return dx, dr


def exnorm(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return ExNormFn.apply(x, residual)


def exnorm_ref(x, residual):
    post_scale = residual.pow(2).mean(dim=-1, keepdim=True).add(1).rsqrt()
    return post_scale * (x + residual)


def benchmark():
    iters = 120
    warmup = 20

    for name, method in (
        ("triton", exnorm),
        ("torch", exnorm_ref),
        ("torch-compiled", torch.compile(exnorm_ref, mode="max-autotune")),
    ):
        elapsed_ms = 0
        for i in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)

            x = torch.randn(16384, 1024, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            x.requires_grad_(True)
            r.requires_grad_(True)

            start.record()
            y = method(x, r)
            y.mean().backward()
            stop.record()

            if i >= warmup:
                torch.cuda.synchronize()
                elapsed_ms += start.elapsed_time(stop)
        print(f"{name}: {elapsed_ms / (iters - warmup):.3f} ms")


def memory_test(method):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(16384, 1024, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    x.requires_grad_(True)
    r.requires_grad_(True)
    y = method(x, r)
    y.mean().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def test_accuracy():
    x = torch.randn(1024, 128, device="cuda", dtype=torch.float32)
    r = torch.randn_like(x)

    x_tri = x.clone()
    r_tri = r.clone()
    x_tri.requires_grad_(True)
    r_tri.requires_grad_(True)
    y_tri = exnorm(x_tri, r_tri)
    y_tri.mean().backward()
    xg_tri = x_tri.grad
    rg_tri = r_tri.grad

    x_ref = x.clone()
    r_ref = r.clone()
    x_ref.requires_grad_(True)
    r_ref.requires_grad_(True)
    y_ref = exnorm_ref(x_ref, r_ref)
    y_ref.mean().backward()
    xg_ref = x_ref.grad
    rg_ref = r_ref.grad
    print("fwd error:", (y_tri - y_ref).abs().max().item())
    print("bwd error (x):", (xg_tri - xg_ref).abs().max().item())
    print("bwd error (r):", (rg_tri - rg_ref).abs().max().item())
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=1e-3)
    assert torch.allclose(xg_tri, xg_ref, atol=1e-5, rtol=1e-3)
    assert torch.allclose(rg_tri, rg_ref, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if args.memory:
        print(memory_test(exnorm) / 1024**2, "MB triton")
        print(memory_test(exnorm_ref) / 1024**2, "MB torch")
        print(
            memory_test(torch.compile(exnorm_ref, mode="max-autotune")) / 1024**2,
            "MB torch-compiled",
        )
    else:
        benchmark()
