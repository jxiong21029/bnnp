import torch
from torch.cuda import max_memory_allocated as peak_vram

from bnnp.nn.patchify import grid_sample_batched
from bnnp.nn.tests.test_patchify import grid_sample_batched_ref

grid_sample_batched_ref_c = torch.compile(grid_sample_batched_ref)
grid_sample_batched_c = torch.compile(grid_sample_batched)


def main():
    torch.manual_seed(0)

    iters = 100
    warmup = 10
    batch_size = 16
    channels = 3
    image_size = 128
    n_grids = 224
    grid_size = 16

    test_image = torch.randn(batch_size, channels, image_size, image_size).cuda()
    test_grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1

    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    out_ref = grid_sample_batched_ref(test_image, test_grid)
    assert out_ref.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (ref, uncompiled): {(peak_vram() - pre) // (1 << 20):,} MiB")

    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    out_ref_c = grid_sample_batched_ref_c(test_image, test_grid)
    assert out_ref_c.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (ref, compiled): {(peak_vram() - pre) // (1 << 20):,} MiB")

    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    out_ours = grid_sample_batched(test_image, test_grid)
    assert out_ours.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (uncompiled): {(peak_vram() - pre) // (1 << 20):,} MiB")
    print(f"error (uncompiled): {(out_ours - out_ref).abs().max():.8f}")

    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    out_ours_c = grid_sample_batched_c(test_image, test_grid)
    assert out_ours_c.shape == (batch_size, n_grids, channels, grid_size, grid_size)
    print(f"peak vram (compiled): {(peak_vram() - pre) // (1 << 20):,} MiB")
    print(f"error (compiled): {(out_ours_c - out_ref).abs().max():.8f}")

    del out_ref, out_ref_c, out_ours, out_ours_c

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = grid_sample_batched_ref(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (ref, uncompiled): {elapsed / (iters - warmup)} ms")

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = grid_sample_batched_ref_c(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (ref, compiled): {elapsed / (iters - warmup)} ms")

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = grid_sample_batched(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (ours, uncompiled): {elapsed / (iters - warmup)} ms")

    elapsed = 0.0
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        image = torch.randn(batch_size, channels, image_size, image_size).cuda()
        grid = 2 * torch.rand(batch_size, n_grids, grid_size, grid_size, 2).cuda() - 1
        start.record()
        _ = grid_sample_batched_c(image, grid)
        stop.record()

        if i >= warmup:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(stop)
    print(f"time/iter (ours, compiled): {elapsed / (iters - warmup)} ms")


if __name__ == "__main__":
    main()
