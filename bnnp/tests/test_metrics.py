import pytest
import torch

from bnnp import Metrics, MetricsV2


def test_timing_cpu():
    metrics = Metrics(enabled=True, use_wandb=False, use_cuda_events=False)
    metrics.tick("forward")
    W = torch.randn(32, 16)
    W.requires_grad_(True)
    x = torch.randn(16) @ W.t()
    loss = x.mean()

    metrics.tick("backward")
    loss.backward()
    assert hasattr(W, "grad")
    assert isinstance(W.grad, torch.Tensor)
    assert W.grad.std() > 0.0

    metrics.tick(None)
    metrics.tock()

    assert "forward_sec" in metrics.n
    assert "forward_sec" in metrics.mean
    assert "backward_sec" in metrics.n
    assert "backward_sec" in metrics.mean
    assert metrics.mean["forward_sec"] > 0.0
    assert metrics.mean["backward_sec"] > 0.0
    assert len(metrics.timed_events) == 0
    metrics.report()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda_unavailable")
def test_timing_cuda_event():
    metrics = Metrics(enabled=True, use_wandb=False, use_cuda_events=True)
    metrics.tick("forward")
    W = torch.randn(32, 16, device="cuda")
    W.requires_grad_(True)
    x = torch.randn(16, device="cuda") @ W.t()
    loss = x.mean()

    metrics.tick("backward")
    loss.backward()
    assert hasattr(W, "grad")
    assert isinstance(W.grad, torch.Tensor)
    assert W.grad.std() > 0.0

    metrics.tick(None)
    metrics.tock()

    assert "forward_sec" in metrics.n
    assert "forward_sec" in metrics.mean
    assert "backward_sec" in metrics.n
    assert "backward_sec" in metrics.mean
    assert metrics.mean["forward_sec"] > 0.0
    assert metrics.mean["backward_sec"] > 0.0
    assert len(metrics.timed_events) == 0


def test_metricsv2(caplog):
    metrics = MetricsV2(use_wandb=False)
    metrics.tick("forward")
    W = torch.randn(32, 16)
    W.requires_grad_(True)
    x = torch.randn(16) @ W.t()
    loss = x.mean()

    metrics.tick("backward")
    loss.backward()
    assert hasattr(W, "grad")
    assert isinstance(W.grad, torch.Tensor)
    assert W.grad.std() > 0.0

    with caplog.at_level("INFO"):
        metrics.log(train_loss=loss)
    assert "forward_sec" in caplog.text
    assert "backward_sec" in caplog.text
    assert "train_loss" in caplog.text
