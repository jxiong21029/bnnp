import logging
import time
from collections import defaultdict

import torch

try:
    import wandb

    is_wandb_available = True
except ImportError:
    wandb = None
    is_wandb_available = False

logger = logging.getLogger(__name__)


# Provided for convenience.
@torch.no_grad()
def rms(x, dim=-1):
    return x.float().pow(2).mean(dim=dim).sqrt().mean()


class Metrics:
    def __init__(
        self,
        enabled: bool = True,
        use_wandb: bool = False,
        use_cuda_events: bool = False,
        log_at_most_once: bool = False,
    ):
        if use_wandb:
            assert is_wandb_available, "wandb is not installed"
        self.enabled = enabled
        self.use_wandb = use_wandb
        self.use_cuda_events = use_cuda_events
        self.report_logging_gt_once = log_at_most_once

        self.context = ""
        self.n = defaultdict(int)
        self.mean = defaultdict(float)

        self.timed_events = []
        self.curr_event = None
        self.start_t = None
        self.stop_t = None

    @torch.compiler.disable()
    @torch.no_grad()
    def push(self, **metrics):
        """Push a metric which is later averaged and logged."""
        if not self.enabled:
            return

        for k, v in metrics.items():
            ck = self.context + k
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1, f"{k} shape={v.shape}"
                delta = v.detach().clone().float().view(()) - self.mean[ck]
            else:
                delta = float(v) - self.mean[ck]
            self.n[ck] += 1
            self.mean[ck] += delta / self.n[ck]

    @torch.compiler.disable()
    def tick(self, name: str | None):
        """Indicate the boundary of a timed interval.

        Example usage:
        >>> metrics = MetricsV2()
        >>> metrics.tick("forward")
        >>> loss = ...  # forward pass
        >>> metrics.tick("backward")
        >>> loss.backward()
        >>> metrics.tick(None)  # stop timing
        >>> ...
        >>> metrics.report()  # adds forward_sec and backward_sec to logged metrics
        """
        if not self.enabled:
            return

        if self.curr_event is not None:
            if self.use_cuda_events:
                self.stop_t.record()
            else:
                self.stop_t = time.perf_counter()
            self.timed_events.append((self.curr_event, self.start_t, self.stop_t))
            if len(self.timed_events) >= 1024:
                self.push_ticks()

        if name is not None:
            self.curr_event = self.context + name
            if self.use_cuda_events:
                self.start_t = torch.cuda.Event(enable_timing=True)
                self.stop_t = torch.cuda.Event(enable_timing=True)
                self.start_t.record()
            else:
                self.start_t = time.perf_counter()
        else:
            self.curr_event = None

    @torch.compiler.disable()
    def push_ticks(self):
        """Push buffer of timing events."""
        if not self.enabled:
            return

        if self.use_cuda_events and len(self.timed_events) > 0:
            torch.cuda.synchronize()
        for k, start, end in self.timed_events:
            if self.use_cuda_events:
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                elapsed = end - start
            k = str(k) + "_sec"
            delta = elapsed - self.mean[k]
            self.n[k] += 1
            self.mean[k] += delta / self.n[k]
        self.timed_events.clear()

    @torch.compiler.disable()
    def commit(self, _step: int | None = None, **metrics):
        """Log metrics to wandb if use_wandb, otherwise to standard library logging."""
        if not self.enabled:
            return
        if metrics:
            self.push(**metrics)

        self.tick(None)
        self.push_ticks()
        results = {}
        for k in self.n:
            if self.n[k] == 0:
                continue
            if self.n[k] > 1 and self.report_logging_gt_once:
                logger.error(
                    f"Logged metric {k} more than once when log_at_most_once=True"
                )
                self.report_logging_gt_once = False
            if isinstance(self.mean[k], torch.Tensor):
                results[k] = self.mean[k].to("cpu", non_blocking=True)
            else:
                results[k] = self.mean[k]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results = {k: float(v) for k, v in results.items()}

        if self.use_wandb and is_wandb_available:
            if _step is not None:
                wandb.log(results, step=_step)
            else:
                wandb.log(results)
        else:
            logger.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        for k in self.n:
            self.n[k] = 0
            if isinstance(self.mean[k], torch.Tensor):
                self.mean[k].zero_()
            else:
                self.mean[k] = 0.0
