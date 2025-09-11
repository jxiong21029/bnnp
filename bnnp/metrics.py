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


# Provided for convenience due to often being used for logged metrics.
@torch.no_grad()
def rms(x, dim=-1):
    return x.float().pow(2).mean(dim=dim).sqrt().mean()


class Metrics:
    def __init__(
        self,
        enabled: bool = True,
        use_wandb: bool = False,
        use_cuda_events: bool = False,
    ):
        if use_wandb:
            assert is_wandb_available, "wandb is not installed"
        self.enabled = enabled
        self.use_wandb = use_wandb
        self.use_cuda_events = use_cuda_events

        self.context = ""
        self.n = defaultdict(int)
        self.mean = defaultdict(float)

        self.timed_events = []
        self.last_t = None
        self.curr_event = None

    @torch.no_grad()
    def push(self, **metrics):
        """Push a metric which is later averaged and logged."""
        if not self.enabled:
            return

        for k, v in metrics.items():
            ck = self.context + k
            if isinstance(v, torch.Tensor):
                assert v.numel() == 1, f"{k} shape={v.shape}"
                delta = v.float().detach().view(()) - self.mean[ck]
            else:
                delta = float(v) - self.mean[ck]
            self.n[ck] += 1
            self.mean[ck] += delta / self.n[ck]

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
                now = torch.cuda.Event(enable_timing=True)
                now.record()
            else:
                now = time.perf_counter()
            self.timed_events.append((self.curr_event, self.last_t, now))
            if len(self.timed_events) >= 1024:
                self.tock()

        if name is not None:
            self.curr_event = self.context + name
            if self.use_cuda_events:
                self.last_t = torch.cuda.Event(enable_timing=True)
                self.last_t.record()
            else:
                self.last_t = time.perf_counter()
        else:
            self.curr_event = None

    def tock(self):
        """Synchronize CPU-GPU and push timed interval lengths."""
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

    def report(self):
        """Log metrics to wandb if use_wandb, otherwise to standard library logging."""
        if not self.enabled:
            return

        self.tick(None)
        results = {}
        for k in self.n:
            if self.n[k] == 0:
                continue
            if isinstance(self.mean[k], torch.Tensor):
                results[k] = self.mean[k].to("cpu", non_blocking=True)
            else:
                results[k] = self.mean[k]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results = {k: float(v) for k, v in results.items()}

        if self.use_wandb:
            wandb.log(results)
        else:
            logger.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        for k in self.n:
            self.n[k] = 0
            if isinstance(self.mean[k], torch.Tensor):
                self.mean[k].zero_()
            else:
                self.mean[k] = 0.0


class MetricsV2:
    def __init__(
        self, use_wandb: bool, enabled: bool = True, use_cuda_events: bool | None = None
    ):
        """
        Args:
            use_wandb: if True, log to wandb; otherwise log to standard library logging
            enabled: set to False to disable all methods
            use_cuda_events: if True, use torch.cuda.Event for timing (requires CUDA);
                defaults to True if CUDA is available, False otherwise
        """

        self.enabled = enabled
        self.use_wandb = use_wandb
        if enabled and (use_cuda_events or use_cuda_events is None):
            assert torch.cuda.is_available()
            self.use_cuda_events = True
        else:
            self.use_cuda_events = False
        if use_wandb:
            assert is_wandb_available, "wandb is not installed"

        self.context = ""
        self.data = {}

        self.timed_events = []
        self.last_t = None
        self.curr_event = None
        self.stream = torch.cuda.current_stream()

    @torch.no_grad()
    def push(self, **metrics):
        if not self.enabled:
            return

        for k, v in metrics.items():
            ck = self.context + k
            if isinstance(v, torch.Tensor):
                self.data[ck] = v.float().detach().view(())
            else:
                self.data[ck] = float(v)

    def tick(self, name: str | None):
        """Indicate the boundary of a timed interval.

        Example usage:
        >>> metrics = MetricsV2()
        >>> metrics.tick("forward")
        >>> loss = ...  # forward pass
        >>> metrics.tick("backward")
        >>> loss.backward()
        >>> metrics.log(train_loss=loss)  # automatically logs forward_sec, backward_sec
        """
        if not self.enabled:
            return

        if self.curr_event is not None:
            if self.use_cuda_events:
                now = torch.cuda.Event(enable_timing=True)
                now.record()
            else:
                now = time.perf_counter()
            self.timed_events.append((self.curr_event, self.last_t, now))

        if name is not None:
            self.curr_event = self.context + name
            if self.use_cuda_events:
                self.last_t = torch.cuda.Event(enable_timing=True)
                self.last_t.record()
            else:
                self.last_t = time.perf_counter()
        else:
            self.curr_event = None

    def log(self, commit: bool = True, **metrics):
        """Log metrics to wandb if use_wandb, otherwise to standard library logging."""

        if not self.enabled:
            return
        if metrics:
            self.push(**metrics)
        if not commit:
            return

        self.tick(None)
        if self.use_cuda_events and len(self.timed_events) > 0:
            torch.cuda.synchronize()
        for k, start, end in self.timed_events:
            k = str(k) + "_sec"
            if self.use_cuda_events:
                self.data[k] = start.elapsed_time(end) / 1000.0
            else:
                self.data[k] = end - start
        self.timed_events.clear()

        results = {}
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.to("cpu", non_blocking=True)
            else:
                results[k] = v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results = {k: float(v) for k, v in results.items()}

        if self.use_wandb and is_wandb_available:
            wandb.log(results)
        else:
            logger.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        self.data.clear()
