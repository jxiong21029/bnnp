import logging
import time
from collections import defaultdict
from typing import Literal

import torch

try:
    import wandb

except ImportError:
    wandb = None

logger = logging.getLogger(__name__)
CUDA_AVAILABLE = torch.cuda.is_available()


class Metrics:
    def __init__(
        self,
        enabled: bool = True,
        use_wandb: bool = False,
        use_cuda_events: bool = CUDA_AVAILABLE,
    ):
        if use_wandb:
            assert wandb is not None, "wandb is not installed"
        self.enabled = enabled
        self.use_wandb = use_wandb
        self.use_cuda_events = use_cuda_events

        self.context = ""
        self.n = defaultdict(int)
        self.mean: dict[str, float | torch.Tensor] = defaultdict(float)

        self.timed_events = []
        self.event_reductions: dict[str, Literal["mean", "sum"]] = {}
        self.curr_event = None
        self.start_t = None
        self.stop_t = None
        self.start_stream = None

    @torch.compiler.disable
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

    @torch.compiler.disable
    def tick(self, name: str | None, reduction: Literal["mean", "sum"] = "mean"):
        """Indicate the boundary of a timed interval.

        Repeated intervals with the same name are averaged between commits, or summed if
        reduction="sum". A name must use the same reduction until the next commit.

        Example usage:
            metrics = Metrics()
            metrics.tick("forward")
            loss = ...  # forward pass
            metrics.tick("backward")
            loss.backward()
            metrics.commit()  # adds forward_sec and backward_sec to logged metrics
        """
        if not self.enabled:
            return

        event_name = self.context + name if name is not None else None
        if event_name is not None:
            if reduction not in ("mean", "sum"):
                raise ValueError(f"Invalid timing reduction: {reduction}")
            previous = self.event_reductions.get(event_name)
            if previous is not None and previous != reduction:
                raise ValueError(
                    f"Conflicting reduction for {event_name!r}: {previous!r} vs {reduction!r}. "
                    "Use the same reduction until the next commit."
                )

        if self.curr_event is not None:
            if self.use_cuda_events:
                self.stop_t.record(self.start_stream)  # ty: ignore
            else:
                self.stop_t = time.perf_counter()
            self.timed_events.append((self.curr_event, self.start_t, self.stop_t))
            if len(self.timed_events) >= 1024:
                self.push_ticks()

        if event_name is not None:
            self.curr_event = event_name
            self.event_reductions[event_name] = reduction
            if self.use_cuda_events:
                self.start_t = torch.cuda.Event(enable_timing=True)
                self.stop_t = torch.cuda.Event(enable_timing=True)
                self.start_stream = torch.cuda.current_stream()
                self.start_t.record(self.start_stream)
            else:
                self.start_t = time.perf_counter()
        else:
            self.curr_event = None

    @torch.compiler.disable
    def push_ticks(self):
        """Push buffer of timing events."""
        if not self.enabled:
            return

        for k, start, end in self.timed_events:
            if self.use_cuda_events:
                end.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                elapsed = end - start
            summed = self.event_reductions[k] == "sum"
            k = str(k) + "_sec"
            self.n[k] += 1
            if summed:
                self.mean[k] += elapsed
            else:
                delta = elapsed - self.mean[k]
                self.mean[k] += delta / self.n[k]
        self.timed_events.clear()

    @torch.compiler.disable
    def commit(self, _step: int | None = None, **metrics):
        """Log metrics to wandb if use_wandb, otherwise to (standard library) logging."""
        if not self.enabled:
            return
        if metrics:
            self.push(**metrics)

        self.tick(None)
        self.push_ticks()
        results = {}
        cuda_devices = set()
        for k in self.n:
            if self.n[k] == 0:
                continue
            value = self.mean[k]
            if isinstance(value, torch.Tensor):
                results[k] = value.to("cpu", non_blocking=value.is_cuda)
                if value.is_cuda:
                    cuda_devices.add(value.device)
            else:
                results[k] = value
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        results = {k: float(v) for k, v in results.items()}

        if self.use_wandb and wandb is not None:
            if _step is not None:
                wandb.log(results, step=_step, commit=True)
            else:
                wandb.log(results)
        else:
            logger.info(", ".join(f"{k}: {v:.4g}" for k, v in results.items()))

        for k in self.n:
            self.n[k] = 0
            if isinstance(self.mean[k], torch.Tensor):
                self.mean[k].zero_()  # ty: ignore
            else:
                self.mean[k] = 0.0
        self.event_reductions.clear()
