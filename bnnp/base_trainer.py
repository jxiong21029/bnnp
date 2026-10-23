import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from pydantic import BaseModel, ConfigDict

try:
    import wandb
except ImportError:
    wandb = None

from .metrics import Metrics

logger = logging.getLogger(__name__)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_name: str | None = None
    run_base_dir: str
    use_wandb: bool
    wandb_entity: str = "archibox"
    wandb_project: str = "wddq"
    wandb_group: str
    seed: int

    n_steps: int
    early_valid_steps: list[int]
    valid_interval: int
    tqdm_kwargs: dict = {"ncols": 100, "mininterval": 3.0, "smoothing": 1.0}

    profile: bool = False
    profile_dir: str | None = "logs/profile/"
    profile_record_shapes: bool = False
    profile_with_stack: bool = False


MASK64 = 2**64 - 1


def splitmix64(x: int) -> int:
    """Scramble a 64-bit int with one step of SplitMix64."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def split_rng(seed: int, n: int = 2) -> list[int]:
    """Split a 64-bit seed into n statistically independent 64-bit seeds."""
    seed &= MASK64
    children = []
    x = seed
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) & MASK64
        children.append(splitmix64(x))
    return children


class BaseTrainer:
    """Simple DDP trainer."""

    def __init__(self, config: TrainerConfig):
        self.config = config

        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s][%(name)s:%(levelname)s] %(message)s"
        )

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0
        self.is_distributed = self.world_size > 1
        self.device = torch.device("cuda", self.local_rank)

        self.metrics = Metrics(
            use_wandb=self.is_main_process and config.use_wandb,
            enabled=self.is_main_process,
            use_cuda_events=True,
        )
        self.profiler: torch.profiler.profile | None = None

        self.metrics.enabled = self.is_main_process
        torch.cuda.set_device(self.device)
        self._local_seed = split_rng(config.seed, self.world_size)[self.rank]
        random.seed(self.next_seed())
        np.random.seed(self.next_seed() % (1 << 32))
        torch.manual_seed(self.next_seed())
        torch.cuda.manual_seed(self.next_seed())

        if self.is_distributed:
            dist.init_process_group(backend="nccl", device_id=self.device)
            logger.info(
                f"running distributed: rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}"
            )
        else:
            assert self.rank == self.local_rank == 0
            logger.info("running non-distributed.")

        logging.getLogger().setLevel(logging.INFO if self.is_main_process else logging.WARNING)
        logger.info("config: " + config.model_dump_json(indent=4))

        run_base_dir = Path(config.run_base_dir)
        run_base_dir.mkdir(parents=True, exist_ok=True)
        if config.use_wandb:
            if self.is_main_process and wandb is not None:
                wandb.init(
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    name=config.run_name,
                    dir=run_base_dir,
                    config=config.model_dump(),
                    group=config.wandb_group,
                    resume="never",
                )
                assert wandb.run is not None
                self.run_dir = Path(wandb.run.dir)
            if self.is_distributed:
                obj = [str(self.run_dir) if self.is_main_process else None]
                dist.broadcast_object_list(obj, src=0)
                if not self.is_main_process:
                    assert obj[0] is not None
                    self.run_dir = Path(obj[0])
        else:
            self.run_dir = run_base_dir / "local"
        logger.info(f"[rank{self.rank}] run_dir: {self.run_dir}")

        if self.is_main_process:
            self.run_dir.mkdir(exist_ok=True)

        if self.is_main_process:

            def _diagnostic_output(command: list[str]) -> str | None:
                """Collect optional diagnostics without aborting training if a command fails."""
                try:
                    return subprocess.check_output(
                        command, text=True, stderr=subprocess.STDOUT
                    ).strip()
                except (OSError, subprocess.CalledProcessError) as exc:
                    logger.warning(f"Could not collect diagnostic {' '.join(command)!r}: {exc}")
                    return None

            commit_hash = _diagnostic_output(["git", "rev-parse", "--short", "HEAD"])
            diff = _diagnostic_output(["git", "diff", "HEAD"])
            hostname = _diagnostic_output(["hostname"])
            nvidia_smi = _diagnostic_output(["nvidia-smi"])
            logger.info(f"git commit hash: {commit_hash}")
            logger.info(f"git diff:\n{diff}")
            logger.info(f"running on host: {hostname}")
            logger.info(f"nvidia-smi:\n{nvidia_smi}")
            self.info = {
                "git_commit": commit_hash,
                "git_diff": diff,
                "hostname": hostname,
                "nvidia_smi": nvidia_smi,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

        if self.config.profile and self.is_main_process:
            profile_dir = (
                Path(self.config.profile_dir)
                if self.config.profile_dir is not None
                else (self.run_dir / "profiler")
            )
            profile_dir.mkdir(parents=True, exist_ok=True)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            profile_schedule = torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1)

            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=profile_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(profile_dir), worker_name=f"rank{self.rank}"
                ),
                record_shapes=self.config.profile_record_shapes,
                profile_memory=True,
                with_stack=self.config.profile_with_stack,
            )
            self.profiler.start()
            logger.info(f"profiler enabled. Writing traces to {profile_dir}")

        self.step = 0
        self.metrics.context = "train_"

    def next_seed(self) -> int:
        self._local_seed, subseed = split_rng(self._local_seed, 2)
        return subseed

    def should_run_validation(self) -> bool:
        return (
            ((self.step + 1) in self.config.early_valid_steps)
            or (self.step + 1 == self.config.n_steps)
            or (
                self.config.valid_interval > 0
                and (
                    len(self.config.early_valid_steps) == 0
                    or (self.step + 1) > max(self.config.early_valid_steps)
                )
                and (self.step + 1) % self.config.valid_interval == 0
            )
        )

    def run(self):
        try:
            with tqdm.tqdm(
                total=self.config.n_steps,
                desc="train",
                disable=(not self.is_main_process) or (not self.config.use_wandb),
                **self.config.tqdm_kwargs,
            ) as pbar:
                if self.step > 0:
                    pbar.update(self.step)
                while self.step < self.config.n_steps:
                    self.train_step()

                    if self.should_run_validation():
                        self.valid_epoch()
                    self.metrics.commit(_step=self.step + 1)

                    self.step += 1
                    pbar.update(1)
                    if self.profiler is not None:
                        self.profiler.step()
        finally:
            if self.profiler is not None:
                self.profiler.stop()

    def train_step(self):
        pass

    @torch.no_grad()
    def valid_epoch(self):
        pass

    def gather_dict(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.is_distributed:
            return data

        ret = {}
        for k in sorted(data):
            global_v = data[k].detach().clone().float()
            dist.all_reduce(global_v, op=dist.ReduceOp.SUM)
            global_v /= self.world_size
            ret[k] = global_v
        return ret
