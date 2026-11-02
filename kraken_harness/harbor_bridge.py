"""The one boundary to the pinned Harbor release.

Harbor owns orchestration: it builds each instance's environment, runs the agent under
test, and runs a bundle's own ``tests/`` when it regrades a recorded trial. This module
names the pinned release, refuses any other, validates layers with Harbor's own models,
and composes the commands Harbor runs. It scores nothing.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMAND = "harbor"
LOCK_FILE = "harbor.lock"


@dataclass(frozen=True, slots=True)
class Harbor:
    release: str
    commit: str

    def require_lock(self, root: Path) -> None:
        """Refuse a parent whose ``harbor.lock`` records another Harbor release or commit."""
        lock = root / LOCK_FILE
        if not lock.is_file():
            raise RuntimeError(f"{lock} is absent, so {root} pins no Harbor release")
        recorded = json.loads(lock.read_text(encoding="utf-8"))
        release, commit = recorded.get("reference_release"), recorded.get("commit")
        if (release, commit) != (self.release, self.commit):
            raise RuntimeError(
                f"{lock} pins harbor {release} at {commit}; "
                f"this benchmark pins {self.release} at {self.commit}"
            )

    def require_installed(self) -> None:
        """Refuse an environment whose installed Harbor is not the pinned release."""
        installed = importlib.metadata.version("harbor")
        if installed != self.release:
            raise RuntimeError(
                f"harbor {installed} is installed; this benchmark pins {self.release}"
            )

    @staticmethod
    def validate_trial(layer: dict[str, Any]) -> None:
        """Validate one trial layer with Harbor's own ``TrialConfig`` model."""
        from harbor.models.trial.config import TrialConfig

        TrialConfig.model_validate(layer)

    @staticmethod
    def validate_job(config: dict[str, Any]) -> None:
        """Validate one job config with Harbor's own ``JobConfig`` model."""
        from harbor.models.job.config import JobConfig

        JobConfig.model_validate(config)

    @staticmethod
    def validate_task(task_dir: Path, *, disable_verification: bool) -> None:
        """Refuse a task directory Harbor itself would not load."""
        from harbor.models.task.task import Task

        if not Task.is_valid_dir(task_dir, disable_verification=disable_verification):
            raise ValueError(f"{task_dir}: Harbor does not accept this task directory")

    @staticmethod
    def validate_regrade(task_dir: Path, task_name: str | None) -> None:
        """Refuse a regrade Harbor's own ``trial regrade`` would refuse, with its own checks."""
        from harbor.trial.regrade import check_task_regradable, local_task_name

        if task_name is not None and local_task_name(task_dir) != task_name:
            raise ValueError(
                f"{task_dir}: provides task {local_task_name(task_dir)!r}, "
                f"but the rollout ran {task_name!r}"
            )
        reason = check_task_regradable(task_dir)
        if reason is not None:
            raise ValueError(reason)

    @staticmethod
    def trial_start(layer: Path) -> list[str]:
        return [COMMAND, "trial", "start", "--config", str(layer)]

    @staticmethod
    def job_run(config: Path) -> list[str]:
        return [COMMAND, "run", "--config", str(config), "--yes"]

    @staticmethod
    def trial_regrade(
        source: Path, task_path: Path, trial_name: str, trials_dir: Path
    ) -> list[str]:
        return [
            COMMAND,
            "trial",
            "regrade",
            str(source),
            "--task-path",
            str(task_path),
            "--trial-name",
            trial_name,
            "--trials-dir",
            str(trials_dir),
        ]

    @staticmethod
    def run(command: list[str], env: Mapping[str, str] | None = None) -> int:
        executable = shutil.which(command[0])
        if executable is None:
            raise RuntimeError(
                f"{command[0]} is not on PATH; run inside the harness environment"
            )
        return subprocess.run(
            [executable, *command[1:]],
            check=False,
            env=None if env is None else dict(env),
        ).returncode
