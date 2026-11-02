"""Summarize one bundle's graded rollouts as the group counts an author pilot records.

Rollouts are taken in attempt order and cut into consecutive groups of ``group_size``. A
rollout passes when Harbor's regrade reward meets the threshold, and a group passes when any
member passes, the best-case tally Trinity's saturation bound reads. The harness only counts
rewards Harbor wrote; it computes no reward of its own. The summary also stages the graded
trial directories, secret-scanned, for ``.seed/pilots/rollouts/<uuid>/``, and prints the
``pipeline.py self-pilot`` command that records them. It never runs that command.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kraken_harness import secrets

CONFIDENCE = 0.95
SATURATION = 0.40
RESULT = "result.json"
REWARD_KEY = "reward"
_BISECTION_STEPS = 200


def _tail(successes: int, trials: int, rate: float) -> float:
    """P(X >= successes) for X ~ Binomial(trials, rate), summed exactly term by term."""
    return math.fsum(
        math.comb(trials, k) * rate**k * (1.0 - rate) ** (trials - k)
        for k in range(successes, trials + 1)
    )


def clopper_pearson_lower(
    successes: int, trials: int, confidence: float = CONFIDENCE
) -> float:
    """The one-sided exact Clopper-Pearson lower bound on the success rate.

    It is the rate at which seeing ``successes`` or more has probability ``1 - confidence``;
    with no success it is exactly 0, and with all successes it is ``(1 - confidence)**(1/n)``.
    """
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(
            f"successes must lie in [0, trials] with trials >= 1, got {successes}/{trials}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha = 1.0 - confidence
    if successes == 0:
        return 0.0
    if successes == trials:
        return alpha ** (1.0 / trials)
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if _tail(successes, trials, middle) < alpha:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class Graded:
    attempt: int
    reward: float | None
    rollout_dir: Path
    evaluation_dir: Path


@dataclass(frozen=True, slots=True)
class Summary:
    uuid: str
    group_size: int
    groups: int
    passing_groups: int
    lower_bound: float
    rollouts: tuple[Graded, ...]

    @property
    def saturated(self) -> bool:
        return self.lower_bound >= SATURATION


def reward_of(evaluation_dir: Path) -> float | None:
    """The reward Harbor recorded in a regrade's ``result.json``, or None when it wrote none."""
    path = evaluation_dir / RESULT
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    verifier = result.get("verifier_result") if isinstance(result, dict) else None
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    value = rewards.get(REWARD_KEY) if isinstance(rewards, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def summarize(
    uuid: str, graded: list[Graded], group_size: int, threshold: float
) -> Summary:
    """Cut ``graded`` into groups and count those holding a rollout that meets ``threshold``."""
    if group_size < 1:
        raise ValueError("group size must be at least 1")
    if not graded:
        raise LookupError(f"bundle {uuid} has no graded rollout in this work directory")
    ordered = sorted(graded, key=lambda row: row.attempt)
    attempts = [row.attempt for row in ordered]
    if len(set(attempts)) != len(attempts):
        raise ValueError(f"bundle {uuid} was graded twice for one attempt")
    if len(ordered) % group_size:
        raise ValueError(
            f"bundle {uuid} has {len(ordered)} graded rollouts, not a whole number of groups "
            f"of {group_size}"
        )
    groups = [ordered[i : i + group_size] for i in range(0, len(ordered), group_size)]
    passing = sum(
        any(row.reward is not None and row.reward >= threshold for row in group)
        for group in groups
    )
    return Summary(
        uuid,
        group_size,
        len(groups),
        passing,
        clopper_pearson_lower(passing, len(groups)),
        tuple(ordered),
    )


def _copy_regular(source: Path, target: Path) -> None:
    """Copy every regular file under ``source``, refusing a symbolic link anywhere."""
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{path}: a staged rollout holds no symbolic link")
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def stage(summary: Summary, staging: Path, env_names: tuple[str, ...]) -> Path:
    """Copy the graded trial directories under ``staging/<uuid>/``, refused if any holds a secret."""
    target = staging / summary.uuid
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; choose a fresh staging directory"
        )
    hits: list[tuple[str, str]] = []
    for row in summary.rollouts:
        for label, directory in (
            ("rollout", row.rollout_dir),
            ("evaluation", row.evaluation_dir),
        ):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"{directory}: the {label} trial directory is absent"
                )
            hits += [
                (f"r{row.attempt}/{label}/{name}", found)
                for name, found in secrets.scan_tree(directory, env_names)
            ]
    if hits:
        listed = ", ".join(f"{path} ({name})" for path, name in hits[:20])
        raise ValueError(f"refusing to stage: secret material found in {listed}")
    try:
        for row in summary.rollouts:
            _copy_regular(row.rollout_dir, target / f"r{row.attempt}" / "rollout")
            _copy_regular(row.evaluation_dir, target / f"r{row.attempt}" / "evaluation")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def self_pilot_command(summary: Summary, run_id: str, digest: str) -> str:
    return (
        "uv run --project ./trinity --no-dev ./trinity/tools/pipeline.py self-pilot ./ "
        f"{summary.uuid} --run-id {run_id} --harness-config-digest {digest} "
        f"--group-size {summary.group_size} --groups {summary.groups} "
        f"--passing-groups {summary.passing_groups}"
    )


def as_record(summary: Summary, threshold: float) -> dict[str, Any]:
    return {
        "bundle_uuid": summary.uuid,
        "pass_threshold": threshold,
        "group_size": summary.group_size,
        "groups": summary.groups,
        "passing_groups": summary.passing_groups,
        "confidence": CONFIDENCE,
        "clopper_pearson_lower": summary.lower_bound,
        "saturated": summary.saturated,
    }
