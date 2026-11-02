"""Compose the Harbor layers a rollout or an evaluation runs from: agent, run and profile.

Every layer is plain data that Harbor's own ``TrialConfig`` or ``JobConfig`` validates.
Inference layers always carry ``verifier.disable``; the ``[verifier]`` table reaches only an
evaluation regrade or an operator's probe job. Credentials appear only as ``${NAME}``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from kraken_harness import secrets
from kraken_harness.run_config import TRIAL_LEVEL, Profile, RunConfig


def agent_table(
    agent: dict[str, Any], run: RunConfig, profile: Profile | None
) -> dict[str, Any]:
    """The agent under test with the run's allowlisted extras and the profile's gate address."""
    secrets.refuse_literal_secrets({"agent": agent}, "agent table", run.env_from_host)
    merged = {**agent, **{k: v for k, v in run.agent.items() if k != TRIAL_LEVEL}}
    clash = sorted(k for k in run.agent if k in agent and agent[k] != run.agent[k])
    if clash:
        raise ValueError(f"agent.toml and run.toml set {', '.join(clash)} differently")
    if profile is not None:
        explicit = merged.get("base_url")
        if explicit is not None and explicit != profile.base_url:
            raise ValueError(
                f"base_url {explicit} disagrees with --compute {profile.name} "
                f"({profile.base_url}); drop it so the profile fills it"
            )
        merged["base_url"] = profile.base_url
    key_name = merged.get("model_api_key_name")
    if isinstance(key_name, str) and key_name in run.env_from_host:
        env = dict(merged.get("env") or {})
        env.setdefault(key_name, secrets.template(key_name))
        merged["env"] = env
    return merged


def environment_table(run: RunConfig) -> dict[str, Any]:
    table: dict[str, Any] = {}
    compose = run.compose_files()
    if compose:
        table["extra_docker_compose"] = [str(path) for path in compose]
    hosts = run.environment.get("extra_allowed_hosts")
    if hosts:
        table["extra_allowed_hosts"] = list(hosts)
    return table


def verifier_table(run: RunConfig, profile: Profile | None) -> dict[str, Any]:
    """The regrade verifier: the run's env and hosts, plus the profile's judge address."""
    env = dict(run.verifier.get("env", {}))
    hosts = list(run.verifier.get("extra_allowed_hosts", []))
    if profile is not None:
        explicit = env.get(profile.judge_env)
        if explicit is not None and explicit != profile.judge_url:
            raise ValueError(
                f"[verifier] env {profile.judge_env} disagrees with --compute {profile.name} "
                f"({profile.judge_url}); drop it so the profile fills it"
            )
        env[profile.judge_env] = profile.judge_url
        if profile.gate_host not in hosts:
            hosts.append(profile.gate_host)
    table: dict[str, Any] = {}
    if env:
        table["env"] = env
    if hosts:
        table["extra_allowed_hosts"] = hosts
    return table


def _trial_level(run: RunConfig) -> dict[str, Any]:
    value = run.agent.get(TRIAL_LEVEL)
    return {} if value is None else {TRIAL_LEVEL: float(value)}


def trial_layer(
    task_dir: Path,
    trial_name: str,
    trials_dir: Path,
    agent: dict[str, Any],
    run: RunConfig,
) -> dict[str, Any]:
    """A ``harbor trial start`` layer for one rollout, verification off."""
    return {
        "task": {"path": str(task_dir)},
        "trial_name": trial_name,
        "trials_dir": str(trials_dir),
        "agent": agent,
        "environment": environment_table(run),
        "verifier": {"disable": True},
        **_trial_level(run),
    }


def job_config(
    task_dir: Path,
    job_name: str,
    jobs_dir: Path,
    agent: dict[str, Any],
    run: RunConfig,
    *,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A one-trial ``harbor run`` job; verification is off unless ``verifier`` is a probe's."""
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "agents": [agent],
        "tasks": [{"path": str(task_dir)}],
        "environment": environment_table(run),
        "verifier": {"disable": True} if verifier is None else verifier,
        **_trial_level(run),
    }


def regrade_layer(
    bundle: Path,
    source_trial: Path,
    source_agent: dict[str, Any],
    trial_name: str,
    trials_dir: Path,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    """The layer Harbor's own regrade builds, plus the verifier hosts its command cannot take."""
    return {
        "task": {"path": str(bundle)},
        "trial_name": trial_name,
        "trials_dir": str(trials_dir),
        "agent": source_agent,
        "environment": {},
        "verifier": verifier,
        "source_trial": {
            "action": "regrade",
            "type": "local",
            "path": str(source_trial),
        },
    }


def compose_digests(run: RunConfig) -> list[dict[str, str]]:
    """The SHA-256 of every extra compose file, so a network policy change shows in the pin."""
    out = []
    for path in run.compose_files():
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "absent"
        )
        out.append({"name": path.name, "sha256": digest})
    return out
