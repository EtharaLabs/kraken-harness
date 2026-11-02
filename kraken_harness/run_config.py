"""The ``run.toml`` an operator passes with ``--run-config``: allowlisted Harbor extras.

Only the tables and keys below are accepted; any other key is refused. ``[agent]``,
``[environment]`` and ``[verifier]`` merge into each Harbor layer, which Harbor's own
models then validate. ``env_from_host`` names the host variables Harbor resolves at launch,
and each ``[compute.<name>]`` profile says where Docker runs and which gate host the
containers reach, so one agent table serves every host.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kraken_harness import secrets

TOP_LEVEL = frozenset({"env_from_host", "agent", "environment", "verifier", "compute"})
AGENT_KEYS = frozenset(
    {
        "base_url",
        "model_api_format",
        "model_api_key_name",
        "agent_setup_timeout_multiplier",
    }
)
ENVIRONMENT_KEYS = frozenset({"extra_docker_compose", "extra_allowed_hosts"})
VERIFIER_KEYS = frozenset({"env", "extra_allowed_hosts"})
PROFILE_KEYS = frozenset(
    {
        "docker_host",
        "gate_host",
        "agent_port",
        "judge_port",
        "judge_env",
        "trinity_target",
    }
)
# Harbor reads this one at the trial level, not inside the agent table.
TRIAL_LEVEL = "agent_setup_timeout_multiplier"
DEFAULT_AGENT_PORT = 8642
DEFAULT_JUDGE_PORT = 8790
DEFAULT_JUDGE_ENV = "LITELLM_PROXY_API_BASE"
SHARED_COMPUTE = "compute.yaml"


@dataclass(frozen=True, slots=True)
class Profile:
    """Where one host runs Docker and the gate address its containers reach."""

    name: str
    gate_host: str
    docker_host: str | None = None
    agent_port: int = DEFAULT_AGENT_PORT
    judge_port: int = DEFAULT_JUDGE_PORT
    judge_env: str = DEFAULT_JUDGE_ENV
    source: str = "run.toml"

    @property
    def base_url(self) -> str:
        return f"http://{self.gate_host}:{self.agent_port}"

    @property
    def judge_url(self) -> str:
        return f"http://{self.gate_host}:{self.judge_port}"

    def record(self) -> dict[str, Any]:
        """The profile as a rollout or evaluation row records it: no credential lives here."""
        return {
            "profile": self.name,
            "docker_host": self.docker_host,
            "gate_host": self.gate_host,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    env_from_host: tuple[str, ...] = ()
    agent: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    directory: Path = Path(".")

    def compose_files(self) -> list[Path]:
        """The ``extra_docker_compose`` paths, resolved against the run.toml directory."""
        raw = self.environment.get("extra_docker_compose", [])
        return [
            path
            if (path := Path(str(item)).expanduser()).is_absolute()
            else self.directory / path
            for item in raw
        ]


def _table(
    document: Mapping[str, Any], name: str, keys: frozenset[str], where: str
) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{where}: [{name}] must be a table")
    unknown = sorted(set(value) - keys)
    if unknown:
        raise ValueError(
            f"{where}: [{name}] refuses {', '.join(unknown)}; allowed: {', '.join(sorted(keys))}"
        )
    return dict(value)


def _string_list(table: Mapping[str, Any], key: str, where: str) -> None:
    value = table.get(key)
    if value is not None and (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{where}: {key} must be a list of non-empty strings")


def parse(document: Mapping[str, Any], where: str, directory: Path) -> RunConfig:
    unknown = sorted(set(document) - TOP_LEVEL)
    if unknown:
        raise ValueError(
            f"{where}: refuses {', '.join(unknown)}; allowed: {', '.join(sorted(TOP_LEVEL))}"
        )
    names = secrets.env_names(document.get("env_from_host", []), where)
    agent = _table(document, "agent", AGENT_KEYS, where)
    environment = _table(document, "environment", ENVIRONMENT_KEYS, where)
    verifier = _table(document, "verifier", VERIFIER_KEYS, where)
    for table, key in (
        (environment, "extra_docker_compose"),
        (environment, "extra_allowed_hosts"),
        (verifier, "extra_allowed_hosts"),
    ):
        _string_list(table, key, where)
    env = verifier.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError(f"{where}: [verifier] env must map names to strings")
    multiplier = agent.get(TRIAL_LEVEL)
    if multiplier is not None and (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or multiplier <= 0
    ):
        raise ValueError(f"{where}: {TRIAL_LEVEL} must be a positive number")
    compute = document.get("compute", {})
    if not isinstance(compute, dict) or not all(
        isinstance(v, dict) for v in compute.values()
    ):
        raise ValueError(f"{where}: [compute] holds one table per profile")
    profiles: dict[str, dict[str, Any]] = {}
    for name, table in compute.items():
        profiles[name] = _table(compute, name, PROFILE_KEYS, f"{where} [compute]")
        if "gate_host" not in table:
            raise ValueError(f"{where}: [compute.{name}] needs gate_host")
    secrets.refuse_literal_secrets(document, where, names)
    return RunConfig(names, agent, environment, verifier, profiles, directory)


def load(path: Path | None) -> RunConfig:
    """The run.toml at ``path``, or the empty configuration when no file is given."""
    if path is None:
        return RunConfig()
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return parse(document, str(path), path.resolve().parent)


def _yaml(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        import yaml  # Harbor's own dependency; the harness pins it through harbor's lock.
    except ImportError:
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _user_compute(environ: Mapping[str, str]) -> Path:
    named = environ.get("TRINITY_COMPUTE_USER")
    if named:
        return Path(named).expanduser()
    base = environ.get("XDG_CONFIG_HOME") or str(Path("~/.config").expanduser())
    return Path(base) / "trinity" / SHARED_COMPUTE


def trinity_docker_host(
    root: Path, target: str, environ: Mapping[str, str] | None = None
) -> tuple[str, str] | None:
    """``(DOCKER_HOST, source)`` for an ssh target in Trinity's compute files, or None.

    The parent's shared ``compute.yaml`` is read first, then the operator's user file.
    """
    source = os.environ if environ is None else environ
    for path, label in (
        (root / SHARED_COMPUTE, "./compute.yaml"),
        (_user_compute(source), "user compute.yaml"),
    ):
        document = _yaml(path)
        targets = document.get("targets") if document is not None else None
        entry = targets.get(target) if isinstance(targets, dict) else None
        if not isinstance(entry, dict) or entry.get("kind") != "ssh":
            continue
        hosts = entry.get("hosts")
        if not isinstance(hosts, list) or not hosts or not isinstance(hosts[0], str):
            continue
        port = entry.get("port", 22)
        suffix = "" if port in (22, None) else f":{port}"
        return f"ssh://{hosts[0]}{suffix}", f"{label} target {target}"
    return None


def resolve_profile(
    config: RunConfig, name: str, root: Path, environ: Mapping[str, str] | None = None
) -> Profile:
    """Profile ``name``, preferring Trinity's compute file for the ssh host when it names one."""
    table = config.profiles.get(name)
    if table is None:
        known = ", ".join(sorted(config.profiles)) or "none"
        raise LookupError(
            f"--compute {name}: run.toml defines no such profile (defined: {known})"
        )
    docker_host = table.get("docker_host")
    source = "run.toml"
    target = table.get("trinity_target")
    if isinstance(target, str) and target:
        found = trinity_docker_host(root, target, environ)
        if found is not None:
            docker_host, source = found
    if isinstance(docker_host, str) and docker_host.startswith("unix://~"):
        docker_host = "unix://" + str(
            Path(docker_host.removeprefix("unix://")).expanduser()
        )
    ports = {
        key: table.get(key, default)
        for key, default in (
            ("agent_port", DEFAULT_AGENT_PORT),
            ("judge_port", DEFAULT_JUDGE_PORT),
        )
    }
    if not all(type(value) is int and 0 < value < 65536 for value in ports.values()):
        raise ValueError(f"[compute.{name}]: ports must be integers in 1-65535")
    return Profile(
        name=name,
        gate_host=str(table["gate_host"]),
        docker_host=str(docker_host) if docker_host else None,
        agent_port=ports["agent_port"],
        judge_port=ports["judge_port"],
        judge_env=str(table.get("judge_env", DEFAULT_JUDGE_ENV)),
        source=source,
    )


def harbor_environ(
    profile: Profile | None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment Harbor runs under: the host's own, plus the profile's Docker host."""
    source = dict(os.environ if environ is None else environ)
    if profile is None or profile.docker_host is None:
        return source
    current = source.get("DOCKER_HOST")
    if current and current != profile.docker_host:
        raise RuntimeError(
            f"DOCKER_HOST is already set to another host; unset it or drop --compute {profile.name}"
        )
    if source.get("DOCKER_CONTEXT"):
        raise RuntimeError(
            f"DOCKER_CONTEXT is set; unset it to use --compute {profile.name}"
        )
    source["DOCKER_HOST"] = profile.docker_host
    return source
