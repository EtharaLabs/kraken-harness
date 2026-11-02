"""The resolved run a rollout used, as one canonical document and its SHA-256 pin.

``kraken-infer --emit-harness-config`` writes the Harbor release and commit, the agent and
model, the timeouts, the network policy and the compute profile, never a credential value,
in Trinity's canonical byte form (``trinity.jcs-lf.v1``: RFC 8785 JSON plus one LF), so the
digest it prints is the one Trinity computes over the same bytes. ``--harness-config``
refuses any run whose resolved document hashes differently from the given pin.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from kraken_harness.compose import compose_digests
from kraken_harness.config import BenchmarkConfig
from kraken_harness.run_config import TRIAL_LEVEL, Profile, RunConfig

SCHEMA = "kraken.harness-run-config/v1"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
AGENT_FIELDS = (
    "name",
    "import_path",
    "model_name",
    "model_api_format",
    "model_api_key_name",
)


# RFC 8785 number form, as ECMAScript renders a double: fixed notation for decimal
# exponents in (-7, 21], exponent notation outside it, shortest round-trip digits.
MAX_FIXED = 21
MIN_FIXED = -6
_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON holds finite numbers only")
    if value == 0.0:
        return "0"
    rendered = repr(value)
    sign = ""
    if rendered.startswith("-"):
        sign, rendered = "-", rendered[1:]
    mantissa, separator, exponent_text = rendered.partition("e")
    exponent = int(exponent_text) if separator else 0
    whole, point, fraction = mantissa.partition(".")
    digits = whole + fraction if point else whole
    leading = len(digits) - len(digits.lstrip("0"))
    digits = digits.lstrip("0").rstrip("0")
    position = len(whole) + exponent - leading
    if 0 < position <= MAX_FIXED:
        if len(digits) <= position:
            return sign + digits + "0" * (position - len(digits))
        return sign + digits[:position] + "." + digits[position:]
    if MIN_FIXED < position <= 0:
        return sign + "0." + "0" * -position + digits
    power = position - 1
    significand = digits[0] if len(digits) == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{sign}{significand}e{'+' if power >= 0 else ''}{power}"


def _string(value: str) -> str:
    out = ['"']
    for character in value:
        escaped = _SHORT_ESCAPES.get(character)
        if escaped is not None:
            out.append(escaped)
        elif ord(character) < 0x20:
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def _render(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number(float(value))
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        return "[" + ",".join(_render(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: str(key).encode("utf-16-be"))
        return "{" + ",".join(f"{_string(str(k))}:{_render(value[k])}" for k in keys) + "}"
    raise TypeError(f"{type(value).__name__} has no canonical JSON form")


def canonical_bytes(document: dict[str, Any]) -> bytes:
    """``trinity.jcs-lf.v1`` bytes: RFC 8785 serialization followed by exactly one LF."""
    return (_render(document) + "\n").encode("utf-8")


def digest_of(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def resolve(
    config: BenchmarkConfig,
    agent: dict[str, Any],
    run: RunConfig,
    profile: Profile | None,
    *,
    via: str,
    rollouts: int,
) -> dict[str, Any]:
    """The run's pinned shape; host paths, credentials and per-trial names never enter it."""
    multiplier = run.agent.get(TRIAL_LEVEL)
    return {
        "schema": SCHEMA,
        "benchmark": config.name,
        "harbor": {"release": config.harbor_release, "commit": config.harbor_commit},
        "agent": {
            key: agent[key] for key in AGENT_FIELDS if agent.get(key) is not None
        },
        "inference": {
            "via": via,
            "verification": "disabled",
            "rollouts_per_instance": rollouts,
        },
        "timeouts": {
            "timeout_multiplier": 1.0,
            TRIAL_LEVEL: 1.0 if multiplier is None else float(multiplier),
        },
        "network": {
            "environment_allowed_hosts": sorted(
                run.environment.get("extra_allowed_hosts", [])
            ),
            "extra_docker_compose": sorted(
                compose_digests(run), key=lambda row: row["name"]
            ),
            "verifier_allowed_hosts": sorted(
                run.verifier.get("extra_allowed_hosts", [])
            ),
        },
        "env_from_host": sorted(run.env_from_host),
        "compute": None
        if profile is None
        else {
            "profile": profile.name,
            "gate_host": profile.gate_host,
            "agent_port": profile.agent_port,
            "judge_port": profile.judge_port,
            "docker_host": profile.docker_host,
        },
    }


def emit(path: Path, document: dict[str, Any]) -> str:
    """Write the canonical bytes to ``path`` and return their SHA-256."""
    data = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def pinned_digest(pin: str) -> str:
    """A pin given as 64 hex digits, or as a file whose canonical form is hashed."""
    if DIGEST.fullmatch(pin):
        return pin
    path = Path(pin)
    if not path.is_file():
        raise ValueError(
            f"--harness-config {pin}: neither a sha256 digest nor a readable file"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"--harness-config {pin}: must hold one JSON object")
    return digest_of(document)


def require_pin(pin: str | None, document: dict[str, Any]) -> str:
    """The resolved digest, refused when a pin was given and it differs."""
    resolved = digest_of(document)
    if pin is not None and pinned_digest(pin) != resolved:
        raise ValueError(
            f"the resolved harness config hashes to {resolved}, not the pinned "
            f"{pinned_digest(pin)}; emit it again with --emit-harness-config to see what moved"
        )
    return resolved
