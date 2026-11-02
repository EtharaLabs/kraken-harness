"""Keep credential values off disk: names in configs, ``${NAME}`` templates in layers.

A run names the host environment variables it needs in ``env_from_host``. The harness
writes only ``${NAME}`` templates into the Harbor layers it composes, and Harbor resolves
them from its own process environment when the trial starts, so no value ever reaches a
layer file, a record or a command line. Any literal value shaped like a credential in a
config the operator hands the harness is refused before anything is written.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
TEMPLATE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}\Z")
# A key whose name says it holds a credential, matched as Harbor's own scrubber does.
SENSITIVE_KEY = re.compile(
    r"(KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE
)
# Credential prefixes are assembled from parts so no source here carries a key-shaped literal.
_SHAPES = (
    ("anthropic-key", "sk" + r"-ant-[A-Za-z0-9_-]{20,}"),
    ("openai-key", "sk" + r"-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    ("aws-access-key", "AK" + r"IA[0-9A-Z]{16}"),
    ("github-token", "gh" + r"[pousr]_[A-Za-z0-9]{30,}"),
    ("slack-token", "xo" + r"x[abprs]-[A-Za-z0-9-]{10,}"),
    ("google-api-key", "AI" + r"za[0-9A-Za-z_-]{35}"),
    ("private-key-block", "-----BEGIN " + r"[A-Z ]*PRIVATE KEY-----"),
)
KEY_SHAPES = tuple((name, re.compile(pattern)) for name, pattern in _SHAPES)


def template(name: str) -> str:
    return "${" + name + "}"


def env_names(value: object, where: str) -> tuple[str, ...]:
    """The ``env_from_host`` list: distinct upper-case environment variable names."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{where}: env_from_host must be a list of variable names")
    names = tuple(str(item) for item in value)
    bad = [name for name in names if ENV_NAME.fullmatch(name) is None]
    if bad:
        raise ValueError(
            f"{where}: env_from_host holds names that are not variables: {bad}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: env_from_host names a variable twice")
    return names


def shape_of(text: str) -> str | None:
    """The name of the credential shape ``text`` carries, or None."""
    for name, pattern in KEY_SHAPES:
        if pattern.search(text):
            return name
    return None


def _walk(value: object, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def refuse_literal_secrets(
    document: Mapping[str, Any], where: str, allowed: tuple[str, ...]
) -> None:
    """Refuse a credential-shaped literal, or a credential key not given as an allowed template.

    A key that names a credential may hold only ``${NAME}`` for a name in ``allowed``; a key
    ending in ``_name`` may instead hold a bare variable name, as ``model_api_key_name`` does.
    """
    for path, text in _walk(document, ""):
        shape = shape_of(text)
        if shape is not None:
            raise ValueError(
                f"{where}: {path} holds a literal {shape}; name it in env_from_host"
            )
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        found = TEMPLATE.fullmatch(text)
        if found is not None:
            if found.group(1) not in allowed:
                raise ValueError(
                    f"{where}: {path} references {found.group(1)}, which env_from_host does not name"
                )
            continue
        if SENSITIVE_KEY.search(key) is None:
            continue
        if key.lower().endswith("_name") and ENV_NAME.fullmatch(text):
            continue
        raise ValueError(
            f"{where}: {path} holds a literal credential value; write ${{NAME}} and list NAME "
            "in env_from_host"
        )


def missing_from_host(
    names: tuple[str, ...], environ: Mapping[str, str] | None = None
) -> list[str]:
    """The names in ``names`` the host environment does not set to a non-empty value."""
    source = os.environ if environ is None else environ
    return [name for name in names if not source.get(name)]


def scan_tree(
    tree: Path, names: tuple[str, ...], environ: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """Every ``(relative path, finding name)`` where a file under ``tree`` carries a secret.

    A finding is either the host value of a variable in ``names`` or a credential shape.
    The value itself is never returned, so a report names the file and the variable only.
    """
    source = os.environ if environ is None else environ
    values = {name: source[name] for name in names if len(source.get(name, "")) >= 8}
    hits: list[tuple[str, str]] = []
    for path in sorted(tree.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        relative = path.relative_to(tree).as_posix()
        for name, value in values.items():
            if value in text:
                hits.append((relative, name))
        shape = shape_of(text)
        if shape is not None:
            hits.append((relative, shape))
    return hits
