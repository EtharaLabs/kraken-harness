"""Write each Harbor layer or job file owner-only, and remove it once its trial is over.

A layer holds only ``${NAME}`` templates, never a credential value, yet it still names the
operator's gate addresses and host paths, so it is created mode 0600 with an exclusive open
and deleted when the trial ends. A dry run keeps its layers so the operator can read them.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MODE = 0o600


def write_private(path: Path, document: dict[str, Any]) -> Path:
    """Create ``path`` exclusively with mode 0600 and write ``document`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, MODE
    )
    try:
        os.fchmod(descriptor, MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return path


def is_private(path: Path) -> bool:
    return stat.S_IMODE(path.lstat().st_mode) == MODE


@contextmanager
def private_layer(
    path: Path, document: dict[str, Any], *, keep: bool
) -> Iterator[Path]:
    """Hold ``document`` at ``path`` for the life of one trial, then remove it unless kept.

    A layer an earlier dry run kept is replaced, so a dry run can be repeated.
    """
    if path.is_file() and not path.is_symlink():
        path.unlink()
    write_private(path, document)
    try:
        yield path
    finally:
        if not keep:
            path.unlink(missing_ok=True)
