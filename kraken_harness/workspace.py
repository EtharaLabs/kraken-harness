"""Prepare a per-instance sandboxed workspace from a bundle's bound environment form.

The workspace is a fresh Harbor task directory holding only the bytes ``layout`` carries,
plus a manifest beside it that binds every carried file and the ``environment/`` tree by
SHA-256, so evaluation can refuse a bundle whose environment moved after inference.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kraken_harness.lanes import Instance
from kraken_harness.layout import carried_files

REQUIRED = (PurePosixPath("task.toml"), PurePosixPath("instruction.md"))
ENVIRONMENT = "environment"
MANIFEST = "workspace.json"


@dataclass(frozen=True, slots=True)
class Workspace:
    instance: Instance
    task_dir: Path
    manifest: Path
    digest: str
    environment_digest: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(rows: list[tuple[str, str]]) -> str:
    """The SHA-256 of the sorted ``<sha256>  <path>`` lines of ``rows``."""
    text = "".join(f"{sha}  {path}\n" for path, sha in sorted(rows))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle_rows(bundle: Path) -> list[tuple[str, str]]:
    return [
        (relative.as_posix(), file_sha256(bundle / relative))
        for relative in carried_files(bundle)
    ]


def carried_digest(bundle: Path) -> str:
    """The digest of every byte a workspace of ``bundle`` carries, read from the bundle itself."""
    return tree_digest(_bundle_rows(bundle))


def environment_digest(bundle: Path) -> str:
    """The digest of the bundle's ``environment/`` tree, read from the bundle itself."""
    return tree_digest(
        [row for row in _bundle_rows(bundle) if row[0].split("/", 1)[0] == ENVIRONMENT]
    )


def prepare(instance: Instance, work_dir: Path) -> Workspace:
    """Copy the carried bytes of ``instance`` into a fresh workspace under ``work_dir``."""
    files = carried_files(instance.bundle)
    missing = [str(path) for path in REQUIRED if path not in files]
    if not any(path.parts[0] == ENVIRONMENT for path in files):
        missing.append(f"{ENVIRONMENT}/")
    if missing:
        raise ValueError(f"{instance.bundle}: bundle lacks {', '.join(missing)}")
    root = work_dir / "workspaces" / instance.uuid
    if root.exists():
        raise FileExistsError(
            f"{root} already holds a workspace; choose a fresh work directory"
        )
    # Harbor names a trial after its task directory, so naming it by the bundle uuid gives the
    # same trial directory names a plain `harbor run` on the bundle itself would.
    task_dir = root / instance.uuid
    rows: list[tuple[str, str]] = []
    for relative in files:
        target = task_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(instance.bundle / relative, target)
        rows.append((relative.as_posix(), file_sha256(target)))
    digest = tree_digest(rows)
    environment = tree_digest(
        [row for row in rows if row[0].split("/", 1)[0] == ENVIRONMENT]
    )
    manifest = root / MANIFEST
    manifest.write_text(
        json.dumps(
            {
                "bundle_uuid": instance.uuid,
                "lane": instance.lane,
                "bundle": str(instance.bundle),
                "files": [{"path": path, "sha256": sha} for path, sha in rows],
                "workspace_digest": digest,
                "environment_digest": environment,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return Workspace(instance, task_dir, manifest, digest, environment)
