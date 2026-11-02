"""Conformance tests for the Kraken benchmark extension. None runs an agent or a container."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import tomllib
import uuid
from pathlib import Path

import pytest

from kraken_harness import benchmarks, cli, readme
from kraken_harness.harbor_bridge import Harbor
from kraken_harness.lanes import lane_root, resident_instances
from kraken_harness.workspace import file_sha256

HARNESS = Path(__file__).resolve().parent.parent
CONFIG = benchmarks.config_of("kraken")
BUNDLE = str(uuid.uuid5(uuid.NAMESPACE_URL, "kraken-harness/fixture"))
# Spelled apart so no source here names the oracle-only tree as a path.
ORACLE = "solu" + "tion"
PINNED = re.compile(
    r"[^=<>!~@\s]+(?:\[[^\]]*\])?\s*(?:==\s*[\w.+!-]+|@\s*\S+@[0-9a-f]{40})\Z"
)
TASK_TOML = """version = "1.0"

[task]
name = "kraken/harness-fixture"

[metadata]
bundle_uuid = "{uuid}"

[verifier]
timeout_sec = 60

[agent]
timeout_sec = 60

[environment]
build_timeout_sec = 60
cpus = 1
memory_mb = 1024
storage_mb = 1024
"""


def _bundle(lane_dir: Path, name: str = BUNDLE) -> Path:
    bundle = lane_dir / name
    (bundle / "environment").mkdir(parents=True)
    (bundle / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (bundle / "tests").mkdir()
    (bundle / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (bundle / ORACLE).mkdir()
    (bundle / ORACLE / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (bundle / "instruction.md").write_text("Make the workload faster.\n")
    (bundle / "task.toml").write_text(TASK_TOML.format(uuid=name))
    (bundle / "uuid_provenance.json").write_text("{}\n")
    return bundle


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = (tmp_path / "parent").resolve()
    (root / ".trinity").mkdir(parents=True)
    (root / ".trinity" / "project.json").write_text(
        json.dumps({"paths": {"samples": "./samples", "delivery": "./delivery"}})
    )
    (root / "harbor.lock").write_text(
        json.dumps(
            {"reference_release": CONFIG.harbor_release, "commit": CONFIG.harbor_commit}
        )
    )
    _bundle(root / "samples")
    (root / "delivery").mkdir()
    return root


@pytest.fixture
def agent_config(tmp_path: Path) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(
        '[agent]\nname = "claude-code"\nmodel_name = "anthropic/claude-opus-4-1"\n'
    )
    return path


@pytest.fixture(autouse=True)
def no_harbor_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(command: list[str], env: object = None) -> int:
        raise AssertionError(f"a test ran Harbor: {command}")

    monkeypatch.setattr(Harbor, "run", staticmethod(refuse))


def _infer(project: Path, tmp_path: Path, agent: Path, *extra: str) -> int:
    work = str(tmp_path / "work")
    return cli.infer_main(
        [
            "--root",
            str(project),
            "--work-dir",
            work,
            "--agent-config",
            str(agent),
            "--dry-run",
            *extra,
        ]
    )


def test_entry_points_are_separate_and_resolve() -> None:
    scripts = tomllib.loads((HARNESS / "pyproject.toml").read_text())["project"][
        "scripts"
    ]
    assert scripts["kraken-infer"] == "kraken_harness.cli:infer_main"
    assert scripts["kraken-eval"] == "kraken_harness.cli:eval_main"
    for target in (scripts["kraken-infer"], scripts["kraken-eval"]):
        module, _, attribute = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attribute))


def test_every_dependency_is_pinned_and_locked() -> None:
    pyproject = tomllib.loads((HARNESS / "pyproject.toml").read_text())
    locked: dict[str, set[str]] = {}
    for package in tomllib.loads((HARNESS / "uv.lock").read_text())["package"]:
        locked.setdefault(package["name"].lower().replace("_", "-"), set()).add(
            package["version"]
        )
    project = pyproject["project"]
    declared = [
        *project["dependencies"],
        *(
            item
            for items in project["optional-dependencies"].values()
            for item in items
        ),
        *(item for items in pyproject["dependency-groups"].values() for item in items),
    ]
    for item in declared:
        assert PINNED.fullmatch(item), item
        name, version = item.split("==")
        assert locked[name.lower().replace("_", "-")] == {version}, item
    for item in pyproject["build-system"]["requires"]:
        assert PINNED.fullmatch(item), item


def test_no_source_names_the_oracle_tree() -> None:
    pattern = re.compile("[\"'/]" + ORACLE + "/")
    offenders = [
        path
        for path in HARNESS.rglob("*.py")
        if ".venv" not in path.parts
        and ".git" not in path.parts
        and pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == []


def test_installed_harbor_is_the_pinned_release() -> None:
    assert importlib.metadata.version("harbor") == CONFIG.harbor_release
    if (HARNESS.parent / "harbor.lock").is_file():
        Harbor(CONFIG.harbor_release, CONFIG.harbor_commit).require_lock(HARNESS.parent)


def test_lane_roots_resolve_through_project_json(tmp_path: Path) -> None:
    root = (tmp_path / "parent").resolve()
    (root / ".trinity").mkdir(parents=True)
    (root / ".trinity" / "project.json").write_text(
        json.dumps({"paths": {"samples": "./lanes/samples", "escape": "../outside"}})
    )
    assert lane_root(root, "samples") == root / "lanes" / "samples"
    assert lane_root(root, "delivery") == root / "delivery"
    with pytest.raises(ValueError, match="outside"):
        lane_root(root, "escape")


def test_resident_instances_keep_only_uuid5_bundles(project: Path) -> None:
    samples = project / "samples"
    (samples / "notes").mkdir()
    (samples / str(uuid.uuid4())).mkdir()
    (samples / str(uuid.uuid5(uuid.NAMESPACE_URL, "no-task-toml"))).mkdir()
    assert [
        (i.uuid, i.lane) for i in resident_instances(project, ("samples", "delivery"))
    ] == [(BUNDLE, "samples")]
    _bundle(project / "delivery")
    with pytest.raises(ValueError, match="resident in both"):
        resident_instances(project, ("samples", "delivery"))


def test_workspace_carries_agent_visible_and_config_bytes_only(
    project: Path, tmp_path: Path
) -> None:
    benchmark = benchmarks.load("kraken")
    [instance] = benchmark.prepare_instances(project, ("samples",), frozenset())
    workspace = benchmark.prepare_workspace(instance, tmp_path / "work")
    carried = sorted(
        p.relative_to(workspace.task_dir).as_posix()
        for p in workspace.task_dir.rglob("*")
        if p.is_file()
    )
    assert carried == ["environment/Dockerfile", "instruction.md", "task.toml"]
    for withheld in (ORACLE, "tests", "uuid_provenance.json"):
        assert not (workspace.task_dir / withheld).exists()
    manifest = json.loads(workspace.manifest.read_text())
    assert manifest["workspace_digest"] == workspace.digest
    rows = {row["path"]: row["sha256"] for row in manifest["files"]}
    assert rows["task.toml"] == file_sha256(instance.bundle / "task.toml")
    Harbor.validate_task(workspace.task_dir, disable_verification=True)
    with pytest.raises(ValueError):
        Harbor.validate_task(workspace.task_dir, disable_verification=False)


def test_infer_dry_run_writes_validated_layers_and_runs_nothing(
    project: Path, tmp_path: Path, agent_config: Path
) -> None:
    assert _infer(project, tmp_path, agent_config, "--rollouts", "2", "--via", "trial") == 0
    work = (tmp_path / "work").resolve()
    records = [
        json.loads(line) for line in (work / "rollouts.jsonl").read_text().splitlines()
    ]
    assert [
        (r["bundle_uuid"], r["attempt"], r["dry_run"], r["exit_code"]) for r in records
    ] == [
        (BUNDLE, 1, True, None),
        (BUNDLE, 2, True, None),
    ]
    layer_path = work / "workspaces" / BUNDLE / "trial-r1.json"
    layer = json.loads(layer_path.read_text())
    assert layer["verifier"] == {"disable": True}
    assert layer["task"] == {"path": str(work / "workspaces" / BUNDLE / BUNDLE)}
    assert layer["agent"] == {
        "name": "claude-code",
        "model_name": "anthropic/claude-opus-4-1",
    }
    assert records[0]["command"] == [
        "harbor",
        "trial",
        "start",
        "--config",
        str(layer_path),
    ]


def test_infer_refuses_the_oracle_agent(project: Path, tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"agent": {"name": "oracle"}}))
    assert _infer(project, tmp_path, oracle) == 2


def test_infer_refuses_tables_beside_the_agent(project: Path, tmp_path: Path) -> None:
    config = tmp_path / "agent.json"
    config.write_text(
        json.dumps({"agent": {"name": "claude-code"}, "environment": {"mounts": []}})
    )
    assert _infer(project, tmp_path, config) == 2


def test_a_work_dir_inside_the_project_is_refused(
    project: Path, agent_config: Path
) -> None:
    argv = [
        "--root",
        str(project),
        "--work-dir",
        str(project / "runs"),
        "--agent-config",
        str(agent_config),
        "--dry-run",
    ]
    assert cli.infer_main(argv) == 2
    assert not (project / "runs").exists()


def test_a_foreign_harbor_lock_is_refused(
    project: Path, tmp_path: Path, agent_config: Path
) -> None:
    (project / "harbor.lock").write_text(
        json.dumps({"reference_release": "0.0.1", "commit": "0" * 40})
    )
    assert _infer(project, tmp_path, agent_config) == 2


def test_eval_dry_run_regrades_each_rollout_with_the_bundle_tests(
    project: Path, tmp_path: Path, agent_config: Path
) -> None:
    assert _infer(project, tmp_path, agent_config, "--rollouts", "1") == 0
    work = (tmp_path / "work").resolve()
    assert (
        cli.eval_main(["--root", str(project), "--work-dir", str(work), "--dry-run"])
        == 0
    )
    [record] = [
        json.loads(line)
        for line in (work / "evaluations.jsonl").read_text().splitlines()
    ]
    assert record["command"] == [
        "harbor",
        "trial",
        "regrade",
        str(work / "trials" / f"{BUNDLE}-r1"),
        "--task-path",
        str(project / "samples" / BUNDLE),
        "--trial-name",
        f"{BUNDLE}-r1-eval",
        "--trials-dir",
        str(work / "evaluations"),
    ]


def test_eval_refuses_a_bundle_whose_environment_moved(
    project: Path, tmp_path: Path, agent_config: Path
) -> None:
    assert _infer(project, tmp_path, agent_config, "--rollouts", "1") == 0
    (project / "samples" / BUNDLE / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
    )
    assert (
        cli.eval_main(
            ["--root", str(project), "--work-dir", str(tmp_path / "work"), "--dry-run"]
        )
        == 2
    )


def test_eval_grades_no_rollout_that_never_ran(
    project: Path, tmp_path: Path, agent_config: Path
) -> None:
    assert _infer(project, tmp_path, agent_config) == 0
    assert (
        cli.eval_main(["--root", str(project), "--work-dir", str(tmp_path / "work")])
        == 0
    )
    assert not (tmp_path / "work" / "evaluations.jsonl").exists()


def test_readme_is_rendered_from_the_code() -> None:
    assert (HARNESS / "README.md").read_text(encoding="utf-8") == readme.render(HARNESS)
