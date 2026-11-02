"""Run configs, secrets, job mode, compute profiles, harness-config pins and pilot summaries.

Every test here fires on bad input and passes on good input. None runs an agent or a
container: Harbor is replaced by a recorder that inspects what the harness hands it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kraken_harness import cli, harness_config, pilot, run_config, secrets
from kraken_harness.harbor_bridge import Harbor
from test_kraken_harness import BUNDLE, CONFIG, _bundle

HARNESS = Path(__file__).resolve().parent.parent
# Assembled at run time so no stored byte here is credential-shaped.
FAKE_KEY = "sk" + "-ant-" + "Q" * 12 + "testonly" + "Z" * 12
KEY_NAME = "KRAKEN_TEST_KEY"
JUDGE_KEY_NAME = "KRAKEN_TEST_JUDGE_KEY"


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
def agent_toml(tmp_path: Path) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(
        '[agent]\nname = "claude-code"\nmodel_name = "anthropic/claude-opus-5-5"\n'
        f'model_api_key_name = "{KEY_NAME}"\n'
    )
    return path


RUN_TOML = f"""env_from_host = ["{KEY_NAME}", "{JUDGE_KEY_NAME}"]

[agent]
model_api_format = "anthropic_messages"
agent_setup_timeout_multiplier = 10

[environment]
extra_allowed_hosts = ["deb.debian.org", "registry.npmjs.org"]
extra_docker_compose = ["agent-hosts.compose.yaml"]

[verifier]
extra_allowed_hosts = []
env = {{ REWARDKIT_JUDGE = "1", LITELLM_PROXY_API_KEY = "${{{JUDGE_KEY_NAME}}}" }}

[compute.local]
docker_host = "unix:///tmp/kraken-test-colima.sock"
gate_host = "host.docker.internal"

[compute.ec2]
trinity_target = "@trinity"
docker_host = "ssh://fallback@203.0.113.9"
gate_host = "172.17.0.1"
"""


@pytest.fixture
def run_toml(tmp_path: Path) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(RUN_TOML)
    (tmp_path / "agent-hosts.compose.yaml").write_text("services: {}\n")
    return path


class Recorder:
    """Stands in for ``Harbor.run``: records each launch and the layer it found on disk."""

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], env: dict[str, str] | None = None) -> int:
        layer = Path(command[command.index("--config") + 1])
        self.calls.append(
            {
                "command": command,
                "docker_host": (env or {}).get("DOCKER_HOST"),
                "env": dict(env or {}),
                "present": layer.exists(),
                "mode": stat.S_IMODE(layer.stat().st_mode) if layer.exists() else None,
                "text": layer.read_text() if layer.exists() else "",
            }
        )
        document = json.loads(self.calls[-1]["text"] or "{}")
        if "trial_name" in document and "trials_dir" in document:
            trial = Path(document["trials_dir"]) / document["trial_name"]
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "result.json").write_text(json.dumps({"exception_info": None}))
        return self.exit_code


@pytest.fixture(autouse=True)
def no_harbor_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(command: list[str], env: object = None) -> int:
        raise AssertionError(f"a test ran Harbor: {command}")

    monkeypatch.setattr(Harbor, "run", staticmethod(refuse))


@pytest.fixture(autouse=True)
def host_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(KEY_NAME, FAKE_KEY)
    monkeypatch.setenv(JUDGE_KEY_NAME, "judge-" + "value-" * 3)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("TRINITY_COMPUTE_USER", str(tmp_path / "no-user-compute.yaml"))


def _infer(project: Path, tmp_path: Path, agent: Path, *extra: str) -> int:
    return cli.infer_main(
        [
            "--root",
            str(project),
            "--work-dir",
            str(tmp_path / "work"),
            "--agent-config",
            str(agent),
            *extra,
        ]
    )


def _eval(project: Path, tmp_path: Path, *extra: str) -> int:
    return cli.eval_main(
        ["--root", str(project), "--work-dir", str(tmp_path / "work"), *extra]
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# 1. run.toml allowlist ---------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        '[harbor]\nrelease = "0.26.0"\n',
        "[agent]\nkwargs = { max_turns = 5 }\n",
        "[environment]\nmounts = []\n",
        '[verifier]\nimport_path = "x:Y"\n',
        '[compute.ec2]\ngate_host = "172.17.0.1"\ncpus = 4\n',
    ],
)
def test_run_config_refuses_keys_outside_the_allowlist(
    tmp_path: Path, extra: str
) -> None:
    path = tmp_path / "run.toml"
    path.write_text(extra)
    with pytest.raises(ValueError, match="refuses"):
        run_config.load(path)


def test_run_config_merges_allowlisted_extras_into_a_validated_layer(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--via",
            "trial",
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    layer = json.loads(
        (tmp_path / "work" / "workspaces" / BUNDLE / "trial-r1.json").read_text()
    )
    Harbor.validate_trial(layer)
    assert layer["agent"]["model_api_format"] == "anthropic_messages"
    assert layer["agent_setup_timeout_multiplier"] == 10.0
    assert "agent_setup_timeout_multiplier" not in layer["agent"]
    assert layer["environment"]["extra_allowed_hosts"] == [
        "deb.debian.org",
        "registry.npmjs.org",
    ]
    assert layer["environment"]["extra_docker_compose"] == [
        str(tmp_path / "agent-hosts.compose.yaml")
    ]
    assert layer["verifier"] == {"disable": True}


def test_a_run_toml_value_conflicting_with_agent_toml_is_refused(
    project: Path, tmp_path: Path, agent_toml: Path
) -> None:
    run = tmp_path / "run.toml"
    run.write_text(
        f'env_from_host = ["{KEY_NAME}"]\n[agent]\nmodel_api_key_name = "OTHER_KEY"\n'
    )
    assert (
        _infer(project, tmp_path, agent_toml, "--run-config", str(run), "--dry-run")
        == 2
    )


# 2. secrets never at rest --------------------------------------------------------------


def test_a_literal_key_shaped_value_is_refused_in_run_toml(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(f'[verifier]\nenv = {{ JUDGE = "{FAKE_KEY}" }}\n')
    with pytest.raises(ValueError, match="anthropic-key"):
        run_config.load(path)


def test_a_literal_value_under_a_credential_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        '[verifier]\nenv = { LITELLM_PROXY_API_KEY = "plain-but-secret" }\n'
    )
    with pytest.raises(ValueError, match="literal credential"):
        run_config.load(path)


def test_a_template_must_name_a_variable_env_from_host_lists(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text('[verifier]\nenv = { LITELLM_PROXY_API_KEY = "${UNLISTED_KEY}" }\n')
    with pytest.raises(ValueError, match="does not name"):
        run_config.load(path)
    path.write_text(
        'env_from_host = ["LISTED_KEY"]\n[verifier]\nenv = { LITELLM_PROXY_API_KEY = "${LISTED_KEY}" }\n'
    )
    assert run_config.load(path).env_from_host == ("LISTED_KEY",)


def test_a_literal_key_in_agent_toml_is_refused(project: Path, tmp_path: Path) -> None:
    agent = tmp_path / "agent.json"
    agent.write_text(
        json.dumps(
            {"agent": {"name": "claude-code", "env": {"ANTHROPIC_API_KEY": FAKE_KEY}}}
        )
    )
    assert _infer(project, tmp_path, agent, "--dry-run") == 2
    assert not (tmp_path / "work" / "rollouts.jsonl").exists()


def test_layers_are_owner_only_during_the_trial_and_removed_after(
    project: Path,
    tmp_path: Path,
    agent_toml: Path,
    run_toml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(exit_code=1)
    monkeypatch.setattr(Harbor, "run", staticmethod(recorder))
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--via",
            "trial",
            "--rollouts",
            "2",
        )
        == 1
    )
    assert [call["present"] for call in recorder.calls] == [True, True]
    assert [call["mode"] for call in recorder.calls] == [0o600, 0o600]
    workspace = tmp_path / "work" / "workspaces" / BUNDLE
    assert not list(workspace.glob("trial-r*.json"))
    for call in recorder.calls:
        assert FAKE_KEY not in call["text"]
        assert "${" + KEY_NAME + "}" in call["text"]
    text = (tmp_path / "work" / "rollouts.jsonl").read_text()
    assert FAKE_KEY not in text
    assert all(
        row["env_from_host"] == [KEY_NAME, JUDGE_KEY_NAME]
        for row in _rows(tmp_path / "work" / "rollouts.jsonl")
    )


def test_a_dry_run_keeps_its_owner_only_layers(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    layer = tmp_path / "work" / "workspaces" / BUNDLE / "job-r1.json"
    assert stat.S_IMODE(layer.stat().st_mode) == 0o600
    assert FAKE_KEY not in layer.read_text()


def test_a_host_variable_the_run_names_must_be_set(
    project: Path,
    tmp_path: Path,
    agent_toml: Path,
    run_toml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Harbor, "run", staticmethod(Recorder()))
    monkeypatch.delenv(JUDGE_KEY_NAME)
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--rollouts",
            "1",
        )
        == 2
    )
    assert not list((tmp_path / "work" / "workspaces" / BUNDLE).glob("*-r1.json"))


def test_harbor_gets_the_verifier_variables_a_task_names_without_a_file_holding_them(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(Harbor, "run", staticmethod(recorder))
    args = ("--run-config", str(run_toml), "--compute", "local", "--rollouts", "1", "--via", "trial")
    assert _infer(project, tmp_path, agent_toml, *args) == 0
    [call] = recorder.calls
    assert call["env"]["REWARDKIT_JUDGE"] == "1"
    assert call["env"]["LITELLM_PROXY_API_BASE"] == "http://host.docker.internal:8790"
    assert call["env"]["LITELLM_PROXY_API_KEY"] == "judge-" + "value-" * 3
    assert "judge-value" not in call["text"] and "judge-value" not in " ".join(call["command"])
    assert json.loads(call["text"])["verifier"] == {"disable": True}


# 3. harbor run job mode ----------------------------------------------------------------


def test_job_mode_is_the_default_and_disables_verification(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    work = tmp_path / "work"
    layer = work / "workspaces" / BUNDLE / "job-r1.json"
    job = json.loads(layer.read_text())
    Harbor.validate_job(job)
    assert job["verifier"] == {"disable": True}
    assert job["n_attempts"] == 1 and len(job["agents"]) == 1 and len(job["tasks"]) == 1
    [row] = _rows(work / "rollouts.jsonl")
    assert row["via"] == "job" and row["probe"] is False
    assert row["command"] == ["harbor", "run", "--config", str(layer), "--yes"]


def test_a_finished_job_records_its_one_trial_directory(
    project: Path, tmp_path: Path, agent_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def harbor(command: list[str], env: object = None) -> int:
        job = json.loads(Path(command[3]).read_text())
        trial = Path(job["jobs_dir"]) / job["job_name"] / "harness-fixture__abc123"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text("{}")
        (trial / "config.json").write_text("{}")
        return 0

    monkeypatch.setattr(Harbor, "run", staticmethod(harbor))
    assert _infer(project, tmp_path, agent_toml, "--rollouts", "1") == 0
    [row] = _rows(tmp_path / "work" / "rollouts.jsonl")
    assert row["trial_dir"] == f"trials/{BUNDLE}-r1/harness-fixture__abc123"


def test_a_probe_runs_the_verifier_in_its_job_and_is_never_graded(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    args = ("--run-config", str(run_toml), "--rollouts", "1", "--dry-run", "--probe")
    assert _infer(project, tmp_path, agent_toml, *args, "--via", "trial") == 2
    assert _infer(project, tmp_path, agent_toml, *args) == 0
    job = json.loads(
        (tmp_path / "work" / "workspaces" / BUNDLE / "job-r1.json").read_text()
    )
    assert job["verifier"]["env"]["REWARDKIT_JUDGE"] == "1"
    assert job["jobs_dir"].endswith("/probes")
    assert _eval(project, tmp_path, "--dry-run") == 0
    assert not (tmp_path / "work" / "evaluations.jsonl").exists()


# 4. compute profiles --------------------------------------------------------------------


def test_profiles_fill_the_gate_address_for_each_host(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    run = run_config.load(run_toml)
    local = run_config.resolve_profile(run, "local", project)
    ec2 = run_config.resolve_profile(run, "ec2", project)
    assert local.base_url == "http://host.docker.internal:8642"
    assert local.judge_url == "http://host.docker.internal:8790"
    assert ec2.base_url == "http://172.17.0.1:8642"
    assert (ec2.docker_host, ec2.source) == ("ssh://fallback@203.0.113.9", "run.toml")
    with pytest.raises(LookupError, match="no such profile"):
        run_config.resolve_profile(run, "gpu", project)
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--compute",
            "ec2",
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    [row] = _rows(tmp_path / "work" / "rollouts.jsonl")
    assert (
        row["compute"]["profile"] == "ec2"
        and row["compute"]["gate_host"] == "172.17.0.1"
    )
    assert row["agent"]["base_url"] == "http://172.17.0.1:8642"


def test_trinity_compute_file_supplies_the_ssh_host_when_it_names_the_target(
    project: Path, tmp_path: Path, run_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("yaml")
    user = tmp_path / "user-compute.yaml"
    user.write_text(
        'schema: "trinity.compute-user/v1"\ntargets:\n  "@trinity":\n    kind: ssh\n'
        "    hosts:\n      - ops@198.51.100.7\n    port: 22\n"
    )
    monkeypatch.setenv("TRINITY_COMPUTE_USER", str(user))
    ec2 = run_config.resolve_profile(run_config.load(run_toml), "ec2", project)
    assert ec2.docker_host == "ssh://ops@198.51.100.7"
    assert ec2.source == "user compute.yaml target @trinity"


def test_a_profile_refuses_a_conflicting_docker_host(
    project: Path,
    tmp_path: Path,
    agent_toml: Path,
    run_toml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(Harbor, "run", staticmethod(recorder))
    monkeypatch.setenv("DOCKER_HOST", "tcp://elsewhere:2375")
    args = (
        "--run-config",
        str(run_toml),
        "--compute",
        "local",
        "--rollouts",
        "1",
        "--via",
        "trial",
    )
    assert _infer(project, tmp_path, agent_toml, *args) == 2
    monkeypatch.delenv("DOCKER_HOST")
    assert _infer(project, tmp_path / "again", agent_toml, *args) == 0
    assert recorder.calls[-1]["docker_host"] == "unix:///tmp/kraken-test-colima.sock"


def test_a_profile_refuses_an_explicit_base_url_it_would_override(
    project: Path, tmp_path: Path, run_toml: Path
) -> None:
    agent = tmp_path / "agent.toml"
    agent.write_text('[agent]\nname = "claude-code"\nbase_url = "http://10.0.0.1:1"\n')
    assert (
        _infer(
            project,
            tmp_path,
            agent,
            "--run-config",
            str(run_toml),
            "--compute",
            "ec2",
            "--dry-run",
        )
        == 2
    )


def test_evaluation_fills_the_judge_and_its_host_from_the_profile(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    assert (
        _eval(
            project,
            tmp_path,
            "--run-config",
            str(run_toml),
            "--compute",
            "ec2",
            "--dry-run",
        )
        == 0
    )
    [row] = _rows(tmp_path / "work" / "evaluations.jsonl")
    layer = json.loads((tmp_path / "work" / row["layer"]).read_text())
    Harbor.validate_trial(layer)
    assert layer["source_trial"]["action"] == "regrade"
    assert (
        layer["verifier"]["env"]["LITELLM_PROXY_API_BASE"] == "http://172.17.0.1:8790"
    )
    assert (
        layer["verifier"]["env"]["LITELLM_PROXY_API_KEY"] == "${" + JUDGE_KEY_NAME + "}"
    )
    assert layer["verifier"]["extra_allowed_hosts"] == ["172.17.0.1"]
    assert row["verifier_env_names"] == [
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
        "REWARDKIT_JUDGE",
    ]
    assert row["compute"]["profile"] == "ec2"


def test_grading_one_bundle_on_two_hosts_is_refused_unless_allowed(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path
) -> None:
    assert (
        _infer(
            project,
            tmp_path,
            agent_toml,
            "--run-config",
            str(run_toml),
            "--rollouts",
            "1",
            "--dry-run",
        )
        == 0
    )
    graded = {"bundle_uuid": BUNDLE, "dry_run": False, "compute": {"profile": "ec2"}}
    (tmp_path / "work" / "evaluations.jsonl").write_text(json.dumps(graded) + "\n")
    base = ("--run-config", str(run_toml), "--dry-run")
    assert _eval(project, tmp_path, *base, "--compute", "local") == 2
    assert _eval(project, tmp_path, *base, "--compute", "ec2") == 0
    assert (
        _eval(project, tmp_path, *base, "--compute", "local", "--allow-mixed-grading")
        == 0
    )


# 5. harness config pin -----------------------------------------------------------------


def test_the_harness_config_digest_is_stable_and_names_no_secret(
    project: Path,
    tmp_path: Path,
    agent_toml: Path,
    run_toml: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "harness-config.json"
    args = ("--run-config", str(run_toml), "--compute", "ec2", "--emit-harness-config")
    assert _infer(project, tmp_path, agent_toml, *args, str(out)) == 0
    first = out.read_bytes()
    assert (
        _infer(project, tmp_path, agent_toml, *args, str(tmp_path / "again.json")) == 0
    )
    assert (tmp_path / "again.json").read_bytes() == first
    digest = hashlib.sha256(first).hexdigest()
    assert f"harness config sha256 {digest}" in capsys.readouterr().out
    assert first.endswith(b"}\n") and FAKE_KEY.encode() not in first
    document = json.loads(first)
    assert document["agent"]["model_name"] == "anthropic/claude-opus-5-5"
    assert document["harbor"] == {
        "release": CONFIG.harbor_release,
        "commit": CONFIG.harbor_commit,
    }
    assert document["compute"]["gate_host"] == "172.17.0.1"
    assert harness_config.digest_of(json.loads(json.dumps(document))) == digest
    assert not (tmp_path / "work").exists()


def test_the_canonical_bytes_match_trinity_when_trinity_is_present() -> None:
    """Trinity is Python 3.14 code, so its canonicalizer runs under its own pinned interpreter."""
    trinity = HARNESS.parent / "trinity"
    uv = shutil.which("uv")
    if uv is None or not (trinity / "tools" / "attest" / "canonical.py").is_file():
        pytest.skip("no trinity checkout and uv beside the harness")
    document = {
        "b": [1, 2.5, 10.0, None, True, -0.0, 123456789012345680000.0, 5e-324],
        "a": {"é": "x\u2028\u0001\"q", "z": 1e-7, "y": 0.000001},
        "c": 1e21,
        "\U0001f600": "emoji key",
    }
    script = (
        "import json, sys; from tools.attest.canonical import canonicalize; "
        "sys.stdout.buffer.write(canonicalize(json.loads(sys.stdin.read())))"
    )
    done = subprocess.run(
        [uv, "run", "--quiet", "--frozen", "--no-dev", "--project", str(trinity), "python", "-c", script],
        cwd=trinity,
        input=json.dumps(document).encode(),
        capture_output=True,
        check=True,
        env={
            **{k: v for k, v in os.environ.items() if k != "UV_PROJECT_ENVIRONMENT"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert harness_config.canonical_bytes(document) == done.stdout


def test_a_run_whose_config_differs_from_the_pin_is_refused(
    project: Path,
    tmp_path: Path,
    agent_toml: Path,
    run_toml: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pin = tmp_path / "pin.json"
    base = ("--run-config", str(run_toml), "--compute", "ec2", "--rollouts", "1")
    assert (
        _infer(project, tmp_path, agent_toml, *base, "--emit-harness-config", str(pin))
        == 0
    )
    digest = hashlib.sha256(pin.read_bytes()).hexdigest()
    run = ("--rollouts", "1", "--dry-run")
    assert (
        _infer(project, tmp_path, agent_toml, *base, *run, "--harness-config", digest)
        == 0
    )
    assert (
        _infer(
            project,
            tmp_path / "b",
            agent_toml,
            *base,
            *run,
            "--harness-config",
            str(pin),
        )
        == 0
    )
    assert (
        _infer(
            project,
            tmp_path / "c",
            agent_toml,
            "--run-config",
            str(run_toml),
            "--compute",
            "local",
            *run,
            "--harness-config",
            digest,
        )
        == 2
    )
    assert "not the pinned" in capsys.readouterr().err
    assert (
        _infer(
            project,
            tmp_path / "d",
            agent_toml,
            *base,
            *run,
            "--harness-config",
            "0" * 64,
        )
        == 2
    )


# 6. pilot summary ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (0, 5, 0.0),
        (1, 1, 0.05),
        (5, 5, 0.05 ** (1 / 5)),
        (5, 10, 0.2224),
        (1, 10, 0.0051),
    ],
)
def test_clopper_pearson_lower_bound(
    successes: int, trials: int, expected: float
) -> None:
    assert pilot.clopper_pearson_lower(successes, trials) == pytest.approx(
        expected, abs=5e-5
    )


def test_clopper_pearson_refuses_impossible_counts() -> None:
    for successes, trials in ((3, 2), (-1, 4), (0, 0)):
        with pytest.raises(ValueError):
            pilot.clopper_pearson_lower(successes, trials)


def _graded(rewards: list[float | None]) -> list[pilot.Graded]:
    return [
        pilot.Graded(i + 1, reward, Path("r"), Path("e"))
        for i, reward in enumerate(rewards)
    ]


def test_pilot_groups_pass_when_any_member_meets_the_threshold() -> None:
    summary = pilot.summarize(
        BUNDLE, _graded([0.0, 1.0, 0.2, None, 0.5, 0.9, 0.0, 0.1]), 4, 1.0
    )
    assert (summary.group_size, summary.groups, summary.passing_groups) == (4, 2, 1)
    assert not summary.saturated
    loose = pilot.summarize(BUNDLE, _graded([0.0, 1.0, 0.2, None, 0.5, 0.9, 0.0, 0.1]), 4, 0.5)
    assert loose.passing_groups == 2 and not loose.saturated
    full = pilot.summarize(BUNDLE, _graded([1.0] * 8), 1, 1.0)
    assert full.lower_bound == pytest.approx(0.05 ** (1 / 8)) and full.saturated
    few = pilot.summarize(BUNDLE, _graded([1.0] * 3), 1, 1.0)
    assert few.lower_bound == pytest.approx(0.05 ** (1 / 3)) and not few.saturated
    with pytest.raises(ValueError, match="whole number of groups"):
        pilot.summarize(BUNDLE, _graded([1.0, 1.0, 1.0]), 2, 1.0)
    with pytest.raises(LookupError):
        pilot.summarize(BUNDLE, [], 4, 1.0)


def _graded_work(
    work: Path, rewards: list[float], digest: str, leak: str | None = None
) -> None:
    rollouts, evaluations = [], []
    for attempt, reward in enumerate(rewards, start=1):
        trial = work / "trials" / f"{BUNDLE}-r{attempt}" / "t"
        evaluation = work / "evaluations" / f"{BUNDLE}-r{attempt}-eval"
        trial.mkdir(parents=True)
        evaluation.mkdir(parents=True)
        (trial / "agent.log").write_text(
            "did work\n" if leak is None or attempt > 1 else leak
        )
        (evaluation / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
        )
        rollouts.append(
            {
                "bundle_uuid": BUNDLE,
                "dry_run": False,
                "probe": False,
                "harness_config_digest": digest,
                "env_from_host": [KEY_NAME],
            }
        )
        evaluations.append(
            {
                "bundle_uuid": BUNDLE,
                "attempt": attempt,
                "dry_run": False,
                "exit_code": 0,
                "evaluation_dir": evaluation.relative_to(work).as_posix(),
                "rollout_trial_dir": trial.relative_to(work).as_posix(),
            }
        )
    (work / "rollouts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rollouts)
    )
    (work / "evaluations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in evaluations)
    )


def test_pilot_summary_stages_rollouts_and_prints_the_self_pilot_command(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    _graded_work(work, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "a" * 64)
    staging = tmp_path / "staging"
    args = (
        "--pilot-summary",
        BUNDLE,
        "--group-size",
        "4",
        "--staging",
        str(staging),
        "--run-id",
        "forge-x-1",
    )
    assert _eval(project, tmp_path, *args) == 0
    out = capsys.readouterr().out
    assert '"groups": 2' in out and '"passing_groups": 1' in out
    assert (
        f"pipeline.py self-pilot ./ {BUNDLE} --run-id forge-x-1 --harness-config-digest {'a' * 64} "
        "--group-size 4 --groups 2 --passing-groups 1"
    ) in out
    copied = staging / BUNDLE / "r3" / "evaluation" / "result.json"
    assert copied.is_file() and not copied.is_symlink()
    assert sorted(p.name for p in (staging / BUNDLE).iterdir()) == [
        f"r{i}" for i in range(1, 9)
    ]


def test_pilot_summary_refuses_to_stage_a_rollout_holding_a_host_secret(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _graded_work(
        tmp_path / "work", [1.0, 0.0, 0.0, 0.0], "b" * 64, leak=f"token={FAKE_KEY}\n"
    )
    staging = tmp_path / "staging"
    assert (
        _eval(project, tmp_path, "--pilot-summary", BUNDLE, "--staging", str(staging))
        == 2
    )
    err = capsys.readouterr().err
    assert "r1/rollout/agent.log" in err and KEY_NAME in err and FAKE_KEY not in err
    assert not (staging / BUNDLE).exists()


def test_pilot_summary_refuses_rollouts_under_two_harness_configs(
    project: Path, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    _graded_work(work, [1.0, 0.0, 0.0, 0.0], "c" * 64)
    rows = _rows(work / "rollouts.jsonl")
    rows[1]["harness_config_digest"] = "d" * 64
    (work / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _eval(project, tmp_path, "--pilot-summary", BUNDLE) == 2


def test_secret_scan_reports_names_never_values(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text(f"x {FAKE_KEY} y")
    hits = secrets.scan_tree(tmp_path, (KEY_NAME,), {KEY_NAME: FAKE_KEY})
    assert hits == [("log.txt", KEY_NAME), ("log.txt", "anthropic-key")]
    assert all(FAKE_KEY not in name for _, name in hits)


# 7. Harbor's exit code hides a failed trial -----------------------------------------------


def _fake_job(exception: str | None) -> Any:
    def harbor(command: list[str], env: object = None) -> int:
        job = json.loads(Path(command[3]).read_text())
        trial = Path(job["jobs_dir"]) / job["job_name"] / "harness-fixture__x1"
        trial.mkdir(parents=True)
        info = None if exception is None else {"exception_type": exception, "exception_message": "m"}
        (trial / "result.json").write_text(json.dumps({"task_name": "kraken/harness-fixture", "exception_info": info}))
        (trial / "config.json").write_text(json.dumps({"agent": {"name": "claude-code"}}))
        return 0

    return harbor


def test_a_rollout_whose_trial_failed_inside_harbor_exits_nonzero_and_is_never_graded(
    project: Path, tmp_path: Path, agent_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Harbor, "run", staticmethod(_fake_job("AgentSetupTimeoutError")))
    assert _infer(project, tmp_path, agent_toml, "--rollouts", "1") == 1
    [row] = _rows(tmp_path / "work" / "rollouts.jsonl")
    assert (row["exit_code"], row["exception"]) == (0, "AgentSetupTimeoutError")
    monkeypatch.setattr(Harbor, "run", staticmethod(Recorder()))
    assert _eval(project, tmp_path) == 0
    assert not (tmp_path / "work" / "evaluations.jsonl").exists()


def _fake_regrade(reward: float | None, exception: str | None) -> Any:
    def harbor(command: list[str], env: object = None) -> int:
        layer = json.loads(Path(command[-1]).read_text())
        trial = Path(layer["trials_dir"]) / layer["trial_name"]
        trial.mkdir(parents=True)
        result: dict[str, Any] = {"exception_info": None if exception is None else {"exception_type": exception}}
        if reward is not None:
            result["verifier_result"] = {"rewards": {"reward": reward}}
        (trial / "result.json").write_text(json.dumps(result))
        return 0

    return harbor


def test_a_failed_regrade_is_recorded_retried_under_a_new_name_and_blocks_the_summary(
    project: Path, tmp_path: Path, agent_toml: Path, run_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Harbor, "validate_regrade", staticmethod(lambda task_dir, task_name: None))
    monkeypatch.setattr(Harbor, "run", staticmethod(_fake_job(None)))
    assert _infer(project, tmp_path, agent_toml, "--rollouts", "1") == 0
    monkeypatch.setattr(Harbor, "run", staticmethod(_fake_regrade(None, "RewardFileNotFoundError")))
    assert _eval(project, tmp_path, "--run-config", str(run_toml)) == 1
    [failed] = _rows(tmp_path / "work" / "evaluations.jsonl")
    assert (failed["exception"], failed["reward"]) == ("RewardFileNotFoundError", None)
    assert _eval(project, tmp_path, "--pilot-summary", BUNDLE, "--group-size", "1") == 2
    monkeypatch.setattr(Harbor, "run", staticmethod(_fake_regrade(0.75, None)))
    assert _eval(project, tmp_path, "--run-config", str(run_toml)) == 0
    _, retried = _rows(tmp_path / "work" / "evaluations.jsonl")
    assert retried["evaluation_dir"].endswith("-r1-eval-2")
    assert (retried["exception"], retried["reward"]) == (None, 0.75)
    assert _eval(project, tmp_path, "--pilot-summary", BUNDLE, "--group-size", "1", "--pass-threshold", "0.5") == 0
