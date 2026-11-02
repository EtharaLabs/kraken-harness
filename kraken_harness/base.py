"""The shared evaluation interface and the orchestration every benchmark inherits.

A benchmark implements three duties: prepare instances from the resident bundles, prepare
a per-instance workspace, and evaluate one instance. The two run loops below are final:
a benchmark inherits them as they are and never reimplements them, and the loops hand all
environment, agent and verifier work to the pinned Harbor release.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, final

from kraken_harness import compose, pilot, secrets
from kraken_harness.config import BenchmarkConfig
from kraken_harness.harbor_bridge import Harbor
from kraken_harness.lanes import Instance
from kraken_harness.layers import private_layer
from kraken_harness.run_config import Profile, RunConfig, harbor_environ
from kraken_harness.workspace import Workspace, carried_digest, environment_digest

ROLLOUTS = "rollouts.jsonl"
EVALUATIONS = "evaluations.jsonl"
# Harbor's oracle agent runs a bundle's reference answer, so it never stands in for the agent under test.
ORACLE_AGENT = "oracle"
VIA_JOB = "job"
VIA_TRIAL = "trial"
VIAS = (VIA_JOB, VIA_TRIAL)
TRIAL_FILES = ("result.json", "config.json")


@dataclass(frozen=True, slots=True)
class Rollout:
    instance: Instance
    attempt: int
    trial_name: str
    trial_dir: Path
    profile: str | None = None
    task_name: str | None = None
    source_agent: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Options:
    """How one invocation runs: everything beyond the benchmark and the agent table."""

    run: RunConfig = field(default_factory=RunConfig)
    profile: Profile | None = None
    via: str = VIA_JOB
    probe: bool = False
    dry_run: bool = False
    harness_config_digest: str | None = None
    allow_mixed_grading: bool = False
    environ: Mapping[str, str] | None = None


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _relative(path: Path, work_dir: Path) -> str:
    return path.relative_to(work_dir).as_posix()


def checked_agent(agent: dict[str, Any]) -> dict[str, Any]:
    """The agent table of the agent under test, refused when it names no agent or the oracle."""
    name, import_path = agent.get("name"), agent.get("import_path")
    if not name and not import_path:
        raise ValueError("the agent table names no agent: set name or import_path")
    if name == ORACLE_AGENT:
        raise ValueError("the oracle agent is not an agent under test")
    return dict(agent)


def job_trial_dir(job_dir: Path) -> Path:
    """The one trial directory a one-trial Harbor job wrote under ``job_dir``."""
    found: list[Path] = []
    if job_dir.is_dir():
        for entry in sorted(job_dir.iterdir()):
            if entry.is_dir() and not entry.is_symlink():
                if all((entry / name).is_file() for name in TRIAL_FILES):
                    found.append(entry)
    if len(found) != 1:
        raise LookupError(
            f"{job_dir}: expected one trial directory, found {len(found)}"
        )
    return found[0]


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    body = json.loads(path.read_text(encoding="utf-8"))
    return body if isinstance(body, dict) else {}


def trial_exception(trial_dir: Path | None) -> str | None:
    """The exception type Harbor recorded for a finished trial, which its exit code hides.

    ``harbor run`` and ``harbor trial start`` exit 0 when a trial fails inside, so a rollout
    or a regrade is judged by its own ``result.json``. A trial that wrote no result failed.
    """
    if trial_dir is None:
        return "TrialDirectoryMissing"
    path = trial_dir / "result.json"
    if not path.is_file():
        return "ResultMissing"
    info = _json_object(path).get("exception_info")
    if isinstance(info, dict) and info.get("exception_type"):
        return str(info["exception_type"])
    return None


def _source_trial(trial_dir: Path) -> tuple[str | None, dict[str, Any]]:
    """The task name and agent table a recorded trial ran with, read from its own files."""
    name = _json_object(trial_dir / "result.json").get("task_name")
    agent = _json_object(trial_dir / "config.json").get("agent")
    return (
        name if isinstance(name, str) else None,
        dict(agent) if isinstance(agent, dict) else {},
    )


class Benchmark(abc.ABC):
    """One benchmark: its three duties, its configuration defaults, and the pinned Harbor."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.harbor = Harbor(config.harbor_release, config.harbor_commit)

    @abc.abstractmethod
    def prepare_instances(
        self, root: Path, lanes: tuple[str, ...], only: frozenset[str]
    ) -> list[Instance]:
        """Find the instances to run among the resident bundles under the lane roots."""

    @abc.abstractmethod
    def prepare_workspace(self, instance: Instance, work_dir: Path) -> Workspace:
        """Build the per-instance sandboxed workspace Harbor runs the agent in."""

    @abc.abstractmethod
    def evaluate_instance(self, rollout: Rollout, work_dir: Path) -> list[str]:
        """Return the Harbor command that scores one rollout with the bundle's own tests."""

    def _launch(self, command: list[str], options: Options) -> int | None:
        if options.dry_run:
            return None
        missing = secrets.missing_from_host(options.run.env_from_host, options.environ)
        if missing:
            raise RuntimeError(
                f"the host environment does not set {', '.join(missing)}"
            )
        env = harbor_environ(options.profile, options.environ)
        # Harbor's credential preflight wants every variable a task's [verifier.env] names, even
        # with verification off. The run's own verifier values are non-secret literals, the
        # profile's judge address, or ${NAME} templates the host already sets, so exporting them
        # to the Harbor process puts no credential on disk or on the command line.
        for name, value in compose.verifier_table(options.run, options.profile).get("env", {}).items():
            found = secrets.TEMPLATE.fullmatch(value)
            env.setdefault(name, env.get(found.group(1), "") if found else value)
        return self.harbor.run(command, env=env)

    def _row(
        self, instance: Instance, attempt: int, options: Options
    ) -> dict[str, Any]:
        return {
            "benchmark": self.config.name,
            "bundle_uuid": instance.uuid,
            "lane": instance.lane,
            "attempt": attempt,
            "harbor_release": self.config.harbor_release,
            "compute": None if options.profile is None else options.profile.record(),
            "env_from_host": list(options.run.env_from_host),
            "harness_config_digest": options.harness_config_digest,
            "dry_run": options.dry_run,
        }

    def _inference_document(
        self,
        task_dir: Path,
        trial_name: str,
        trials_dir: Path,
        agent: dict[str, Any],
        options: Options,
    ) -> tuple[dict[str, Any], str]:
        """The validated Harbor document of one rollout, and the layer file stem it goes in."""
        if options.via == VIA_JOB:
            verifier = (
                compose.verifier_table(options.run, options.profile)
                if options.probe
                else None
            )
            document = compose.job_config(
                task_dir,
                trial_name,
                trials_dir,
                agent,
                options.run,
                verifier=verifier,
            )
            self.harbor.validate_job(document)
            return document, "job"
        document = compose.trial_layer(
            task_dir, trial_name, trials_dir, agent, options.run
        )
        self.harbor.validate_trial(document)
        return document, "trial"

    @final
    def run_inference(
        self,
        root: Path,
        work_dir: Path,
        agent: dict[str, Any],
        *,
        lanes: tuple[str, ...] | None = None,
        only: frozenset[str] = frozenset(),
        rollouts: int | None = None,
        options: Options | None = None,
    ) -> list[dict[str, Any]]:
        """Run the agent under test on each instance through Harbor with verification off."""
        options = options or Options()
        if options.via not in VIAS:
            raise ValueError(f"--via must be one of {', '.join(VIAS)}")
        if options.probe and options.via != VIA_JOB:
            raise ValueError("a probe runs as one Harbor job; use --via job")
        self.harbor.require_lock(root)
        self.harbor.require_installed()
        agent = compose.agent_table(checked_agent(agent), options.run, options.profile)
        count = self.config.rollouts if rollouts is None else rollouts
        if count < 1:
            raise ValueError("rollouts must be at least 1")
        trials_dir = work_dir / ("probes" if options.probe else "trials")
        records: list[dict[str, Any]] = []
        for instance in self.prepare_instances(root, lanes or self.config.lanes, only):
            workspace = self.prepare_workspace(instance, work_dir)
            # A probe runs the bundle's own verifier in the same job, so Harbor reads the
            # bundle directory itself; a graded rollout runs in the workspace, which has no tests/.
            task_dir = instance.bundle if options.probe else workspace.task_dir
            self.harbor.validate_task(task_dir, disable_verification=not options.probe)
            for attempt in range(1, count + 1):
                trial_name = f"{instance.uuid}-r{attempt}"
                document, stem = self._inference_document(
                    task_dir, trial_name, trials_dir, agent, options
                )
                layer_path = workspace.manifest.parent / f"{stem}-r{attempt}.json"
                start = (
                    self.harbor.job_run if stem == "job" else self.harbor.trial_start
                )
                with private_layer(layer_path, document, keep=options.dry_run) as path:
                    command = start(path)
                    exit_code = self._launch(command, options)
                job_dir = trials_dir / trial_name if stem == "job" else None
                trial_dir = trials_dir / trial_name if job_dir is None else None
                if job_dir is not None and exit_code == 0:
                    try:
                        trial_dir = job_trial_dir(job_dir)
                    except LookupError:
                        trial_dir = None
                exception = None if exit_code is None else trial_exception(trial_dir)
                record = {
                    **self._row(instance, attempt, options),
                    "via": options.via,
                    "probe": options.probe,
                    "trial_name": trial_name,
                    "trial_dir": None
                    if trial_dir is None
                    else _relative(trial_dir, work_dir),
                    "job_dir": None
                    if job_dir is None
                    else _relative(job_dir, work_dir),
                    "layer": _relative(layer_path, work_dir),
                    "layer_kept": options.dry_run,
                    "workspace_digest": workspace.digest,
                    "environment_digest": workspace.environment_digest,
                    "agent": {
                        key: agent.get(key)
                        for key in ("name", "import_path", "model_name", "base_url")
                    },
                    "command": command,
                    "exit_code": exit_code,
                    "exception": exception,
                }
                _append(work_dir / ROLLOUTS, record)
                records.append(record)
        return records

    @final
    def run_evaluation(
        self, root: Path, work_dir: Path, *, options: Options | None = None
    ) -> list[dict[str, Any]]:
        """Score every recorded rollout by regrading its trial through Harbor."""
        options = options or Options()
        self.harbor.require_lock(root)
        self.harbor.require_installed()
        rollouts = self._recorded_rollouts(root, work_dir, dry_run=options.dry_run)
        self._refuse_mixed_grading(work_dir, rollouts, options)
        verifier = compose.verifier_table(options.run, options.profile)
        records: list[dict[str, Any]] = []
        for rollout in rollouts:
            evaluations = work_dir / "evaluations"
            eval_name = f"{rollout.trial_name}-eval"
            if not options.dry_run:
                # A regrade never reuses a trial directory: a second grading of the same
                # rollout, after a fix to the judge, gets the next free name.
                attempt = 2
                while (evaluations / eval_name).exists():
                    eval_name = f"{rollout.trial_name}-eval-{attempt}"
                    attempt += 1
            if not options.dry_run:
                self.harbor.validate_regrade(rollout.instance.bundle, rollout.task_name)
            layer_rel = None
            if verifier:
                # `harbor trial regrade` takes verifier env but no verifier hosts, so the same
                # regrade is handed to Harbor as a TrialConfig carrying source_trial.
                document = compose.regrade_layer(
                    rollout.instance.bundle,
                    rollout.trial_dir,
                    rollout.source_agent,
                    eval_name,
                    evaluations,
                    verifier,
                )
                self.harbor.validate_trial(document)
                layer_path = work_dir / "regrades" / f"{eval_name}.json"
                layer_rel = _relative(layer_path, work_dir)
                with private_layer(layer_path, document, keep=options.dry_run) as path:
                    command = self.harbor.trial_start(path)
                    exit_code = self._launch(command, options)
            else:
                command = self.evaluate_instance(rollout, work_dir)
                exit_code = self._launch(command, options)
            record = {
                **self._row(rollout.instance, rollout.attempt, options),
                "rollout_trial_dir": _relative(rollout.trial_dir, work_dir),
                "rollout_compute": rollout.profile,
                "evaluation_dir": _relative(evaluations / eval_name, work_dir),
                "layer": layer_rel,
                "verifier_env_names": sorted(verifier.get("env", {})),
                "verifier_allowed_hosts": verifier.get("extra_allowed_hosts", []),
                "command": command,
                "exit_code": exit_code,
                "exception": None if exit_code is None else trial_exception(evaluations / eval_name),
                "reward": None if exit_code is None else pilot.reward_of(evaluations / eval_name),
            }
            _append(work_dir / EVALUATIONS, record)
            records.append(record)
        return records

    def _refuse_mixed_grading(
        self, work_dir: Path, rollouts: list[Rollout], options: Options
    ) -> None:
        """Refuse grading a bundle on a host other than the one that graded it already."""
        if options.allow_mixed_grading:
            return
        here = None if options.profile is None else options.profile.name
        graded: dict[str, set[str | None]] = {}
        path = work_dir / EVALUATIONS
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("dry_run"):
                    continue
                compute = row.get("compute")
                profile = compute.get("profile") if isinstance(compute, dict) else None
                graded.setdefault(row["bundle_uuid"], set()).add(profile)
        for rollout in rollouts:
            earlier = graded.get(rollout.instance.uuid, set()) - {here}
            if earlier:
                raise ValueError(
                    f"bundle {rollout.instance.uuid} was graded on "
                    f"{sorted(str(name) for name in earlier)}; grading it on {here} mixes hosts. "
                    "Grade every rollout of a bundle on one profile, or pass --allow-mixed-grading"
                )

    def _recorded_rollouts(
        self, root: Path, work_dir: Path, *, dry_run: bool
    ) -> list[Rollout]:
        path = work_dir / ROLLOUTS
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} records no rollout; run kraken-infer first"
            )
        out: list[Rollout] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["benchmark"] != self.config.name or record.get("probe"):
                continue
            if record["dry_run"] and not dry_run:
                continue
            if not record["dry_run"] and (record.get("exit_code") != 0 or record.get("exception")):
                continue
            relative = record.get("trial_dir") or record.get("job_dir")
            trial_dir = (
                work_dir / relative
                if relative
                else work_dir / "trials" / record["trial_name"]
            )
            if not dry_run and not trial_dir.is_dir():
                raise FileNotFoundError(
                    f"{trial_dir}: the rollout's trial directory is absent"
                )
            found = self.prepare_instances(
                root, (record["lane"],), frozenset({record["bundle_uuid"]})
            )
            if len(found) != 1:
                raise LookupError(
                    f"bundle {record['bundle_uuid']} is no longer resident in {record['lane']}"
                )
            instance = found[0]
            if carried_digest(instance.bundle) != record["workspace_digest"]:
                raise ValueError(
                    f"{instance.bundle}: carried bytes moved since inference; rerun kraken-infer"
                )
            if environment_digest(instance.bundle) != record["environment_digest"]:
                raise ValueError(
                    f"{instance.bundle}: environment moved since inference; rerun kraken-infer"
                )
            task_name, source_agent = (
                _source_trial(trial_dir) if trial_dir.is_dir() else (None, {})
            )
            if not source_agent:
                source_agent = {"name": record.get("agent", {}).get("name")}
            compute = record.get("compute")
            out.append(
                Rollout(
                    instance,
                    int(record["attempt"]),
                    record["trial_name"],
                    trial_dir,
                    compute.get("profile") if isinstance(compute, dict) else None,
                    task_name,
                    source_agent,
                )
            )
        return out
