"""The shared command line: ``kraken-infer`` and ``kraken-eval``."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kraken_harness import benchmarks, compose, harness_config, pilot, run_config
from kraken_harness.base import (
    EVALUATIONS,
    ROLLOUTS,
    VIA_JOB,
    VIAS,
    Options,
    checked_agent,
)

# A fixed width keeps --help, and the README rendered from it, the same on every terminal.
HELP_WIDTH = 88
RUN_ERRORS = (ValueError, RuntimeError, LookupError, OSError)
SEALING_RUN = "<the sealing FORGE run>"


def _formatter(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, width=HELP_WIDTH)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="the parent project root that holds the lane roots and harbor.lock (default: .)",
    )
    parser.add_argument(
        "--benchmark",
        default="kraken",
        help="the benchmark directory to run (default: kraken)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="a directory outside --root for workspaces, trial layers and records",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare and validate everything, print the Harbor commands, and run none",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="a run.toml of allowlisted Harbor extras, env_from_host names and compute "
        "profiles; any other key is refused",
    )
    parser.add_argument(
        "--compute",
        metavar="PROFILE",
        help="the [compute.<PROFILE>] of --run-config: its Docker host and the gate host its "
        "containers reach",
    )


def infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-infer",
        description="Run the agent under test on resident bundles through the pinned Harbor "
        "release with verification off. Each trial directory is a rollout log for kraken-eval.",
        formatter_class=_formatter,
    )
    _common(parser)
    parser.add_argument(
        "--agent-config",
        type=Path,
        required=True,
        help="a JSON or TOML file holding one agent table: the agent under test and its model",
    )
    parser.add_argument(
        "--lane",
        action="append",
        help="a lane to read; repeatable (default: the benchmark's resident lanes)",
    )
    parser.add_argument(
        "--instance",
        action="append",
        help="a bundle uuid to run; repeatable (default: every resident bundle)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        help="rollouts per instance (default: the benchmark's rollouts)",
    )
    parser.add_argument(
        "--via",
        choices=VIAS,
        default=VIA_JOB,
        help="run each rollout as a one-trial `harbor run` job, or as `harbor trial start` "
        "(default: job)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="an author-side probe: one job runs the agent and the bundle's verifier together; "
        "a probe is never a rollout kraken-eval grades and never difficulty evidence",
    )
    parser.add_argument(
        "--harness-config",
        metavar="PIN",
        help="refuse the run unless its resolved harness config hashes to PIN, a sha256 or a "
        "file --emit-harness-config wrote",
    )
    parser.add_argument(
        "--emit-harness-config",
        type=Path,
        metavar="PATH",
        help="write the resolved harness config to PATH, print its sha256, and run nothing",
    )
    return parser


def eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-eval",
        description="Score each rollout kraken-infer recorded in --work-dir by regrading its "
        "trial through the pinned Harbor release with the bundle's own tests.",
        formatter_class=_formatter,
    )
    _common(parser)
    parser.add_argument(
        "--allow-mixed-grading",
        action="store_true",
        help="grade a bundle on this profile though another profile already graded it",
    )
    parser.add_argument(
        "--pilot-summary",
        metavar="UUID",
        help="grade nothing; summarize the graded rollouts of bundle UUID as pilot groups",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=1.0,
        help="the regrade reward a rollout must reach to pass (default: 1.0)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="rollouts per pilot group (default: the benchmark's rollouts)",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        help="a directory outside --root where --pilot-summary copies the graded trial "
        "directories for .seed/pilots/rollouts/<UUID>/",
    )
    parser.add_argument(
        "--run-id",
        help="the sealing FORGE run the printed self-pilot command names",
    )
    return parser


def _outside(root: Path, path: Path, flag: str) -> Path:
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(
            f"{flag} {resolved} lies inside --root {root}; choose a directory outside it"
        )
    return resolved


def _work_dir(root: Path, work_dir: Path) -> Path:
    work = _outside(root, work_dir, "--work-dir")
    work.mkdir(parents=True, exist_ok=True)
    return work


def _agent(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    document = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
    if (
        not isinstance(document, dict)
        or set(document) != {"agent"}
        or not isinstance(document["agent"], dict)
    ):
        raise ValueError(f"{path}: must hold exactly one top-level agent table")
    return document["agent"]


def _settings(
    root: Path, args: argparse.Namespace
) -> tuple[run_config.RunConfig, run_config.Profile | None]:
    run = run_config.load(args.run_config)
    profile = (
        None
        if args.compute is None
        else run_config.resolve_profile(run, args.compute, root)
    )
    return run, profile


def _report(prog: str, records: list[dict[str, Any]], dry_run: bool) -> int:
    for record in records:
        state = "dry run" if dry_run else f"exit {record['exit_code']}"
        if record.get("exception"):
            state += f", trial failed: {record['exception']}"
        if record.get("reward") is not None:
            state += f", reward {record['reward']}"
        where = record.get("compute") or {}
        host = f" @{where['profile']}" if where else ""
        print(
            f"{prog}: {record['bundle_uuid']} r{record['attempt']}{host} [{state}]: "
            f"{' '.join(record['command'])}"
        )
    print(f"{prog}: {len(records)} trial(s)" + (", none run" if dry_run else ""))
    ok = all(
        record["exit_code"] in (0, None) and not record.get("exception") for record in records
    )
    return 0 if ok else 1


def infer_main(argv: Sequence[str] | None = None) -> int:
    args = infer_parser().parse_args(argv)
    root = args.root.resolve()
    try:
        benchmark = benchmarks.load(args.benchmark)
        run, profile = _settings(root, args)
        raw = _agent(args.agent_config)
        rollouts = benchmark.config.rollouts if args.rollouts is None else args.rollouts
        agent = compose.agent_table(checked_agent(raw), run, profile)
        document = harness_config.resolve(
            benchmark.config, agent, run, profile, via=args.via, rollouts=rollouts
        )
        if args.emit_harness_config is not None:
            digest = harness_config.emit(args.emit_harness_config, document)
            print(f"kraken-infer: wrote {args.emit_harness_config.resolve()}")
            print(f"kraken-infer: harness config sha256 {digest}")
            return 0
        digest = harness_config.require_pin(args.harness_config, document)
        records = benchmark.run_inference(
            root,
            _work_dir(root, args.work_dir),
            raw,
            lanes=tuple(args.lane) if args.lane else None,
            only=frozenset(args.instance or ()),
            rollouts=args.rollouts,
            options=Options(
                run=run,
                profile=profile,
                via=args.via,
                probe=args.probe,
                dry_run=args.dry_run,
                harness_config_digest=digest,
            ),
        )
    except RUN_ERRORS as error:
        print(f"kraken-infer: {error}", file=sys.stderr)
        return 2
    print(f"kraken-infer: harness config sha256 {digest}")
    return _report("kraken-infer", records, args.dry_run)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pilot_summary(
    root: Path, work: Path, args: argparse.Namespace, group_default: int
) -> int:
    uuid = args.pilot_summary
    graded: dict[int, pilot.Graded] = {}
    failed: dict[int, str] = {}
    for row in _rows(work / EVALUATIONS):
        if row.get("bundle_uuid") != uuid or row.get("dry_run"):
            continue
        attempt = int(row["attempt"])
        if row.get("exit_code") != 0 or row.get("exception"):
            failed[attempt] = str(row.get("exception") or f"exit {row.get('exit_code')}")
            continue
        evaluation = work / row["evaluation_dir"]
        failed.pop(attempt, None)
        graded[attempt] = pilot.Graded(
            attempt,
            pilot.reward_of(evaluation),
            work / row["rollout_trial_dir"],
            evaluation,
        )
    still_failed = {a: why for a, why in failed.items() if a not in graded}
    if still_failed:
        listed = ", ".join(f"r{a} ({why})" for a, why in sorted(still_failed.items()))
        raise ValueError(
            f"bundle {uuid} has evaluations that failed inside Harbor: {listed}; a failed "
            "grading is not a failed rollout, so regrade them before summarizing"
        )
    group_size = group_default if args.group_size is None else args.group_size
    summary = pilot.summarize(
        uuid, list(graded.values()), group_size, args.pass_threshold
    )
    digests = {
        row.get("harness_config_digest")
        for row in _rows(work / ROLLOUTS)
        if row.get("bundle_uuid") == uuid
        and not row.get("dry_run")
        and not row.get("probe")
    }
    if len(digests) != 1 or None in digests:
        raise ValueError(
            f"bundle {uuid}'s rollouts carry harness config digests {sorted(map(str, digests))}; "
            "a pilot summary needs one pinned config"
        )
    digest = str(digests.pop())
    names = tuple(
        sorted(
            {
                name
                for row in _rows(work / ROLLOUTS)
                for name in row.get("env_from_host", [])
            }
        )
    )
    record = pilot.as_record(summary, args.pass_threshold)
    print(json.dumps(record, indent=2, sort_keys=True))
    if summary.saturated:
        print(
            f"kraken-eval: SATURATED: the {pilot.CONFIDENCE:.0%} lower bound "
            f"{summary.lower_bound:.4f} reaches {pilot.SATURATION:.2f}"
        )
    if args.staging is not None:
        staged = pilot.stage(summary, _outside(root, args.staging, "--staging"), names)
        print(
            f"kraken-eval: staged {len(summary.rollouts)} graded rollouts at {staged}"
        )
        print(
            f"kraken-eval: copy it to ./.seed/pilots/rollouts/{uuid}/ in the sealing run"
        )
    print("kraken-eval: record it from the parent root with (not run here):")
    print(pilot.self_pilot_command(summary, args.run_id or SEALING_RUN, digest))
    return 0


def eval_main(argv: Sequence[str] | None = None) -> int:
    args = eval_parser().parse_args(argv)
    root = args.root.resolve()
    try:
        benchmark = benchmarks.load(args.benchmark)
        work = _work_dir(root, args.work_dir)
        if args.pilot_summary is not None:
            return _pilot_summary(root, work, args, benchmark.config.rollouts)
        run, profile = _settings(root, args)
        records = benchmark.run_evaluation(
            root,
            work,
            options=Options(
                run=run,
                profile=profile,
                dry_run=args.dry_run,
                allow_mixed_grading=args.allow_mixed_grading,
            ),
        )
    except RUN_ERRORS as error:
        print(f"kraken-eval: {error}", file=sys.stderr)
        return 2
    return _report("kraken-eval", records, args.dry_run)
