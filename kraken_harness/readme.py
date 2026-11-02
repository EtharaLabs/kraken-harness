"""Render ``harness/README.md`` from the harness code, so the page cannot drift from it.

From the harness root, ``python -m kraken_harness.readme`` rewrites the page and
``python -m kraken_harness.readme --check`` fails when the page is stale.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from kraken_harness import (
    base,
    benchmarks,
    cli,
    harness_config,
    layers,
    layout,
    pilot,
    run_config,
    secrets,
)

README = "README.md"
UPSTREAM = Path("docs") / "upstream" / "swefficiency-README.md"
STAMP = (
    "<!-- Rendered by `python -m kraken_harness.readme` from the harness code. "
    "Edit the code, not this page. -->"
)
INTRO = (
    "The benchmark extension an external runner uses to run inference and evaluation over the "
    "Kraken parent's resident bundles. It vendors the SWE-fficiency harness, whose own README "
    "closes this page, and adds the `kraken_harness` package. Every trial runs through the "
    "pinned Harbor release, every rollout is scored by the bundle's own tests, and the "
    "extension carries no grading logic of its own."
)
RUNNING = (
    "FORGE never runs this extension against a solver. An author-side run is a "
    "`.seed/probe.yaml` record through the `rig` door, never difficulty evidence, and every "
    "measured rollout comes from the out-of-band Phase evidence runner."
)


def _doc(target: object) -> str:
    text = inspect.getdoc(target) or ""
    return " ".join(text.split("\n\n", 1)[0].split())


def _codes(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _entry_points(project: dict) -> list[str]:
    out = ["## Entry points", "", "| command | target |", "| --- | --- |"]
    out += [f"| `{name}` | `{target}` |" for name, target in project["scripts"].items()]
    for parser in (cli.infer_parser(), cli.eval_parser()):
        out += [
            "",
            f"### `{parser.prog}`",
            "",
            "```text",
            parser.format_help().rstrip(),
            "```",
        ]
    return out


def _interface() -> list[str]:
    out = [
        "## Shared evaluation interface",
        "",
        _doc(base),
        "",
        "| duty | what it does |",
        "| --- | --- |",
    ]
    for name in ("prepare_instances", "prepare_workspace", "evaluate_instance"):
        out.append(f"| `{name}` | {_doc(getattr(base.Benchmark, name))} |")
    out += ["", "| inherited loop | what it does |", "| --- | --- |"]
    for name in ("run_inference", "run_evaluation"):
        out.append(f"| `{name}` | {_doc(getattr(base.Benchmark, name))} |")
    return out


def _benchmarks() -> list[str]:
    out = ["## Benchmarks", ""]
    for name in benchmarks.available():
        config = benchmarks.config_of(name)
        out += [
            f"### `{name}`",
            "",
            config.description,
            "",
            "| default | value |",
            "| --- | --- |",
            f"| directory | `kraken_harness/benchmarks/{name}/` |",
            f"| Harbor release | `{config.harbor_release}` at `{config.harbor_commit}` |",
            f"| resident lanes | {_codes(config.lanes)} |",
            f"| rollouts per instance | {config.rollouts} |",
            "",
        ]
    return out


def _layout() -> list[str]:
    return [
        "## Workspace layout",
        "",
        _doc(layout),
        "",
        "| carried | paths |",
        "| --- | --- |",
        f"| agent-visible files | {_codes(layout.AGENT_VISIBLE_FILES)} |",
        f"| agent-visible trees | {_codes([name + '/' for name in layout.AGENT_VISIBLE_DIRS])} |",
        f"| config files | {_codes(layout.CONFIG_FILES)} |",
        f"| per step under `{layout.STEPS_DIR}/<s>/` | "
        f"{_codes(layout.STEP_VISIBLE_FILES)}, {_codes([n + '/' for n in layout.STEP_VISIBLE_DIRS])} |",
    ]


def _pins(pyproject: dict) -> list[str]:
    project = pyproject["project"]
    out = [
        "## Pinned environment",
        "",
        f"Python `{project['requires-python']}`. `uv.lock` pins every transitive dependency.",
        "",
        "| dependency | where |",
        "| --- | --- |",
    ]
    out += [f"| `{item}` | runtime |" for item in project["dependencies"]]
    for group, items in project.get("optional-dependencies", {}).items():
        out += [f"| `{item}` | extra `{group}` |" for item in items]
    for group, items in pyproject.get("dependency-groups", {}).items():
        out += [f"| `{item}` | group `{group}` |" for item in items]
    out += [f"| `{item}` | build |" for item in pyproject["build-system"]["requires"]]
    return out


def _running() -> list[str]:
    return [
        "## Running it",
        "",
        RUNNING,
        "",
        "```sh",
        "uv sync --frozen",
        "uv run --frozen kraken-infer --root <parent> --work-dir <dir outside the parent> --agent-config agent.toml \\",
        "    --run-config run.toml --compute ec2",
        "uv run --frozen kraken-eval --root <parent> --work-dir <the same dir> --run-config run.toml --compute ec2",
        "uv run --frozen python -m kraken_harness.readme --check",
        "```",
        "",
        "`agent.toml` holds the agent under test and its model, outside the harness code:",
        "",
        "```toml",
        "[agent]",
        'name = "<a Harbor agent name>"',
        'model_name = "<provider/model>"',
        'model_api_key_name = "<the host variable holding the model key>"',
        "```",
        "",
        _doc(run_config),
        "",
        "| table | allowed keys |",
        "| --- | --- |",
        f"| top level | {_codes(sorted(run_config.TOP_LEVEL))} |",
        f"| `[agent]` | {_codes(sorted(run_config.AGENT_KEYS))} |",
        f"| `[environment]` | {_codes(sorted(run_config.ENVIRONMENT_KEYS))} |",
        f"| `[verifier]` (evaluation and probes only) | {_codes(sorted(run_config.VERIFIER_KEYS))} |",
        f"| `[compute.<name>]` | {_codes(sorted(run_config.PROFILE_KEYS))} |",
        "",
        "```toml",
        'env_from_host = ["ANTHROPIC_API_KEY", "LITELLM_PROXY_API_KEY"]',
        "",
        "[agent]",
        'model_api_format = "anthropic_messages"',
        "agent_setup_timeout_multiplier = 10",
        "",
        "[environment]",
        'extra_allowed_hosts = ["deb.debian.org", "registry.npmjs.org"]',
        'extra_docker_compose = ["agent-hosts.compose.yaml"]',
        "",
        "[verifier]",
        'env = { REWARDKIT_JUDGE = "<judge>", LITELLM_PROXY_API_KEY = "${LITELLM_PROXY_API_KEY}" }',
        "",
        "[compute.local]",
        'docker_host = "unix://~/.colima/default/docker.sock"',
        'gate_host = "host.docker.internal"',
        "",
        "[compute.ec2]",
        'trinity_target = "@trinity"',
        'gate_host = "172.17.0.1"',
        "```",
        "",
        _doc(secrets),
        "",
        _doc(layers),
        "",
        "`--via job` (the default) runs each rollout as a one-trial `harbor run` job with "
        "verification off; `--via trial` uses `harbor trial start`. `--probe` runs the agent and "
        "the bundle's verifier in one job under `probes/`: an author-side probe, never graded and "
        f"never difficulty evidence. {_doc(base.Benchmark._refuse_mixed_grading)}",
        "",
        _doc(harness_config),
        "",
        _doc(pilot),
    ]


def render(harness: Path) -> str:
    """The README the harness code at ``harness`` renders."""
    pyproject = tomllib.loads((harness / "pyproject.toml").read_text(encoding="utf-8"))
    upstream = (harness / UPSTREAM).read_text(encoding="utf-8").strip("\n")
    sections = [
        [STAMP, "", "# kraken-harness", "", INTRO],
        _entry_points(pyproject["project"]),
        _interface(),
        _benchmarks(),
        _layout(),
        _pins(pyproject),
        _running(),
        [
            "## Upstream SWE-fficiency documentation",
            "",
            f"The vendored harness's own README, kept verbatim at `{UPSTREAM.as_posix()}`, follows.",
            "",
            upstream,
        ],
    ]
    return "\n\n".join("\n".join(section).strip("\n") for section in sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kraken_harness.readme", description=_doc(sys.modules[__name__])
    )
    parser.add_argument(
        "--harness", type=Path, default=Path("."), help="the harness root (default: .)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when README.md is stale; write nothing",
    )
    args = parser.parse_args(argv)
    harness = args.harness.resolve()
    expected = render(harness)
    path = harness / README
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if args.check:
        if current != expected:
            print(
                f"{path} is stale; run python -m kraken_harness.readme", file=sys.stderr
            )
            return 1
        return 0
    if current != expected:
        path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
