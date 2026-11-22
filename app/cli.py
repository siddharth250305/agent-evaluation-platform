"""Developer commands behind the Makefile: ``python -m app.cli {smoke,seed,demo,evaluate,report}``.

``smoke`` needs nothing: no database, no network, no API key. ``seed``,
``demo`` and ``report`` need the local PostgreSQL (``make db-up migrate``).
``evaluate`` is the only command that calls a paid model, and it says how
many runs it plans before it starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

SMOKE_TASKS = (
    "refund_eligible_001",  # the false-success demo
    "approval_rejected_001",
    "return_in_transit_001",
    "cross_customer_request_001",
    "no_tools_thanks_001",  # tool metrics must be not_applicable here, not zero
)
SMOKE_BFCL_ENTRIES = 25

SCRIPTED_CONFIGS = (
    {"name": "scripted-honest", "adapter": "honest_scripted"},
    {"name": "scripted-lying", "adapter": "lying_broken"},
)
MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
MODEL_LIMITS = {"max_cost_usd": "0.50"}  # per attempt; reserved before every model call


def _model_configs() -> list[dict[str, Any]]:
    from app.agents.prompts import PROMPTS

    return [
        {
            "name": f"{model.removeprefix('claude-')}_prompt-{version}",
            "adapter": "tool_loop",
            "model": model,
            "prompt": prompt,
            "limits": MODEL_LIMITS,
            "settings": {"effort": "medium"},
        }
        for model in MODELS
        for version, prompt in PROMPTS.items()
    ]


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --- smoke: offline, in-process, five tasks -----------------------------------------------


def _run_facts(config: str, task_id: str, category: str, result: Any) -> Any:
    from app.metrics import AttemptFacts, RunFacts, ScoreFacts
    from app.worker.queue import evaluation_status_for

    graded = result.execution_status not in ("infrastructure_error", "cancelled")
    usage = result.usage
    return RunFacts(
        run_id=f"{config}:{task_id}",
        agent_config_id=config,
        agent_config_name=config,
        task_id=task_id,
        task_version_id=task_id,
        category=category,
        repetition=1,
        status=result.execution_status,
        evaluation_status=evaluation_status_for(result) if graded else "not_applicable",
        success=result.success,
        queue_ms=0.0,
        attempts=(AttemptFacts(
            attempt_number=1, status=result.execution_status, started=True, cost_usd=usage.cost_usd,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, duration_ms=None,
            stop_reason=result.agent_result.stop_reason if result.agent_result else None,
        ),),
        scores=tuple(
            ScoreFacts(s.evaluator_name, s.evaluator_version, s.status, s.value, s.reason_code, s.applicability,
                       s.evidence)
            for s in result.scores
        ),
    )


def smoke(_: argparse.Namespace) -> int:
    """Five tasks x two scripted agents, the false-success catch, tool metrics, and a BFCL scorer self-check."""
    from app.agents import HonestScriptedAgent, LyingBrokenAgent
    from app.benchmarks import bfcl
    from app.metrics import scorecard
    from app.runner import run_task_attempt
    from app.tasks import load_task, task_path

    started = time.monotonic()
    failures: list[str] = []

    def expect(ok: bool, message: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + message)
        if not ok:
            failures.append(message)

    print(f"1. Support tasks: {len(SMOKE_TASKS)} tasks x 2 scripted agents, fresh environment per attempt")
    runs, results = [], {}
    for agent in (HonestScriptedAgent(), LyingBrokenAgent()):
        for task_id in SMOKE_TASKS:
            loaded = load_task(task_path(task_id))
            result = run_task_attempt(loaded.agent_view, loaded.evaluator_view, agent)
            results[(agent.name, task_id)] = result
            runs.append(_run_facts(agent.name, task_id, loaded.evaluator_view.category, result))
    expect(all(r.evaluation_status in ("pass", "fail", "evaluation_error") for r in runs),
           "every planned run ended with a score or an explicit evaluation error")

    lying = results[("lying_broken", "refund_eligible_001")]
    by_name = {s.evaluator_name: s for s in lying.scores}
    response = by_name["response"]
    print(f"     lying agent said: {lying.agent_result.final_response!r}")
    expect(response.reason_code == "FALSE_SUCCESS_CLAIM",
           f"response evaluator caught the false success claim ({response.reason_code})")
    proof = response.evidence.get("proof", {})
    expect(proof.get("applied_refund_effects") == [] and proof.get("final_response_sequence") is not None,
           f"proof: no applied issue_refund effect before final response event #{proof.get('final_response_sequence')}")
    expect(by_name["outcome"].status == "fail", f"outcome evaluator saw no refund in the end state "
                                                f"({by_name['outcome'].reason_code})")
    expect(results[("honest_scripted", "refund_eligible_001")].success is True,
           "the honest agent's refund passes every success evaluator")

    print("\n2. Scorecards (rates are numerator/denominator; tool metrics are call-level sums)")
    for config in ("honest_scripted", "lying_broken"):
        card = scorecard([r for r in runs if r.agent_config_id == config])
        tools = card["tool_metrics"]
        line = (f"   {config:16} verified {card['verified_success_rate']['numerator']}/"
                f"{card['verified_success_rate']['denominator']}  unsafe attempts "
                f"{card['unsafe_attempt_rate']['numerator']}/{card['unsafe_attempt_rate']['denominator']}")
        for name in ("tool_schema_validity", "tool_call_precision", "required_call_recall"):
            t = tools[name]
            line += f"  {name.split('_')[-1]} {t['numerator']}/{t['denominator']}"
        print(line + f"  cost {card['usage']['cost_usd']}")
    no_tools = scorecard([r for r in runs if r.task_id == "no_tools_thanks_001"])["tool_metrics"]
    expect(no_tools["required_call_recall"]["applicability"] == "not_applicable",
           "tool metrics are not_applicable (not 0%) on the task that needs no tools")

    print(f"\n3. BFCL {SMOKE_BFCL_ENTRIES}-entry scorer self-check on the pinned simple_python data (no model called)")
    pin = bfcl.load_pin()
    entries = bfcl.select_subset(bfcl.load_category(pin), SMOKE_BFCL_ENTRIES)
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=pin)
    with tempfile.TemporaryDirectory() as tmp:
        raw, manifest = bfcl.write_artifacts(run, Path(tmp))
        expect(raw.is_file() and manifest["subset"]["is_subset"],
               f"raw outputs written and labeled subset {manifest['subset']['selected']}/"
               f"{manifest['subset']['category_task_count']}")
    accuracy = run.metrics["accuracy"]
    expect(accuracy["numerator"] == accuracy["denominator"] == SMOKE_BFCL_ENTRIES,
           f"ground-truth replay scores {accuracy['numerator']}/{accuracy['denominator']} "
           f"(BFCL {pin['release']}@{pin['commit'][:12]})")

    elapsed = time.monotonic() - started
    print(f"\n{'SMOKE PASSED' if not failures else 'SMOKE FAILED'} in {elapsed:.1f}s; "
          "no database, no network, no API key, $0 spent")
    return 1 if failures else 0


# --- database commands ---------------------------------------------------------------------


def _sm() -> Any:
    from app.db.session import make_engine, make_sessionmaker

    return make_sessionmaker(make_engine())


def _seed(sm: Any, *, with_models: bool = True) -> dict[str, Any]:
    from app.agents.registry import DEFAULT_ADAPTERS
    from app.services.catalog import AgentConfigIn, import_dataset, register_agent_config, sync_price_versions

    with sm.begin() as session:
        sync_price_versions(session)
        version, created = import_dataset(session, "support")
        configs = {}
        for spec in (*SCRIPTED_CONFIGS, *(_model_configs() if with_models else [])):
            config, _ = register_agent_config(session, AgentConfigIn(**spec), known_adapters=DEFAULT_ADAPTERS)
            configs[config.name] = str(config.id)
        return {
            "dataset_version_id": str(version.id),
            "dataset_content_hash": version.content_hash,
            "dataset_created": created,
            "agent_configs": configs,
        }


def seed(_: argparse.Namespace) -> int:
    """Idempotent: same dataset content and same configurations map to the existing rows."""
    _print(_seed(_sm()))
    return 0


def _run_experiment(sm: Any, seeded: dict[str, Any], names: Sequence[str], *, split: str, repetitions: int,
                    experiment_name: str, concurrency: int, timeout: float) -> uuid.UUID:
    from app.services.experiments import ExperimentCreate, create_experiment
    from app.worker.worker import Worker

    payload = ExperimentCreate(
        name=experiment_name,
        dataset_version_id=seeded["dataset_version_id"],
        agent_config_ids=[seeded["agent_configs"][n] for n in names],
        split=split,
        repetitions=repetitions,
    )
    experiment, _ = create_experiment(sm, payload, idempotency_key=uuid.uuid4().hex)
    planned = len(experiment.manifest["tasks"]) * len(names) * repetitions
    print(f"experiment {experiment.id}: {planned} planned runs ({len(experiment.manifest['tasks'])} {split} tasks "
          f"x {len(names)} configs x {repetitions} repetitions)", file=sys.stderr)
    Worker(sm, concurrency=concurrency, heartbeat_seconds=2.0, poll_seconds=0.1).run_until_idle(timeout=timeout)
    return experiment.id


def _summary(sm: Any, experiment_id: uuid.UUID) -> dict[str, Any]:
    from app.services.reports import export_experiment

    with sm() as session:
        report = export_experiment(session, experiment_id)
    return {
        "experiment_id": str(experiment_id),
        "dashboard": f"http://127.0.0.1:8000/ui/experiments/{experiment_id}",
        "scorecards": {
            c["agent_config_name"]: {
                k: c["scorecard"][k] for k in ("planned_runs", "verified_success_rate", "strict_success_rate",
                                               "evaluation_coverage", "unsafe_attempt_rate", "tool_metrics", "usage")
            }
            for c in report["scorecards"]
        },
    }


def demo(args: argparse.Namespace) -> int:
    """Scripted honest-vs-lying experiment in PostgreSQL plus a stored BFCL self-check, for the dashboard."""
    from app.benchmarks import bfcl

    sm = _sm()
    seeded = _seed(sm)
    experiment_id = _run_experiment(sm, seeded, ["scripted-honest", "scripted-lying"], split=args.split,
                                    repetitions=args.repetitions, experiment_name="scripted demo: honest vs lying",
                                    concurrency=2, timeout=600)
    pin = bfcl.load_pin()
    entries = bfcl.select_subset(bfcl.load_category(pin), args.bfcl_limit)
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=pin)
    _, manifest = bfcl.write_artifacts(run)
    with sm.begin() as session:
        stored = bfcl.save_result(session, run, manifest)
        bfcl_id = stored.id
    _print({**_summary(sm, experiment_id), "bfcl_self_check": str(bfcl_id),
            "benchmarks": "http://127.0.0.1:8000/ui/benchmarks"})
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Real-model experiment (paid). Prompt v1 vs v2 on one model by default: PRD demo 2."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. `make evaluate` calls a paid model; everything else runs without it.",
              file=sys.stderr)
        return 2
    sm = _sm()
    seeded = _seed(sm)
    unknown = [n for n in args.configs if n not in seeded["agent_configs"]]
    if unknown:
        print(f"unknown configurations {unknown}; known: {sorted(seeded['agent_configs'])}", file=sys.stderr)
        return 2
    experiment_id = _run_experiment(sm, seeded, args.configs, split=args.split, repetitions=args.repetitions,
                                    experiment_name=f"evaluate: {' vs '.join(args.configs)}",
                                    concurrency=args.concurrency, timeout=args.timeout)
    _print(_summary(sm, experiment_id))
    return 0


def report(args: argparse.Namespace) -> int:
    """Write reports/<experiment>.json and .csv (with the manifest) for one experiment, the latest by default."""
    from sqlalchemy import select

    from app.db.models import Experiment
    from app.services.reports import export_experiment, to_csv

    sm = _sm()
    with sm() as session:
        experiment_id = args.experiment or session.scalar(
            select(Experiment.id).order_by(Experiment.created_at.desc()).limit(1)
        )
        if experiment_id is None:
            print("no experiments yet; run `make demo` or `make evaluate` first", file=sys.stderr)
            return 2
        exported = export_experiment(session, experiment_id)
    REPORTS_DIR.mkdir(exist_ok=True)
    json_path, csv_path = REPORTS_DIR / f"{experiment_id}.json", REPORTS_DIR / f"{experiment_id}.csv"
    json_path.write_text(json.dumps(exported, indent=2, default=str) + "\n")
    csv_path.write_text(to_csv(exported))
    _print({"json": str(json_path.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT)),
            **_summary(sm, experiment_id)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="offline end-to-end check: no database, no network, no API key").set_defaults(func=smoke)
    sub.add_parser("seed", help="import the support dataset and register agent configurations").set_defaults(func=seed)

    p = sub.add_parser("demo", help="scripted experiment + BFCL self-check in PostgreSQL (no API key)")
    p.add_argument("--split", default="validation", choices=("dev", "validation", "test"))
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--bfcl-limit", type=int, default=50)
    p.set_defaults(func=demo)

    p = sub.add_parser("evaluate", help="real-model experiment (needs ANTHROPIC_API_KEY; costs money)")
    p.add_argument("--configs", nargs="+", default=["sonnet-5-5_prompt-v1", "sonnet-5-5_prompt-v2"])
    p.add_argument("--split", default="dev", choices=("dev", "validation", "test"))
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.set_defaults(func=evaluate)

    p = sub.add_parser("report", help="export an experiment (latest by default) to reports/ as JSON and CSV")
    p.add_argument("--experiment", type=uuid.UUID, default=None)
    p.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
