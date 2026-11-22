"""POST /comparisons against real runs: regressions, denominators, cost honesty, compatibility. Scripted agents only."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Attempt, Experiment, Run, TaskVersion
from tests.postgres.helpers import client_for, experiment_payload, import_support, launch, register, worker


@pytest.fixture
def finished(sm: sessionmaker[Session]):
    """One experiment, honest and lying configurations on the validation split, fully run."""
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])
    worker(sm).run_until_idle()
    ids = {c["name"]: c["id"] for c in experiment["manifest"]["agent_configs"]}
    return client, experiment, ids


def _side(experiment: dict, config_id: str | None = None) -> dict:
    side = {"experiment_id": experiment["id"]}
    if config_id is not None:
        side["agent_config_id"] = config_id
    return side


def _compare(client, baseline: dict, candidate: dict, **extra):
    return client.post("/comparisons", json={"baseline": baseline, "candidate": candidate, **extra})


def _successes(sm: sessionmaker[Session], experiment_id: str, config_id: str) -> dict[str, list[bool | None]]:
    """Ground truth straight from the runs table: task_id -> success per repetition."""
    out: dict[str, list[bool | None]] = defaultdict(list)
    with sm() as session:
        for task_id, success in session.execute(
            select(TaskVersion.task_id, Run.success)
            .join(TaskVersion, TaskVersion.id == Run.task_version_id)
            .where(Run.experiment_id == uuid.UUID(experiment_id), Run.agent_config_id == uuid.UUID(config_id))
        ):
            out[task_id].append(success)
    return out


def test_regressions_are_exactly_the_tasks_the_baseline_passed_and_the_candidate_failed(sm, finished) -> None:
    client, experiment, ids = finished
    honest = _successes(sm, experiment["id"], ids["honest"])
    lying = _successes(sm, experiment["id"], ids["lying"])
    expected = sorted(t for t in honest if all(s is True for s in honest[t]) and any(s is False for s in lying[t]))
    assert expected, "the scripted pair must produce at least one regression for this test to mean anything"

    response = _compare(client, _side(experiment, ids["honest"]), _side(experiment, ids["lying"]))

    assert response.status_code == 200, response.text
    body = response.json()
    summary = body["summary"]
    assert summary["regressed_tasks"] == expected and summary["regressions"] == len(expected)
    assert summary["wins"] == 0 and summary["improvements"] == 0
    assert summary["wins"] + summary["losses"] + summary["ties"] == summary["tasks_compared"] == len(honest)
    regressed_rows = [t for t in body["tasks"] if t["regression"]]
    assert all(t["result"] == "loss" and t["baseline"]["outcome"] == "passed_all" for t in regressed_rows)

    reverse = _compare(client, _side(experiment, ids["lying"]), _side(experiment, ids["honest"])).json()["summary"]
    assert reverse["improved_tasks"] == expected and reverse["regressions"] == 0
    assert (reverse["wins"], reverse["losses"]) == (summary["losses"], summary["wins"])


def test_metrics_carry_explicit_denominators_that_match_the_runs(sm, finished) -> None:
    client, experiment, ids = finished
    honest = _successes(sm, experiment["id"], ids["honest"])
    lying = _successes(sm, experiment["id"], ids["lying"])
    body = _compare(client, _side(experiment, ids["honest"]), _side(experiment, ids["lying"])).json()

    planned = sum(len(v) for v in honest.values())
    verified = body["metrics"]["verified_success_rate"]
    assert verified["baseline"]["numerator"] == sum(s is True for v in honest.values() for s in v)
    assert verified["baseline"]["denominator"] == sum(s is not None for v in honest.values() for s in v)
    assert verified["candidate"]["numerator"] == sum(s is True for v in lying.values() for s in v)
    strict = body["metrics"]["strict_success_rate"]
    assert strict["baseline"]["denominator"] == strict["candidate"]["denominator"] == planned
    assert strict["delta"] == pytest.approx(strict["candidate"]["value"] - strict["baseline"]["value"])
    assert body["metrics"]["evaluation_coverage"]["baseline"] == {"numerator": planned, "denominator": planned, "value": 1.0}
    assert body["metrics"]["passed_all_repetitions"]["baseline"]["denominator"] == len(honest)
    assert body["metrics"]["unsafe_attempt_rate"]["baseline"]["denominator"] == planned  # every run policy-scored
    ci = strict["delta_ci"]
    assert ci["unit"] == "task" and ci["clusters"] == len(honest) and ci["low"] <= strict["delta"] <= ci["high"]
    assert body["latency_ms"]["queue"]["baseline"]["n"] == planned
    assert body["latency_ms"]["execution"]["candidate"]["n"] == planned
    assert body["cost"]["baseline"]["cost_usd"] == "0" and body["cost"]["delta_cost_usd"] == "0"
    assert body["compatibility"]["split"] == "validation" and body["compatibility"]["limits"] is not None
    assert body["config_differences"]["adapter"] == {"baseline": "honest_scripted", "candidate": "lying_broken"}
    assert "accuracy" not in body["metrics"]


def test_the_same_request_gives_the_same_answer(finished) -> None:
    client, experiment, ids = finished
    args = (_side(experiment, ids["honest"]), _side(experiment, ids["lying"]))
    first = _compare(client, *args, seed=3, bootstrap_resamples=300).json()
    again = _compare(client, *args, seed=3, bootstrap_resamples=300).json()
    assert first == again


def test_unknown_usage_stays_unknown_and_retries_are_charged(sm, finished) -> None:
    client, experiment, ids = finished
    with sm.begin() as session:
        lying_attempt = session.scalars(
            select(Attempt).join(Run, Run.id == Attempt.run_id)
            .where(Run.agent_config_id == uuid.UUID(ids["lying"])).limit(1)
        ).one()
        lying_attempt.cost_usd = None  # e.g. a provider that returned no usage
        honest_run = session.scalars(
            select(Run).where(Run.agent_config_id == uuid.UUID(ids["honest"])).limit(1)
        ).one()
        # Make the scored attempt #2 and add an earlier, paid infrastructure failure as #1.
        session.execute(update(Attempt).where(Attempt.run_id == honest_run.id).values(attempt_number=2))
        now = datetime.now(UTC)
        session.add(Attempt(
            id=uuid.uuid4(), run_id=honest_run.id, attempt_number=1, status="infrastructure_error",
            environment_id=f"env-{uuid.uuid4().hex}", error_code="PROVIDER_ERROR",
            cost_usd=Decimal("0.25"), input_tokens=100, output_tokens=20, duration_ms=50.0,
            started_at=now, finished_at=now,
        ))

    cost = _compare(client, _side(experiment, ids["honest"]), _side(experiment, ids["lying"])).json()["cost"]

    assert cost["baseline"]["cost_usd"] == "0.25" and cost["baseline"]["retry_attempts"] == 1
    assert cost["candidate"]["cost_usd"] is None and cost["candidate"]["attempts_with_unknown_cost"] == 1
    assert cost["candidate"]["cost_per_successful_run_usd"]["value"] is None
    assert cost["delta_cost_usd"] is None  # never "0" or "-0.25"


def test_two_single_config_experiments_compare_without_naming_configs(sm) -> None:
    client = client_for(sm)
    first = launch(client, [("honest-a", "honest_scripted")], repetitions=2)
    second = launch(client, [("honest-b", "honest_scripted")], repetitions=2)
    worker(sm).run_until_idle()

    response = _compare(client, _side(first), _side(second))

    assert response.status_code == 200, response.text
    body = response.json()
    tasks = len(first["manifest"]["tasks"])
    assert body["summary"]["tasks_compared"] == tasks and body["summary"]["ties"] == tasks
    assert body["summary"]["regressions"] == 0
    assert body["metrics"]["strict_success_rate"]["baseline"]["denominator"] == tasks * 2
    assert body["metrics"]["passed_all_repetitions"]["baseline"]["denominator"] == tasks
    assert all(t["baseline"]["planned"] == 2 for t in body["tasks"])
    assert body["metrics"]["strict_success_rate"]["delta"] == 0.0


def test_unfinished_experiments_are_compared_with_a_warning(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])
    ids = {c["name"]: c["id"] for c in experiment["manifest"]["agent_configs"]}

    body = _compare(client, _side(experiment, ids["honest"]), _side(experiment, ids["lying"])).json()

    assert len(body["warnings"]) == 2 and not body["baseline"]["complete"]
    assert body["metrics"]["strict_success_rate"]["baseline"]["numerator"] == 0
    assert body["metrics"]["verified_success_rate"]["baseline"]["value"] is None  # 0/0 is unknown, not 0%
    assert body["summary"]["regressions"] == 0


def _two_experiments(client, *, first: dict, second: dict) -> tuple[dict, dict]:
    version = import_support(client)
    out = []
    for spec in (first, second):
        config = register(client, spec["name"], "honest_scripted", **spec.get("extra", {}))
        response = client.post(
            "/experiments",
            json=experiment_payload(version["id"], [config["id"]], split=spec.get("split", "validation")),
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        assert response.status_code == 201, response.text
        out.append(response.json())
    return out[0], out[1]


def test_different_splits_are_not_comparable(sm) -> None:
    client = client_for(sm)
    a, b = _two_experiments(client, first={"name": "a"}, second={"name": "b", "split": "test"})
    response = _compare(client, _side(a), _side(b))
    assert response.status_code == 422 and "split differs" in response.json()["detail"]


def test_different_limits_are_not_comparable(sm) -> None:
    client = client_for(sm)
    a, b = _two_experiments(client, first={"name": "a"}, second={"name": "b", "extra": {"limits": {"max_model_turns": 3}}})
    response = _compare(client, _side(a), _side(b))
    assert response.status_code == 422 and "limits differ" in response.json()["detail"]


@pytest.mark.parametrize(("path", "value", "message"), [
    ("environment", {"name": "support_environment", "version": "9.9.9"}, "environment version differs"),
    ("evaluators", [{"name": "outcome", "version": "0.0.1"}], "evaluator versions differ"),
])
def test_different_environment_or_evaluator_versions_are_not_comparable(sm, path, value, message) -> None:
    client = client_for(sm)
    a, b = _two_experiments(client, first={"name": "a"}, second={"name": "b"})
    with sm.begin() as session:
        experiment = session.get(Experiment, uuid.UUID(b["id"]))
        experiment.manifest = {**experiment.manifest, path: value}
    response = _compare(client, _side(a), _side(b))
    assert response.status_code == 422 and message in response.json()["detail"]


def test_bad_comparison_requests(finished) -> None:
    client, experiment, ids = finished
    multi = _compare(client, _side(experiment), _side(experiment, ids["lying"]))
    assert multi.status_code == 422 and "baseline.agent_config_id" in multi.json()["detail"]
    same = _compare(client, _side(experiment, ids["honest"]), _side(experiment, ids["honest"]))
    assert same.status_code == 422 and "same configuration" in same.json()["detail"]
    foreign = _compare(client, _side(experiment, str(uuid.uuid4())), _side(experiment, ids["lying"]))
    assert foreign.status_code == 422 and "not part of" in foreign.json()["detail"]
    missing = _compare(client, {"experiment_id": str(uuid.uuid4())}, _side(experiment, ids["lying"]))
    assert missing.status_code == 404
    sides = (_side(experiment, ids["honest"]), _side(experiment, ids["lying"]))
    assert _compare(client, *sides, confidence=1.5).status_code == 422
    assert _compare(client, *sides, bootstrap_resamples=-1).status_code == 422
    assert _compare(client, *sides, unexpected=True).status_code == 422


def test_compatibility_error_lists_all_differences_together(sm) -> None:
    client = client_for(sm)
    a, b = _two_experiments(client, first={"name": "a"}, second={
        "name": "b", "split": "test", "extra": {"limits": {"max_model_turns": 3}},
    })
    with sm.begin() as session:
        experiment = session.get(Experiment, uuid.UUID(b["id"]))
        experiment.manifest = {**experiment.manifest,
                               "environment": {"version": "different"},
                               "evaluators": [{"name": "outcome", "version": "different"}]}
    response = _compare(client, _side(a), _side(b))
    assert response.status_code == 422
    for difference in ("split", "limits", "environment version", "evaluator versions"):
        assert f"{difference} differs" in response.json()["detail"]


def test_trajectory_count_version_boundary_is_not_comparable(sm) -> None:
    client = client_for(sm)
    a, b = _two_experiments(client, first={"name": "legacy"}, second={"name": "counts"})
    with sm.begin() as session:
        experiment = session.get(Experiment, uuid.UUID(a["id"]))
        experiment.manifest = {**experiment.manifest, "evaluators": [
            {**e, "version": "1.2.0"} if e["name"] == "trajectory" else e
            for e in experiment.manifest["evaluators"]
        ]}
    response = _compare(client, _side(a), _side(b))
    assert response.status_code == 422
    assert "trajectory" in response.json()["detail"]
    assert "1.2.0" in response.json()["detail"] and "1.3.0" in response.json()["detail"]
