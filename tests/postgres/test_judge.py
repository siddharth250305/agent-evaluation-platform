"""The optional judge against a real database: it changes no primary result and its spend stays separate."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.accounting import price_for
from app.db.models import JudgeScore
from app.judges.communication import CommunicationJudge
from app.providers.base import ModelRequest, ModelResponse, ProviderError
from app.providers.fake import known_usage, tool_calls
from app.services.judging import judge_experiment
from app.services.reports import export_experiment
from tests.postgres.helpers import client_for, launch, worker

GOOD = {d: {"score": 4, "reason": "fine"} for d in ("clarity", "completeness", "tone")}


class CountingJudgeProvider:
    """Scores every message 4/4/4, except that the ``fail_on``-th call fails like an outage."""

    name = "scripted"
    model = "claude-sonnet-5-5"

    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        self.calls += 1
        if self.calls == self.fail_on:
            raise ProviderError("HTTP 529: overloaded", retryable=True, status_code=529)
        return tool_calls(("submit_scores", GOOD), usage=known_usage(300, 60))


def _primary(report: dict) -> dict:
    """Everything the deterministic platform reports, minus the export timestamp and the judge block."""
    return {k: v for k, v in report.items() if k not in ("generated_at", "secondary_judges")}


def test_judging_leaves_every_primary_result_unchanged_and_bills_the_judge_separately(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])
    worker(sm).run_until_idle()
    experiment_id = uuid.UUID(experiment["id"])
    with sm() as session:
        before = export_experiment(session, experiment_id)
    assert before["secondary_judges"] == []

    provider = CountingJudgeProvider(fail_on=2)
    judge = CommunicationJudge(provider, price=price_for("claude-sonnet-5-5"))
    counts = judge_experiment(sm, experiment_id, judge)
    eligible = counts["eligible"]
    assert eligible == len(before["runs"])  # every run was graded and produced a message
    assert counts == {"eligible": eligible, "already_judged": 0, "scored": eligible - 1, "evaluation_errors": 1}

    with sm() as session:
        after = export_experiment(session, experiment_id)
        stored = session.scalar(select(func.count()).select_from(JudgeScore))
    assert stored == eligible
    assert _primary(after) == _primary(before)  # scorecards, runs, success, candidate cost: identical

    judges = {j["agent_config_name"]: j for j in after["secondary_judges"]}
    assert set(judges) == {"honest", "lying"}
    total_calls = sum(j["usage"]["judge_calls"] for j in judges.values())
    assert total_calls == eligible
    errored = next(j for j in judges.values() if j["evaluation_errors"])
    assert errored["error_reasons"] == {"JUDGE_PROVIDER_ERROR": 1}
    assert errored["usage"]["cost_usd"] is None  # the failed call's usage is unknown
    clean = next(j for j in judges.values() if not j["evaluation_errors"])
    assert clean["usage"]["cost_usd"] is not None and clean["dimensions"]["tone"]["mean"] == 4.0
    for card in after["scorecards"]:
        assert card["scorecard"]["usage"]["cost_usd"] == next(
            c for c in before["scorecards"] if c["agent_config_id"] == card["agent_config_id"]
        )["scorecard"]["usage"]["cost_usd"]

    again = judge_experiment(sm, experiment_id, judge)
    assert again["already_judged"] == eligible and again["scored"] == again["evaluation_errors"] == 0
    assert provider.calls == eligible  # nothing was judged twice


def test_the_dashboard_shows_the_judge_as_secondary_and_uncalibrated(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    worker(sm).run_until_idle()
    experiment_id = uuid.UUID(experiment["id"])
    assert 'id="judge"' not in client.get(f"/ui/experiments/{experiment_id}").text

    judge_experiment(sm, experiment_id, CommunicationJudge(CountingJudgeProvider(), price=price_for("claude-sonnet-5-5")))
    html = client.get(f"/ui/experiments/{experiment_id}").text
    assert 'id="judge"' in html and "uncalibrated" in html and "not</strong> included in the candidate" in html

    run_id = client.get(f"/experiments/{experiment_id}/runs").json()["items"][0]["id"]
    run_html = client.get(f"/ui/runs/{run_id}").text
    assert 'id="judge-scores"' in run_html and "does not affect success" in run_html


def test_concurrent_judges_do_not_make_two_paid_calls_for_one_verdict(sm) -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from sqlalchemy import text

    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    worker(sm).run_until_idle()
    entered, release = threading.Event(), threading.Event()

    class BlockingProvider(CountingJudgeProvider):
        def complete(self, request, *, timeout_seconds):
            result = super().complete(request, timeout_seconds=timeout_seconds)
            entered.set()
            assert release.wait(5)
            return result

    provider = BlockingProvider()
    judge = CommunicationJudge(provider, price=price_for(provider.model))
    experiment_id = uuid.UUID(experiment["id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(judge_experiment, sm, experiment_id, judge, limit=1)
        assert entered.wait(5)
        second = pool.submit(judge_experiment, sm, experiment_id, judge, limit=1)
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with sm() as session:
                    waiting = session.scalar(text("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND NOT granted"))
                if waiting or provider.calls > 1:
                    break
                time.sleep(0.01)
            assert waiting or provider.calls > 1, "second invocation did not reach contention"
        finally:
            release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert provider.calls == 1
    assert sum(r["scored"] for r in results) == 1
    with sm() as session:
        assert session.scalar(select(func.count()).select_from(JudgeScore)) == 1


def test_nul_in_judge_reason_and_error_does_not_lose_usage(sm) -> None:
    from app.providers.fake import ScriptedProvider

    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    worker(sm).run_until_idle()
    provider = ScriptedProvider([
        tool_calls(("submit_scores", {d: {"score": 4, "reason": "fine\x00text"} for d in GOOD}),
                   usage=known_usage(70, 20)),
        ProviderError("bad\x00response", retryable=False),
    ], model="claude-sonnet-5-5")
    counts = judge_experiment(sm, uuid.UUID(experiment["id"]), CommunicationJudge(provider, price=price_for(provider.model)), limit=2)
    assert counts["scored"] == counts["evaluation_errors"] == 1
    with sm() as session:
        rows = session.scalars(select(JudgeScore).order_by(JudgeScore.id)).all()
        assert rows[0].scores["tone"]["reason"] == "fine\ufffdtext"
        assert rows[0].input_tokens == 70 and rows[0].cost_usd > 0
        assert rows[1].error == "bad\ufffdresponse"
        assert rows[1].scores is None and rows[1].cost_usd is None


def test_nonpositive_judge_limit_is_rejected_before_spend(sm) -> None:
    import pytest

    provider = CountingJudgeProvider()
    judge = CommunicationJudge(provider, price=price_for(provider.model))
    for limit in (0, -1):
        with pytest.raises(ValueError, match="limit"):
            judge_experiment(sm, uuid.uuid4(), judge, limit=limit)
    assert provider.calls == 0
