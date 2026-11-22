"""Developer commands: `make smoke` passes offline, paid commands refuse to start without a key."""

from __future__ import annotations

import argparse
import os

import pytest

from app import cli
from app.services.catalog import AgentConfigIn
from app.services.hashing import sha256_text


def test_smoke_passes_offline_and_says_what_it_checked(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert "SMOKE PASSED" in out and "FAIL" not in out
    assert "FALSE_SUCCESS_CLAIM" in out and "subset 25/400" in out


def test_evaluate_refuses_to_spend_without_an_api_key(monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = argparse.Namespace(configs=["sonnet-5-5_prompt-v1"], split="dev", repetitions=1, concurrency=1, timeout=1,
                              tasks=None, limit=None, dry_run=False)
    assert cli.evaluate(args) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


SPLIT = ["a_001", "a_002", "b_001", "c_001"]


@pytest.mark.parametrize(("tasks", "limit", "expected"), [
    (None, None, None),  # whole split
    (None, 2, ["a_001", "a_002"]),  # first N in task-id order
    (["c_001", "a_001"], None, ["a_001", "c_001"]),  # explicit ids, split order
    (["c_001", "b_001", "a_002"], 1, ["a_002"]),
])
def test_pilot_task_selection(tasks, limit, expected) -> None:
    assert cli._select_tasks(SPLIT, tasks, limit) == expected


def test_pilot_task_selection_rejects_unknown_ids_and_empty_limits() -> None:
    with pytest.raises(ValueError, match="z_001"):
        cli._select_tasks(SPLIT, ["a_001", "z_001"], None)
    with pytest.raises(ValueError, match="--limit"):
        cli._select_tasks(SPLIT, None, 0)


def test_evaluate_parses_pilot_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "evaluate", lambda args: seen.append(args) or 0)
    assert cli.main(["evaluate", "--tasks", "a_001", "b_001", "--limit", "1", "--dry-run"]) == 0
    assert seen[0].tasks == ["a_001", "b_001"] and seen[0].limit == 1 and seen[0].dry_run is True


def test_seeded_model_configurations_are_three_models_by_two_prompts() -> None:
    configs = [AgentConfigIn(**spec) for spec in cli._model_configs()]
    assert len(configs) == 6
    assert len({(c.model, sha256_text(c.prompt)) for c in configs}) == 6
    assert {c.name for c in configs} == {
        "sonnet-5-5_prompt-v1", "sonnet-5-5_prompt-v2", "opus-5-5_prompt-v1", "opus-5-5_prompt-v2",
        "gpt-4.1-mini_prompt-v1", "gpt-4.1-mini_prompt-v2",
    }
    assert all(c.limits["max_cost_usd"] == "0.5" for c in configs)  # a per-attempt budget is always set
    by_name = {c.name: c for c in configs}
    assert by_name["gpt-4.1-mini_prompt-v1"].settings == {}  # no reasoning_effort for a non-reasoning model
    assert by_name["sonnet-5-5_prompt-v1"].settings == {"effort": "medium"}  # unchanged, so existing hashes hold


def test_evaluate_asks_for_the_key_of_the_selected_configs_vendor(monkeypatch: pytest.MonkeyPatch,
                                                                  capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = argparse.Namespace(configs=["gpt-4.1-mini_prompt-v1"], split="dev", repetitions=1, concurrency=1,
                              timeout=1, tasks=None, limit=1, dry_run=False)
    assert cli.evaluate(args) == 2  # refused before touching the database or any model
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "ANTHROPIC_API_KEY" not in err


def test_dry_run_uses_no_database_or_provider_and_reports_retry_ceiling(monkeypatch, capsys) -> None:
    import json
    from app.providers import factory

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not create a DB session or provider")
    monkeypatch.setattr(cli, "_sm", forbidden)
    monkeypatch.setattr(factory, "make_provider", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-SHOULD-NOT-LEAK")
    assert cli.main(["evaluate", "--configs", "gpt-4.1-mini_prompt-v1", "--limit", "1", "--dry-run"]) == 0
    output = capsys.readouterr().out
    plan = json.loads(output)
    assert plan["planned_runs"] == 1 and plan["max_spend_usd"] == "1.50"
    assert plan["max_first_attempt_spend_usd"] == "0.50"
    assert "sk-offline" not in output


def test_make_dry_run_skips_migration() -> None:
    import subprocess

    result = subprocess.run(["make", "-n", "evaluate", "EVAL_ARGS=--dry-run --limit 1"], capture_output=True, text=True, check=True)
    assert "app.db.migrate" not in result.stdout
    assert "app.cli evaluate" in result.stdout


def test_live_queue_workers_are_this_hosts_running_app_worker_processes_only() -> None:
    ids = ["mac.local:53566:df1065", "mac.local:51342:b6a5ee", "mac.local:53566:aaaaaa", "other:53566:x",
           "mac.local:77:dead", "mac.local:88:reused", "malformed"]
    commands = {53566: "Python -m app.worker", 51342: "Python -m app.cli evaluate", 88: "/usr/bin/vim"}
    assert cli._live_queue_workers(ids, host="mac.local", own_pid=51342,
                                   command_of=lambda pid: commands.get(pid, "")) == [53566]
    assert cli._process_command(os.getpid()) != "" and cli._process_command(2**22 + 1) == ""


def test_evaluate_refuses_to_spend_while_a_background_worker_polls_the_queue(monkeypatch, capsys) -> None:
    # Experiments 6d50a610 and c8c1a7af: an outdated `python -m app.worker` leased 18 of 20 paid pilot
    # runs and failed each one ADAPTER_UNAVAILABLE before the CLI's own worker could take them.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-SHOULD-NOT-LEAK")
    monkeypatch.setattr(cli, "_sm", lambda: object())
    monkeypatch.setattr(cli, "_seed", lambda sm: {"dataset_version_id": "d", "agent_configs": {
        "gpt-4.1-mini_prompt-v1": "c1", "gpt-4.1-mini_prompt-v2": "c2"}})
    monkeypatch.setattr(cli, "_split_task_ids", lambda sm, dvid, split: ["approval_expired_001"])
    monkeypatch.setattr(cli, "_background_workers", lambda sm: [53566])

    def forbidden(*args, **kwargs):
        raise AssertionError("no experiment may be created while a background worker can lease its runs")
    monkeypatch.setattr(cli, "_run_experiment", forbidden)
    assert cli.main(["evaluate", "--configs", "gpt-4.1-mini_prompt-v1", "gpt-4.1-mini_prompt-v2",
                     "--split", "dev", "--limit", "1"]) == 2
    err = capsys.readouterr().err
    assert "53566" in err and "kill 53566" in err and "ADAPTER_UNAVAILABLE" in err and "sk-offline" not in err
