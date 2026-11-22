"""Developer commands: `make smoke` passes offline, paid commands refuse to start without a key."""

from __future__ import annotations

import argparse

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
    args = argparse.Namespace(configs=["sonnet-5-5_prompt-v1"], split="dev", repetitions=1, concurrency=1, timeout=1)
    assert cli.evaluate(args) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_seeded_model_configurations_are_two_models_by_two_prompts() -> None:
    configs = [AgentConfigIn(**spec) for spec in cli._model_configs()]
    assert len(configs) == 4
    assert len({(c.model, sha256_text(c.prompt)) for c in configs}) == 4
    assert {c.name for c in configs} == {
        "sonnet-5-5_prompt-v1", "sonnet-5-5_prompt-v2", "opus-5-5_prompt-v1", "opus-5-5_prompt-v2",
    }
    assert all(c.limits["max_cost_usd"] == "0.5" for c in configs)  # a per-attempt budget is always set
