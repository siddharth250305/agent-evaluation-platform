"""BFCL simple_python: pinned import, scorer port, runner, artifacts. Offline: no model is called."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from app.benchmarks import bfcl
from app.providers.anthropic_http import build_request_body
from app.providers.base import ModelRequest, ModelResponse, ProviderError, ToolCall
from app.providers.fake import ScriptedProvider, final_text

PIN = bfcl.load_pin()
ENTRIES = bfcl.load_category(PIN)
BY_ID = {e.id: e for e in ENTRIES}

TRIANGLE = BY_ID["simple_python_0"]  # calculate_triangle_area(base: int, height: int, unit: str = optional)
FACTORIAL = BY_ID["simple_python_1"]  # math.factorial(number: int)


def check(entry: bfcl.BfclEntry, *calls: dict) -> dict:
    return bfcl.check_simple(entry.functions[0], list(calls), entry.ground_truth[0])


# --- pinned import ---------------------------------------------------------------------


def test_the_pinned_category_loads_whole_and_in_file_order() -> None:
    assert len(ENTRIES) == PIN["category_task_count"] == 400
    assert ENTRIES[0].id == "simple_python_0" and ENTRIES[-1].id == "simple_python_399"
    assert PIN["commit"] == "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"


def test_a_file_that_does_not_match_its_pinned_hash_is_refused(tmp_path: Path) -> None:
    shutil.copytree(bfcl.BFCL_DIR / "data", tmp_path / "data")
    answers = tmp_path / PIN["files"]["answers"]["local_path"]
    answers.write_text(answers.read_text().replace('"base": [10]', '"base": [11]', 1))
    with pytest.raises(bfcl.BenchmarkDataError, match="sha256"):
        bfcl.load_category(PIN, tmp_path)


def test_a_missing_file_is_refused_with_how_to_fetch_it(tmp_path: Path) -> None:
    with pytest.raises(bfcl.BenchmarkDataError, match="bfcl-fetch"):
        bfcl.load_category(PIN, tmp_path)


def test_subset_selection_is_the_first_n_entries() -> None:
    assert [e.id for e in bfcl.select_subset(ENTRIES, 3)] == ["simple_python_0", "simple_python_1", "simple_python_2"]
    assert len(bfcl.select_subset(ENTRIES, None)) == 400
    with pytest.raises(ValueError):
        bfcl.select_subset(ENTRIES, 0)


# --- scorer port -----------------------------------------------------------------------


def test_scorer_accepts_every_ground_truth_answer_in_the_category() -> None:
    """Replaying ground truth through the full pipeline (tool conversion, decoding, scoring) must score 400/400."""
    run = bfcl.run_entries(ENTRIES, bfcl.GroundTruthReplayProvider(ENTRIES), kind="scorer_self_check", pin=PIN)
    assert run.metrics["accuracy"] == {"numerator": 400, "denominator": 400, "value": 1.0}
    assert run.metrics["evaluation_errors"] == 0


@pytest.mark.parametrize(("args", "error_type"), [
    ({"base": 10, "height": 5}, None),  # the optional unit may be omitted
    ({"base": 10, "height": 5, "unit": " Units"}, None),  # strings compare case- and space-insensitively
    ({"base": "10", "height": 5}, "type_error:simple"),
    ({"base": True, "height": 5}, "type_error:simple"),  # a bool is not an integer
    ({"base": 10.0, "height": 5}, "type_error:simple"),  # no float -> int conversion
    ({"base": 10}, "simple_function_checker:missing_required"),
    ({"base": 10, "height": 5, "color": "red"}, "simple_function_checker:unexpected_param"),
    ({"base": 11, "height": 5}, "value_error:others"),
    ({"base": 10, "height": 5, "unit": "cm"}, "value_error:string"),
])
def test_scorer_verdicts_follow_the_official_simple_function_checker(args: dict, error_type: str | None) -> None:
    verdict = check(TRIANGLE, {"calculate_triangle_area": args})
    assert verdict["valid"] is (error_type is None)
    if error_type:
        assert verdict["error_type"] == error_type


def test_scorer_rejects_wrong_count_and_wrong_name() -> None:
    call = {"calculate_triangle_area": {"base": 10, "height": 5}}
    assert check(TRIANGLE, call, call)["error_type"] == "simple_function_checker:wrong_count"
    assert check(TRIANGLE)["error_type"] == "simple_function_checker:wrong_count"
    assert check(TRIANGLE, {"triangle_area": {"base": 10, "height": 5}})["error_type"] == \
        "simple_function_checker:wrong_func_name"


def test_dotted_function_names_are_scored_under_their_wire_names() -> None:
    assert check(FACTORIAL, {"math_factorial": {"number": 5}})["valid"]
    assert not check(FACTORIAL, {"math.factorial": {"number": 5}})["valid"]


def test_float_parameters_accept_integers_and_required_optional_answers_are_enforced() -> None:
    function = {"name": "f", "parameters": {"type": "dict", "required": ["x"], "properties": {
        "x": {"type": "float", "description": ""}, "y": {"type": "integer", "description": ""}}}}
    assert bfcl.check_simple(function, [{"f": {"x": 2}}], {"f": {"x": [2.0], "y": ["", 3]}})["valid"]
    missing = bfcl.check_simple(function, [{"f": {"x": 2.0}}], {"f": {"x": [2.0], "y": [3]}})
    assert missing["error_type"] == "simple_function_checker:missing_optional"


def test_scorer_never_evaluates_model_output() -> None:
    tree = ast.parse(Path(bfcl.__file__).read_text())
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {"eval", "exec", "compile", "__import__"} & called
    assert "literal_eval" not in attrs


# --- request building ------------------------------------------------------------------


def test_tool_specs_follow_bfcl_anthropic_conversion_without_mutating_the_data() -> None:
    original = json.dumps(FACTORIAL.functions, sort_keys=True)
    spec = bfcl.tool_specs(FACTORIAL.functions)[0]
    assert spec.name == "math_factorial"
    assert spec.description.endswith("Note that the provided function is in Python 3 syntax.")
    assert spec.input_schema["type"] == "object"
    assert json.dumps(FACTORIAL.functions, sort_keys=True) == original

    float_entry = next(e for e in ENTRIES if any(p.get("type") == "float" for p in
                                                  e.functions[0]["parameters"]["properties"].values()))
    props = bfcl.tool_specs(float_entry.functions)[0].input_schema["properties"]
    floats = [p for p in props.values() if p.get("format") == "float"]
    assert floats and all(p["type"] == "number" and p["description"].endswith("This is a float type value.")
                          for p in floats)


def test_requests_carry_no_system_prompt_and_the_wire_body_omits_it() -> None:
    request = bfcl.build_request(TRIANGLE, max_output_tokens=256)
    assert request.system == ""
    assert [m.role for m in request.messages] == ["user"]
    assert "system" not in build_request_body("claude-sonnet-5-5", request, effort="medium")


# --- runner ----------------------------------------------------------------------------


def _respond(*calls: tuple[str, object], text: str = "") -> ModelResponse:
    return ModelResponse(text=text, tool_calls=tuple(ToolCall(id=f"c{i}", name=n, arguments=a)
                                                     for i, (n, a) in enumerate(calls)),
                         stop_reason="tool_use" if calls else "end_turn", usage=None, model="scripted-model")


def test_runner_separates_failures_from_infrastructure_errors_and_keeps_unknown_usage_unknown() -> None:
    entries = bfcl.select_subset(ENTRIES, 4)
    provider = ScriptedProvider([
        _respond(("calculate_triangle_area", {"base": 10, "height": 5})),  # pass
        final_text("The factorial of 5 is 120."),  # answered in prose: fail
        _respond(("math_hypot", "x=4, y=5")),  # arguments not an object: fail, and never parsed
        ProviderError("HTTP 529: overloaded", retryable=True, status_code=529),  # infra: neither pass nor fail
    ])
    run = bfcl.run_entries(entries, provider, kind="model_run", pin=PIN, price=None)
    metrics = run.metrics

    assert [e.status for e in run.entries] == ["pass", "fail", "fail", "infrastructure_error"]
    assert metrics["accuracy"] == {"numerator": 1, "denominator": 3, "value": pytest.approx(1 / 3)}
    assert metrics["strict_accuracy"]["denominator"] == 4
    assert metrics["error_types"] == {"decoder:arguments_not_an_object": 1, "decoder:no_function_call": 1}
    assert metrics["usage"]["cost_usd"] is None  # no price and a failed call: unknown, never $0
    assert metrics["usage"]["unknown_usage_calls"] == 4
    assert len(provider.requests) == 4  # one request per entry, no retries


def test_a_scorer_crash_is_an_evaluation_error_not_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> dict:
        raise KeyError("boom")

    monkeypatch.setattr(bfcl, "check_simple", broken)
    entries = bfcl.select_subset(ENTRIES, 1)
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=PIN)
    assert run.entries[0].status == "evaluation_error"
    assert run.metrics["accuracy"]["denominator"] == 0


# --- artifacts, manifest, reproduction ---------------------------------------------------


def test_artifacts_keep_raw_outputs_and_label_the_subset_honestly(tmp_path: Path) -> None:
    entries = bfcl.select_subset(ENTRIES, 5)
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=PIN)
    raw, manifest = bfcl.write_artifacts(run, tmp_path)

    records = [json.loads(line) for line in raw.read_text().splitlines()]
    assert [r["id"] for r in records] == [e.id for e in entries]
    assert all(r["response"]["tool_calls"] and r["request"]["tools"] for r in records)
    assert manifest["raw_outputs"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["subset"] == {"is_subset": True, "selected": 5, "category_task_count": 400,
                                  "method": bfcl.SUBSET_METHOD, "entry_ids": [e.id for e in entries]}
    assert manifest["benchmark"]["commit"] == PIN["commit"]
    assert manifest["scoring"]["official_leaderboard_comparable"] is False
    assert any("subset" in d for d in manifest["deviations_from_official"])
    assert json.loads((raw.parent / "manifest.json").read_text()) == manifest


def test_a_saved_manifest_reproduces_the_same_setup(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert bfcl.main(["--self-check", "--limit", "3", "--no-db", "--artifacts-dir", str(tmp_path / "a")]) == 0
    first = next((tmp_path / "a").rglob("manifest.json"))
    assert bfcl.main(["--from-manifest", str(first), "--no-db", "--artifacts-dir", str(tmp_path / "b")]) == 0
    again = json.loads(next((tmp_path / "b").rglob("manifest.json")).read_text())
    original = json.loads(first.read_text())
    for key in ("benchmark", "subset", "scoring", "result_kind"):
        assert again[key] == original[key]
    assert again["runner"]["settings"] == original["runner"]["settings"]


def test_a_model_run_without_an_api_key_stops_before_any_request(monkeypatch: pytest.MonkeyPatch,
                                                                 tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert bfcl.main(["--limit", "1", "--no-db", "--artifacts-dir", str(tmp_path)]) == 2
    assert not any(tmp_path.iterdir())


def test_build_request_body_still_sends_a_non_empty_system_prompt() -> None:
    request = ModelRequest(system="be brief", messages=(), tools=(), max_output_tokens=10)
    assert build_request_body("m", request, effort="low")["system"] == "be brief"


def test_fetch_writes_a_pinned_file_only_when_its_hash_matches(tmp_path: Path) -> None:
    import httpx

    real = {spec["upstream_path"]: (bfcl.BFCL_DIR / spec["local_path"]).read_bytes() for spec in PIN["files"].values()}

    def serve(tampered: bool) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            assert PIN["commit"] in request.url.path
            path = request.url.path.split(PIN["commit"] + "/", 1)[1]
            return httpx.Response(200, content=real[path] + (b" " if tampered else b""))
        return httpx.Client(transport=httpx.MockTransport(handler))

    written = bfcl.fetch_pinned(PIN, tmp_path, client=serve(False))
    assert len(bfcl.load_category(PIN, tmp_path)) == 400 and len(written) == 2

    with pytest.raises(bfcl.BenchmarkDataError, match="sha256"):
        bfcl.fetch_pinned(PIN, tmp_path / "other", client=serve(True))
    assert not (tmp_path / "other").exists()


def test_openai_model_runs_need_the_openai_key_and_record_their_wire_api(tmp_path, monkeypatch, capsys) -> None:
    from app.providers import factory

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert bfcl.main(["--model", "gpt-4.1-mini", "--limit", "1", "--no-db", "--artifacts-dir", str(tmp_path)]) == 2
    assert "OPENAI_API_KEY is not set (needed for gpt-4.1-mini)" in capsys.readouterr().err

    observed = {}

    def fake(model, *, effort):
        observed.update(model=model, effort=effort)
        return ScriptedProvider([_respond(("calculate_triangle_area", {"base": 10, "height": 5}))], model=model)

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-never-sent")
    monkeypatch.setattr(factory, "make_provider", fake)
    assert bfcl.main(["--model", "gpt-4.1-mini", "--limit", "1", "--no-db", "--artifacts-dir", str(tmp_path)]) == 0
    assert observed == {"model": "gpt-4.1-mini", "effort": None}  # no reasoning_effort for a non-reasoning model
    saved = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert saved["runner"]["settings"]["provider"] == "openai_chat_completions"
    assert saved["runner"]["settings"]["effort"] is None
    assert bfcl._from_manifest(next(tmp_path.rglob("manifest.json")))["runner"]["model"] == "gpt-4.1-mini"


def test_replay_rejects_a_wire_api_that_does_not_match_the_model(tmp_path) -> None:
    run = bfcl.run_entries(ENTRIES[:1], ScriptedProvider([_respond(("calculate_triangle_area", {"base": 1}))],
                                                         model="claude-sonnet-5-5"),
                           kind="model_run", pin=PIN, settings={"effort": "low", "provider": "openai_chat_completions"})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bfcl.manifest(run, raw_path=None, raw_sha256=None)))
    with pytest.raises(bfcl.BenchmarkDataError, match="unsupported model runner"):
        bfcl._from_manifest(path)


@pytest.mark.parametrize("change", ["duplicate", "reorder", "empty", "scorer", "category", "provider", "count", "hash", "prompt", "tokens", "timeout"])
def test_replay_rejects_incompatible_or_mislabeled_manifests(tmp_path, change) -> None:
    entries = ENTRIES[:2]
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=PIN)
    saved = bfcl.manifest(run, raw_path=None, raw_sha256=None)
    if change == "duplicate":
        saved["subset"]["entry_ids"] = [entries[0].id] * 2
    elif change == "reorder":
        saved["subset"]["entry_ids"].reverse()
    elif change == "empty":
        saved["subset"].update(entry_ids=[], selected=0)
    elif change == "scorer":
        saved["scoring"]["scorer_version"] = "0.0.0"
    elif change == "category":
        saved["benchmark"]["category"] = "parallel_python"
    elif change == "provider":
        saved["runner"]["settings"]["provider"] = "unsupported"
    elif change == "count":
        saved["subset"]["category_task_count"] = 2
    elif change == "prompt":
        saved["runner"]["system_prompt"] = "custom system prompt"
    elif change == "tokens":
        saved["runner"]["settings"]["max_output_tokens"] = -1
    elif change == "timeout":
        saved["runner"]["settings"]["timeout_seconds"] = float("nan")
    else:
        saved["benchmark"]["files"]["answers"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(bfcl.BenchmarkDataError):
        bfcl._from_manifest(path)


def test_bfcl_provider_crash_is_recorded_without_losing_prior_work() -> None:
    def crash(request):
        raise RuntimeError("adapter crash")
    provider = ScriptedProvider([_respond(("calculate_triangle_area", {"base": 10, "height": 5})), crash])
    run = bfcl.run_entries(ENTRIES[:2], provider, kind="model_run", pin=PIN)
    assert [e.status for e in run.entries] == ["pass", "infrastructure_error"]
    assert run.usage["model_calls"] == 2 and run.usage["cost_usd"] is None
    assert len(provider.requests) == 2


def test_code_like_model_arguments_are_data_not_executed(tmp_path) -> None:
    marker = tmp_path / "executed"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    run = bfcl.run_entries(ENTRIES[:1], ScriptedProvider([
        _respond(("calculate_triangle_area", {"base": payload, "height": 5}), text=payload)
    ]), kind="model_run", pin=PIN)
    assert run.entries[0].status == "fail" and not marker.exists()


def test_model_manifest_replay_preserves_model_effort_and_request_limits(tmp_path, monkeypatch) -> None:
    from app.providers import factory

    original = bfcl.run_entries(ENTRIES[:1], ScriptedProvider([
        _respond(("calculate_triangle_area", {"base": 10, "height": 5}))
    ], model="claude-sonnet-5-5"), kind="model_run", pin=PIN, max_output_tokens=321,
        timeout_seconds=12.5, settings={"effort": "low", "provider": "anthropic_messages"})
    _, saved = bfcl.write_artifacts(original, tmp_path / "original")
    path = next((tmp_path / "original").rglob("manifest.json"))
    observed = {}

    class ReplayProvider:
        model = "claude-sonnet-5-5"
        name = "test"

        def __init__(self, *, model, effort):
            observed.update(model=model, effort=effort)

        def complete(self, request, *, timeout_seconds):
            observed.update(max_output_tokens=request.max_output_tokens, timeout_seconds=timeout_seconds)
            return _respond(("calculate_triangle_area", {"base": 10, "height": 5}))

    monkeypatch.setattr(factory, "make_provider", lambda model, *, effort: ReplayProvider(model=model, effort=effort))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-never-sent")
    assert bfcl.main(["--from-manifest", str(path), "--no-db", "--artifacts-dir", str(tmp_path / "replay")]) == 0
    assert observed == {"model": "claude-sonnet-5-5", "effort": "low", "max_output_tokens": 321, "timeout_seconds": 12.5}
    replayed = json.loads(next((tmp_path / "replay").rglob("manifest.json")).read_text())
    assert replayed["subset"] == saved["subset"]
    assert replayed["runner"]["settings"] == saved["runner"]["settings"]
    assert replayed["scoring"]["official_leaderboard_comparable"] is False
