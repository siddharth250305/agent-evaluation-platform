"""BFCL ``simple_python``: pinned import, a native tool-calling runner, an AST-rule scorer, stored results.

Scope is deliberately one category. Each entry is one user turn and one
function; the model must answer with exactly one call. The data files are
vendored byte-for-byte from a pinned Gorilla commit and verified by SHA-256
before every run (``benchmarks/bfcl/pin.json``).

Model output is never executed or ``eval``-ed. The model answers through the
provider's native tool-use API, so its call arrives as already-parsed JSON
(``{"name": ..., "input": {...}}``); the scorer only compares those values with
the possible answers. That scorer is a port of the Python path of
``bfcl_eval/eval_checker/ast_eval/ast_checker.py::simple_function_checker`` at
the pinned commit, including its quirks. It is not the official package, and
every stored result says so (``official_leaderboard_comparable: false``).

Honest labeling: a run over fewer entries than the category is a subset, and
the subset size, category size and selection rule are stored as columns and
in the manifest. A ground-truth replay (``--self-check``) validates the
pipeline and scorer on the real data; it is stored as ``scorer_self_check``,
never as a model result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.accounting import PriceVersion, UsageLedger, cost_usd, price_for
from app.db.models import BenchmarkResult
from app.metrics import Ratio, money
from app.providers.base import (
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

ROOT = Path(__file__).resolve().parents[2]
BFCL_DIR = ROOT / "benchmarks" / "bfcl"
PIN_FILE = BFCL_DIR / "pin.json"
ARTIFACTS_DIR = ROOT / "artifacts"

SCORER_VERSION = "1.0.0"  # bump on any change that can alter a verdict
SCORING_METHOD = "AST match: port of bfcl_eval simple_function_checker (Python), scorer 1.0.0"
SUBSET_METHOD = "first N entries in file order"
SELF_CHECK_MODEL = "ground-truth-replay"
_PYTHON_HINT = " Note that the provided function is in Python 3 syntax."  # bfcl_eval/utils.py, Python categories

ResultKind = Literal["model_run", "scorer_self_check"]


class BenchmarkDataError(Exception):
    """The vendored data does not match the pin, or is not in the expected format."""


# --- pinned import ---------------------------------------------------------------------


@dataclass(frozen=True)
class BfclEntry:
    id: str
    turns: tuple[tuple[dict[str, Any], ...], ...]  # question: one tuple of messages per turn
    functions: tuple[dict[str, Any], ...]
    ground_truth: tuple[dict[str, Any], ...]


def load_pin(path: Path = PIN_FILE) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_category(pin: dict[str, Any] | None = None, data_dir: Path = BFCL_DIR) -> list[BfclEntry]:
    """Every entry of the pinned category, after checking each file against its pinned SHA-256."""
    pin = pin or load_pin()
    paths = {}
    for role, spec in pin["files"].items():
        path = data_dir / spec["local_path"]
        if not path.is_file():
            raise BenchmarkDataError(f"missing {role} file {path}; run `make bfcl-fetch`")
        actual = _sha256(path)
        if actual != spec["sha256"]:
            raise BenchmarkDataError(f"{path} has sha256 {actual}, pinned {spec['sha256']}")
        paths[role] = path

    questions, answers = _jsonl(paths["questions"]), _jsonl(paths["answers"])
    by_id = {a["id"]: a for a in answers}
    if len(by_id) != len(answers) or [q["id"] for q in questions] != [a["id"] for a in answers]:
        raise BenchmarkDataError("question and answer files do not list the same ids in the same order")
    if len(questions) != pin["category_task_count"]:
        raise BenchmarkDataError(f"expected {pin['category_task_count']} entries, found {len(questions)}")
    entries = []
    for q in questions:
        if len(q["question"]) != 1 or len(q["function"]) != 1 or len(by_id[q["id"]]["ground_truth"]) != 1:
            raise BenchmarkDataError(f"{q['id']} is not a single-turn, single-function entry")
        entries.append(BfclEntry(
            id=q["id"],
            turns=tuple(tuple(turn) for turn in q["question"]),
            functions=tuple(q["function"]),
            ground_truth=tuple(by_id[q["id"]]["ground_truth"]),
        ))
    return entries


def select_subset(entries: Sequence[BfclEntry], limit: int | None) -> list[BfclEntry]:
    """Deterministic: the first ``limit`` entries in file order (all of them when ``limit`` is None)."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    return list(entries[:limit] if limit is not None else entries)


# --- request building (port of bfcl_eval convert_to_tool, Anthropic style) --------------

_GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
    "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object", "tuple": "array",
    "any": "string",
}


def wire_name(name: str) -> str:
    """Tool names must match ^[a-zA-Z0-9_-]{1,64}$, so BFCL maps '.' to '_' for Anthropic models."""
    return re.sub(r"\.", "_", name)


def _cast_types(properties: dict[str, Any]) -> dict[str, Any]:
    for value in properties.values():
        var_type = value.get("type")
        if var_type is None:
            value["type"] = "string"
        else:
            if var_type == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = _GORILLA_TO_OPENAPI.get(var_type, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast_types(value["properties"])
            elif "items" in value:
                items = value["items"]
                items["type"] = _GORILLA_TO_OPENAPI[items["type"]]
                if items["type"] == "array" and "items" in items:
                    items["items"]["type"] = _GORILLA_TO_OPENAPI[items["items"]["type"]]
                elif items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_types(items["properties"])
    return properties


def tool_specs(functions: Sequence[dict[str, Any]]) -> tuple[ToolSpec, ...]:
    specs = []
    for function in copy.deepcopy(list(functions)):
        parameters = function["parameters"]
        parameters["type"] = "object"
        parameters["properties"] = _cast_types(parameters["properties"])
        specs.append(ToolSpec(
            name=wire_name(function["name"]),
            description=function.get("description", "") + _PYTHON_HINT,
            input_schema=parameters,
        ))
    return tuple(specs)


def build_request(entry: BfclEntry, *, max_output_tokens: int) -> ModelRequest:
    """No system prompt of our own: system messages in the entry (if any) become the system text."""
    messages = entry.turns[0]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return ModelRequest(
        system=system,
        messages=tuple(Message(role="user", content=m["content"]) for m in messages if m["role"] == "user"),
        tools=tool_specs(entry.functions),
        max_output_tokens=max_output_tokens,
    )


def decode(response: ModelResponse) -> list[dict[str, Any]] | None:
    """``[{name: arguments}]`` from native tool calls; None if any call's arguments are not a JSON object."""
    decoded = []
    for call in response.tool_calls:
        if not isinstance(call.arguments, dict):
            return None
        decoded.append({call.name: call.arguments})
    return decoded


# --- scorer: port of ast_checker.simple_function_checker (Python path) -------------------

_PYTHON_TYPES: dict[str, type] = {
    "string": str, "integer": int, "float": float, "boolean": bool, "array": list, "tuple": list, "dict": dict,
    "any": str,
}
_NESTED_TYPES = ("array", "tuple")


def _invalid(error_type: str, error: str) -> dict[str, Any]:
    return {"valid": False, "error": [error], "error_type": error_type}


def _valid() -> dict[str, Any]:
    return {"valid": True, "error": []}


def _answer_type(possible: list[Any]) -> type | None:
    for answer in possible:
        if answer != "":  # "" marks an optional parameter
            return type(answer)
    return None


def _standardize(text: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", text).lower().replace("'", '"')


def _type_check(param: str, value: Any, possible: list[Any], expected_desc: str, expected: type,
                nested: type | None) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}
    is_variable = False
    answer_type = _answer_type(possible)
    if answer_type is not None and answer_type != expected:
        is_variable = True
    if type(value) == expected:  # noqa: E721 - exact type match, as upstream (bool is not int)
        if nested is None:
            result["is_variable"] = is_variable
            return result
        for item in possible:
            ok = True
            if type(item) == list:  # noqa: E721
                for value_item in value:
                    if not _type_check(param, value_item, item, str(nested), nested, None)["valid"]:
                        ok = False
                        break
            if ok:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for parameter {param!r}."]
        result["error_type"] = "type_error:nested"
    if answer_type is not None and type(value) == answer_type:  # noqa: E721
        result["is_variable"] = True
        return result
    result["valid"] = False
    result["error"].append(f"Incorrect type for parameter {param!r}. Expected type {expected_desc}, "
                           f"got {type(value).__name__}.")
    result["error_type"] = "type_error:simple"
    return result


def _string_check(param: str, value: str, possible: list[Any]) -> dict[str, Any]:
    if _standardize(value) not in [_standardize(p) for p in possible if type(p) == str]:  # noqa: E721
        return _invalid("value_error:string", f"Invalid value for parameter {param!r}: {value!r}.")
    return _valid()


def _list_check(param: str, value: list[Any], possible: list[Any]) -> dict[str, Any]:
    standardized = [_standardize(v) if type(v) == str else v for v in value]  # noqa: E721
    answers = [[_standardize(x) if type(x) == str else x for x in answer] for answer in possible]  # noqa: E721
    if standardized not in answers:
        return _invalid("value_error:list/tuple", f"Invalid value for parameter {param!r}: {value!r}.")
    return _valid()


def _dict_check(param: str, value: dict[str, Any], possible_answers: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for possible in possible_answers:
        if possible == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        ok = True
        for key, item in value.items():
            if key not in possible:
                result = _invalid("value_error:dict_key", f"Unexpected dict key parameter: {key!r}.")
                ok = False
                break
            standardized = _standardize(item) if type(item) == str else item  # noqa: E721
            answers = [_standardize(a) if type(a) == str else a for a in possible[key]]  # noqa: E721
            if standardized not in answers:
                result = _invalid("value_error:dict_value", f"Invalid value for parameter {key!r}: {item!r}.")
                ok = False
                break
        for key, answers in possible.items():
            if key not in value and "" not in answers:
                result = _invalid("value_error:dict_key", f"Missing dict key parameter: {key!r}.")
                ok = False
                break
        if ok:
            return _valid()
    return result


def _list_dict_check(param: str, value: list[Any], possible_answers: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer in possible_answers:
        ok = True
        if len(value) != len(answer):
            result = _invalid("value_error:list_dict_count", "Wrong number of dictionaries in the list.")
            continue
        for index, item in enumerate(value):
            result = _dict_check(param, item, [answer[index]])
            if not result["valid"]:
                ok = False
                break
        if ok:
            return _valid()
    return result


def check_simple(function: dict[str, Any], output: list[dict[str, Any]], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """BFCL's verdict for one simple-category answer: ``{"valid": bool, "error": [...], "error_type": str}``."""
    if len(output) != 1:
        return _invalid("simple_function_checker:wrong_count", "Wrong number of functions.")
    possible = list(ground_truth.values())[0]
    name = wire_name(function["name"])
    details = function["parameters"]["properties"]
    model_output = output[0]
    if name not in model_output:
        return _invalid("simple_function_checker:wrong_func_name", f"Function name {name!r} not found in model output.")
    params = model_output[name]
    for param in function["parameters"]["required"]:
        if param not in params:
            return _invalid("simple_function_checker:missing_required", f"Missing required parameter: {param!r}.")
    for param, value in params.items():
        if param not in details or param not in possible:
            return _invalid("simple_function_checker:unexpected_param", f"Unexpected parameter: {param!r}.")
        expected_desc = details[param]["type"]
        expected = _PYTHON_TYPES[expected_desc]
        nested = _PYTHON_TYPES[details[param]["items"]["type"]] if expected_desc in _NESTED_TYPES else None
        if expected_desc == "tuple" and type(value) == tuple:  # noqa: E721
            value = list(value)
        if expected_desc == "float" and type(value) == int:  # noqa: E721 - Python's int -> float promotion
            value = float(value)
        checked = _type_check(param, value, possible[param], expected_desc, expected, nested)
        if not checked["valid"]:
            return {k: checked[k] for k in ("valid", "error", "error_type")}
        if not checked["is_variable"]:
            if expected == dict:
                special = _dict_check(param, value, possible[param])
            elif expected == list and nested == dict:
                special = _list_dict_check(param, value, possible[param])
            elif expected == str:
                special = _string_check(param, value, possible[param])
            elif expected == list:
                special = _list_check(param, value, possible[param])
            else:
                special = None
            if special is not None:
                if not special["valid"]:
                    return special
                continue
        if value not in possible[param]:
            return _invalid("value_error:others", f"Invalid value for parameter {param!r}: {value!r}.")
    for param in possible:
        if param not in params and "" not in possible[param]:
            return _invalid("simple_function_checker:missing_optional",
                            f"Optional parameter {param!r} not provided and not marked as optional.")
    return _valid()


# --- runner ----------------------------------------------------------------------------


@dataclass
class EntryResult:
    id: str
    status: Literal["pass", "fail", "infrastructure_error", "evaluation_error"]
    record: dict[str, Any]


@dataclass
class BenchmarkRun:
    kind: ResultKind
    pin: dict[str, Any]
    entries: list[EntryResult]
    model: str
    settings: dict[str, Any]
    usage: dict[str, Any]
    resolved_models: list[str]
    started_at: datetime
    finished_at: datetime
    selected_ids: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, Any]:
        counts = Counter(e.status for e in self.entries)
        scored = counts["pass"] + counts["fail"]
        return {
            # The headline number: correct / scored. Errors are shown beside it, never folded in silently.
            "accuracy": Ratio(counts["pass"], scored).as_dict(),
            "strict_accuracy": Ratio(counts["pass"], len(self.entries)).as_dict(),
            "coverage": Ratio(scored, len(self.entries)).as_dict(),
            "entries": len(self.entries),
            "passed": counts["pass"],
            "failed": counts["fail"],
            "infrastructure_errors": counts["infrastructure_error"],
            "evaluation_errors": counts["evaluation_error"],
            "error_types": dict(sorted(Counter(
                e.record["score"]["error_type"] for e in self.entries if e.status == "fail"
            ).items())),
            "usage": self.usage,
        }


def _usage_dict(usage: Usage | None) -> dict[str, Any] | None:
    return usage.model_dump() if usage is not None else None


def run_entries(
    entries: Sequence[BfclEntry],
    provider: ModelProvider,
    *,
    kind: ResultKind,
    pin: dict[str, Any],
    max_output_tokens: int = 1024,
    timeout_seconds: float = 60.0,
    settings: dict[str, Any] | None = None,
    price: PriceVersion | None = None,
    on_entry: Callable[[EntryResult], None] | None = None,
) -> BenchmarkRun:
    """One request per entry, no retries. A provider failure is an infrastructure error, never a fail."""
    ledger = UsageLedger()
    results: list[EntryResult] = []
    resolved: set[str] = set()
    started = datetime.now(UTC)
    for entry in entries:
        request = build_request(entry, max_output_tokens=max_output_tokens)
        record: dict[str, Any] = {
            "id": entry.id,
            "request": {"system": request.system or None, "messages": [m.content for m in request.messages],
                        "tools": [t.model_dump() for t in request.tools], "max_output_tokens": max_output_tokens},
        }
        t0 = time.monotonic()
        try:
            response = provider.complete(request, timeout_seconds=timeout_seconds)
            if not isinstance(response, ModelResponse):
                raise TypeError("provider did not return ModelResponse")
        except Exception as exc:  # provider/adapter failures are not benchmark verdicts

            ledger.record(provider.model, None, None, None)  # usage unknown, never free
            record.update(latency_ms=(time.monotonic() - t0) * 1000, response=None,
                          error={"type": "provider_error" if isinstance(exc, ProviderError) else "adapter_error",
                                 "message": str(exc), "retryable": getattr(exc, "retryable", False),
                                 "status_code": getattr(exc, "status_code", None)},
                          score=None)
            result = EntryResult(entry.id, "infrastructure_error", record)
        else:
            resolved.add(response.model)
            cost = cost_usd(response.usage, price)
            ledger.record(response.model, response.usage, cost, price.price_version_id if price else None)
            decoded = decode(response)
            record.update(
                latency_ms=(time.monotonic() - t0) * 1000,
                response={
                    "text": response.text,
                    "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                    "response_id": response.response_id,
                    "usage": _usage_dict(response.usage),
                    "cost_usd": money(cost),
                    "raw_content": list(response.provider_content) if response.provider_content else None,
                },
                decoded=decoded,
                error=None,
            )
            if decoded is None:
                score = _invalid("decoder:arguments_not_an_object", "A tool call's arguments were not a JSON object.")
            elif not decoded:
                score = _invalid("decoder:no_function_call", "The model answered without calling a function.")
            else:
                try:
                    score = check_simple(entry.functions[0], decoded, entry.ground_truth[0])
                except Exception as exc:  # noqa: BLE001 - a scorer crash is evaluation_error, never pass or fail
                    score = None
                    record["error"] = {"type": "evaluation_error", "message": f"{type(exc).__name__}: {exc}"}
            record["score"] = score
            if score is None:
                result = EntryResult(entry.id, "evaluation_error", record)
            else:
                result = EntryResult(entry.id, "pass" if score["valid"] else "fail", record)
        results.append(result)
        if on_entry is not None:
            on_entry(result)
    summary = ledger.summary()
    return BenchmarkRun(
        kind=kind,
        pin=pin,
        entries=results,
        model=provider.model,
        settings={"max_output_tokens": max_output_tokens, "timeout_seconds": timeout_seconds, **(settings or {})},
        usage={
            "model_calls": summary.model_calls,
            "unknown_usage_calls": summary.unknown_usage_calls,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "cost_usd": money(summary.cost_usd),
            "price_version_ids": list(summary.price_version_ids),
        },
        resolved_models=sorted(resolved),
        started_at=started,
        finished_at=datetime.now(UTC),
        selected_ids=[e.id for e in entries],
    )


def _first_acceptable(answers: dict[str, list[Any]]) -> dict[str, Any]:
    """First non-optional acceptable value per key. Dict values nest their own lists of acceptable values."""
    chosen: dict[str, Any] = {}
    for key, values in answers.items():
        value = next((v for v in values if v != ""), None)
        if value is None:
            continue  # only "" (optional): leave it out
        if isinstance(value, dict):
            value = _first_acceptable(value)
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            value = [_first_acceptable(v) for v in value]
        chosen[key] = value
    return chosen


class GroundTruthReplayProvider:
    """Answers each entry with its first acceptable ground-truth call: a scorer and pipeline self-check.

    It calls no model. Usage is reported as zero tokens because none were used,
    and the result is stored as ``scorer_self_check``, never as a model score.
    """

    name = "ground_truth_replay"
    model = SELF_CHECK_MODEL

    def __init__(self, entries: Sequence[BfclEntry]) -> None:
        self._queue = list(entries)

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        entry = self._queue.pop(0)
        name, answers = next(iter(entry.ground_truth[0].items()))
        arguments = _first_acceptable(answers)
        if request.tools[0].name != wire_name(name):
            raise ProviderError(f"replay out of order: expected {wire_name(name)}", retryable=False)
        return ModelResponse(
            text="",
            tool_calls=(ToolCall(id=f"replay_{entry.id}", name=wire_name(name), arguments=arguments),),
            stop_reason="tool_use",
            usage=Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            model=SELF_CHECK_MODEL,
        )


# --- manifest, artifacts and storage ------------------------------------------------------


def manifest(run: BenchmarkRun, *, raw_path: str | None, raw_sha256: str | None) -> dict[str, Any]:
    from app.services.experiments import _code_identity

    pin = run.pin
    size = pin["category_task_count"]
    selected = len(run.selected_ids)
    return {
        "schema_version": 1,
        "result_kind": run.kind,
        "benchmark": {
            "name": pin["benchmark"], "release": pin["release"], "repository": pin["repository"],
            "commit": pin["commit"], "commit_date": pin["commit_date"], "license": pin["license"],
            "category": pin["category"],
            "files": {role: {"upstream_path": f["upstream_path"], "sha256": f["sha256"]} for role, f in pin["files"].items()},
        },
        "subset": {
            "is_subset": selected < size,
            "selected": selected,
            "category_task_count": size,
            "method": SUBSET_METHOD,
            "entry_ids": run.selected_ids,
        },
        "scoring": {
            "method": SCORING_METHOD,
            "scorer_version": SCORER_VERSION,
            "ported_from": f"{pin['repository']}/blob/{pin['commit']}/berkeley-function-call-leaderboard/"
                           "bfcl_eval/eval_checker/ast_eval/ast_checker.py (simple_function_checker, Python)",
            "decoding": "native tool-use JSON; model output is never executed or eval()-ed",
            "headline_metric": "accuracy = passed / scored entries; infrastructure and scorer errors reported separately",
            "official_leaderboard_comparable": False,
        },
        "runner": {
            "kind": run.kind,
            "model": run.model,
            "resolved_models": run.resolved_models,
            "settings": run.settings,
            "tool_conversion": "bfcl_eval convert_to_tool (Anthropic style): '.'->'_' in names, Gorilla->OpenAPI types,"
                               " Python 3 hint appended to descriptions",
            "system_prompt": None,
            "retries": "none; provider errors are recorded as infrastructure errors",
        },
        "deviations_from_official": [
            "scorer is a port of the official AST checker, not the bfcl_eval package",
            "sampling temperature is the provider default (BFCL sets 0.001)",
            *(["subset of the category: " + SUBSET_METHOD] if selected < size else []),
        ],
        "usage": run.usage,
        "raw_outputs": {"path": raw_path, "sha256": raw_sha256, "records": len(run.entries)},
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat(),
        "code": _code_identity(),
    }


def write_artifacts(run: BenchmarkRun, artifacts_dir: Path = ARTIFACTS_DIR) -> tuple[Path, dict[str, Any]]:
    """``raw_outputs.jsonl`` (one record per entry, as received) and ``manifest.json``. Returns the raw path."""
    stamp = run.started_at.strftime("%Y%m%dT%H%M%SZ")
    directory = artifacts_dir / "benchmarks" / "bfcl" / f"{stamp}-{run.kind}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / "raw_outputs.jsonl"
    with raw.open("w") as f:
        for entry in run.entries:
            f.write(json.dumps({"status": entry.status, **entry.record}, sort_keys=True, default=str) + "\n")
    try:
        shown = str(raw.relative_to(ROOT))
    except ValueError:
        shown = str(raw)
    built = manifest(run, raw_path=shown, raw_sha256=_sha256(raw))
    (directory / "manifest.json").write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    return raw, built


def save_result(session: Session, run: BenchmarkRun, built_manifest: dict[str, Any]) -> BenchmarkResult:
    pin = run.pin
    result = BenchmarkResult(
        benchmark=pin["benchmark"],
        benchmark_version=f"{pin['release']}@{pin['commit']}",
        category=pin["category"],
        model=run.model,
        task_count=len(run.selected_ids),
        category_task_count=pin["category_task_count"],
        subset_method=SUBSET_METHOD if len(run.selected_ids) < pin["category_task_count"] else None,
        result_kind=run.kind,
        scoring_method=SCORING_METHOD,
        metrics=run.metrics,
        raw_artifact_uri=built_manifest["raw_outputs"]["path"],
        manifest=built_manifest,
    )
    session.add(result)
    session.flush()
    return result


# --- fetching the pinned files -----------------------------------------------------------

RAW_URL = "https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/{path}"


def fetch_pinned(pin: dict[str, Any] | None = None, data_dir: Path = BFCL_DIR, *, client: Any = None) -> list[Path]:
    """Download each pinned file at the pinned commit; write it only if its SHA-256 matches the pin."""
    import httpx

    pin = pin or load_pin()
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    written = []
    for role, spec in pin["files"].items():
        response = client.get(RAW_URL.format(commit=pin["commit"], path=spec["upstream_path"]))
        response.raise_for_status()
        actual = hashlib.sha256(response.content).hexdigest()
        if actual != spec["sha256"]:
            raise BenchmarkDataError(f"downloaded {role} file has sha256 {actual}, pinned {spec['sha256']}")
        target = data_dir / spec["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        written.append(target)
    return written


# --- command line ----------------------------------------------------------------------


def _from_manifest(path: Path) -> dict[str, Any]:
    """Reproduce a stored setup: same pin, same entries, same model and settings."""
    saved = json.loads(path.read_text())
    pin = load_pin()
    try:
        benchmark, subset, scoring, runner = (saved[k] for k in ("benchmark", "subset", "scoring", "runner"))
        if saved["schema_version"] != 1 or any(
            benchmark[key] != pin[key] for key in ("commit", "release", "repository", "category")
        ) or benchmark["name"] != pin["benchmark"] or any(
            benchmark["files"][role]["sha256"] != spec["sha256"] for role, spec in pin["files"].items()
        ):
            raise BenchmarkDataError("manifest was produced from different BFCL data than the current pin")
        if scoring["scorer_version"] != SCORER_VERSION or scoring["method"] != SCORING_METHOD:
            raise BenchmarkDataError("manifest scorer differs from the current scorer")
        if scoring["official_leaderboard_comparable"] is not False:
            raise BenchmarkDataError("this scorer is not official-leaderboard comparable")
        entries = load_category(pin)  # verify bytes as well as the manifest's claimed hashes
        ids = subset["entry_ids"]
        if (not isinstance(ids, list) or not ids or len(ids) > len(entries)
                or ids != [entry.id for entry in entries[:len(ids)]]
                or subset["selected"] != len(ids)
                or subset["category_task_count"] != len(entries)
                or subset["is_subset"] is not (len(ids) < len(entries))
                or subset["method"] != SUBSET_METHOD):
            raise BenchmarkDataError("manifest selection must be a nonempty first-N subset with accurate labels")
        kind = saved["result_kind"]
        settings = runner["settings"]
        tokens, timeout = settings["max_output_tokens"], settings["timeout_seconds"]
        if (type(tokens) is not int or tokens < 1 or type(timeout) not in (int, float)
                or not math.isfinite(timeout) or timeout <= 0 or runner["system_prompt"] is not None):
            raise BenchmarkDataError("unsupported runner prompt or request limits")
        if kind not in ("scorer_self_check", "model_run") or runner["kind"] != kind:
            raise BenchmarkDataError("invalid manifest runner kind")
        if kind == "scorer_self_check":
            if runner["model"] != SELF_CHECK_MODEL or set(settings) != {"max_output_tokens", "timeout_seconds"}:
                raise BenchmarkDataError("self-check runner/settings mismatch")
        elif (runner["model"] == SELF_CHECK_MODEL or settings.get("provider") != "anthropic_messages"
              or set(settings) != {"max_output_tokens", "timeout_seconds", "effort", "provider"}):
            raise BenchmarkDataError("unsupported model runner/settings")
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkDataError(f"invalid replay manifest: {exc}") from exc
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.benchmarks.bfcl", description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="first N entries (default: whole category)")
    parser.add_argument("--self-check", action="store_true", help="replay ground truth; no model, no API key")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--from-manifest", type=Path, help="re-run the exact setup recorded in a manifest.json")
    parser.add_argument("--no-db", action="store_true", help="write artifacts only; do not store a result row")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--fetch", action="store_true", help="re-download the pinned files, verify, and exit")
    args = parser.parse_args(argv)

    pin = load_pin()
    if args.fetch:
        for path in fetch_pinned(pin):
            print(f"verified {path.relative_to(ROOT)}", file=sys.stderr)
        return 0
    entries = load_category(pin)
    if args.from_manifest:
        saved = _from_manifest(args.from_manifest)
        wanted = saved["subset"]["entry_ids"]
        by_id = {e.id: e for e in entries}
        selected = [by_id[i] for i in wanted]
        args.self_check = saved["result_kind"] == "scorer_self_check"
        settings = saved["runner"]["settings"]
        args.model = saved["runner"]["model"]
        args.effort = settings.get("effort", args.effort)
        args.max_output_tokens = settings["max_output_tokens"]
        args.timeout_seconds = settings["timeout_seconds"]
    else:
        selected = select_subset(entries, args.limit)

    if args.self_check:
        provider: ModelProvider = GroundTruthReplayProvider(selected)
        kind: ResultKind = "scorer_self_check"
        price, settings = None, {}
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. Use --self-check for an offline run.", file=sys.stderr)
            return 2
        from app.providers.anthropic_http import AnthropicMessagesProvider

        provider = AnthropicMessagesProvider(model=args.model, effort=args.effort)
        kind = "model_run"
        price, settings = price_for(args.model), {"effort": args.effort, "provider": "anthropic_messages"}

    label = f"{pin['category']} {len(selected)}/{pin['category_task_count']}"
    print(f"BFCL {pin['release']}@{pin['commit'][:12]} {label} ({kind}, {provider.model})", file=sys.stderr)
    run = run_entries(selected, provider, kind=kind, pin=pin, max_output_tokens=args.max_output_tokens,
                      timeout_seconds=args.timeout_seconds, settings=settings, price=price)
    raw, built = write_artifacts(run, args.artifacts_dir)
    if not args.no_db:
        from app.db.session import make_engine, make_sessionmaker

        with make_sessionmaker(make_engine()).begin() as session:
            stored = save_result(session, run, built)
            print(f"stored benchmark result {stored.id}", file=sys.stderr)
    metrics = run.metrics
    print(json.dumps({
        "category": label, "result_kind": kind, "accuracy": metrics["accuracy"], "coverage": metrics["coverage"],
        "error_types": metrics["error_types"], "usage": metrics["usage"], "raw_outputs": str(raw),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
