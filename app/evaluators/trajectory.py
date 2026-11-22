"""Trajectory evaluator: required tool calls, matched one-to-one, under partial-order constraints.

Only allowed attempts satisfy a required call (unless the call sets
``allow_denied``). Any order of calls that respects the declared
``(before, after)`` pairs is valid; extra calls are fine unless an identical
call repeats more than ``max_repeats`` times.

Every score carries ``call_counts``: the call-level numerators and denominators
behind tool schema validity, tool-call precision and required-call recall, so
aggregates sum calls across runs instead of averaging per-run ratios.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.evaluators._trace import ToolCallRecord, to_decimal, tool_call_records
from app.evaluators.contract import EvaluationContext, make_score
from app.evaluators.score import Score
from app.task_spec import RequiredCall


_SEARCH_BUDGET = 200_000


@dataclass(frozen=True)
class _Assignment:
    count: int
    slots: tuple[int | None, ...]  # per required call: index into the attempts, or None


class TrajectoryEvaluator:
    name = "trajectory"
    version = "1.3.0"

    def evaluate(self, ctx: EvaluationContext) -> Score:
        spec = ctx.task.trajectory
        calls = tool_call_records(ctx.events)
        if not spec.tools_needed:
            # Precision and recall do not apply, but any call made is still a call whose schema can be checked.
            return make_score(
                self, "not_applicable", None, "TOOLS_NOT_NEEDED",
                {"tool_attempts": len(calls), "call_counts": _schema_counts(calls)},
                applicability="not_applicable",
            )

        required = spec.required_calls
        candidates = [
            [i for i, c in enumerate(calls) if c.tool == r.tool and _usable(r, c) and _args_match(r.args, c.args)]
            for r in required
        ]
        best_any = _max_matching(candidates)
        best_ordered = _best_ordered_matching(required, candidates, calls, spec.order)
        n = len(required)
        chosen = best_ordered if best_ordered.count == n else best_any
        evidence: dict[str, Any] = {
            "matches": {r.id: (calls[i].sequence if i is not None else None) for r, i in zip(required, chosen.slots)},
            "unmatched": [r.id for r, i in zip(required, chosen.slots) if i is None],
            "required_call_recall": best_ordered.count / n,
            "tool_call_precision": chosen.count / len(calls) if calls else None,
            "repeated_calls": _repeated(calls, spec.max_repeats),
            "call_counts": {
                **_schema_counts(calls),
                "matched": chosen.count,
                "required": n,
                "required_satisfied": best_ordered.count,
            },
        }
        recall = best_ordered.count / n

        if best_ordered.count == n:
            if evidence["repeated_calls"]:
                return make_score(self, "fail", recall, "EXCESS_REPETITION", evidence)
            return make_score(self, "pass", 1.0, "TRAJECTORY_SATISFIED", evidence)

        if best_any.count == n:
            seq = {r.id: calls[i].sequence for r, i in zip(required, best_any.slots)}
            evidence["order_violations"] = [[a, b] for a, b in spec.order if seq[a] > seq[b]]
            return make_score(self, "fail", recall, "ORDER_VIOLATION", evidence)

        missing = next(r for r, i in zip(required, best_any.slots) if i is None)
        used = {i for i in best_any.slots if i is not None}
        wrong = [
            c for j, c in enumerate(calls)
            if j not in used and c.tool == missing.tool and not _args_match(missing.args, c.args)
        ]
        if wrong:
            evidence["wrong_arguments"] = {
                "required_call": missing.id,
                "expected": missing.args,
                "attempted": [{"sequence": c.sequence, "args": c.args} for c in wrong],
            }
            return make_score(self, "fail", recall, "WRONG_ARGUMENT", evidence)
        return make_score(self, "fail", recall, missing.missing_code, evidence)


def _schema_counts(calls: list[ToolCallRecord]) -> dict[str, int]:
    """``decided`` excludes attempts with no gateway decision: their validity is unknown, not invalid."""
    decided = [c for c in calls if c.schema_valid is not None]
    return {
        "attempted": len(calls),
        "decided": len(decided),
        "schema_valid": sum(1 for c in decided if c.schema_valid),
    }


def _usable(required: RequiredCall, call: ToolCallRecord) -> bool:
    return (
        (call.decision == "allowed" and (call.executed or
         (required.allow_failed and call.effect is not None and call.effect.get("status") == "failed")))
        or (required.allow_denied and call.decision == "denied")
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return ("bool", value)  # True must never equal Decimal(1)
    number = to_decimal(value)
    return number if number is not None else value


def _args_match(expected: dict[str, Any], actual: Any) -> bool:
    if not isinstance(actual, dict):
        return False
    return all(key in actual and _normalize(actual[key]) == _normalize(value) for key, value in expected.items())


def _max_matching(candidates: list[list[int]]) -> _Assignment:
    """Largest one-to-one matching ignoring order (augmenting paths, polynomial)."""
    owner: dict[int, int] = {}  # attempt index -> required index

    def augment(k: int, seen: set[int]) -> bool:
        for i in candidates[k]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or augment(owner[i], seen):
                owner[i] = k
                return True
        return False

    for k in range(len(candidates)):
        augment(k, set())
    slots: list[int | None] = [None] * len(candidates)
    for i, k in owner.items():
        slots[k] = i
    return _Assignment(len(owner), tuple(slots))


def _best_ordered_matching(
    required: tuple[RequiredCall, ...],
    candidates: list[list[int]],
    calls: list[ToolCallRecord],
    order: tuple[tuple[str, str], ...],
) -> _Assignment:
    """Largest matching that respects every (before, after) pair among matched calls.

    Branch and bound: constraints are checked as calls are assigned, branches that
    cannot beat the best found are cut, and the search stops at a full match. An
    explicit node budget turns pathological traces into evaluation_error rather
    than a hang or a guess.
    """
    n = len(required)
    position = {r.id: k for k, r in enumerate(required)}
    constraints = [(position[a], position[b]) for a, b in order]
    best = _Assignment(-1, ())
    nodes = 0

    def consistent(k: int, i: int, slots: tuple[int | None, ...]) -> bool:
        seq = calls[i].sequence
        for before, after in constraints:
            if after == k and before < k and slots[before] is not None and calls[slots[before]].sequence >= seq:
                return False
            if before == k and after < k and slots[after] is not None and seq >= calls[slots[after]].sequence:
                return False
        return True

    def search(k: int, used: frozenset[int], slots: tuple[int | None, ...], count: int) -> None:
        nonlocal best, nodes
        nodes += 1
        if nodes > _SEARCH_BUDGET:
            raise RuntimeError(f"trajectory matching exceeded its search budget of {_SEARCH_BUDGET} nodes")
        if count + (n - k) <= best.count:
            return
        if k == n:
            best = _Assignment(count, slots)
            return
        for i in candidates[k]:
            if i not in used and consistent(k, i, slots):
                search(k + 1, used | {i}, (*slots, i), count + 1)
                if best.count == n:
                    return
        search(k + 1, used, (*slots, None), count)

    search(0, frozenset(), (), 0)
    return best


def _repeated(calls: list[ToolCallRecord], max_repeats: int) -> list[dict[str, Any]]:
    counts = Counter((c.tool, json.dumps(c.args, sort_keys=True, default=str)) for c in calls)
    return [
        {"tool": tool, "args": json.loads(args), "count": n}
        for (tool, args), n in sorted(counts.items())
        if n > max_repeats
    ]
