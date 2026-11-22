"""Scorecard and comparison math. Pure: no database, no I/O.

Every rate is reported as numerator / denominator / value, never as a bare
number, and there is no blended accuracy score. Unknown usage stays unknown:
one attempt with unknown cost makes the total unknown, never $0.

Definitions (PRD §7):

- verified success rate     = successful runs / scored runs (pass + fail)
- strict end-to-end success = successful runs / planned runs (missing = unsuccessful)
- evaluation coverage       = scored runs / planned runs
- unsafe attempt rate       = runs with an attempted policy violation / runs the policy evaluator scored
- executed violation rate   = runs with an executed policy violation / runs the policy evaluator scored
- passed all repetitions    = tasks whose every planned repetition passed / tasks
- latency                   = queue time (run created -> first attempt started) and execution
                              time (the result attempt's monotonic duration), reported separately
- cost                      = every attempt that started, infrastructure retries included
- tool schema validity      = schema-valid tool calls / attempted tool calls
- tool-call precision       = matched required calls / tool calls attempted, tools-needed tasks only
- required-call recall      = required calls satisfied (in a valid order) / required calls, tools-needed tasks only

Tool metrics sum call counts from the trajectory evaluator's evidence across runs;
they never average per-run ratios, so a run with 12 calls weighs 12 times a run
with one. Runs the trajectory evaluator could not score are reported, not dropped.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, TypeVar

T = TypeVar("T")

_ACTIVE = ("queued", "running")
_TERMINAL = ("completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled")
_MONEY_PLACES = Decimal("0.00000001")  # matches NUMERIC(18, 8)

TaskOutcome = Literal["passed_all", "failed_any", "inconclusive"]
TaskResultKind = Literal["win", "loss", "tie"]


# --- facts -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptFacts:
    attempt_number: int
    status: str
    started: bool
    cost_usd: Decimal | None
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: float | None
    stop_reason: str | None = None
    error_code: str | None = None

    @property
    def counted(self) -> bool:
        """It ran and has finished, so whatever it spent is final (known or unknown)."""
        return self.started and self.status not in _ACTIVE


@dataclass(frozen=True)
class ScoreFacts:
    evaluator_name: str
    evaluator_version: str
    status: str
    value: float | None
    reason_code: str
    applicability: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RunFacts:
    """Everything the scorecard needs about one run, already read from the database."""

    run_id: str
    agent_config_id: str
    agent_config_name: str
    task_id: str
    task_version_id: str
    category: str
    repetition: int
    status: str
    evaluation_status: str
    success: bool | None
    queue_ms: float | None
    attempts: tuple[AttemptFacts, ...] = ()
    scores: tuple[ScoreFacts, ...] = ()
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def scored(self) -> bool:
        return self.evaluation_status in ("pass", "fail")

    @property
    def passed(self) -> bool:
        return self.evaluation_status == "pass"

    @property
    def failed(self) -> bool:
        return self.evaluation_status == "fail"

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL

    @property
    def counted_attempts(self) -> tuple[AttemptFacts, ...]:
        return tuple(a for a in self.attempts if a.counted)

    @property
    def result_attempt(self) -> AttemptFacts | None:
        """The last finished attempt: the one whose outcome the run reports."""
        counted = self.counted_attempts
        return max(counted, key=lambda a: a.attempt_number) if counted else None

    @property
    def execution_ms(self) -> float | None:
        attempt = self.result_attempt
        return attempt.duration_ms if attempt is not None else None

    @property
    def cost_usd(self) -> Decimal | None:
        """Every finished attempt, retries included; unknown if any of them is unknown."""
        return _sum_known(a.cost_usd for a in self.counted_attempts)

    @property
    def input_tokens(self) -> int | None:
        return _sum_tokens(a.input_tokens for a in self.counted_attempts)

    @property
    def output_tokens(self) -> int | None:
        return _sum_tokens(a.output_tokens for a in self.counted_attempts)

    def score(self, evaluator_name: str) -> ScoreFacts | None:
        return next((s for s in self.scores if s.evaluator_name == evaluator_name), None)

    @property
    def policy_scored(self) -> bool:
        policy = self.score("policy")
        return policy is not None and policy.applicability == "applicable" and policy.status in ("pass", "fail")

    @property
    def unsafe_attempt(self) -> bool | None:
        """A forbidden action was attempted (even if the gateway blocked it). None: not policy-scored."""
        if not self.policy_scored:
            return None
        return self.score("policy").status == "fail"

    @property
    def executed_violation(self) -> bool | None:
        """A forbidden action actually took effect. None: not policy-scored."""
        if not self.policy_scored:
            return None
        policy = self.score("policy")
        return policy.status == "fail" and policy.evidence.get("executed_violation") is True

    @property
    def violations(self) -> list[dict[str, Any]]:
        policy = self.score("policy")
        if policy is None or policy.status != "fail":
            return []
        found = policy.evidence.get("violations")
        return [v for v in found if isinstance(v, dict)] if isinstance(found, list) else []

    @property
    def failing_scores(self) -> tuple[ScoreFacts, ...]:
        return tuple(s for s in self.scores if s.status in ("fail", "evaluation_error"))


# --- primitives ------------------------------------------------------------------------


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def as_dict(self) -> dict[str, Any]:
        return {"numerator": self.numerator, "denominator": self.denominator, "value": self.value}


def money(value: Decimal | None) -> str | None:
    """Plain decimal string ("0", "0.0024"); None stays None (unknown)."""
    return format(value.normalize(), "f") if value is not None else None


def _sum_known(values: Iterable[Decimal | None]) -> Decimal | None:
    total = Decimal("0")
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _sum_tokens(values: Iterable[int | None]) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    """Summary of the known values; ``unknown`` counts the missing ones instead of treating them as zero."""
    values = list(values)
    known = sorted(v for v in values if v is not None)
    return {
        "n": len(known),
        "unknown": len(values) - len(known),
        "mean": sum(known) / len(known) if known else None,
        "median": quantile(known, 0.5),
        "p95": quantile(known, 0.95),
    }


def _diff(candidate: float | None, baseline: float | None) -> float | None:
    return candidate - baseline if candidate is not None and baseline is not None else None


def _money_diff(candidate: Decimal | None, baseline: Decimal | None) -> str | None:
    return money(candidate - baseline) if candidate is not None and baseline is not None else None


# --- task-level grouping -----------------------------------------------------------------


@dataclass(frozen=True)
class TaskResult:
    """All repetitions of one task for one configuration, kept together."""

    task_id: str
    category: str
    planned: int
    passed: int
    failed: int
    scored: int

    @property
    def pass_rate(self) -> Ratio:
        """Strict: a repetition that was not scored counts as not passed."""
        return Ratio(self.passed, self.planned)

    @property
    def outcome(self) -> TaskOutcome:
        if self.planned and self.passed == self.planned:
            return "passed_all"
        if self.failed:
            return "failed_any"
        return "inconclusive"

    def as_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "passed": self.passed,
            "failed": self.failed,
            "scored": self.scored,
            "pass_rate": self.pass_rate.as_dict(),
            "outcome": self.outcome,
        }


def by_task(runs: Iterable[RunFacts]) -> dict[str, TaskResult]:
    grouped: dict[str, list[RunFacts]] = defaultdict(list)
    for run in runs:
        grouped[run.task_id].append(run)
    return {
        task_id: TaskResult(
            task_id=task_id,
            category=group[0].category,
            planned=len(group),
            passed=sum(r.passed for r in group),
            failed=sum(r.failed for r in group),
            scored=sum(r.scored for r in group),
        )
        for task_id, group in sorted(grouped.items())
    }


# --- bootstrap ---------------------------------------------------------------------------


def cluster_bootstrap(
    clusters: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Percentile bootstrap that resamples whole clusters (tasks), so repetitions of a task stay together.

    Deterministic for a given seed. Resamples where the statistic is undefined
    (e.g. a zero denominator) are dropped and reported as such.
    """
    out: dict[str, Any] = {
        "low": None,
        "high": None,
        "confidence": confidence,
        "resamples": resamples,
        "valid_resamples": 0,
        "seed": seed,
        "unit": "task",
        "clusters": len(clusters),
    }
    if not clusters or resamples <= 0:
        return out
    rng = random.Random(seed)
    k = len(clusters)
    stats = []
    for _ in range(resamples):
        value = statistic([clusters[rng.randrange(k)] for _ in range(k)])
        if value is not None:
            stats.append(value)
    stats.sort()
    alpha = (1 - confidence) / 2
    out.update(low=quantile(stats, alpha), high=quantile(stats, 1 - alpha), valid_resamples=len(stats))
    return out


def _strict(tasks: Sequence[TaskResult]) -> float | None:
    return Ratio(sum(t.passed for t in tasks), sum(t.planned for t in tasks)).value


def _verified(tasks: Sequence[TaskResult]) -> float | None:
    return Ratio(sum(t.passed for t in tasks), sum(t.scored for t in tasks)).value


# --- tool metrics --------------------------------------------------------------------------

# metric -> (numerator count, denominator count, unit, needs a tools-needed task)
TOOL_METRICS: dict[str, tuple[str, str, str, bool]] = {
    "tool_schema_validity": ("schema_valid", "attempted", "tool call", False),
    "tool_call_precision": ("matched", "attempted", "tool call", True),
    "required_call_recall": ("required_satisfied", "required", "required call", True),
}


def _call_counts(score: ScoreFacts) -> dict[str, int] | None:
    counts = score.evidence.get("call_counts")
    if not isinstance(counts, dict):
        return None
    return {k: v for k, v in counts.items() if isinstance(v, int) and not isinstance(v, bool)}


def tool_metrics(runs: Sequence[RunFacts]) -> dict[str, dict[str, Any]]:
    """Call-level tool metrics with an explicit account of every run that did not contribute.

    Missing schema decisions make the schema metric unknown even with a positive
    denominator. Otherwise ``applicability`` is ``applicable`` when it is positive,
    ``not_applicable`` when nothing could have contributed (no tools needed, or no
    calls made) and every run was scored, and ``unknown`` otherwise: then evaluator
    errors, ungraded runs or legacy scores without counts hide the answer.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, (num_key, den_key, unit, tools_needed_only) in TOOL_METRICS.items():
        numerator = denominator = unknown_calls = 0
        runs_by_kind = Counter({"counted": 0, "not_applicable": 0, "evaluation_error": 0, "missing_counts": 0,
                                "ungraded": 0})
        for run in runs:
            score = run.score("trajectory")
            if score is None:
                runs_by_kind["ungraded"] += 1
                continue
            if score.status == "evaluation_error":
                runs_by_kind["evaluation_error"] += 1
                continue
            if tools_needed_only and score.applicability == "not_applicable":
                runs_by_kind["not_applicable"] += 1
                continue
            counts = _call_counts(score)
            if counts is None or num_key not in counts or den_key not in counts:
                runs_by_kind["missing_counts"] += 1  # scored by a trajectory version that did not record counts
                continue
            if name == "tool_schema_validity":
                if "decided" not in counts:
                    runs_by_kind["missing_counts"] += 1
                    continue
                unknown_calls += counts["attempted"] - counts["decided"]
            runs_by_kind["counted"] += 1
            numerator += counts[num_key]
            denominator += counts[den_key]
        unknown = runs_by_kind["evaluation_error"] + runs_by_kind["missing_counts"] + runs_by_kind["ungraded"]
        if unknown_calls:
            applicability = "unknown"
        elif denominator:
            applicability = "applicable"
        elif unknown:
            applicability = "unknown"
        else:
            applicability = "not_applicable"
        out[name] = {
            **Ratio(numerator, denominator).as_dict(),
            "value": None if unknown_calls else Ratio(numerator, denominator).value,
            "unknown_calls": unknown_calls,
            "unit": unit,
            "applicability": applicability,
            "runs": dict(runs_by_kind),
            # Runs whose counts are known / runs not known to be inapplicable.
            "coverage": Ratio(runs_by_kind["counted"], len(runs) - runs_by_kind["not_applicable"]).as_dict(),
        }
    return out


# --- scorecard ---------------------------------------------------------------------------


def _cost(runs: Sequence[RunFacts]) -> tuple[Decimal | None, Decimal | None, int]:
    """Total cost (retries included), cost per successful run, and the successful-run denominator."""
    total = _sum_known(a.cost_usd for r in runs for a in r.counted_attempts)
    successful = sum(r.passed for r in runs)
    per_success = (total / successful).quantize(_MONEY_PLACES) if total is not None and successful else None
    return total, per_success, successful


def usage(runs: Sequence[RunFacts]) -> dict[str, Any]:
    attempts = [a for r in runs for a in r.counted_attempts]
    total, per_success, successful = _cost(runs)
    return {
        "attempts": len(attempts),
        "retry_attempts": sum(max(0, len(r.counted_attempts) - 1) for r in runs),
        "attempts_with_unknown_cost": sum(1 for a in attempts if a.cost_usd is None),
        "cost_usd": money(total),
        "cost_per_successful_run_usd": {
            "numerator_usd": money(total),
            "denominator": successful,
            "value": money(per_success),
        },
        "input_tokens": _sum_tokens(a.input_tokens for a in attempts),
        "output_tokens": _sum_tokens(a.output_tokens for a in attempts),
        "attempts_with_unknown_tokens": sum(
            1 for a in attempts if a.input_tokens is None or a.output_tokens is None
        ),
    }


def rates(runs: Sequence[RunFacts]) -> dict[str, Ratio]:
    planned = len(runs)
    scored = sum(r.scored for r in runs)
    successful = sum(r.passed for r in runs)
    policy_scored = sum(r.policy_scored for r in runs)
    tasks = by_task(runs)
    return {
        "verified_success_rate": Ratio(successful, scored),
        "strict_success_rate": Ratio(successful, planned),
        "evaluation_coverage": Ratio(scored, planned),
        "unsafe_attempt_rate": Ratio(sum(r.unsafe_attempt is True for r in runs), policy_scored),
        "executed_violation_rate": Ratio(sum(r.executed_violation is True for r in runs), policy_scored),
        "passed_all_repetitions": Ratio(sum(t.outcome == "passed_all" for t in tasks.values()), len(tasks)),
    }


def scorecard(runs: Sequence[RunFacts]) -> dict[str, Any]:
    return {
        "planned_runs": len(runs),
        "finished_runs": sum(r.finished for r in runs),
        "scored_runs": sum(r.scored for r in runs),
        "successful_runs": sum(r.passed for r in runs),
        "failed_runs": sum(r.failed for r in runs),
        "evaluation_errors": sum(r.evaluation_status == "evaluation_error" for r in runs),
        "tasks": len({r.task_id for r in runs}),
        **{name: ratio.as_dict() for name, ratio in rates(runs).items()},
        "latency_ms": {
            "queue": distribution(r.queue_ms for r in runs),
            "execution": distribution(r.execution_ms for r in runs),
        },
        "tool_metrics": tool_metrics(runs),
        "usage": usage(runs),
    }


def by_category(runs: Sequence[RunFacts]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunFacts]] = defaultdict(list)
    for run in runs:
        grouped[run.category].append(run)
    out = []
    for category, group in sorted(grouped.items()):
        r = rates(group)
        out.append({
            "category": category,
            "planned_runs": len(group),
            "verified_success_rate": r["verified_success_rate"].as_dict(),
            "strict_success_rate": r["strict_success_rate"].as_dict(),
            "evaluation_coverage": r["evaluation_coverage"].as_dict(),
            "unsafe_attempt_rate": r["unsafe_attempt_rate"].as_dict(),
        })
    return out


def failure_reasons(runs: Iterable[RunFacts]) -> list[dict[str, Any]]:
    """How many runs each (evaluator, status, reason code) explains, most frequent first."""
    counts = Counter((s.evaluator_name, s.status, s.reason_code) for r in runs for s in r.failing_scores)
    return [
        {"evaluator": evaluator, "status": status, "reason_code": code, "runs": n}
        for (evaluator, status, code), n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def hard_violations(runs: Iterable[RunFacts]) -> list[dict[str, Any]]:
    """Runs with a policy violation, executed ones first."""
    out = []
    for run in runs:
        violations = run.violations
        if not violations:
            continue
        out.append({
            "run_id": run.run_id,
            "task_id": run.task_id,
            "agent_config_name": run.agent_config_name,
            "repetition": run.repetition,
            "codes": sorted({str(v.get("code")) for v in violations}),
            "executed": run.executed_violation is True,
            "violations": violations,
        })
    return sorted(out, key=lambda v: (not v["executed"], v["agent_config_name"], v["task_id"], v["repetition"]))


# --- comparison --------------------------------------------------------------------------

DEFINITIONS = {
    "verified_success_rate": "successful runs / scored runs (pass + fail)",
    "strict_success_rate": "successful runs / planned runs; unscored runs count as unsuccessful",
    "evaluation_coverage": "scored runs / planned runs",
    "unsafe_attempt_rate": "runs with an attempted policy violation / runs the policy evaluator scored",
    "executed_violation_rate": "runs with an executed policy violation / runs the policy evaluator scored",
    "passed_all_repetitions": "tasks whose every planned repetition passed / tasks",
    "win": "candidate's strict per-task pass rate (passed repetitions / planned) is higher",
    "loss": "candidate's strict per-task pass rate is lower",
    "regression": "baseline passed every repetition of the task; candidate failed at least one",
    "improvement": "baseline failed at least one repetition; candidate passed every repetition",
    "queue_latency": "run created -> first attempt started (database clock)",
    "execution_latency": "monotonic duration of the attempt that produced the run's result",
    "cost": "sum over every attempt that started, retries included; unknown if any attempt's usage is unknown",
    "tool_schema_validity": "schema-valid tool calls / attempted tool calls, summed over runs",
    "tool_call_precision": "matched required calls / tool calls attempted, summed over tools-needed runs",
    "required_call_recall": "required calls satisfied in a valid order / required calls, summed over tools-needed runs",
}


def _task_result(base: TaskResult, cand: TaskResult) -> TaskResultKind:
    # Cross-multiply: compare passed/planned without floating point.
    left, right = cand.passed * base.planned, base.passed * cand.planned
    return "win" if left > right else "loss" if left < right else "tie"


def compare(
    baseline: Sequence[RunFacts],
    candidate: Sequence[RunFacts],
    *,
    resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Same-task comparison of two configurations' runs. Deltas are candidate minus baseline."""
    base_tasks, cand_tasks = by_task(baseline), by_task(candidate)
    common = sorted(set(base_tasks) & set(cand_tasks))

    tasks = []
    for task_id in common:
        base, cand = base_tasks[task_id], cand_tasks[task_id]
        tasks.append({
            "task_id": task_id,
            "category": base.category,
            "baseline": base.as_dict(),
            "candidate": cand.as_dict(),
            "result": _task_result(base, cand),
            "regression": base.outcome == "passed_all" and cand.outcome == "failed_any",
            "improvement": base.outcome == "failed_any" and cand.outcome == "passed_all",
        })
    kinds = Counter(t["result"] for t in tasks)

    pairs = [(base_tasks[t], cand_tasks[t]) for t in common]
    boot = {"resamples": resamples, "seed": seed, "confidence": confidence}

    def side_ci(index: int, stat: Callable[[Sequence[TaskResult]], float | None]) -> dict[str, Any]:
        return cluster_bootstrap(pairs, lambda sample: stat([p[index] for p in sample]), **boot)

    def delta_ci(stat: Callable[[Sequence[TaskResult]], float | None]) -> dict[str, Any]:
        return cluster_bootstrap(
            pairs, lambda sample: _diff(stat([p[1] for p in sample]), stat([p[0] for p in sample])), **boot
        )

    base_rates, cand_rates = rates(baseline), rates(candidate)
    metrics: dict[str, Any] = {}
    for name in base_rates:
        b, c = base_rates[name], cand_rates[name]
        metrics[name] = {"baseline": b.as_dict(), "candidate": c.as_dict(), "delta": _diff(c.value, b.value)}
    for name, stat in (("strict_success_rate", _strict), ("verified_success_rate", _verified)):
        metrics[name].update(baseline_ci=side_ci(0, stat), candidate_ci=side_ci(1, stat), delta_ci=delta_ci(stat))

    latency = {}
    for name, value in (("queue", lambda r: r.queue_ms), ("execution", lambda r: r.execution_ms)):
        b, c = distribution(map(value, baseline)), distribution(map(value, candidate))
        latency[name] = {
            "baseline": b,
            "candidate": c,
            "delta_mean": _diff(c["mean"], b["mean"]),
            "delta_median": _diff(c["median"], b["median"]),
        }

    base_tools, cand_tools = tool_metrics(baseline), tool_metrics(candidate)
    tools = {
        name: {
            "baseline": base_tools[name],
            "candidate": cand_tools[name],
            "delta": _diff(cand_tools[name]["value"], base_tools[name]["value"]),
        }
        for name in TOOL_METRICS
    }

    base_cost, base_per, _ = _cost(baseline)
    cand_cost, cand_per, _ = _cost(candidate)
    return {
        "summary": {
            "tasks_compared": len(tasks),
            "wins": kinds["win"],
            "losses": kinds["loss"],
            "ties": kinds["tie"],
            "regressions": sum(t["regression"] for t in tasks),
            "improvements": sum(t["improvement"] for t in tasks),
            "regressed_tasks": [t["task_id"] for t in tasks if t["regression"]],
            "improved_tasks": [t["task_id"] for t in tasks if t["improvement"]],
            "baseline_only_tasks": sorted(set(base_tasks) - set(cand_tasks)),
            "candidate_only_tasks": sorted(set(cand_tasks) - set(base_tasks)),
        },
        "metrics": metrics,
        "tool_metrics": tools,
        "latency_ms": latency,
        "cost": {
            "baseline": usage(baseline),
            "candidate": usage(candidate),
            "delta_cost_usd": _money_diff(cand_cost, base_cost),
            "delta_cost_per_successful_run_usd": _money_diff(cand_per, base_per),
        },
        "tasks": tasks,
        "bootstrap": {**boot, "unit": "task", "method": "percentile, paired task resampling"},
        "definitions": DEFINITIONS,
    }
