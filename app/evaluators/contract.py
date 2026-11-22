"""What every evaluator receives and returns, and how crashes are contained."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.evaluators.score import EvaluationStatus, Score
from app.tasks import EvaluatorTaskView
from app.trace import TraceEvent


@dataclass(frozen=True)
class EvaluationContext:
    """Hidden task spec plus the attempt's evidence: initial and final state and the ordered trace."""

    task: EvaluatorTaskView
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    events: tuple[TraceEvent, ...]


class Evaluator(Protocol):
    name: str
    version: str  # semver; bump on any change that can alter a score

    def evaluate(self, ctx: EvaluationContext) -> Score: ...


def make_score(
    evaluator: Evaluator,
    status: EvaluationStatus,
    value: float | None,
    reason_code: str,
    evidence: dict[str, Any] | None = None,
    *,
    applicability: Literal["applicable", "not_applicable"] = "applicable",
) -> Score:
    return Score(
        evaluator_name=evaluator.name,
        evaluator_version=evaluator.version,
        status=status,
        value=value,
        reason_code=reason_code,
        evidence=evidence or {},
        applicability=applicability,
    )


def run_evaluator(evaluator: Evaluator, ctx: EvaluationContext) -> Score:
    """Run one evaluator. Any exception, or a malformed result, becomes ``evaluation_error``."""
    try:
        score = evaluator.evaluate(ctx)
        if not isinstance(score, Score):
            raise TypeError(f"evaluator returned {type(score).__name__}, not Score")
        if (score.evaluator_name, score.evaluator_version) != (evaluator.name, evaluator.version):
            raise ValueError(
                f"score claims {score.evaluator_name}@{score.evaluator_version}, "
                f"expected {evaluator.name}@{evaluator.version}"
            )
        return score
    except Exception as exc:  # noqa: BLE001 - an evaluator crash is never a pass or a fail
        return Score(
            evaluator_name=evaluator.name,
            evaluator_version=evaluator.version,
            status="evaluation_error",
            value=None,
            reason_code="EVALUATOR_EXCEPTION",
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
        )
