from app.evaluators.contract import EvaluationContext, Evaluator, run_evaluator
from app.evaluators.outcome import OutcomeEvaluator, evaluate_outcome
from app.evaluators.policy import PolicyEvaluator
from app.evaluators.response import ResponseEvaluator
from app.evaluators.score import EvaluationStatus, Score
from app.evaluators.trajectory import TrajectoryEvaluator

DEFAULT_EVALUATORS: tuple[Evaluator, ...] = (
    OutcomeEvaluator(),
    PolicyEvaluator(),
    TrajectoryEvaluator(),
    ResponseEvaluator(),
)


def evaluate_all(
    ctx: EvaluationContext, evaluators: tuple[Evaluator, ...] = DEFAULT_EVALUATORS
) -> tuple[Score, ...]:
    return tuple(run_evaluator(evaluator, ctx) for evaluator in evaluators)


__all__ = [
    "DEFAULT_EVALUATORS",
    "EvaluationContext",
    "EvaluationStatus",
    "Evaluator",
    "OutcomeEvaluator",
    "PolicyEvaluator",
    "ResponseEvaluator",
    "Score",
    "TrajectoryEvaluator",
    "evaluate_all",
    "evaluate_outcome",
    "run_evaluator",
]
