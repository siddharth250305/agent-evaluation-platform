"""The score contract shared by every evaluator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvaluationStatus = Literal["pending", "pass", "fail", "partial", "not_applicable", "evaluation_error"]


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_name: str
    evaluator_version: str
    status: EvaluationStatus
    value: float | None = Field(allow_inf_nan=False)  # None when the evaluator could not decide (evaluation_error)
    reason_code: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    applicability: Literal["applicable", "not_applicable"] = "applicable"

    @model_validator(mode="after")
    def error_has_no_value(self) -> "Score":
        if self.status == "evaluation_error" and self.value is not None:
            raise ValueError("evaluation_error must have no value")
        return self
