"""Task loading with strict view separation.

A task file has an agent-visible ``agent`` section and evaluator-only sections:
``fixture`` (initial state, inline or a reference to a shared world),
``expected`` (outcome), ``trajectory`` (tool-use constraints) and
``reference`` (a known-good solution). Only the ``agent`` section can ever
reach an ``AgentTaskView``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.task_spec import Category, ExpectedOutcome, ReferenceSolution, Review, TrajectorySpec
from app.tools.schemas import TOOLS

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

_SLUG = r"^[a-z0-9_]+$"
_SLUG_RE = re.compile(_SLUG)
_FIXTURE_REF_KEYS = {"ref", "session", "overrides"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _AgentSection(_Strict):
    user_request: str = Field(min_length=1)
    tools: tuple[str, ...] = Field(min_length=1)

    @field_validator("tools")
    @classmethod
    def _known_unique_tools(cls, tools: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [t for t in tools if t not in TOOLS]
        if unknown:
            raise ValueError(f"unknown tools: {unknown}")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must be unique")
        return tools


class TaskFile(_Strict):
    task_id: str = Field(pattern=_SLUG)
    version: int = Field(ge=1)
    family: str = Field(pattern=_SLUG)
    category: Category
    paraphrase_of: str | None = None
    review: Review
    agent: _AgentSection
    fixture: dict[str, Any]
    expected: dict[str, Any]
    trajectory: TrajectorySpec
    reference: ReferenceSolution

    @field_validator("expected")
    @classmethod
    def _valid_expected(cls, expected: dict[str, Any]) -> dict[str, Any]:
        try:
            ExpectedOutcome.model_validate(expected)
        except ValidationError as exc:
            raise ValueError(str(exc)) from None
        return expected

    @model_validator(mode="after")
    def _tools_are_available(self) -> TaskFile:
        available = set(self.agent.tools)
        used = {c.tool for c in self.trajectory.required_calls} | {c.tool for c in self.reference.calls}
        if used - available:
            raise ValueError(f"trajectory/reference use tools the agent does not have: {sorted(used - available)}")
        return self


class AgentTaskView(_Strict):
    """Everything the agent is allowed to see, and nothing else.

    ``task_id`` is deliberately absent: ids like ``refund_eligible_001`` hint
    at the expected outcome.
    """

    user_request: str
    tools: tuple[str, ...]


class EvaluatorTaskView(_Strict):
    """Evaluator-only view: resolved fixture, expected outcome, trajectory spec, reference."""

    task_id: str
    version: int
    family: str
    category: Category
    fixture: dict[str, Any]
    expected: dict[str, Any]
    trajectory: TrajectorySpec
    reference: ReferenceSolution


@dataclass(frozen=True)
class LoadedTask:
    file: TaskFile
    agent_view: AgentTaskView
    evaluator_view: EvaluatorTaskView


def task_path(task_id: str, dataset: str = "support") -> Path:
    return DATASETS_DIR / dataset / f"{task_id}.yaml"


def load_task(path: Path) -> LoadedTask:
    with path.open(encoding="utf-8") as f:
        task = TaskFile.model_validate(yaml.safe_load(f))
    return LoadedTask(
        file=task,
        agent_view=AgentTaskView(user_request=task.agent.user_request, tools=task.agent.tools),
        evaluator_view=EvaluatorTaskView(
            task_id=task.task_id,
            version=task.version,
            family=task.family,
            category=task.category,
            fixture=_resolve_fixture(task.fixture, path),
            expected=task.expected,
            trajectory=task.trajectory,
            reference=task.reference,
        ),
    )


def load_agent_view(path: Path) -> AgentTaskView:
    return load_task(path).agent_view


def load_evaluator_view(path: Path) -> EvaluatorTaskView:
    return load_task(path).evaluator_view


def _resolve_fixture(fixture: dict[str, Any], path: Path) -> dict[str, Any]:
    """Inline fixtures pass through; ``{ref, session, overrides}`` expands a shared world file.

    Overrides merge per entity: ``orders: {ORD-1: {status: x}}`` changes one field of
    ORD-1 and keeps the rest. Non-mapping values (lists, scalars) replace outright.
    """
    if "ref" not in fixture:
        return fixture
    unknown = set(fixture) - _FIXTURE_REF_KEYS
    if unknown:
        raise ValueError(f"unknown fixture reference keys: {sorted(unknown)}")
    ref = fixture["ref"]
    if not isinstance(ref, str) or not _SLUG_RE.fullmatch(ref):
        raise ValueError(f"invalid fixture ref {ref!r}")
    session = fixture.get("session")
    if not isinstance(session, dict):
        raise ValueError("a fixture reference needs a session mapping")
    overrides = fixture.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("fixture overrides must be a mapping")

    with (path.parent / "fixtures" / f"{ref}.yaml").open(encoding="utf-8") as f:
        world = yaml.safe_load(f)
    if not isinstance(world, dict):
        raise ValueError(f"fixture {ref!r} is empty or not a mapping")
    for key, value in overrides.items():
        base = world.get(key)
        if isinstance(base, dict):
            if not isinstance(value, dict):
                raise ValueError(f"override for {key!r} must be a mapping")
            world[key] = {
                **base,
                **{k: {**base[k], **v} if isinstance(base.get(k), dict) and isinstance(v, dict) else v
                   for k, v in value.items()},
            }
        else:
            world[key] = value
    world["session"] = session
    return world
