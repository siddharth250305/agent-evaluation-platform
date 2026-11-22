"""Dataset versions, agent configurations and price versions: the immutable inputs of an experiment."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.accounting import PRICE_VERSIONS
from app.agents.registry import MODEL_ADAPTERS, limits_from
from app.datasets import MANIFEST_FILE, DatasetError, load_dataset
from app.db.models import AgentConfig, Dataset, DatasetVersion, PriceVersion, TaskVersion
from app.services.errors import Conflict, InvalidRequest, NotFound
from app.services.hashing import sha256_json, sha256_text
from app.tasks import DATASETS_DIR, load_task
from app.tools.schemas import TOOLS

_DATASET_NAME = re.compile(r"^[a-z0-9_]+$")


# --- price versions ------------------------------------------------------------


def sync_price_versions(session: Session) -> None:
    """Insert the dated price versions from code; a changed existing row is an error, never an update."""
    for price in PRICE_VERSIONS:
        values = {f.name: getattr(price, f.name) for f in dataclasses.fields(price)}
        session.execute(insert(PriceVersion).values(**values).on_conflict_do_nothing())
        stored = session.scalars(
            select(PriceVersion).where(
                PriceVersion.price_version_id == price.price_version_id, PriceVersion.model == price.model
            )
        ).one()
        if any(getattr(stored, key) != value for key, value in values.items()):
            raise RuntimeError(
                f"price version {price.price_version_id}/{price.model} changed; add a new dated version instead"
            )


def price_version_pk(session: Session, price_version_id: str | None, model: str | None) -> int | None:
    if not price_version_id or not model:
        return None
    return session.scalar(
        select(PriceVersion.id).where(PriceVersion.price_version_id == price_version_id, PriceVersion.model == model)
    )


# --- datasets ------------------------------------------------------------------


def import_dataset(session: Session, name: str, *, datasets_dir: Path = DATASETS_DIR) -> tuple[DatasetVersion, bool]:
    """Validate ``datasets/<name>`` and store it as an immutable version. Same content -> same version."""
    if not _DATASET_NAME.fullmatch(name):
        raise InvalidRequest(f"invalid dataset name {name!r}")
    directory = datasets_dir / name
    if not (directory / MANIFEST_FILE).is_file():
        raise NotFound(f"dataset {name!r} not found")
    try:
        loaded = load_dataset(directory)
    except DatasetError as exc:
        raise InvalidRequest(f"dataset {name!r} is invalid: {exc}") from exc

    dataset = _get_or_create_dataset(session, loaded.name)
    existing = _version_by_hash(session, dataset.id, loaded.content_hash)
    if existing is not None:
        return existing, False

    version = DatasetVersion(
        dataset_id=dataset.id,
        content_hash=loaded.content_hash,
        split_seed=loaded.split_seed,
        task_count=len(loaded.tasks),
        manifest={
            "dataset": loaded.name,
            "split_seed": loaded.split_seed,
            "families": {t.family: t.split for t in loaded.tasks},
            "tasks": [
                {"task_id": t.task_id, "family": t.family, "split": t.split, "content_hash": t.content_hash}
                for t in loaded.tasks
            ],
        },
    )
    try:
        with session.begin_nested():
            session.add(version)
            session.flush()
            for task in loaded.tasks:
                views = load_task(task.path)
                session.add(
                    TaskVersion(
                        dataset_version_id=version.id,
                        task_id=task.task_id,
                        version=views.file.version,
                        family=task.family,
                        category=task.category,
                        split=task.split,
                        paraphrase_of=task.paraphrase_of,
                        review_status=task.review_status,
                        content_hash=task.content_hash,
                        agent_view=views.agent_view.model_dump(mode="json"),
                        evaluator_view=views.evaluator_view.model_dump(mode="json"),
                    )
                )
            session.flush()
    except IntegrityError:
        # A concurrent import of the same content won; it is the same immutable version.
        existing = _version_by_hash(session, dataset.id, loaded.content_hash)
        if existing is None:
            raise
        return existing, False
    return version, True


def _get_or_create_dataset(session: Session, name: str) -> Dataset:
    dataset = session.scalar(select(Dataset).where(Dataset.name == name))
    if dataset is not None:
        return dataset
    try:
        with session.begin_nested():
            dataset = Dataset(name=name)
            session.add(dataset)
            session.flush()
    except IntegrityError:
        dataset = session.scalars(select(Dataset).where(Dataset.name == name)).one()
    return dataset


def _version_by_hash(session: Session, dataset_id: Any, content_hash: str) -> DatasetVersion | None:
    return session.scalar(
        select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id, DatasetVersion.content_hash == content_hash)
    )


def split_counts(version: DatasetVersion) -> dict[str, int]:
    counts = Counter(task["split"] for task in version.manifest["tasks"])
    return {split: counts.get(split, 0) for split in ("dev", "validation", "test")}


# --- agent configurations ----------------------------------------------------------


class AgentConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=100)
    adapter: str = Field(min_length=1, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    prompt: str | None = Field(default=None, min_length=1)
    tools: list[str] | None = None  # None: every tool the task offers
    # validate_default: an omitted ``limits`` is canonicalised exactly like an explicit {}.
    limits: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tools")
    @classmethod
    def _known_unique_tools(cls, tools: list[str] | None) -> list[str] | None:
        if tools is None:
            return None
        unknown = sorted(set(tools) - set(TOOLS))
        if unknown:
            raise ValueError(f"unknown tools: {unknown}")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must be unique")
        return tools

    @field_validator("limits")
    @classmethod
    def _canonical_limits(cls, limits: dict[str, Any]) -> dict[str, Any]:
        # Defaults filled in, so {} and the explicit defaults are the same configuration.
        resolved = dataclasses.asdict(limits_from(limits))
        if resolved["max_cost_usd"] is not None:
            # 0.5, "0.5" and "0.50" are one budget, so they must hash the same.
            resolved["max_cost_usd"] = format(resolved["max_cost_usd"].normalize(), "f")
        return resolved


def register_agent_config(
    session: Session, spec: AgentConfigIn, *, known_adapters: Collection[str]
) -> tuple[AgentConfig, bool]:
    """Register an immutable configuration. Same name and content -> existing; same name, new content -> 409."""
    if spec.adapter not in known_adapters:
        raise InvalidRequest(f"unknown adapter {spec.adapter!r}; known: {sorted(known_adapters)}")
    if spec.adapter in MODEL_ADAPTERS and not spec.model:
        raise InvalidRequest(f"adapter {spec.adapter!r} needs a model")

    content = spec.model_dump(mode="json", exclude={"name"})
    config_hash = sha256_json(content)
    existing = session.scalar(select(AgentConfig).where(AgentConfig.name == spec.name))
    if existing is not None:
        return _same_config(existing, config_hash), False
    config = AgentConfig(
        name=spec.name,
        adapter=spec.adapter,
        model=spec.model,
        prompt=spec.prompt,
        prompt_hash=sha256_text(spec.prompt) if spec.prompt is not None else None,
        tools=spec.tools,
        limits=spec.limits,
        settings=spec.settings,
        config_hash=config_hash,
    )
    try:
        with session.begin_nested():
            session.add(config)
            session.flush()
    except IntegrityError:
        existing = session.scalars(select(AgentConfig).where(AgentConfig.name == spec.name)).one()
        return _same_config(existing, config_hash), False
    return config, True


def _same_config(existing: AgentConfig, config_hash: str) -> AgentConfig:
    if existing.config_hash != config_hash:
        raise Conflict(
            f"agent configuration {existing.name!r} already exists with different content; "
            "configurations are immutable, register a new name"
        )
    return existing


def tool_schema_hash(tools: list[str] | None) -> str:
    names = tools if tools is not None else sorted(TOOLS)
    return sha256_json([TOOLS[name].spec().model_dump(mode="json") for name in names])
