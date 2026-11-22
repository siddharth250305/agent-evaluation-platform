"""Dataset versions: validated task files, content hashes, and family-level splits.

Splits are assigned per scenario *family* (60/20/20 dev/validation/test)
before any paraphrasing, and recorded in ``splits.yaml``. Every task inherits
its family's split, so paraphrases of one scenario can never leak across splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.task_spec import Category
from app.tasks import load_task

Split = Literal["dev", "validation", "test"]
SPLITS: tuple[Split, ...] = ("dev", "validation", "test")
DEFAULT_RATIOS: dict[Split, float] = {"dev": 0.6, "validation": 0.2, "test": 0.2}
MANIFEST_FILE = "splits.yaml"


class DatasetError(ValueError):
    """The dataset on disk is inconsistent; the message says where."""


class _SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    split_seed: str
    ratios: dict[Split, float] = Field(default_factory=lambda: dict(DEFAULT_RATIOS))
    families: dict[str, Split]

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> _SplitManifest:
        if (set(self.ratios) != set(SPLITS) or any(not 0 < v < 1 for v in self.ratios.values())
                or abs(sum(self.ratios.values()) - 1.0) > 1e-9):
            raise ValueError("ratios must cover dev, validation and test and sum to 1")
        return self


@dataclass(frozen=True)
class DatasetTask:
    task_id: str
    path: Path
    family: str
    category: Category
    split: Split
    paraphrase_of: str | None
    review_status: str
    content_hash: str


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    split_seed: str
    content_hash: str
    tasks: tuple[DatasetTask, ...]

    def task(self, task_id: str) -> DatasetTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def split(self, split: Split) -> tuple[DatasetTask, ...]:
        return tuple(t for t in self.tasks if t.split == split)


def assign_splits(
    families: Iterable[str],
    *,
    seed: str,
    ratios: dict[Split, float] = DEFAULT_RATIOS,
    existing: dict[str, Split] | None = None,
) -> dict[str, Split]:
    """Deterministic family -> split assignment.

    Fresh: shuffle families by seeded hash and cut 60/20/20, with at least one
    family per split. With ``existing``: keep every existing assignment and put
    each new family in the split furthest below its target share, so adding a
    family never moves an old one (which would leak test scenarios into dev).
    """
    names = set(families)

    def by_hash(family: str) -> str:
        return hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()

    if existing:
        result = {f: s for f, s in existing.items() if f in names}
        for family in sorted(names - set(result), key=by_hash):
            counts = Counter(result.values())
            total = len(result) + 1
            result[family] = max(SPLITS, key=lambda s: (ratios[s] * total - counts[s], -SPLITS.index(s)))
        return result

    ordered = sorted(names, key=by_hash)
    n = len(ordered)
    n_validation = max(1, round(n * ratios["validation"]))
    n_test = max(1, round(n * ratios["test"]))
    n_dev = n - n_validation - n_test
    if n_dev < 1:
        raise ValueError(f"need at least 3 families to fill dev, validation and test (got {n})")
    return {
        family: "dev" if i < n_dev else "validation" if i < n_dev + n_validation else "test"
        for i, family in enumerate(ordered)
    }


def load_dataset(directory: Path) -> DatasetVersion:
    manifest = _load_manifest(directory / MANIFEST_FILE)
    loaded = {}
    for path in sorted(directory.glob("*.yaml")):
        if path.name == MANIFEST_FILE:
            continue
        try:
            task = load_task(path)
        except (ValidationError, ValueError, OSError, yaml.YAMLError) as exc:
            raise DatasetError(f"{path.name}: {exc}") from exc
        if task.file.task_id != path.stem:
            raise DatasetError(f"{path.name}: task_id {task.file.task_id!r} does not match its file name")
        loaded[task.file.task_id] = (path, task)

    _check_paraphrases(loaded)
    _check_families(manifest, {task.file.family for _, task in loaded.values()})

    tasks = tuple(
        DatasetTask(
            task_id=task_id,
            path=path,
            family=task.file.family,
            category=task.file.category,
            split=manifest.families[task.file.family],
            paraphrase_of=task.file.paraphrase_of,
            review_status=task.file.review.status,
            content_hash=_hash(
                {
                    "agent_view": task.agent_view.model_dump(mode="json"),
                    "evaluator_view": task.evaluator_view.model_dump(mode="json"),
                    "paraphrase_of": task.file.paraphrase_of,
                }
            ),
        )
        for task_id, (path, task) in sorted(loaded.items())
    )
    return DatasetVersion(
        name=manifest.dataset,
        split_seed=manifest.split_seed,
        content_hash=_hash(
            {
                "dataset": manifest.dataset,
                "split_seed": manifest.split_seed,
                "tasks": [[t.task_id, t.split, t.content_hash] for t in tasks],
            }
        ),
        tasks=tasks,
    )


def _load_manifest(path: Path) -> _SplitManifest:
    try:
        with path.open(encoding="utf-8") as f:
            return _SplitManifest.model_validate(yaml.safe_load(f))
    except (ValidationError, OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"{path.name}: {exc}") from exc


def _check_paraphrases(loaded: dict[str, Any]) -> None:
    # Exact normalized prompts in different families are leakage even without a declared link.
    prompts: dict[str, str] = {}
    categories: dict[str, str] = {}
    for task_id, (_, task) in loaded.items():
        previous_category = categories.setdefault(task.file.family, task.file.category)
        if previous_category != task.file.category:
            raise DatasetError(f"{task_id}: inconsistent category within family {task.file.family}")
        prompt = " ".join(task.agent_view.user_request.casefold().split())
        previous = prompts.get(prompt)
        if previous and loaded[previous][1].file.family != task.file.family:
            raise DatasetError(f"{task_id}: duplicate prompt across families ({previous})")
        prompts[prompt] = task_id
    for task_id, (_, task) in loaded.items():
        target = task.file.paraphrase_of
        if target is None:
            continue
        if target not in loaded or target == task_id:
            raise DatasetError(f"{task_id}: paraphrase_of {target!r} is not another task in this dataset")
        seen = {task_id}
        cursor = target
        while cursor is not None and cursor in loaded:
            if cursor in seen:
                raise DatasetError(f"{task_id}: paraphrase cycle")
            seen.add(cursor)
            cursor = loaded[cursor][1].file.paraphrase_of
        if loaded[target][1].file.family != task.file.family:
            raise DatasetError(f"{task_id}: paraphrase_of {target!r} is in a different family")


def _check_families(manifest: _SplitManifest, families: set[str]) -> None:
    missing, extra = families - set(manifest.families), set(manifest.families) - families
    if missing:
        raise DatasetError(f"families without a split in {MANIFEST_FILE}: {sorted(missing)}")
    if extra:
        raise DatasetError(f"{MANIFEST_FILE} lists families with no tasks: {sorted(extra)}")
    counts = Counter(manifest.families.values())
    n = len(families)
    empty = [split for split in SPLITS if counts[split] == 0]
    if empty:
        raise DatasetError(f"splits with no families: {empty}")
    for split in SPLITS:
        target = manifest.ratios[split] * n
        if abs(counts[split] - target) > 1:  # +-1 absorbs rounding
            raise DatasetError(
                f"split ratio violated: {counts[split]} {split} families, expected about {target:g} of {n}"
            )


def _hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
