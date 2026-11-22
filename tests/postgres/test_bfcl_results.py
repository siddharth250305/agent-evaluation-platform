"""BFCL results in PostgreSQL and on /ui/benchmarks: subset labels, kind labels, raw artifacts. Zero paid calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from app.benchmarks import bfcl
from app.db.models import BenchmarkResult
from app.web import routes
from tests.postgres.helpers import client_for

PIN = bfcl.load_pin()


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(routes, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


def _self_check(n: int, artifacts: Path) -> tuple[bfcl.BenchmarkRun, dict]:
    entries = bfcl.select_subset(bfcl.load_category(PIN), n)
    run = bfcl.run_entries(entries, bfcl.GroundTruthReplayProvider(entries), kind="scorer_self_check", pin=PIN)
    _, manifest = bfcl.write_artifacts(run, artifacts)
    return run, manifest


def test_a_stored_subset_is_labeled_and_its_raw_outputs_are_downloadable(sm, artifacts: Path) -> None:
    run, manifest = _self_check(5, artifacts)
    with sm.begin() as session:
        stored = bfcl.save_result(session, run, manifest)
        result_id = stored.id
    with sm() as session:
        row = session.get(BenchmarkResult, result_id)
        assert (row.task_count, row.category_task_count, row.subset_method) == (5, 400, bfcl.SUBSET_METHOD)
        assert row.result_kind == "scorer_self_check"
        assert row.benchmark_version == f"v4@{PIN['commit']}"
        assert row.metrics["accuracy"] == {"numerator": 5, "denominator": 5, "value": 1.0}

    client = client_for(sm)
    html = client.get("/ui/benchmarks").text
    assert "subset 5/400" in html and "scorer self-check" in html and "not a model score" in html
    assert PIN["commit"] in html and "not leaderboard-comparable" in html
    assert f"/ui/benchmarks/{result_id}/raw" in html

    raw = client.get(f"/ui/benchmarks/{result_id}/raw")
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("application/x-ndjson")
    assert raw.headers["x-content-type-options"] == "nosniff"
    assert raw.text.count("\n") == 5


def test_the_raw_route_only_serves_files_inside_the_artifacts_directory(sm, artifacts: Path) -> None:
    with sm.begin() as session:
        outside = BenchmarkResult(
            benchmark="BFCL", benchmark_version="v4@x", category="simple_python", task_count=0,
            scoring_method="m", metrics={}, raw_artifact_uri="pyproject.toml", manifest={},
        )
        session.add(outside)
        session.flush()
        outside_id = outside.id
    client = client_for(sm)
    assert client.get(f"/ui/benchmarks/{outside_id}/raw").status_code == 404
    assert client.get("/ui/benchmarks/00000000-0000-0000-0000-000000000000/raw").status_code == 404


def test_an_unlabeled_subset_cannot_be_stored(sm) -> None:
    with pytest.raises(IntegrityError, match="subset_is_labeled"):
        with sm.begin() as session:
            session.add(BenchmarkResult(
                benchmark="BFCL", benchmark_version="v4@x", category="simple_python", task_count=5,
                category_task_count=400, scoring_method="m", metrics={}, manifest={},
            ))
