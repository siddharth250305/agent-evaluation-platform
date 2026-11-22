# Phase 5 TDD evidence: comparison, export and dashboard

**Source plan:** the Phase 5 request (comparison engine, CSV/JSON export with manifest, 5-screen Jinja2 dashboard, tests) plus PRD §6 (FR-25/26/27), §7 metric definitions and §11. No `*.plan.md`. The repo is not under git, so this report is the record.

## User journeys

1. As an FDE, I want to compare config A vs B on the same tasks and name exactly which tasks improved or regressed (US-6, Phase 5 gate).
2. As a stakeholder, I want success, violations, latency and cost as separate numbers with their denominators, never one blended score (US-7).
3. As anyone, I want an export that carries the manifest needed to reproduce it (US-9).
4. As an agent engineer, I want to open a failed run and see what the agent saw, what it did, and why it failed, in under a minute (US-5).
5. As a reviewer, I want public benchmark numbers kept apart from custom support scores (US-10).

## What was built

| Piece | Where |
|---|---|
| Pure scorecard / comparison / task-level bootstrap math | `app/metrics.py` |
| Run facts loader (runs, attempts, scores) | `app/services/results.py` |
| `POST /comparisons` (compatibility checks, deterministic, not stored) | `app/services/comparisons.py`, `app/api/routes.py` |
| `GET /experiments/{id}/export?format=json\|csv` | `app/services/reports.py`, `app/api/routes.py` |
| Manifest now records `fault_config` | `app/services/experiments.py` |
| Dashboard: `/ui/experiments`, `/ui/experiments/{id}`, `/ui/runs/{id}`, `/ui/compare`, `/ui/benchmarks` | `app/web/` |
| `benchmark_results` table for Phase 6 BFCL import | `app/db/models.py`, `migrations/versions/0002_benchmark_results.py` |

## Definitions chosen

- **Compatible** = same dataset version, split, environment version, evaluator versions and agent-config limits; otherwise 422 listing every difference.
- **Win / loss / tie** by strict per-task pass rate (passed repetitions / planned repetitions), compared by cross-multiplication.
- **Regression** = baseline passed every repetition and candidate failed at least one (a definite `fail`; an `evaluation_error` is a loss, not a regression). **Improvement** is the mirror.
- **Bootstrap**: percentile, paired resampling of tasks (repetitions stay together), seeded, 2000 resamples by default.
- **Cost** sums every attempt that started and finished, infrastructure retries included; one unknown attempt makes the total, the per-success cost and the delta unknown (`null`), never `0`.
- **Latency**: queue = run created → first attempt started (database clock); execution = the result attempt's monotonic `duration_ms`.

## Results

| Step | Command | Result |
|---|---|---|
| Baseline before Phase 5 | `pytest -q` (local Postgres 16 up) | `555 passed, 1 xfailed` |
| New unit tests | `pytest -q tests/unit/test_metrics.py tests/unit/test_reporting_offline.py` | `42 passed` |
| Mutation check | regression rule loosened / unknown cost summed as 0 / bootstrap without resampling | each caught (1, 2, 2 failures) |
| New Postgres tests (first run) | `REQUIRE_POSTGRES=1 pytest tests/postgres/test_phase5_*.py` | 4 failures, all test bugs: `refund_eligible_001` is not in the dev split; a selector assumed `id` came before `class`; a raw `&` in an href (template fixed to `&amp;`) |
| GREEN | `pytest -q` | `628 passed, 1 xfailed` |
| GREEN | `make test-db` | `109 passed, 1 xfailed` |
| Offline | `pytest -q` with Postgres unreachable | `520 passed, 109 skipped` |

## Known gaps

- Comparisons are computed on request, not stored; `GET /comparisons/{id}` would need a table.
- The dashboard has no authentication (PRD §10: auth before a public dashboard). Bind the API to localhost.
- `git_commit` comes from `GIT_COMMIT` and `lockfile_sha256` from `uv.lock`; both are `null` in this checkout (no git repo, no lockfile).
- The run list renders at most 500 rows per filter; the API's paginated `/runs` has no cap.
- "Second prompt, second model" configurations (PRD §11 Phase 5) are registrations, not code; real-model comparisons need an API key and stay out of the deterministic suite.
