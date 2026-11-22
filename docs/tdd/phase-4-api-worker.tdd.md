# Phase 4 TDD evidence: API, PostgreSQL queue and workers

**Source plan:** the Phase 4 request (Alembic + SQLAlchemy schema, FastAPI endpoints with idempotent experiment creation, SKIP LOCKED worker with leases, heartbeats, sweeper, cancellation and bounded concurrency, integration and recovery tests) plus PRD §8–§11. No `*.plan.md`. The repo is not under git, so this report is the RED/GREEN record.

## User journeys

1. As an agent engineer, I want to launch an experiment and close my laptop, so long runs finish in the background (US-3).
2. As an agent engineer, I want a retried `POST /experiments` to never create a second experiment, so a flaky network can't double my spend.
3. As anyone watching an experiment, I want a killed worker to show up as an interrupted attempt followed by a clean replacement, never as a silent loss or a duplicate result (PRD demo 3).
4. As an agent engineer, I want to cancel a bad experiment and have no further tool calls executed (FR-22).
5. As an eval engineer, I want results tied to immutable dataset versions and agent configurations, with a manifest that pins everything needed to reproduce them.

## RED → GREEN

| Step | Command | Result |
|---|---|---|
| RED (compile-time) | `pytest -q tests/unit/test_phase4_runtime.py tests/postgres` before any implementation | `ModuleNotFoundError: No module named 'app.db'` / `'app.agents.registry'`, the intended missing implementation |
| GREEN (offline) | `pytest -q` after runner, registry and services | `472 passed, 55 skipped` (Postgres unavailable: Docker Desktop unresponsive) |
| Review round 1 | `ecc:python-reviewer` ran the suite on a local Postgres 15 | `525 passed, 2 failed`, both test bugs; plus 2 high, 7 medium and 10 low findings (below) |
| RED (real DB, after fixes) | `REQUIRE_POSTGRES=1 pytest -q` against local Postgres 15.14 | `5 failed, 527 passed`: **deadlock** between finalizers (4) introduced by the review's M4 lock reordering; the CHECK-constraint test also matched catalog domains (1) |
| GREEN | same, twice | `532 passed` ×2; 0 deadlocks after the fix (34 before, all in the failing run) |
| Compose | `make db-up` then `make test-db` against `compose.yaml` Postgres 16 | container `healthy` in ~6 s (`pg_isready`: accepting connections; port bound to `127.0.0.1` only); `78 passed, 1 xfailed` (includes `test_phase4_audit.py`, added outside this session) |
| Live smoke | `make migrate`, `uvicorn --factory app.api.app:create_app`, `python -m app.worker`, curl | ready at `0001_initial`; replay 200, conflicting replay 409; 20/20 runs completed; honest 0.6, lying 0.0 verified success; no errors in API or worker logs |

## Review fixes

| Finding | Severity | Fix |
|---|---|---|
| NUL bytes / NaN / over-long names from the agent made `finalize_attempt` raise; lease expired and the sweeper re-ran the (paid) agent 3×, mislabelled `WORKER_LEASE_EXPIRED`, usage lost | high | Payloads sanitised before writing (`\x00` → U+FFFD, NaN/Inf → null, lengths truncated); any other persistence failure is recorded once as `PERSIST_FAILED` **with its usage**, never retried |
| Provider errors retried even when the provider said not to (400, 401, missing key) | high | Only `provider_error` with `retryable=True` gets a new attempt; adapter exceptions are no longer retried (decision: agent code that crashed will most likely crash again and may bill again) |
| `compare_metadata` cannot see CHECK constraints, partial-index predicates or server defaults | medium | Compare types and server defaults; assert every CHECK definition against the models; assert both partial-index predicates |
| Python `None` stored as JSON `null` | medium | `JSONB(none_as_null=True)` on nullable JSONB columns |
| A failing experiment-status update after a claim stranded the lease | medium | Logged and ignored (it is display status only) |
| Finalizers held the experiment lock while inserting traces | medium | Lock taken after persisting; **then** found to deadlock (FK `KEY SHARE` vs `FOR UPDATE`) and fixed with `FOR NO KEY UPDATE` on every experiment lock |
| Vacuous test; missing coverage for lost heartbeat, provider retry policy, hostile output, persist failure | medium | Vacuous test removed; 5 tests added |
| Whole-table aggregates in list endpoints; missing FK indexes | medium | Correlated subqueries; indexes on `scores.run_id`, `runs.agent_config_id`, `runs.task_version_id`, `jobs.current_attempt_id` |
| Low items: never-started attempts counted as unknown cost, spurious "lost lease" log, inconsistent backoff, client vs DB clock, `synchronize_session` fetch, `0.5` vs `"0.50"` budgets hashing differently, Alembic `path_separator` | low | All fixed |

## Guarantees

| # | What is guaranteed | Test | Type |
|---|---|---|---|
| 1 | Migration creates all 13 tables; models and migration agree on columns, types, defaults, CHECKs and partial indexes; downgrade/upgrade round-trips | `tests/postgres/test_migrations.py` (6) | integration |
| 2 | PRD unique constraints: `runs(experiment, config, task_version, repetition)`, `attempts(run, attempt_number)`, `trace_events(attempt, sequence)`, `scores(attempt, evaluator, version, dimension)` | `test_required_unique_constraints` | integration |
| 3 | `POST /experiments` creates experiment, runs and jobs in one transaction; a failure mid-way leaves nothing | `test_creates_experiment_runs_and_jobs_together`, `test_creation_is_all_or_nothing` | integration |
| 4 | Missing `Idempotency-Key` → 400; same key + payload → same experiment (200); different payload → 409; 4 concurrent same-key requests → 1 experiment | `test_api_experiments.py` (4) | integration |
| 5 | Datasets import as immutable, hash-identified versions; stored agent view holds only `user_request` and `tools`; agent configs are immutable (409 on change) | `test_api_catalog.py` | integration |
| 6 | `/health/ready` is 200 only with a reachable DB at the head migration, 503 otherwise | `test_health_live_and_ready`, `test_ready_is_503_when_the_database_is_unreachable` | integration |
| 7 | Claims use SKIP LOCKED (never block on a locked job); concurrent claims never share a job | `test_claim_skips_rows_locked_by_another_transaction`, `test_concurrent_claims_never_share_a_job` | integration |
| 8 | A stale lease can never write results; a worker that loses its lease mid-run executes no further tools | `test_a_stale_worker_cannot_write_results`, `test_a_worker_that_loses_its_lease_stops_and_writes_nothing` | recovery |
| 9 | The sweeper marks the attempt `WORKER_LEASE_EXPIRED` and queues a replacement with a new `environment_id`, which runs on a clean fixture state | `test_recovery.py` (3) | recovery |
| 10 | `kill -9` of a real worker process → interrupted attempts, clean replacements, exactly one scored attempt per run | `test_killed_worker_process_yields_interrupted_attempt_and_clean_recovery` | recovery |
| 11 | An API restart loses no queued work | `test_api_restart_loses_no_queued_work` | recovery |
| 12 | Cancellation cancels queued runs at once and stops a running attempt before its next tool call; no scores for cancelled work | `test_cancel_stops_queued_runs`, `test_cancellation_stops_new_tool_execution`, unit cancellation tests | integration / unit |
| 13 | Default concurrency is 2 and is never exceeded; two workers process each run exactly once | `test_default_concurrency_is_two_and_is_never_exceeded`, `test_two_workers_process_each_run_exactly_once` | integration |
| 14 | Agent behaviour is never retried; only transient provider errors and lost workers get new attempts, capped at `max_attempts`; every planned run ends with a result or an explicit error | `test_agent_failures_are_never_retried`, provider-error tests, `test_expired_leases_are_given_up_after_max_attempts` | integration |
| 15 | Missing usage stays unknown (NULL), never zero; known usage survives a failed save | `test_transient_provider_errors_get_new_attempts_with_unknown_usage`, `test_a_result_that_cannot_be_saved_fails_once_and_keeps_its_usage` | integration |

Stress (reviewer's script, 4 seeds, 2 with mid-run cancellation; 3 workers × concurrency 3, 1 s leases, random forced expiries): no deadlocks, no run scored from more than one attempt, no stuck attempts, no job over `max_attempts`.

## Known gaps

- **Double persistence failure.** If both the result write and the failure record fail, the lease expires and the sweeper re-runs an agent that already finished. Tracked as a strict `xfail` in `test_phase4_audit.py`; closing it needs a durable "execution complete" checkpoint before persisting.
- **Interrupted attempts keep no partial trace.** Events are written when an attempt finishes, so a killed attempt shows only `attempt_interrupted`; its tool calls are lost (its usage is correctly unknown).
- **Read consistency.** `GET /experiments/{id}` reads with several READ COMMITTED statements; numbers can be momentarily inconsistent while workers write.
- **Shutdown.** SIGTERM waits for in-flight attempts with no upper bound.
- **Compose** runs only Postgres; `migrate`, `api` and `worker` services need a Dockerfile (PRD §10).
- **Trace redaction** (FR-23) is not implemented.
