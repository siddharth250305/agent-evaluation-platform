# Architecture

A small modular monolith: two processes, an **API** and a **worker**, sharing
PostgreSQL. The tool gateway is the only place where an agent's actions touch
the environment, so it is the only place that has to be trusted.

```
            HTTP / dashboard                               background
  ┌──────────────────────────────┐            ┌─────────────────────────────────────────┐
  │ API (FastAPI + Jinja2)       │            │ Worker (python -m app.worker)           │
  │  /datasets /agent-configs    │            │  claim job (FOR UPDATE SKIP LOCKED)     │
  │  /experiments /comparisons   │  jobs      │  lease + heartbeat, sweeper             │
  │  /runs /attempts/{id}/events │──────────▶ │  fresh SupportEnvironment per attempt   │
  │  /ui/* (5 screens)           │  table     │   ┌─────────┐  call   ┌──────────────┐ │
  └──────────────┬───────────────┘            │   │  agent  │───────▶ │ tool gateway │ │
                 │                            │   │ adapter │◀─────── │ allowlist,   │ │
                 │                            │   └─────────┘ result  │ schema, auth │ │
                 │                            │        │              └──────┬───────┘ │
                 │                            │        ▼ trace events        ▼ effects │
                 │                            │   evaluators read hidden end state +   │
                 │                            │   ordered trace → scores (lease-checked)│
                 ▼                            └──────────────────┬──────────────────────┘
        ┌─────────────────────────────────────────────────────────▼─┐
        │ PostgreSQL: datasets, task_versions, agent_configs,        │
        │ experiments, runs, attempts, jobs, trace_events,           │
        │ model_calls, tool_calls, scores, price_versions,           │
        │ benchmark_results, judge_scores                            │
        └────────────────────────────────────────────────────────────┘
```

## Modules

| Path | Responsibility |
| --- | --- |
| `app/tasks.py`, `app/datasets.py`, `datasets/support/` | Task YAML, split manifest, content hashes. Each task loads into an **agent view** (request + offered tools) and an **evaluator view** (fixture, expected outcome, trajectory spec). |
| `app/environment.py` | Fictional retailer state (customers, orders, returns, refunds, approvals, tickets) built from a deep copy of the fixture, with its own `environment_id`, approval simulator and fault injection keyed to logical operations. |
| `app/tools/` | Seven tool schemas (Pydantic, `extra=forbid`) and the **gateway**: allowlist → strict schema → server-side identity and authorization → preconditions → execution. It records `tool_attempt`, `tool_decision` and `tool_effect` as separate trace facts. |
| `app/agents/` | `AgentAdapter` interface; scripted honest and lying agents; the hand-written tool loop; versioned prompts (`prompts.py`); registry. |
| `app/providers/` | Provider-neutral request and response types, the Claude Messages adapter over raw HTTPX, and a scripted provider for tests. No automatic retries. |
| `app/runner.py` | One attempt: fresh environment, gateway, agent, trace, then every evaluator. Computes run success. |
| `app/evaluators/` | Outcome, policy, trajectory, response. Each is versioned; crashes become `evaluation_error`. |
| `app/accounting.py` | Dated price versions, exact cost, budget reservation before each call. Unknown usage stays unknown. |
| `app/worker/` | PostgreSQL queue, leases, heartbeats, the expired-lease sweeper, cancellation, bounded concurrency. Every write needs the current lease token. |
| `app/metrics.py` | Pure scorecard and comparison math: ratios with numerator and denominator, call-level tool metrics, task-cluster bootstrap. |
| `app/services/` | Catalog, experiments (one transaction, idempotency key, manifest), results loading, comparisons, exports, judging. |
| `app/web/` | Five server-rendered screens: experiments, experiment detail, run detail, comparison, benchmarks. |
| `app/benchmarks/bfcl.py`, `benchmarks/bfcl/` | Pinned BFCL `simple_python` import, runner, scorer port and artifacts. Results live in `benchmark_results`, never in experiment scorecards. |
| `app/judges/` | Optional communication-quality LLM judge. Its results live in `judge_scores`, never in `scores`. |
| `app/cli.py` | `smoke`, `seed`, `demo`, `evaluate` and `report`, behind the Makefile. |

## Information boundaries

* **Agent-visible input** is exactly `AgentTaskView`: the user request, the
  offered tools and the configuration's system prompt. Fixtures, expected
  outcomes, trajectory specs and grading rules exist only in the evaluator
  view. Tests check that the agent's requests never contain them.
* **Identity** comes from the session (`CustomerContext`), never from tool
  arguments. A cross-customer lookup gets the same `order_not_found` error as
  a missing order, so other customers stay invisible.
* **Evaluators** read the hidden initial and final snapshots and the ordered
  trace. The agent's final message is one piece of evidence among many.
* **Benchmarks and judges** are kept out of the primary results by schema, not
  just by convention. Separate tables mean no query that builds a scorecard can
  pick them up.

## Data model notes

* Execution status (`queued` … `cancelled`) and evaluation status (`pending`
  … `evaluation_error`) are separate columns with CHECK constraints.
* `UNIQUE(experiment_id, agent_config_id, task_version_id, repetition)` on
  runs, `UNIQUE(run_id, attempt_number)` on attempts and
  `UNIQUE(attempt_id, event_sequence)` on trace events. Scores are unique per
  attempt, evaluator, version and dimension.
* An experiment stores an immutable **manifest**: dataset hash and split,
  task content hashes, configuration and prompt hashes, tool-schema hash,
  evaluator versions, environment version, fault config, price versions and
  code identity. Exports carry it.
* `benchmark_results` records `task_count`, `category_task_count`,
  `subset_method` and `result_kind`. A CHECK constraint refuses a subset with
  no stated selection method, and another refuses a `task_count` larger than
  the category.
* `judge_scores` holds the judge's verdict, its rubric hash, and its own
  tokens and cost. That spend is never added to the candidate's.

## Durability

Execution is at least once. A worker that dies mid-attempt loses its lease.
The sweeper then marks that attempt as an explicit `infrastructure_error`
(interrupted) and requeues the run as a new attempt, and the stale worker's
late writes are rejected by the lease-token check. Only infrastructure
failures are retried, never a failed agent behavior. Infrastructure retries
count toward cost. One known gap is tracked as a strict `xfail`: if both
persistence writes fail after an agent completes, there is no durable
"execution complete" checkpoint.

## Deployment (local)

`docker compose up -d postgres` runs PostgreSQL 16 bound to `127.0.0.1`.
The API and worker run from the virtualenv (`make api`, `make worker`).
Secrets come from the environment (`ANTHROPIC_API_KEY`), never from
configurations or images. Containerized api, worker and one-shot migrate
services are not part of v1 yet.

See [evaluation-methodology.md](evaluation-methodology.md) for what is scored and how.
