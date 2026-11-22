# AI Agent Evaluation Platform

Run repeatable experiments against tool-using AI agents, **verify what they actually did**, compare
configurations task by task, and explain every failure.

> An agent's final message is evidence to inspect, not proof that the task succeeded.

A support agent can say "your refund has been processed" when no refund exists. This platform runs
each agent configuration against a versioned set of 50 support tasks for a fictional electronics
retailer, each in its own isolated environment. It records every model call and tool call, scores the
**hidden end state** with deterministic evaluators, and reports success, safety, tool use, latency and
cost as a scorecard, never as one blended number.

## Quickstart: no API key, no database, under a minute

```bash
make setup     # .venv + pip install -e ".[dev]"   (Python 3.12+)
make smoke     # offline tests + 5-task scripted run + BFCL scorer self-check
```

`make smoke` blocks network sockets and spends $0. It checks that:

- a scripted agent that **claims a refund it never issued** fails the response evaluator with
  `FALSE_SUCCESS_CLAIM`, citing the trace events that prove it;
- every planned run ends with a score or an explicit evaluation error;
- tool metrics are call-level sums, and `not_applicable` (not 0%) on a task that needs no tools;
- the pinned BFCL data loads with verified hashes, and replaying its ground truth scores 25/25.

## With the dashboard (still no API key)

```bash
make db-up     # PostgreSQL 16 in Docker, bound to 127.0.0.1
make demo      # migrate, seed, scripted honest-vs-lying experiment (x3 reps), BFCL self-check
make api       # http://127.0.0.1:8000
```

The five screens: **Experiments**, **Experiment detail** (scorecard, tool metrics, failure reasons,
filterable runs), **Run detail** (agent-visible input vs. hidden ground truth, evaluator evidence,
final-state diff, trace with cited events highlighted), **Compare**, and **Benchmarks**.

## Commands

| Command | What it does | Needs |
| --- | --- | --- |
| `make setup` | Create `.venv`, install the app and test dependencies | Python 3.12+ |
| `make smoke` | Offline suite + scripted end-to-end check (about 15 s) | nothing |
| `make test` | Full deterministic suite; PostgreSQL tests skip if the DB is down | nothing |
| `make test-db` | Full suite with PostgreSQL required | Docker |
| `make db-up` / `make migrate` | Start PostgreSQL / apply Alembic migrations | Docker |
| `make seed` | Import `datasets/support`, register 2 scripted + 4 model configs (2 models x 2 prompts) | DB |
| `make demo` | Scripted experiment + stored BFCL self-check for the dashboard | DB |
| `make evaluate` | **Paid.** Real-model experiment; default `sonnet-5-5_prompt-v1` vs `_prompt-v2` on `dev` | DB, `ANTHROPIC_API_KEY` |
| `make report [EXPERIMENT=id]` | Export the latest (or given) experiment to `reports/` as JSON + CSV with its manifest | DB |
| `make bfcl-self-check` | Replay BFCL ground truth through the full pipeline, store as a self-check | DB |
| `make bfcl [BFCL_LIMIT=50]` | **Paid.** Run BFCL `simple_python` on a model, store with raw outputs | DB, `ANTHROPIC_API_KEY` |
| `make judge EXPERIMENT=id` | **Paid, optional.** LLM judge of communication quality (secondary) | DB, `ANTHROPIC_API_KEY` |
| `make bfcl-fetch` | Re-download the pinned BFCL files and verify SHA-256 | network |
| `make api` / `make worker` | Serve the API + dashboard / run a background worker | DB |

`make evaluate EVAL_ARGS="--configs sonnet-5-5_prompt-v1 opus-5-5_prompt-v1 --split validation --repetitions 3"`
compares any registered configurations. Every model configuration carries a per-attempt budget
(`max_cost_usd = 0.50`) that is reserved before each model call.

## The four demos

1. **Convincing but wrong.** `make smoke`, or open a `scripted-lying` run after `make demo`. The agent
   says the $149.00 refund was processed. The final-state diff shows no refund, the response evaluator
   fails with `FALSE_SUCCESS_CLAIM`, and the trace highlights the events it cites (no applied
   `issue_refund` effect before the final response).
2. **Better prompt, real tradeoff.** `make evaluate`, then **Compare** the two configurations. You get
   the tasks that improved or regressed, success deltas with task-level bootstrap intervals, tool-call
   precision/recall deltas, and latency and cost deltas. This one needs an API key; the comparison
   screen itself also works on the scripted demo experiment.
3. **Worker recovery.** `pytest tests/postgres/test_recovery.py -v` runs a real worker process,
   `kill -9`s it mid-attempt, and shows the interrupted attempt (`WORKER_LEASE_EXPIRED`), the lease
   expiry, a clean replacement attempt in a fresh environment, and no duplicate result.
4. **Public benchmark.** **Benchmarks** shows each pinned BFCL `simple_python` result with:
   - the version (Gorilla commit `6ea5797…`), task count, and subset label (e.g. "subset 50/400");
   - the scoring method, and whether the row is a model run or a scorer self-check;
   - a link to the raw outputs. The manifest beside it reproduces the setup:
     `python -m app.benchmarks.bfcl --from-manifest artifacts/benchmarks/bfcl/<run>/manifest.json`.

## How it is graded

Four deterministic evaluators, each versioned:

| Evaluator | Checks |
| --- | --- |
| **Outcome** | The hidden end state |
| **Policy** | Every attempted tool call |
| **Trajectory** | Required calls under partial-order constraints |
| **Response** | The final message against reality |

Success = outcome ∧ no policy violation ∧ response consistent. An evaluator crash is
`evaluation_error`, never a pass or fail. Missing token usage is unknown, never $0.

**Tool metrics** are computed per call, not per run:

| Metric | Definition |
| --- | --- |
| Tool schema validity | valid calls / attempted calls |
| Tool-call precision | matched / attempted |
| Required-call recall | satisfied / required |

Each metric's counts are summed across runs, and the dashboard shows which runs were counted,
not applicable, or lost to evaluator errors.

**Optional LLM judge.** It rates clarity, completeness and tone. It lives in its own table, never
changes a primary result, bills its own tokens separately from the candidate's, and is labeled
uncalibrated.

**BFCL** results live on their own screen and are never blended into experiment scorecards.

Measured numbers: [results.md](results.md) ·
Details: [docs/evaluation-methodology.md](docs/evaluation-methodology.md) ·
[docs/architecture.md](docs/architecture.md) ·
[PRD](PRD%20AI%20Agent%20Evaluation%20Platform.md).

## Layout

```
app/            agents, providers, tools (gateway), environment, evaluators, worker, services, web,
                benchmarks (BFCL), judges, metrics, cli
datasets/       support: 50 tasks, 25 families, split manifest (dev 30 / validation 10 / test 10)
benchmarks/     bfcl: pinned simple_python data (Apache-2.0, vendored unmodified) + pin.json
migrations/     Alembic (0001 initial, 0002 benchmark_results, 0003 judge scores + subset labels)
tests/          unit, evaluators, integration (offline) and postgres (local DB); zero paid calls
```

## Testing

All tests use scripted agents and scripted providers; `tests/conftest.py` blocks every outbound
socket, so a paid API call cannot happen by accident. `make test-db` runs everything against a
throwaway PostgreSQL database per session.

## Non-goals (v1)

Public code uploads, real payments or customer data, browser agents, fine-tuning, multi-agent judge
committees, multi-tenancy/SSO, and Kubernetes.

## Third-party data

`benchmarks/bfcl/data/` contains files from the Berkeley Function Calling Leaderboard
([gorilla](https://github.com/ShishirPatil/gorilla), Apache-2.0), copied unmodified at a pinned commit.
See [benchmarks/bfcl/NOTICE.md](benchmarks/bfcl/NOTICE.md).
