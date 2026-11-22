# Developer commands. Everything except `evaluate`, `bfcl` and `judge` runs with no API key and $0 spent.
VENV ?= .venv
PY := $(VENV)/bin/python
PYTHON ?= python3

# make evaluate EVAL_ARGS="--configs sonnet-5-5_prompt-v1 opus-5-5_prompt-v1 --split validation --repetitions 3"
# Pilot subsets: add --tasks <id ...> and/or --limit N; add --dry-run to print the plan and spend ceiling ($0, no key)
# make evaluate EVAL_ARGS="--configs sonnet-5-5_prompt-v1 --split dev --tasks return_in_transit_001 --dry-run"
EVAL_ARGS ?=
BFCL_LIMIT ?= 50
BFCL_MODEL ?= claude-sonnet-5-5
EXPERIMENT ?=
# make judge EXPERIMENT=<id> JUDGE_ARGS="--model gpt-4.1-mini --limit 5"
JUDGE_ARGS ?=
# Paid targets need the key for the selected model's vendor: ANTHROPIC_API_KEY for claude-*,
# OPENAI_API_KEY for gpt-* (e.g. EVAL_ARGS="--configs gpt-4.1-mini_prompt-v1", BFCL_MODEL=gpt-4.1-mini).

.PHONY: help setup db-up db-down migrate seed test test-db smoke demo evaluate report judge \
	bfcl bfcl-self-check bfcl-fetch api worker

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup: ## Create .venv and install the app with test dependencies
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

db-up: ## Start local PostgreSQL 16 (Docker, bound to 127.0.0.1)
	docker compose up -d --wait postgres

db-down: ## Stop PostgreSQL (data volume kept)
	docker compose down

migrate: ## Apply Alembic migrations to DATABASE_URL (default: the compose database)
	$(PY) -m app.db.migrate

seed: migrate ## Import the support dataset and register scripted + model agent configurations (idempotent)
	$(PY) -m app.cli seed

test: ## Full deterministic suite (PostgreSQL tests skip, with the reason, if the database is down)
	$(PY) -m pytest

test-db: db-up ## Full suite; an unreachable PostgreSQL is a failure, not a skip
	REQUIRE_POSTGRES=1 $(PY) -m pytest -q

# No database, no network (sockets are blocked in tests/conftest.py), no API key. Well under a minute.
smoke: ## Offline check: unit/evaluator/integration tests + 5-task scripted run + BFCL self-check
	$(PY) -m pytest -q tests/unit tests/evaluators tests/integration
	$(PY) -m app.cli smoke

demo: migrate ## Scripted honest-vs-lying experiment and a stored BFCL self-check for the dashboard ($0)
	$(PY) -m app.cli demo

evaluate: $(if $(findstring --dry-run,$(EVAL_ARGS)),,migrate) ## PAID: real-model experiment, prompt v1 vs v2 by default (key for the configs' vendor)
	$(PY) -m app.cli evaluate $(EVAL_ARGS)

report: ## Export an experiment (EXPERIMENT=<id>, default latest) to reports/ as JSON + CSV with its manifest
	$(PY) -m app.cli report $(if $(EXPERIMENT),--experiment $(EXPERIMENT))

judge: ## PAID, optional: LLM communication judge for EXPERIMENT=<id> (secondary, uncalibrated; JUDGE_ARGS)
	@test -n "$(EXPERIMENT)" || (echo "usage: make judge EXPERIMENT=<experiment id>" && exit 2)
	$(PY) -m app.judges --experiment $(EXPERIMENT) $(JUDGE_ARGS)

bfcl-self-check: migrate ## Replay BFCL ground truth through the full pipeline and store it ($0, no model)
	$(PY) -m app.benchmarks.bfcl --self-check

bfcl: migrate ## PAID: BFCL simple_python on BFCL_MODEL (claude-* or gpt-*), first BFCL_LIMIT entries
	$(PY) -m app.benchmarks.bfcl --model $(BFCL_MODEL) --limit $(BFCL_LIMIT)

bfcl-fetch: ## Re-download the pinned BFCL files from GitHub and verify their SHA-256
	$(PY) -m app.benchmarks.bfcl --fetch

api: ## Serve the API and dashboard on http://127.0.0.1:8000
	$(VENV)/bin/uvicorn --factory app.api.app:create_app --host 127.0.0.1 --port 8000

worker: ## Run a background worker (concurrency 2)
	$(PY) -m app.worker
