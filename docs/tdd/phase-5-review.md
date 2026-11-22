# Phase 5 focused review

Reviewed against `claude.md` and PRD sections 7, 10 and 11 on 2026-11-21.

## Findings

- **P1, fixed — CSV metadata bypassed formula protection.** `to_csv` wrote the manifest and supplement as raw JSON comment lines. A comma inside a user-controlled configuration setting could become a new spreadsheet cell beginning with `=`; CSV does not define comment lines. Metadata now uses the CSV writer as single-cell records. Row/header protection also handles newline and whitespace-prefixed formulas. Manifest JSON round-trips unchanged. Consumers must parse CSV records first and then recognize single-cell records beginning `# `; filtering physical lines before CSV parsing is unsafe for multiline cells.
- **P2, open — aggregate tool metrics are missing.** `metrics.rates` exposes six run/task rates but no tool schema validity, tool-call precision or required-call recall from PRD section 7. Consequently comparison results, exported scorecards and dashboard scorecards omit them. Individual trajectory evidence contains precision/recall, and its `not_applicable` status and null value survive JSON/CSV exports. Completing these aggregates requires explicit call-level numerators/denominators, applicability and evaluator-error coverage; averaging per-run scores would give incorrect weighting. This review does not implement the missing metric feature.

## Verified boundaries and semantics

- Verified success uses pass + fail as its denominator; strict success and coverage use all planned runs. Empty denominators and undefined deltas/intervals return null.
- Finalized started attempts include infrastructure retries in cost and tokens. Any unknown finalized cost propagates to total, cost per success and deltas. Never-started work contributes zero. The existing implementation intentionally excludes active attempts: displayed spend during an unfinished experiment is a finalized-attempt subtotal, not a final bill.
- Bootstrap uses sorted task clusters, keeps repetitions together, pairs both configurations and uses a local seeded RNG. Undefined resamples are counted/excluded. Tests cover permutation invariance and no global RNG mutation.
- A regression requires all baseline repetitions passing and at least one explicit candidate failure. Candidate evaluator errors can produce a strict-rate loss without being a regression. Baseline evaluator errors prevent a passed-all classification.
- Compatibility checks dataset version, split, environment version, evaluator versions and limits. A 422 reports all differences; the new integration test exercises simultaneous differences.
- Run Detail constructs the agent-visible panel from request, offered tools and configuration prompt only. Fixtures/expected/reference/trajectory stay in evaluator sections; Jinja escapes hostile markup. Worker input remains the strict `AgentTaskView`.
- JSON reports deliberately include evaluator scores/evidence and reproducibility manifests for operators. These reports and the full Run Detail page are not agent inputs; evaluator evidence may include expected values. They must not be reused as agent prompts. No reporting-to-agent input path was found.

## Validation

- Targeted metrics/reporting unit suite: **56 passed**.
- `REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q --tb=short`: **645 passed, 1 xfailed, no skips** in 96.96 seconds against local Docker PostgreSQL 16.
- The sole strict xfail predates this review: Phase 4 has no durable completed-execution checkpoint if both primary and fallback persistence fail.
- Added 17 test cases covering CSV metadata/formula prefixes, multiline cells, null score values, bootstrap invariance, evaluator-error regressions, simultaneous incompatibilities, hidden-markup escaping and no-tool JSON/CSV exports.
- Tests use scripted agents and no paid provider calls.
