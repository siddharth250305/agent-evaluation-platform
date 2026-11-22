# Phase 6 focused review

Reviewed against `claude.md`, PRD sections 7/10/11 and FR-20/28 on 2026-11-21.

## Findings and changes

1. **P1, fixed: concurrent judging could spend twice and silently lose one call's accounting.** Both invocations read the same pending attempts and called the provider before the unique constraint selected a winner. The loser caught every `IntegrityError` and discarded its usage. Each attempt/judge-version now obtains a PostgreSQL transaction advisory lock and rechecks existence before the provider call. Other integrity failures propagate. A real PostgreSQL contention test verifies exactly one call and one stored result.
2. **P2, fixed: schema validity used decided calls rather than attempted calls.** A missing gateway decision disappeared from the denominator; all-undecided calls looked not applicable. The denominator now includes attempted calls, `unknown_calls` records missing decisions, and any such call makes the value null/applicability unknown. Precision and recall still sum call-level counts; no-tools and legacy/evaluator-error coverage remain explicit. No trajectory verdict/evidence changes were needed: the producer stays at 1.3.0, and a comparison test verifies rejection against 1.2.0.
3. **P2, fixed: replay could silently misrepresent the benchmark setup.** Matching only commit/file hashes allowed duplicates, reordered or empty selections, inaccurate subset counts, obsolete scorers and unsupported runner settings. Replay now validates the current data bytes, benchmark identity, scorer version, non-comparability label, nonempty first-N IDs/counts, runner kind/provider/settings, prompt and positive finite limits before making requests. Both self-check and mocked model-run replay are tested.
4. **P2, fixed: BFCL adapter crashes aborted the batch before artifacts were produced.** Unexpected provider exceptions or invalid response objects now become infrastructure-error entries with unknown usage, with no retries. Earlier entry results survive.
5. **P2, fixed: malformed judge returns escaped error handling, and extra tool calls could be accepted.** Both now produce evaluation errors without scores; received usage remains billable judge usage. Parsing changes bump the judge version to 1.0.1. The rubric hash remains tied to the unchanged rubric/tool schema.
6. **P2, fixed: NUL bytes in judge explanations or provider-error messages rejected PostgreSQL writes.** Those free-text fields now replace NUL with U+FFFD before storage, preserving usage and error status. Nonpositive judge limits are rejected before any paid work.

## Verified behavior

- Tool ratios are sums of explicit call counts, not averages of run ratios. Unknown/empty values remain null. Partially covered metrics from scored runs retain their separate run coverage; missing schema decisions additionally invalidate the reported schema percentage.
- The BFCL scorer compares parsed native tool-call data; it never executes model output or uses `eval`/`literal_eval`. Static inspection and a code-like payload test cover this boundary.
- Question/answer SHA-256 checks remain on load and fetch. Raw-output artifacts retain hashes. Replay verifies the input pin/data; it makes a new run rather than trusting or rescoring the saved raw-output file.
- Subsets and scorer self-checks remain labeled separately from model results and the official leaderboard. The scorer stays at 1.0.0: no scoring-rule changes were made. Reviewed the local Python matching paths against the [pinned upstream checker](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py); this remains a port, not the official benchmark harness.
- Judge rows remain outside primary evaluator scores and candidate accounting. The integration test compares primary export contents before/after judging. Errors carry null scores, unknown usage remains unknown, and summaries/dashboard retain `calibrated: false`/uncalibrated labels, model identities and rubric hashes.
- Makefile commands distinguish paid targets from offline tests/self-checks. Dry-run checked smoke, BFCL self-check/fetch and judge commands; the offline CLI smoke passed with zero API calls. README/methodology now describe attempted-call denominators and replay/judge safeguards.

## Remaining limitations

- **FR-20 calibration remains incomplete:** no 30–50 human-label calibration set is supplied. The judge is honestly labeled uncalibrated and must remain secondary.
- Judge deduplication is intentionally per attempt/name/version, not per model or effort. Switching models with the same judge version skips existing rows; partial runs can produce a summary listing multiple models. Provider effort and request limits are not stored per judge row. This is insufficient for a reproducible comparison of judge configurations, though candidate accounting and primary comparisons remain separate.
- The advisory lock prevents concurrent duplicate spend, not process/database failure after a paid response. Such persistence failures propagate; there is no durable external-call receipt/outbox for recovering that usage.
- The pre-existing Phase 4 double-persistence durability test remains an expected failure.

## Validation

- Added 22 regression cases; the first targeted run reproduced 12 failures before fixes.
- Targeted unit/PostgreSQL suite: 81 passed (before the final four replay cases).
- Full suite: `REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q --tb=short` — **721 passed, 1 xfailed, no skips**, in 100.92 seconds.
- The sole xfail is the pre-existing Phase 4 double-persistence durability gap.
- `.venv/bin/python -m app.cli smoke` passed: support demo and 25-entry pinned BFCL self-check, no database/network/API key.
- All provider calls in tests are scripted; outbound Python socket connections are blocked.
