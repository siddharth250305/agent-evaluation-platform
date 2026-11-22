# OpenAI paid-pilot audit — 2026-11-22

Scope: `claude.md`, PRD FR-16/FR-24, OpenAI HTTP adapter and factory, dated pricing,
tool-loop limits, CLI/Makefile planning, worker scope, persisted traces and exports.
All provider execution used offline fixtures; no paid API request was made.

## Findings fixed

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | Vendor errors and unexpected decoder/client exceptions could expose credentials, including masked/truncated key fragments. HTTP debug logging was unguarded. | Discard vendor error bodies; emit local error descriptions; sanitize successful response data and HTTPX/httpcore diagnostics; convert unexpected failures to a generic non-retryable error. |
| P1 | The CLI worker could drain unrelated queued experiments during a small paid pilot. | Scope claiming, pending-work checks and lease recovery to the newly created experiment. General workers keep their existing global scope. |
| P1 | Character-based token estimates could under-reserve the $0.50 cap. | GPT-4.1 mini reserves the full documented context capacity plus configured output maximum at Standard rates; known usage settles actual cost, unknown usage retains the reservation. Pin `service_tier=default`. |
| P2 | Quota errors expressed in `error.type` could retry. | Inspect both type and code; insufficient quota is non-retryable. Offline PostgreSQL tests check one attempt for 401/quota and three for transient rate limits. |
| P2 | Dry-run accessed/seeded PostgreSQL, and Makefile ran migrations. Spend preview omitted retry exposure. | Build plans from local files before database/provider setup, skip migration prerequisite, show first-attempt and three-attempt ceilings separately. |
| P2 | Non-finite timeouts/budgets and invalid usage counts were insufficiently validated. | Reject invalid limits; negative, boolean, string or inconsistent counts stay unknown rather than becoming a zero/negative cost. |
| P2 | Per-I/O timeouts did not bound a slowly arriving response; late final responses could pass the deadline. | Production transport uses an async cancellable total deadline inside the synchronous provider interface; loop rejects late responses after recording usage. Synchronous injected clients remain an offline testing seam. |
| P2 | Python JSON parsing accepted NaN/Infinity in tool arguments. | Preserve these invalid strings for the tool gateway to reject, like other invalid JSON. |

## Verified behavior

The factory selects OpenAI from the pinned model name and exposes only key-presence booleans
or environment variable names. Credentials are not configuration or manifest fields. Tests cover
fake full keys, masked 401 fragments, Authorization headers, decoder exceptions, debug logs,
persisted trace payloads, JSON exports and CSV exports.

The pilot retains 10 model turns, 20 tool calls, 120 seconds, 16,000 output tokens and $0.50 per
attempt. One run can have three infrastructure attempts: the retry-inclusive ceiling is $1.50.
There are no adapter-level automatic retries. Non-retryable failures end the run; invalid tool
arguments are agent behavior, not a reason to rerun an attempt.

At `openai-2026-11-22`, GPT-4.1 mini rates are $0.40 uncached input, $0.10 cached input and
$1.60 output per million tokens. Prompt tokens include cached tokens, so the adapter subtracts
cached tokens exactly once. Missing cache breakdown or usage remains unknown. Reasoning tokens
are already included in completion tokens. Sources: [model pricing and limits](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[Chat Completions usage](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[API errors](https://developers.openai.com/api/docs/guides/error-codes).

The conservative full-context reservation can stop an attempt while its recorded spend is still
well below $0.50. This is deliberate. Transport cancellation cannot retract provider-side billing;
failed calls therefore retain unknown accounting. These controls cover the configured text-only
pilot at the pinned rates, not arbitrary models, external charges or user-installed logging code.

## Validation and existing limitation

`REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q --tb=short`: **801 passed, 1 expected
failure**, no skips (107.07 seconds). The local one-task OpenAI dry-run also completed without
an API key, reporting $0.50 first-attempt and $1.50 retry-inclusive ceilings.

Offline tests cover parsing, cached usage, unknown accounting, hard call caps, invalid limits,
total-deadline cancellation, late-response accounting, dry-run socket/DB/provider exclusion,
Makefile migration exclusion, retry counts, persisted/exported redaction and queue isolation.

The pre-existing strict xfail in `test_phase4_audit.py` remains: if both normal and fallback
persistence fail after agent execution, lease recovery can rerun completed work. This is an
existing durability issue, separate from provider retry classification; the audit does not
certify exactly-once billing under that failure.
