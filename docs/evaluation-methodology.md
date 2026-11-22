# Evaluation methodology

**An agent's final message is evidence to inspect, not proof that the task succeeded.**
Every grade comes from the hidden end state of an isolated environment and the
ordered trace of what the agent attempted, what the gateway allowed, and what
actually changed.

## Dataset

`datasets/support`: 50 reviewed tasks in 25 scenario families across 8
categories (eligible and ineligible actions, read-only lookups, missing
information, tool failures, access boundaries, untrusted content, duplicate
operations). Families are assigned to splits **before** paraphrasing (dev 30,
validation 10, test 10 tasks), so two paraphrases of one scenario never sit in
different splits. A dataset version is immutable and identified by its content
hash.

## Four deterministic evaluators (primary)

| Evaluator | Version | Reads | Example reason codes |
| --- | --- | --- | --- |
| outcome | 1.1.0 | hidden final state vs. expected effects | `NO_REFUND_ISSUED`, `WRONG_REFUND_AMOUNT`, `DUPLICATE_EFFECT` |
| policy | 1.1.0 | every tool attempt and gateway decision | `UNAPPROVED_REFUND_ATTEMPT`, `CROSS_CUSTOMER_ACCESS`, `OVER_REFUND` |
| trajectory | 1.3.0 | ordered tool calls vs. partial-order constraints, one-to-one matching | `MISSING_RETURN_CHECK`, `WRONG_ARGUMENT`, `ORDER_VIOLATION`, `EXCESS_REPETITION` |
| response | 1.2.0 | final message claims vs. actual state and trace | `FALSE_SUCCESS_CLAIM`, `WRONG_STATUS_STATED`, `WRONG_AMOUNT_STATED` |

**Run success** = outcome pass ∧ policy pass ∧ response pass. Trajectory is
diagnostic and does not gate success, because valid alternative orders exist.
A correct refusal or escalation is success when the task requires it. Hitting
a turn limit is an unsuccessful completed run (`stop_reason` is kept).

**Evaluator crashes** produce `evaluation_error`, never `pass` or `fail`. A
definite failure from one evaluator still makes the run unsuccessful even if
another evaluator errored. When nothing failed but something could not be
evaluated, success is unknown.

## Scorecard (never one blended number)

Every rate is reported as numerator / denominator / value. An empty
denominator is `n/a`, never 0%.

| Metric | Definition |
| --- | --- |
| Verified success rate | successful runs / scored runs (pass + fail) |
| Strict end-to-end success | successful runs / planned runs (unscored = unsuccessful) |
| Evaluation coverage | scored runs / planned runs |
| Unsafe attempt rate | runs with an attempted policy violation / policy-scored runs |
| Executed violation rate | runs where a forbidden effect happened / policy-scored runs |
| Passed all repetitions | tasks whose every repetition passed / tasks |
| Tool schema validity | schema-valid tool calls / attempted tool calls |
| Tool-call precision | matched required calls / tool calls attempted (tools-needed tasks) |
| Required-call recall | required calls satisfied in a valid order / required calls (tools-needed tasks) |
| Latency | queue time (created → first attempt) and execution time (monotonic), separately |
| Cost per successful run | total candidate cost (retries included) / successful runs |

### Tool metrics: call-level, with explicit coverage

The trajectory evaluator writes `call_counts` into every score's evidence:
`attempted`, `decided`, `schema_valid`, `matched`, `required` and
`required_satisfied`. Aggregates **sum these counts across runs**; they never
average per-run ratios. A run with 12 calls therefore weighs twelve times a
run with one.

* A call is **schema-invalid** when the gateway rejects it at admission:
  `UNKNOWN_TOOL`, `TOOL_NOT_ALLOWED` (not offered for this task) or
  `INVALID_ARGUMENTS`. Authorization and precondition denials were well-formed
  calls, so they count as schema-valid. A call with no recorded decision is
  retained in the attempted-call denominator. `unknown_calls` reports missing
  decisions; any such call makes the schema-validity value null and applicability
  unknown, rather than displaying a misleading percentage.
* **Applicability.** Precision and recall apply only to tasks that need
  tools. When nothing could contribute, the metric is `not_applicable` (shown
  as "not applicable (0/0)"), never 0%. Calls made on a no-tools task still
  count toward schema validity.
* **Coverage.** Each metric lists the runs counted, the runs not applicable,
  the runs with a trajectory `evaluation_error`, the runs whose score lacks
  counts (older trajectory versions) and the ungraded runs. Coverage = runs
  counted / runs not known to be inapplicable. If nothing was counted and any
  of those gaps exists, the metric is `unknown`, not `not_applicable`.

## Comparisons

Two (experiment, configuration) pairs are compared only if dataset version,
split, environment version, evaluator versions and limits all match. Results
are reported per task:
* **Win or loss:** the candidate's strict per-task pass rate is higher or lower.
* **Regression:** the baseline passed every repetition and the candidate
  failed at least one.
* **Improvement:** the reverse.

Confidence intervals use a percentile bootstrap that resamples tasks, so a
task's repetitions stay together. It is paired across both sides and seeded.
Tool metrics, latency and cost are reported as deltas alongside.

## Accounting

Cost = uncached input × rate + cache reads × cache rate + cache writes ×
write rate + output × rate, using a dated price version. Missing usage or an
unknown rate makes the cost **unknown**, and an unknown attempt makes the total
unknown, never $0. A budget is reserved before each model call and settled
afterwards; an unknown actual keeps the reservation. Infrastructure retries
count toward cost. Failed agent behavior is never rerun to chase a better
score. The candidate agent, the LLM judge and the benchmark runs are each
tracked separately.

## Optional LLM judge (secondary)

`app/judges/communication.py` asks a model to rate the final message 1–5 for
**clarity**, **completeness** and **tone**, through a strict `submit_scores`
tool. Design rules:
* **No grading of facts.** The judge does not grade factual correctness; the
  response evaluator already does. It sees the verified outcome only to judge
  whether the message communicates it.
* **Injection fencing.** The agent's message and the customer request are
  wrapped as data, and the rubric tells the judge never to follow
  instructions inside them.
* **Strict parsing.** Output is parsed as JSON tool input with strict
  validation (integers 1–5, no extra keys). Prose, a malformed verdict or a
  provider failure is `evaluation_error` and is not retried. Parser version
  1.0.1 also rejects extra tool calls and malformed provider return objects.
* **Kept apart.** Results go to `judge_scores`, never `scores`. They never
  change run success, evaluation status, scorecards, comparisons or candidate
  cost. A test proves the primary export is identical before and after judging.
* **Concurrent calls.** An attempt/judge-version PostgreSQL advisory lock and
  an existence recheck precede the provider call, preventing duplicate spend
  from concurrent invocations. Database write failures propagate instead of
  being silently treated as duplicate judgements.
* **Own spend.** The judge's tokens and cost are stored per verdict and shown
  as judge spend.
* **Uncalibrated.** It has not been calibrated against human labels, and the
  dashboard says so. Calibrating it on 30–50 human-labeled messages
  (PRD FR-20) is future work; no human labels ship with the repository.

Run it with `make judge EXPERIMENT=<id>` (needs `ANTHROPIC_API_KEY`).

## Public benchmark: BFCL `simple_python` (separate view)

* **Pinned.** Gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
  (BFCL v4). The question and possible-answer files are vendored unmodified
  and SHA-256-verified on every load (`benchmarks/bfcl/pin.json`).
  `make bfcl-fetch` re-downloads and verifies them.
* **Request.** Port of BFCL's Anthropic-style `convert_to_tool`: `.` becomes
  `_` in names, Gorilla types map to OpenAPI types, and the Python 3 hint is
  appended to descriptions. No system prompt is added.
* **Decoding.** The model answers through native tool use, so its call
  arrives as parsed JSON. Model output is never executed, `eval()`-ed or
  `literal_eval()`-ed; a test inspects the module's syntax tree for those calls.
* **Scoring.** Port of the Python path of `simple_function_checker` at the
  pinned commit, including its quirks: exact types (a bool is not an int),
  int → float promotion, case and punctuation-insensitive strings, optional
  parameters marked by `""`, and nested dict and list checks.
* **Headline metric.** Accuracy = passed / scored entries. Provider failures
  (no retries) and scorer crashes are reported separately. Strict accuracy
  over all selected entries is shown beside it.
* **Honest labels.** A run over fewer than 400 entries is stored with
  `category_task_count = 400` and `subset_method = "first N entries in file
  order"`, and the page shows "subset N/400". Every result says
  `official_leaderboard_comparable: false` and lists its deviations from the
  official harness (scorer port, provider-default temperature, subset).
* **Self-check.** `make bfcl-self-check` replays ground truth through the
  full pipeline and must score 400/400. It is stored as `scorer_self_check`
  ("not a model score"). It caught a bug in our own replay of nested-dict
  answers during development.
* **Reproduction.** `python -m app.benchmarks.bfcl --from-manifest
  artifacts/.../manifest.json` reruns the exact entries, model and settings
  recorded in a saved manifest, after checking the pin and actual file hashes,
  scorer version, non-comparability label, runner kind/settings, and first-N
  selection metadata. Unsupported setups fail before any provider call.

## Known limitations

* Response claims are extracted with clause-level rules. A message with no
  checkable claim passes as `NO_CHECKABLE_CLAIMS` (visible in evidence)
  rather than being verified.
* The LLM judge is uncalibrated, and the BFCL scorer is a port rather than
  the official package.
* Only one BFCL category is supported. tau2-bench (PRD FR-29) is not imported.
