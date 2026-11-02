# Moderation: Killable Standard-Library Regex Execution

Date: 2026-11-02
Status: Draft for approval; not an implementation authorization
Task: TASK-15414
Baseline inspected: `aec70514cd`, following merged Moderation PR #3396

## Intent and Decisions

Keep Python's standard-library `re` engine. Run moderation matching and
redaction in a separate, parent-supervised, killable process so a pathological
expression cannot monopolize the application's Python execution thread.

The requester explicitly selected process isolation and rejected substituting
the third-party `regex` engine. On timeout or budget exhaustion, their required
behavior is: block the affected content and surface the failure to an admin.
This is a deliberate behavior-changing security follow-up, not another strictly
structural extraction.

Other details below are engineering proposals for approval, not decisions
previously made by the requester.

## Goals and Non-Goals

- Bound active regex work and isolate worker hangs/crashes from application code.
- Preserve ordinary stdlib pattern, flag, rule-order, category, phase, replacement,
  and result behavior when execution succeeds within limits.
- Fail closed without exposing partially redacted content or retrying inline.
- Make failures operational incidents, not fabricated content-policy matches.
- Preserve supported imports and service override dispatch.
- Avoid pools, per-pattern process launches, new regex dependencies, and a
  generalized cross-module execution framework in this slice.

Do not expand moderation to currently unmoderated LLM surfaces (TASK-13950.27),
rewrite Web Scraping, fix the existing long-span search geometry, change
zero-length/count semantics, or automatically file public GitHub issues.
Initial policy parsing/compilation and admin linting remain existing compiler
work in this slice; this design does not claim those paths are process-isolated.

## Current Code and Prior Art

`ModerationService` delegates deterministic work to `PolicyEvaluator`.
`find_match_span()` uses stdlib searches, including chunk/window and fallback
searches. `collect_rule_matches()` uses a full-text `finditer()`; short-text
redaction uses `sub()`/`subn()`. None has a parent-enforced execution deadline.

The service deliberately evaluates first, then dispatches through
`self.redact_text()` for redaction. Preserve that extension point rather than
shipping a service instance or an override callback into a child.

Web Scraping's `safe_regex.py` and `safe_regex_worker.py` provide local examples
of bounded admission, JSON IPC, resource limits, and kill/reap cleanup. Its
wrapper also has third-party-engine/in-process execution paths and different
replacement semantics, so calling it directly would violate this design.
`Utils/cpu_bound_handler.py` is another local process-offload example, but a
shared `ProcessPoolExecutor` is not the ownership boundary needed here.

Normal moderation review capture defaults off and is best-effort.
`review_service._safe_excerpt()` truncates; it does not sanitize arbitrary
failed content. Neither that helper nor optional capture is an adequate
mandatory operational-incident path.

## Approaches Considered

| Approach | Tradeoff and disposition |
| --- | --- |
| Native timeouts using third-party `regex` | Rejected by requester: changes engine and does not provide the requested process boundary. |
| Standard-library `re` in a killable child | Selected by requester: retains engine semantics and gives the parent explicit termination authority; costs startup and IPC. |
| Thread timeout or timed-out pool future | Rejected: abandoning a wait is not termination of the running regex operation. |

Use a fresh child per evaluator batch, not per pattern. A persistent pool may
be evaluated separately only after profiling and its own ownership design.

## Process Boundary

Add Moderation-owned runner, protocol, and worker modules. Names are left to
the implementation plan. The worker executes existing evaluator algorithms;
do not create a second moderation semantics implementation.

Supported requests are one complete `evaluate` batch (without service-level
redaction), `redact`, or `redact_with_count` batch. All rules for that batch
execute inside that child, including short-text search/substitution, long-text
iteration, and nested helper calls. Child-side reconstruction compiles patterns
with stdlib `re`; there is no ASCII, literal-rule, short-text, overload, or
worker-error inline matching fallback.

The parent snapshots supported policy values into bounded plain data. Preserve
the distinction between `PatternRule` and legacy raw `re.Pattern` entries;
carry pattern text and integer flags, including inline-flag semantics. Preserve
phase `None`, ordered rules, replacement literals, and current limit coercions.
No pickle, arbitrary object deserialization, callbacks, or executable policy data.
Unsupported objects fail closed rather than being coerced into a different rule.

Launch the configured Python interpreter with a fixed trusted worker entry
point, argument vector, no shell, no content in argv, closed unrelated file
descriptors, and only the environment needed to start that interpreter.
Worker startup must not initialize the application, import providers, connect
to databases, or start other workers. Packaging and import-isolation tests must
verify this for installed distributions, not only the checkout. Protocol output
must be separate from diagnostic output, including `re.DEBUG` compilation output.

Use versioned, size-bounded JSON request/response frames through private pipes.
Responses contain only validated result fields, counts, spans, and final text.
Do not transfer match objects. Reject unknown tags, invalid types, impossible
spans/counts, trailing protocol data, nonzero exits, and incomplete responses.
The parent must enforce read limits while reading, not after unbounded capture.
Drain/write concurrently where needed so a full pipe cannot deadlock the parent.

## Deadlines, Capacity, and Cleanup

The parent owns a monotonic deadline for a complete service operation. It starts
before admission/serialization and is shared across decision and redaction
batches, so two launches or many rules cannot each reset the budget. Store the
operation context per call, not on mutable shared service state. Nested service
dispatch inherits it; concurrent calls cannot steal each other's budget.

Proposed initial defaults for approval: 2 seconds per service operation and at
most 4 active workers per API process. Admission is immediate and nonblocking;
no unbounded queue or automatic retry. Deployment-wide capacity is the per-process
limit multiplied by API worker count and must be documented.

Proposed safety ceilings: 1,000,000 input characters, 1,000,000 output characters,
256 rules, 4,096 characters per pattern/replacement, 256 KiB total encoded policy,
32 MiB request frame, and 8 MiB response frame. Validate counts/lengths before
allocating encoded copies; enforce UTF-8 byte limits as well as character limits.
Oversize input, policy, intermediate output, or final output blocks; never silently
truncate the input or keep only part of the rule set. These are explicit new
admission limits, not reinterpretations of the existing scan chunk settings.

On timeout, cancellation, disconnect, abnormal exit, protocol failure, or shutdown:
discard all output, kill an owned live child, close pipes, and reap it with bounded
cleanup. No graceful retry of a pathological regex. Keep a slot quarantined while
its child is still alive; a bounded lifecycle reaper owns any process not immediately
reaped. Do not release admission merely because the request stopped waiting.
Lifecycle shutdown rejects new work and drains/kills all registered children.

Do not overpromise a hard real-time wall-clock bound: OS process creation and
process reaping can exceed an application's timer. Track these separately and
never perform an unbounded join on the event loop. Python documents process
creation timeout limitations in its subprocess API.

Apply supported child-only CPU/address-space/core-dump limits before decoding
policy and executing regex. On Linux, propose a 512 MiB address-space ceiling
and CPU backstop; failure to install required Linux limits denies execution.
Document platform-specific guarantees: Unix `resource` is not portable to Windows,
and address-space limits are not equally effective on macOS. Parent deadlines,
byte/count caps, and bounded concurrency remain mandatory everywhere. Process
isolation is not a complete OS sandbox or a guarantee against host-wide OOM;
deployment/container memory limits remain necessary.

## Service and Caller Contract

Use one typed `ModerationExecutionError` with sanitized reason codes such as
`deadline_exceeded`, `capacity_exhausted`, `size_exceeded`, `worker_unavailable`,
`worker_crashed`, `invalid_response`, and `cleanup_failed`.
All service evaluation/redaction entry points propagate it after reporting.
No string/tuple-only method can truthfully represent incomplete redaction as a
successful ordinary return. Do not invent a matched pattern/span or classify the
user as having violated a content rule.

Existing normal success shapes and `moderation_service.py` imports remain intact.
`evaluate_action*()` still dispatch through `self.evaluate_text()`; evaluation
still uses `self.redact_text()` when required. The guarantees cover the default
implementation, not arbitrary trusted application overrides that execute their
own regex. Direct evaluator tests may run benign patterns locally; live production
callers must go through the isolated service path.

Async production callers must use a cancellable async runner/offload boundary;
regex and blocking process waits must not run on the event loop. Cancellation
must reach the process owner, not merely abandon an awaiting thread. Retain sync
service compatibility for sync callers; adapt streaming's synchronous transform
boundary before enabling isolation there. Do not hide blocking waits in a sync
callback executed on the event loop.

Audit all current service callers, not only the new runner. In particular:
Chat input/output paths and `moderation_pipeline.py`, Chat streaming transforms
and final flushes, the alternate endpoint Chat path, `/moderation/test`, Workflow
moderation adapters, and Audio STT policy redaction. Add explicit typed-error
handling before broad noncritical catches. Do not retry the failed request through
`check_text()`, another compatibility method, or a fail-open fallback.

- Input failure: do not invoke the model or continue ingestion with affected text.
- Non-stream output failure: return no affected generated text and do not save it
  as a successful completion.
- Streaming failure: latch a terminal stop, discard current chunk and holdback,
  stop/cancel the upstream producer, emit the established structured stream error,
  and prevent finalizers from flushing content or marking success. Bytes already
  emitted cannot be recalled; this work does not promise whole-response buffering.
- HTTP surfaces: return a sanitized operational-unavailability error (proposed
  HTTP 503, `moderation_unavailable`), not a fabricated content-policy violation.
- Workflows/audio: return their existing structured failure envelope without
  original/partially redacted text in success fields or downstream continuation.
- Client cancellation: kill/reap and preserve cancellation semantics. Ordinary
  disconnects are not admin incidents; unexpected cleanup failure is.

The 2-second budget is per service call, not a lifetime budget for an unlimited
stream. Each streaming moderation batch is bounded and a first execution failure
terminates that stream. A cumulative stream quota is separate work.

## Admin Incidents and Privacy

Add a dedicated parent-side operational-incident reporter, independent of
`MODERATION_REVIEW_CAPTURE_ENABLED`. Reuse `ModerationReviewService` and its
DB-management store, with an explicit metadata-only construction path:

- category `moderation_engine_failure`, severity `high`, status `needs_review`,
  source type `moderation_engine`, recommendation `escalate`;
- literal excerpt `[Moderation execution failed; content withheld]`;
- allowlisted reason, operation, actual phase when known, bounded sizes/timings,
  server-generated correlation ID, and reporter version;
- no input, output, snippets, regex text, replacement values, matched data,
  arbitrary caller context, or raw exception/child stderr;
- no synthetic content match and no automatic public issue.

For phase `None`, use an explicit phase-unspecified marker in context and create
separate input/output administrative rows only if required by the existing DB
phase constraint; neither row asserts a real content violation. The builder must
avoid attributing an unknown phase to a user's input or output.

Deduplicate atomically by reporter version, phase marker, reason, operation, and
UTC five-minute bucket using the existing unique idempotency key. Do not include
content hashes or user-controlled source identifiers. Every occurrence increments
a bounded-label operational metric even when its review row is deduplicated.
The existing admin list/detail/audit surfaces must show the safe metadata and
distinguish these rows from content reports. Dismissing an incident never authorizes
the blocked content or disables safety enforcement.

Use the existing DB abstraction, not raw SQL in Moderation. Add a narrowly scoped
incident-write timeout through that abstraction (proposed 250 ms busy wait),
including first-use initialization; do not globally change normal review behavior.
Reporting cannot hold the event loop, retry indefinitely, or undo a block. If the
store fails, emit a sanitized ERROR event and failure metric, then preserve the
original block. Durable admin visibility cannot be guaranteed while all stores
are unavailable; document and test that degraded state explicitly.

## Validation and Rollout

1. Characterize successful existing evaluator behavior before moving execution:
   flags/Unicode/anchors, raw rules, phase/category filters, ordering/ties, literal
   replacements, zero-length and count quirks, short/long scans, and overrides.
2. Deterministic runner tests with controlled clock/process/pipe doubles: startup
   stalls, partial/oversize IPC, saturation, deadline sharing, crash, malformed
   results, cancellation, slot quarantine, reaping, and lifecycle shutdown.
3. Real child tests with an outer harness timeout: catastrophic backtracking is
   killed, the parent survives and handles a subsequent safe request, async
   heartbeat remains responsive, and no child/pipe/reader is leaked. Never run
   adversarial patterns in the pytest/application main process.
4. Integration tests for every caller family above: no model invocation on input
   failure, no affected output/save, no fallback retry, terminal stream behavior,
   and bounded cancellation. Assert operational versus content-block distinction.
5. Incident tests with capture disabled, concurrent deduplication, absent phase,
   DB lock/failure, safe admin rendering, and secret-bearing sentinel content to
   prove it cannot reach logs, metrics, incidents, stderr, or error responses.
6. Package/import tests and Linux/macOS/Windows process tests; record actual
   platform guarantees. Benchmark process-start and streaming overhead. Revisit
   proposed numerical defaults before approval of the implementation plan if
   measurements make them unsuitable; never silently weaken isolation.

Run focused suites and relevant Chat/Workflow/Audio/admin integration tests,
compile checks, lint, touched-scope Bandit, Backlog canonical/ID gates, and a fresh
code review before a PR. No unsafe inline execution rollout toggle. If a deployment
cannot spawn a child, enabled moderation fails closed with an admin incident.

## ADR Check

ADR required: yes. Proposed record:
`Docs/ADR/086-moderation-killable-stdlib-regex-execution.md`.
The security failure contract and parent/child ownership are durable decisions;
the existing ADR index does not specify this Moderation boundary. Related ADRs:
021 (lifecycle ownership), 028 (avoid general pickle), and 006 (security gate).
This draft does not change those accepted records.

## Builder Discretion and Approval

The builder may select module names and concrete bounded IPC/offload primitives,
but not substitute engines, execute matching inline, weaken fail-closed behavior,
skip current caller families, or bypass public override dispatch. No new admin
dashboard is required if existing review surfaces can display the metadata safely.

Fresh builder check: completed by an independent Python backend/security reviewer
on 2026-11-02; result: "nothing important left". No blocking questions raised.
Requester approval of this written design: pending.

## References

- [Python re](https://docs.python.org/3.12/library/re.html)
- [Python subprocess, including timeouts and kill](https://docs.python.org/3.12/library/subprocess.html)
- [Python resource platform limitations](https://docs.python.org/3.12/library/resource.html)
