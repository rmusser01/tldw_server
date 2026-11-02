# Moderation Killable Standard-Library Regex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate Moderation's runtime stdlib `re` matching/redaction in killable children, block affected content on execution failure, and surface sanitized admin incidents.

**Architecture:** A Moderation-owned supervisor launches one short-lived worker per evaluator batch, with a parent-owned deadline shared across service decision/redaction dispatch. Typed failures propagate through every current caller without fallback; async callers offload orchestration with explicit cancellation ownership. Existing review persistence/UI display metadata-only operational incidents independently of optional content capture.

**Tech Stack:** Python stdlib `re`, `subprocess`, bounded JSON pipes, threading/asyncio and existing lifecycle helpers; pytest/Hypothesis; existing SQLite review store; React/TypeScript/Vitest for the existing review UI. No new dependency or engine substitution.

**Spec:** `Docs/superpowers/specs/2026-11-02-moderation-killable-stdlib-re-design.md` (approved 2026-11-02).

**Tracking:** TASK-15415 (planning); TASK-15414 (approved design).

## Global Constraints

- Python >=3.12, matching `pyproject.toml`; do not rely on newer process-management APIs without a supported implementation on 3.12.
- Standard-library `re` only in the runtime worker; no inline matching fallback, generic pool, or per-pattern launch.
- 2 seconds per service operation; at most 4 active workers per API process; immediate nonblocking admission.
- 1,000,000 input characters; 1,000,000 output characters; 256 rules; 4,096 characters per pattern/replacement; 256 KiB total encoded policy; 32 MiB request frame; 8 MiB response frame.
- Linux child address-space ceiling: 512 MiB; CPU/core-dump backstops before policy decoding. Parent deadlines, byte/count caps, and bounded concurrency remain mandatory everywhere.
- Incident category `moderation_engine_failure`, severity `high`, status `needs_review`, source type `moderation_engine`, recommendation `escalate`; excerpt `[Moderation execution failed; content withheld]`.
- Atomic five-minute UTC-bucket incident deduplication; 250 ms incident DB busy wait; capture flag must not disable incidents. A busy wait is not a guarantee against arbitrary filesystem stalls.
- Preserve supported imports, service override dispatch, normal result/tuple shapes, existing search geometry, zero-length/count quirks, literal replacements, and raw `re.Pattern` compatibility.
- Scope excludes initial compiler/lint isolation, new LLM-surface coverage (TASK-13950.27), and whole-response stream buffering. Already-emitted bytes cannot be recalled.
- No product code in the planning task. Before implementation edits, find/create one implementation Backlog task linked to this plan; keep its notes/stage state current and commit it with related work.
- One integration PR must contain the complete default-service activation and caller handling. Do not ship an intermediate service activation with unconverted callers.
- Read current root and nested AGENTS instructions before execution, activate the project venv for Python tools, use apply_patch, and do not revert unrelated work.

## Review Focus

- A client disconnects during process creation: the eventual child remains owned and is killed/reaped, not detached (Task 2).
- Redaction expansion exceeds limits before response encoding: no giant successful intermediate/final result reaches the parent (Task 1).
- A stream fails after a safe prefix: no holdback, final flush, partial-save callback, or late success turns the failure into success (Task 4).
- `phase=None`, disabled moderation, or an overridden method: preserve established guards/dispatch without false input/output attribution (Tasks 3 and 4).
- Incident DB contention or malformed child data containing secrets: remain blocked, preserve cancellation, and never expose failed content in diagnostics (Tasks 2-4).

## File Ownership and Interfaces

New files under `tldw_Server_API/app/core/Moderation/`:

| File | Responsibility |
| --- | --- |
| `execution_models.py` | Frozen safety ceilings, request/response types, per-call deadline/cancellation scope and stable reason/operation types. No I/O or regex execution. |
| `execution_protocol.py` | Bounded policy snapshot and versioned frame validation/encoding; no pattern compilation. |
| `regex_worker.py` | Minimal trusted child entry point, resource posture, policy reconstruction, existing evaluator dispatch, bounded output. |
| `execution_runner.py` | PID-owned admission, process/pipe registry, deadlines, kill/reap/quarantine, sync batch execution and lifecycle shutdown. |
| `async_execution.py` | Cancellable orchestration offload preserving ContextVars and trusted public method dispatch. |
| `execution_incidents.py` | Allowlisted metadata-only incident construction/reporting and bounded-cardinality metrics. |

Shared exception: define `ModerationExecutionError(RuntimeError)` in existing
`core/exception_types.py`, re-export it from `core/exceptions.py`. Worker-side
imports use the framework-neutral owner, not FastAPI's exception translation.
Its constructor is `__init__(self, reason: str) -> None`; allow only the seven
reason codes in the spec. `str(error)`/`repr(error)` must not include content.
Detailed sizes/timings belong to the operation scope, not arbitrary exception args.

`ExecutionBounds` is frozen and contains the exact defaults above; nonfinite,
nonpositive, boolean, or above-ceiling safety configuration fails validation.
Existing `EvaluationLimits` remain separate and preserve characterized coercions.
`ExecutionScope` contains `deadline: float`, `cancelled: threading.Event`,
`correlation_id: str`, and operation-local reporting state. Use a ContextVar
context manager `execution_scope(*, timeout_s: float = 2.0) -> Iterator[ExecutionScope]`;
nested calls inherit the outer scope instead of resetting it. Inject clocks and
process factories through runner construction for deterministic tests, not policy.

`Operation = Literal["evaluate", "redact", "redact_with_count"]`.
`WorkerRequest` carries operation, text, phase, a plain-data policy snapshot,
evaluation-limit snapshot, and safety ceilings. `WorkerResponse.result` is
`ModerationEvaluationResult | str | tuple[str, int]`, constrained by operation.
Neither wire object carries service instances, callbacks, match objects, or pickle.

## Stage 1: Bounded Worker and Compatibility

**Goal:** A standalone worker runs existing evaluator semantics with bounded data.
**Success Criteria:** Benign parity, strict protocol validation, and bounded output; production service remains unchanged.
**Tests:** New protocol/worker suites plus existing evaluator characterization.
**Status:** Not Started

### Task 1: Protocol, Worker, and Output Guard

**Files:** Create the first three modules in the ownership table; modify
`core/exception_types.py`, `core/exceptions.py`, and
`Moderation/policy_evaluator.py` only for a shared optional output-allocation guard.
Create `tests/unit/test_moderation_execution_protocol.py` and
`tests/unit/test_moderation_regex_worker.py`; extend
`tests/unit/test_moderation_policy_evaluator_characterization.py`.
All Python paths here are relative to `tldw_Server_API/` unless prefixed `core/`
or `Moderation/`, which mean `tldw_Server_API/app/core/` and its Moderation child.

**Interfaces:**
- `snapshot_request(operation: Operation, text: str, policy: ModerationPolicy, phase: str | None, limits: EvaluationLimits, bounds: ExecutionBounds) -> WorkerRequest`.
- `encode_request(request: WorkerRequest) -> bytes`; `decode_request(frame: bytes) -> WorkerRequest`.
- `encode_response(response: WorkerResponse) -> bytes`; `decode_response(frame: bytes, request: WorkerRequest) -> WorkerResponse`.
- `execute_request(request: WorkerRequest) -> WorkerResponse`; `regex_worker.main() -> int`.
- `OutputBudget(max_chars: int)` in `execution_models.py`; evaluator redaction entry points gain keyword-only `output_budget: OutputBudget | None = None`. Existing positional/default behavior remains unchanged.

- [ ] **Step 1: Write failing protocol and successful-parity tests.**

```python
def test_unknown_wire_version_is_rejected():
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(b'{"version":999}')

def test_requester_defaults_are_fixed():
    bounds = ExecutionBounds()
    assert (bounds.timeout_s, bounds.max_workers, bounds.max_output_chars) == (2.0, 4, 1_000_000)
```

Add named cases `test_raw_pattern_flags_round_trip`,
`test_worker_matches_evaluator_characterization`,
`test_literal_backslash_replacement_stays_literal`,
`test_zero_length_and_raw_count_limits_keep_legacy_behavior`,
`test_debug_flag_cannot_write_protocol_or_stderr`,
`test_oversized_policy_is_rejected_before_encoding`,
`test_response_span_and_count_are_validated`, and
`test_intermediate_expansion_aborts_before_join`. Compare complete benign results
against the existing evaluator; assert the expansion case raises `size_exceeded`
without allocating the oversized joined result. Property-test bounded JSON frames,
Unicode/control characters, and malformed fields. Reject NaN/infinity and bool-as-int.

- [ ] **Step 2: Run new tests red.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_protocol.py tldw_Server_API/tests/unit/test_moderation_regex_worker.py -q` in the activated venv.
Expected: missing new modules/interfaces, not an environment or collection failure unrelated to the task.

- [ ] **Step 3: Implement these interfaces.** Use one four-byte big-endian length-prefixed UTF-8 JSON frame and EOF, with exact version/tag/schema validation and strict size checks. Snapshot `.pattern`/integer `.flags` without compiling in the parent, distinguish raw rules, and reconstruct canonical models in the child. Call the existing evaluator, not a copied algorithm. Check output growth during substitution callbacks and list/slice assembly before large joins, preserving literal replacement and zero-length semantics. No callbacks cross IPC.

Launch entry is a fixed trusted file with `sys.executable`, `-I`, and `-S`;
resolve its installed package root from that file, not cwd/PYTHONPATH. Only the
minimal model/evaluator dependency graph may load. Install Linux address-space
512 MiB, CPU 3 seconds, and core-dump zero limits before decoding requests;
validate/report resource posture before execution. On other platforms implement
only supported child limits and document the actual guarantee. Required Linux
limit failure yields `worker_unavailable`. Capture compilation debug output to
the null sink; stdout is protocol-only, stderr is not a content-bearing diagnostic.
Unit tests call the worker directly only with benign patterns.

- [ ] **Step 4: Run new tests and existing evaluator tests green.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_protocol.py tldw_Server_API/tests/unit/test_moderation_regex_worker.py tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py -q`.
Expected: all pass, including unchanged benign baseline behavior.
- [ ] **Step 5: Review the diff and commit** as `feat(moderation): add bounded stdlib regex worker (<implementation task ID>)` with only Task 1 paths and tracking updates.

## Stage 2: Parent Supervision and Cancellation

**Goal:** A parent can terminate, contain, and reap any admitted worker.
**Success Criteria:** Deadline/capacity/cancellation failures are typed and bounded; no application-loop blocking.
**Tests:** Deterministic process doubles plus outer-timeout real-child recovery tests.
**Status:** Not Started

### Task 2: Supervisor and Async Orchestration

**Files:** Create `Moderation/execution_runner.py`, `Moderation/async_execution.py`,
`tests/unit/test_moderation_execution_runner.py`, and
`tests/unit/test_moderation_async_execution.py`.
Create `app/services/startup_moderation_workers.py`; register its provider in
`app/services/startup_worker_groups.py`. Extend `tests/Services/test_lifecycle_worker_catalog.py`.

**Interfaces:**
- `ModerationExecutionRunner(bounds: ExecutionBounds, *, clock: Callable[[], float] = time.monotonic, process_factory: Callable[..., subprocess.Popen] = subprocess.Popen)`.
- `runner.execute(request: WorkerRequest, scope: ExecutionScope) -> WorkerResponse`; `runner.close(*, timeout_s: float = 0.5) -> None`.
- `get_execution_runner() -> ModerationExecutionRunner`: one PID-owned instance; no process spawn at import time.
- `async run_moderation_async(call: Callable[[], T]) -> T`: establish/inherit scope, offload trusted public method invocation, propagate cancellation to the shared event and process owner.
- `provide_moderation_worker_specs(context: WorkerLifecycleContext) -> Sequence[WorkerSpec]`: lifecycle registration for lazy supervisor ownership and bounded shutdown.

- [ ] **Step 1: Write failing runner/async tests.** Assert
`test_fifth_active_worker_is_refused_without_queue` gives `capacity_exhausted`;
`test_decision_and_redaction_share_original_deadline` never grants a fresh two seconds;
`test_cancel_during_spawn_kills_eventual_child` retains registry ownership;
`test_unreaped_child_keeps_slot_quarantined` does not admit replacement work;
`test_partial_oversize_and_trailing_frames_are_rejected` bounds reads before buffering;
`test_shutdown_rejects_admission_and_reaps_registry` leaves no live child;
`test_context_does_not_leak_between_concurrent_calls` preserves independent scopes.
Add `test_async_heartbeat_runs_while_worker_stalls` and
`test_cancelled_async_call_signals_owner_without_admin_incident`.
- [ ] **Step 2: Run both new unit suites red.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_runner.py tldw_Server_API/tests/unit/test_moderation_async_execution.py -q`.
Expected: missing runner/async interface assertions fail.
- [ ] **Step 3: Implement supervision.** Use a fixed argument vector, `shell=False`, closed unrelated descriptors, and allowlisted interpreter environment. Count PID-local slots with thread-safe immediate admission. Register ownership across spawn/cancel races. Concurrent bounded pipe transfer must terminate on deadline or cancellation; do not use unbounded `communicate()` capture. Kill immediately on execution failure, allow 0.5 seconds for cleanup, then retain the slot/registry entry for bounded reaper work. At most four owned children/readers exist; stop admission if cleanup cannot be completed. Reaper work may not create unbounded threads. Do not log raw child output, argv content, or exception messages.

Use one bounded orchestration-thread admission per active child, with a dedicated
cancel event and copied ContextVars. `run_moderation_async()` must keep ownership
through cancellation and signal the synchronous supervisor; merely cancelling a
future is insufficient. Audit the slot/pipe/reader lifecycle on every exit.
Reset PID-owned admission safely after an API-worker fork and reject new work
during lifecycle shutdown. Do not spawn processes from a daemon/pool worker.
- [ ] **Step 4: Run deterministic suites green, then real-child tests under an outer pytest timeout.**
Create `test_pathological_re_child_is_killed_and_next_safe_request_succeeds`,
marked integration and `pytest.mark.timeout(10)`. The adversarial expression is
sent only to the child; assert typed deadline failure, dead/reaped owned child,
and successful later benign request. Test nonzero exit, unsupported limit posture,
installed-path launch, no imported `regex`, and event-loop heartbeat. Run
`python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_runner.py tldw_Server_API/tests/unit/test_moderation_async_execution.py tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py -q`.
- [ ] **Step 5: Review and commit** as `feat(moderation): supervise regex worker deadlines and cancellation (<implementation task ID>)`.

## Stage 3: Safe Admin Incidents

**Goal:** Execution failure reporting is visible without retaining affected content.
**Success Criteria:** Independent of optional capture, atomically deduplicated, lock-bounded, and accurately rendered.
**Tests:** Backend report/store/privacy tests and shared review UI tests.
**Status:** Not Started

### Task 3: Operational Incident Reporting

**Files:** Create `Moderation/execution_incidents.py` and
`tests/unit/test_moderation_execution_incidents.py`; modify
`Moderation/review_service.py` and `core/DB_Management/Moderation_Review_DB.py`.
Modify the existing shared UI files under
`apps/packages/ui/src/components/Option/ModerationReview/`:
`review-utils.ts`, `ReviewQueueList.tsx`, `ReviewItemDetail.tsx`, and
`__tests__/review-utils.test.ts`; create `__tests__/ExecutionIncident.test.tsx`.
Extend `tests/unit/test_moderation_review_store.py` and
`apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review.spec.ts`.

**Interfaces:** `report_execution_failure(error: ModerationExecutionError, *, operation: Operation, phase: str | None, scope: ExecutionScope, input_chars: int) -> None`.
The reporter owns no regex/content argument and catches store failure without
masking the original execution error. Add keyword-only `busy_timeout_ms: int | None = None`
to `ModerationReviewStore.__init__`; apply it to connect, PRAGMA, and schema-first-use
operations for the incident store only, leaving the default store unchanged.

- [ ] **Step 1: Write failing backend and renderer tests.**

```python
def test_failure_excerpt_is_constant():
    assert ENGINE_FAILURE_EXCERPT == "[Moderation execution failed; content withheld]"
```

Add `test_capture_disabled_still_records_incident`,
`test_concurrent_failures_deduplicate_in_same_five_minute_bucket`,
`test_new_bucket_records_new_incident`,
`test_unknown_phase_has_no_false_attribution`,
`test_store_locked_or_unavailable_preserves_original_block`,
`test_failure_sentinels_never_enter_logs_metrics_or_review`, and
`test_normal_disconnect_is_not_reported`. Assert the exact category/source/status/
severity/recommendation values in Global Constraints, no match metadata, and one
occurrence metric increment per failure even if persistence deduplicates.
- [ ] **Step 2: Run backend suite red and new UI tests red.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_incidents.py -q`.
Run `pnpm --dir apps/packages/ui exec vitest run src/components/Option/ModerationReview/__tests__/ExecutionIncident.test.tsx`.
Expected: missing reporter/operational renderer behavior, not dependency failures.
- [ ] **Step 3: Implement reporting and narrow UI presentation.** Metadata is an explicit allowlist; never pass `policy.to_dict()`, exception traceback, raw stderr, snippets, or arbitrary context. Dedup key is reporter version, operation, actual phase marker, reason, and five-minute UTC bucket. Report at most once per failed top-level service operation; nested propagation must not duplicate metrics/incidents.

For unspecified phase, preserve the existing DB phase constraint: write two
administrative projections (`input`, `output`) with context `phase="unspecified"`
and the same server-generated incident-group ID; include the storage phase in
each row's idempotency key. Render both as operational failures with phase
"Unspecified", not alleged input/output violations. Known-phase failures have
one row. No schema widening or new dashboard. Review actions do not replay
content or change enforcement. Show allowlisted reason/operation/correlation
metadata and ensure incident wording remains distinct from user-policy matches.
Use bounded-cardinality reason/operation/phase labels only; no correlation IDs in metrics.
- [ ] **Step 4: Run backend and UI suites green.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_incidents.py tldw_Server_API/tests/unit/test_moderation_review_store.py tldw_Server_API/tests/unit/test_moderation_review_service.py -q` and
`pnpm --dir apps/packages/ui exec vitest run src/components/Option/ModerationReview/__tests__`.
Expected: existing capture/review behavior unchanged and operational incidents correctly distinguished.
- [ ] **Step 5: Review and commit** as `feat(moderation): surface sanitized execution incidents (<implementation task ID>)`.

## Stage 4: Atomic Service and Caller Activation

**Goal:** All current moderation callers use isolated execution without fail-open fallback or event-loop blocking.
**Success Criteria:** Sync API compatibility, correct override dispatch, awaited streaming transforms, terminal failure propagation, and no affected success/save.
**Tests:** Cross-surface failures plus existing moderation, streaming, Workflow and Audio regressions.
**Status:** Not Started

### Task 4: Activate Facade, Async Callers, and Stream Stop

**Files:** Modify `Moderation/moderation_service.py`; Chat
`chat_service.py`, `moderation_pipeline.py`, `streaming_utils.py`,
`streaming_pipeline.py`; endpoints `chat.py`, `moderation.py`,
`audio/audio_streaming.py`, `audio/audio_transcriptions.py`;
`core/Workflows/adapters/llm/moderation.py`;
`core/Ingestion_Media_Processing/Audio/stt_policy.py`;
`core/Audio/Realtime/default_pipeline.py`.
Create `tests/unit/test_moderation_execution_service.py`,
`tests/unit/test_moderation_execution_surfaces.py`,
`tests/Chat/unit/test_streaming_moderation_execution_failure.py`.
Extend existing `tests/unit/test_moderation_policy_evaluator_delegation.py`,
`tests/Chat/unit/test_streaming_utils.py`,
`tests/Chat_NEW/integration/test_moderation.py`, and
`tests/Audio/test_stt_policy_payloads.py`.

**Interfaces:** Existing public service methods/signatures stay intact and gain
the typed failure contract. Each opens/inherits one `execution_scope`, runs the
correct isolated batch, reports once, and re-raises with sanitized exception
chaining. `_evaluate_text_core()` retains `self.redact_text()` dispatch; tuple
facades retain `self.evaluate_text()` dispatch. Disabled/no-op guards are specific
to each existing entry point; do not make redaction obey a guard it did not
previously obey. No need to spawn for a genuinely no-op request.

In `streaming_utils.py`, add
`async _apply_text_transform(transform: Callable[[str], str | Awaitable[str]], text: str) -> str`
and `async _flush_text_transform(transform: object) -> str`.
Accept existing synchronous transforms and await actual awaitables. Change the
pipeline request type to `Callable[[str], str | Awaitable[str]] | None`.
Make `_absorb_transform_tail()` async and await it at every call site.

- [ ] **Step 1: Write failing service/caller/stream tests.**
For every seven execution reasons, assert `test_input_failure_never_calls_provider`,
`test_output_failure_never_returns_or_saves_affected_reply`,
`test_compatibility_fallback_does_not_retry_failed_evaluation`,
`test_direct_redaction_never_returns_partial_text`, and
`test_disabled_and_noop_guards_preserve_existing_semantics`.
Assert HTTP status 503 and safe code `moderation_unavailable` for operational HTTP
failures; preserve ordinary content-block status behavior.
Add `test_override_dispatch_preserved_with_shared_budget` and
`test_evaluate_then_redact_does_not_reset_deadline`.

Streaming tests cover structured/text provider chunks, normal/error/cancel flush,
partial settlement, before-success callbacks, and provider close/cancel. Assert
`test_failure_after_safe_prefix_never_flushes_or_saves_tail` emits only the safe
prefix plus structured `moderation_unavailable` error, never success completion;
`test_awaitable_transform_is_awaited_in_every_path` emits strings, not coroutine
representations; `test_sync_transform_remains_compatible` keeps existing behavior.
Audio/Workflow tests assert failure envelopes contain no original text, timed
segment, cached snapshot, or success marker; no downstream continuation.
- [ ] **Step 2: Run the three new suites red.**
Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_service.py tldw_Server_API/tests/unit/test_moderation_execution_surfaces.py tldw_Server_API/tests/Chat/unit/test_streaming_moderation_execution_failure.py -q`.
Expected: current inline service/caller behavior fails the new assertions.
- [ ] **Step 3: Implement facade routing and all caller adaptations together.** Replace direct evaluator execution in the default service with Task 2 runner batches. Preserve decision/redaction as two batches under one deadline and public overrides. Re-raise `ModerationExecutionError` before broad noncritical catches; callers must not fall back through `check_text()` or reinterpret failure as a content match. Async callers invoke public methods via `await run_moderation_async(lambda: ...)`, including whole transcript/payload operations where necessary.

Update all transform/flush call sites, including partial-settlement/error paths,
to await safely. On execution failure latch a terminal stop before any flush,
clear holdback, close/cancel upstream, emit the existing stream error structure
with the safe operational code, and prevent partial save or final success. Typed
failure must reach audio HTTP/WebSocket adapters before the existing generic STT
catch can convert it into a successful placeholder transcript. Do not change
unrelated STT fallback behavior or partial-transcript opt-out policy.
- [ ] **Step 4: Run focused suites green and audit call sites.**
Run the Step 2 command plus existing evaluator-delegation, Chat_NEW moderation,
`tests/Chat/unit/test_streaming_utils.py`, `tests/Audio/test_stt_policy_payloads.py`,
`tests/Audio/test_audio_transcription_retention_and_redaction.py`, and
`tests/Workflows/adapters/test_workflow_adapters_user_scope.py` under pytest.
Run `rg -n 'evaluate_text\(|evaluate_action|redact_text|check_text\(' tldw_Server_API/app`;
classify all actual Moderation calls and indirect transcript-policy calls. Add any
missed current caller and regression to this task rather than shipping a bypass.
- [ ] **Step 5: Review the complete activation diff and commit** as `feat(moderation): enforce isolated fail-closed execution across current callers (<implementation task ID>)`. No partial activation release.

## Stage 5: Release Evidence and Final Review

**Goal:** Verify semantics, failure containment, packaging, platforms, and operational cost.
**Success Criteria:** All changed-scope checks pass; residual limitations are explicit; independent whole-branch review has no unresolved issues.
**Tests:** Full focused regression manifest, platform/packaging probes, UI smoke and touched-scope security checks.
**Status:** Not Started

### Task 5: Validation and Documentation

**Files:** Create `tests/unit/test_moderation_execution_packaging.py` and
`Docs/superpowers/reviews/moderation-backend/2026-11-02-killable-re-validation.md`;
update `Moderation/README.md` if present (otherwise create it), the implementation
Backlog task and this plan's statuses. Inspect current CI shard registration and
register new suites in its existing manifest when explicit registration is needed;
do not invent a parallel runner or dispatch required gates manually.

**Interfaces:** No new runtime API; consume Tasks 1-4 and document exact defaults,
platform posture, process-count multiplication, degraded reporting, and safe errors.

- [ ] **Step 1: Add failing package/import and end-to-end fault-injection tests.** Assert installed-distribution worker launch outside the repo cwd, no `regex` import or app/provider/DB initialization, no content in argv/logs, missing interpreter/worker blocks, normal `re` flags preserved, Linux limit-install failure denied, and platform-appropriate cancellation/kill behavior. Run adversarial tests only behind the outer timeout. Extend the existing Moderation review Playwright test with an operational incident and verify desktop/mobile labels and safe detail fields.
- [ ] **Step 2: Run package/UI tests red, implement only missing packaging/registration/docs, then run green.** Use the repo's existing wheel/build and frontend Playwright setup, not a new build system. Do not mark an unavailable OS/browser run as passed; record the exact environment and remaining gate. A platform guarantee cannot be claimed without its platform test evidence.
- [ ] **Step 3: Execute the full regression manifest and measurements.** Run all new suites plus the existing Moderation unit files, Chat_NEW moderation/category tests, relevant streaming, Audio, Workflow, review/admin and lifecycle suites. Record commands, revisions, counts, warnings, and failures, not only totals. Measure cold start, benign short/long batch latency, streaming overhead, saturation and cancellation with four workers; no inline fallback if overhead is unacceptable. If the 2-second/default ceilings fail legitimate representative workloads, return with measurements for a design amendment before changing approved limits.
- [ ] **Step 4: Run compilation, lint, Bandit and metadata gates.** Stage the new Task 5 files so the diff inventory includes them, then activate the venv and run these commands from the worktree root:

```bash
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API | rg '\.py$' | xargs python -m py_compile
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API | rg '\.py$' | xargs python -m ruff check
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API/app | rg '\.py$' | xargs python -m bandit -r -f json -o /tmp/bandit_moderation_killable_re.json
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API/tests | rg '\.py$' | xargs python -m bandit -r -s B101 -f json -o /tmp/bandit_moderation_killable_re_tests.json
pnpm --dir apps/packages/ui exec vitest run src/components/Option/ModerationReview/__tests__
pnpm --dir apps/tldw-frontend typecheck
PYTHONPATH=tools/backlog-py/src python -m backlog_py task normalize --check
python Helper_Scripts/ci/check_backlog_task_ids.py .
PYTHONPATH=tools/backlog-py/src python -m pytest tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py -q
git diff --check
git diff --cached --check
```

Expected: exit 0 for compilation/lint/test/metadata checks; inspect both Bandit
JSON reports and resolve new findings. The diff base is the recorded pre-plan
commit; do not scan unrelated paths or silently omit staged new files. Scope
normalization to changed task files if whole-tree checks expose unrelated legacy
records; do not rewrite unrelated records. Run touched-file ESLint with the existing
shared UI config and record the command. Explain pre-existing/environment failures
precisely rather than claiming the gate passed.
- [ ] **Step 5: Request whole-branch code review and commit verified evidence.** Use requesting-code-review, address every substantiated finding with tests, rerun affected checks, and update task notes. Preserve subagent per-task reviews. No PR publish/merge is authorized by this plan approval alone; follow the user's eventual publishing request, seven required gates, current merge-queue rules, and human-written Change summary policy. Remove only this implementation plan after all stages finish, per AGENTS, retaining its history/reference in the task and final report.

## ADR Assessment and Self-Review

ADR required: yes; governed by accepted
`Docs/ADR/086-moderation-killable-stdlib-regex-execution.md` (plus lifecycle ADR-021).
This plan implements its process ownership and security contract; it introduces
no engine change, generic sandbox framework, or replacement persistence authority.

Coverage checked: process/wire/output safety (1); deadline/admission/cancellation/
resource/lifecycle behavior (1-2); incident/admin/privacy behavior (3); compatibility
dispatch, every current caller family, async waits and terminal streaming (4);
platform/packaging/benchmark/security/metadata/review evidence (5). The five Review
Focus conditions each have named tests in their owning tasks. Interfaces and paths
were checked against the current source; new names are defined above. No product
code has been written while preparing this plan.

Execution handoff: subagent-driven, preserving the requester's earlier choice.
Plan review/confirmation is pending. Each stage is Not Started; do not mark tests
or runtime guarantees passed based on this document.
