# Audit Strict Remediation Design

Date: 2026-04-09
Status: Approved for planning
Scope: Confirmed Audit defects, strict durability for selected security/state-changing flows, bounded compatibility cleanup
Related: 2026-04-08 Audit module review, GitHub issue `rmusser01/tldw_server#1053`

## Summary

This tranche fixes the confirmed Audit and Sharing defects from the Audit module review and hardens a narrow set of high-value call sites so they do not report success unless their audit contract is satisfied.

It is intentionally not the broad repo-wide audit unification effort. That larger refactor remains tracked separately in `rmusser01/tldw_server#1053`.

The design decision for this tranche is:

- fix confirmed Audit durability and migration defects first
- make selected security and state-changing paths fail closed on audit persistence failure
- keep compatibility layers only where they avoid immediate breakage
- avoid expanding into a repo-wide audit architecture rewrite

## Why This Exists

The review surfaced a mix of confirmed bugs and risky behavior:

- Sharing legacy backfill is broken because the migration calls a missing writer method
- `UnifiedAuditService.stop()` can silently lose buffered events during final flush failure
- shared audit migration can over-report inserted rows on commit failure
- DI shutdown helpers can miss shared-mode services or hide real stop failures
- Evaluations run creation persists and announces a run before the mandatory audit write succeeds
- Chat moderation enforcement paths still treat audit as background best-effort work
- Jobs audit currently relies on side-channel best-effort bridge behavior in places where operators would reasonably expect durable audit intent
- AuthNZ API-key management history still bypasses unified audit as the source of truth

These are reliability problems first, not taxonomy or UX problems. The safest response is a narrow remediation tranche with explicit boundaries.

## Goals

- Fix the Sharing legacy backfill regression.
- Make final Audit shutdown flushes durable-or-loud instead of silently lossy.
- Make shared audit migration counts and checkpoints commit-bound.
- Make DI shutdown behavior correct in shared mode and explicit about stop failures.
- Make Evaluation run creation fail closed if the mandatory audit write cannot persist.
- Make Chat moderation enforcement stop reporting successful completion when its mandatory audit write fails.
- Require durable audit intent for a narrow Jobs subset that has clean source-side transaction boundaries in this tranche.
- Make API-key create, virtual-create, rotate, and revoke use unified audit as the mandatory source of truth.
- Keep the remediation bounded enough to implement and verify with targeted tests.

## Non-Goals

- Repo-wide audit unification across every remaining domain-local or logger-only audit path.
- Full Jobs lifecycle audit redesign across every worker/internal path.
- A full streaming abstraction rewrite to support async-per-chunk moderation transforms.
- Removal of every legacy compatibility table in the same tranche.
- Broad audit taxonomy normalization beyond the touched paths.

## Review-Driven Corrections

This design incorporates the following corrections from the design self-review:

- Jobs strictness is narrowed to source-side durable audit intent for `job.created` in this tranche. Full worker-lifecycle unification stays with issue `#1053`.
- Streaming Chat moderation does not attempt to await audit persistence inside the synchronous text transform. Instead, mandatory moderation audit tasks are tracked and must succeed before the stream is allowed to complete successfully.
- `shutdown_all_audit_services()` does not become raising-by-default, because app shutdown and cleanup-heavy tests currently rely on best-effort teardown. It will instead report aggregated failures and support an explicit raising mode for targeted callers/tests.
- Evaluations “run started” webhook emission moves behind the new audit-plus-run-persistence success boundary so external systems are not notified about runs that never committed.
- AuthNZ compatibility is one-way: unified audit is mandatory, while the legacy `api_key_audit_log` mirror remains best-effort during the transition.

## Architectural Decisions

### 1. Strict Audit Means Surface-Specific Success Boundaries

“Strict audit” does not mean the same mechanism everywhere. It means the parent operation must not report success until the audit contract for that surface is satisfied.

For this tranche:

- Audit core shutdown: buffered events must either commit or be durably spilled to fallback before shutdown is considered clean.
- Evaluations: the unified audit event must flush successfully before the run is persisted and before the start webhook is emitted.
- Chat moderation:
  - input enforcement and non-stream output enforcement must await mandatory audit persistence before returning their moderated result or error
  - streaming output enforcement must not allow successful stream completion if the tracked mandatory moderation audit task failed
- Jobs: strictness means durable audit intent at the source transaction boundary for `job.created`, not immediate direct unified-audit success for every internal lifecycle edge
- AuthNZ API-key management: unified audit must persist before the action is considered successful; the legacy table mirror is secondary

### 2. Typed Audit Failure Signaling

Introduce explicit typed failures for mandatory audit behavior instead of leaking raw runtime or DB exceptions.

Planned exception shape:

- `MandatoryAuditWriteError`
- `AuditShutdownError`

These let call sites choose between:

- surfacing a clean API error
- aborting an internal operation
- aggregating shutdown/reporting state without losing the underlying cause

### 3. Compatibility Is Transitional, Not Co-Primary

When a touched path moves to unified audit, unified audit becomes the only mandatory store.

Legacy tables may remain temporarily for:

- read compatibility
- transition queries
- low-risk best-effort mirroring

But they do not become second mandatory durability dependencies.

## Scope Of Changes

### A. Sharing Backfill And Unified Writer Repair

Touched areas:

- `tldw_Server_API/app/core/Sharing/unified_share_audit.py`
- `tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py`
- related Sharing tests

Changes:

- Add an explicit `import_legacy_event(...)` API on `UnifiedShareAuditWriter`.
- Preserve:
  - `compatibility_id == legacy_share_audit_id`
  - stable legacy-derived `event_id`
  - original `created_at` timestamp
  - idempotent replay behavior
- Keep compatibility-floor handling aligned with imported legacy ids.

Constraints:

- rerunning the migration must not duplicate imported rows
- imported history must remain queryable through existing Sharing compatibility reads
- no silent fallback to synthetic timestamps or ids when legacy values are present

### B. Audit Core Shutdown Durability

Touched areas:

- `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- related Audit tests

Changes:

- Final `stop()` flush failures are no longer silently swallowed.
- If the final flush fails, remaining buffered events are durably appended to the fallback queue before shutdown fails.
- `stop()` raises `AuditShutdownError` when it cannot complete a clean durable shutdown.

Constraints:

- cancellation behavior still preserves buffered events
- fallback queue append must remain bounded and deterministic
- `stop()` failure must include enough context for lifecycle callers and tests

### C. Shared Migration Commit-Bound Accounting

Touched areas:

- `tldw_Server_API/app/core/Audit/audit_shared_migration.py`
- related Audit migration tests

Changes:

- per-chunk inserted/skipped/stat counters remain provisional until checkpoint save and commit succeed
- report totals update only after successful commit
- commit failure leaves returned counters consistent with durable state

Constraints:

- duplicate detection logging remains intact
- checkpoint semantics stay monotonic
- no inflated success counts on failed chunks

### D. DI And Lifecycle Hardening

Touched areas:

- `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- shutdown-related tests
- app shutdown integration

Changes:

- `shutdown_user_audit_service()` resolves the correct cache key semantics in shared mode instead of only looking up the raw user id
- targeted shutdown helpers collect and report real stop failures instead of only logging a narrow allowlist
- `shutdown_all_audit_services()` returns clean aggregated reporting by default and accepts an explicit raising mode for targeted callers/tests

Constraints:

- app shutdown remains best-effort by default
- cleanup-heavy tests can continue to tear down without widespread breakage
- targeted tests can still assert that real stop failures surfaced

### E. Evaluations Strict Run-Creation Ordering

Touched areas:

- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py`
- Evaluations audit adapter/tests
- Evaluations API tests

Changes:

- pre-generate `run_id`
- write and flush the mandatory unified audit event using that `run_id`
- persist the run row only after the audit succeeds
- schedule async execution only after the run row exists
- emit the “run started” webhook only after the audit-plus-run-persistence boundary succeeds

Result:

- no persisted run row when mandatory audit persistence fails
- no external start webhook for a run that never committed

Constraints:

- no change to public response schema
- no new id generation contract beyond optional `run_id` support already present in the DB layer

### F. Chat Moderation Strictness

Touched areas:

- `tldw_Server_API/app/core/Chat/chat_service.py`
- Chat moderation tests

Changes:

- Introduce a shared helper for mandatory moderation audit writes that performs `log_event(...)` plus `flush(raise_on_failure=True)`.
- Input moderation and non-stream output moderation await that helper before returning a moderated result or raising a moderation error.
- Streaming moderation uses tracked mandatory audit tasks:
  - first block/redact schedules a mandatory audit task
  - the stream may not finish successfully if any tracked mandatory moderation audit task failed
  - block paths remain failing responses; audit failure is observed before successful completion is possible

Important bound:

- This tranche does not redesign streaming transforms to await per-chunk audit writes before every emitted moderated chunk.
- For streaming redaction, the strict guarantee is “no successful completion without mandatory audit success,” not “no moderated bytes emitted before audit completion.”

### G. Jobs Durable Source Boundary

Touched areas:

- `tldw_Server_API/app/core/Jobs/manager.py`
- `tldw_Server_API/app/core/Jobs/event_stream.py`
- `tldw_Server_API/app/core/Jobs/audit_bridge.py`
- Jobs tests

Changes:

- Narrow this tranche to the `job.created` path, which already has a source-side transactional seam.
- For `job.created`, require durable audit intent at the source transaction boundary rather than relying only on the side-channel audit bridge.
- Keep the bridge/outbox replay path as the transport into unified audit where that remains the least-invasive implementation seam.

Result:

- parent success for `job.created` depends on durable audit intent existing
- the implementation does not attempt a repo-wide rewrite of every worker lifecycle emission path

Deferred:

- broad direct unified-audit enforcement across all Jobs worker/internal lifecycle paths is tracked under issue `#1053`

### H. AuthNZ API-Key Management Unification

Touched areas:

- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- supporting AuthNZ repo/helpers as needed
- AuthNZ integration tests

Changes:

- API-key create, virtual-create, rotate, and revoke write mandatory unified audit events
- these actions fail closed if the mandatory unified audit write cannot persist
- legacy `api_key_audit_log` writes become best-effort compatibility mirroring only
- API-key usage-touch auditing remains best-effort and does not gate ordinary request authentication

Reason:

- management actions are security-significant and reviewable
- usage heartbeat writes occur on normal request validation and should not turn audit trouble into broad auth outages

## Failure Semantics

### API/Service Calls

For the touched strict surfaces:

- mandatory audit persistence failure aborts the parent operation
- call sites should raise `MandatoryAuditWriteError` (or a surface-specific wrapper) rather than raw DB/runtime errors
- HTTP-facing paths in this tranche should map that failure to `503 Service Unavailable` with a stable audit-persistence failure detail

### Shutdown

- individual service `stop()` may raise `AuditShutdownError`
- global shutdown helpers return an aggregated shutdown summary by default and log its failures
- targeted callers/tests may opt into a raising mode when they need hard assertions

## Testing Strategy

All implementation work in this tranche follows TDD for each defect or behavior change.

Required regression coverage:

- Sharing legacy import remains idempotent and preserves ids/timestamps
- final `stop()` flush failure spills to fallback and raises `AuditShutdownError`
- shared migration counters are not inflated on commit failure
- shared-mode `shutdown_user_audit_service()` actually stops the shared singleton
- shutdown helpers can report non-allowlisted stop failures in targeted tests
- Evaluations run creation leaves no run row and no run-start webhook when mandatory audit fails
- Chat input moderation fails closed on mandatory audit failure
- Chat streaming moderation does not end successfully if tracked mandatory moderation audit tasks fail
- Jobs `job.created` requires durable audit intent, and no worker/internal lifecycle path is required to become direct unified-audit-strict in this tranche
- AuthNZ API-key management actions write unified audit and fail closed when that write fails
- legacy API-key audit compatibility mirror, if retained, remains best-effort and non-blocking

## Risks And Mitigations

### Risk: Over-broad Jobs remediation

Mitigation:

- explicitly narrow the Jobs scope to source-side transactional boundaries in this tranche
- require `job.created` as the only mandatory Jobs write in the minimum acceptable implementation for this tranche
- track broader lifecycle unification separately under `#1053`

### Risk: Streaming moderation strictness is overstated

Mitigation:

- define the exact guarantee as completion-bound strictness for streaming redaction
- avoid promising async-per-chunk audit enforcement without a broader stream refactor

### Risk: Shutdown raising breaks existing cleanup flows

Mitigation:

- raise at service-stop granularity
- keep global shutdown aggregated/best-effort by default
- add explicit raising mode for targeted tests and narrow callers

### Risk: Dual-write compatibility increases failure surface

Mitigation:

- make unified audit the only mandatory store
- keep legacy mirrors best-effort only

## Acceptance Criteria

- The Sharing backfill regression is fixed and covered by tests.
- `UnifiedAuditService.stop()` no longer silently loses buffered events during final flush failure.
- shared migration reports match durable commit state.
- shared-mode audit shutdown works correctly and targeted shutdown tests can detect real stop failures.
- Evaluations run creation, selected Chat moderation paths, Jobs `job.created`, and AuthNZ API-key management actions all enforce their revised audit contract.
- The remediation remains bounded and does not absorb the broader unification work tracked in `#1053`.
