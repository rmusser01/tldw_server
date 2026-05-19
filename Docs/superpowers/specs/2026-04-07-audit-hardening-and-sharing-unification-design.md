# Audit Hardening And Sharing Unification Design

Date: 2026-04-07
Status: Approved for planning
Scope: Audit reliability fixes, shutdown hardening, Sharing audit unification, bounded durability-policy cleanup

## Summary

This tranche hardens the Audit subsystem around two confirmed defects and removes the long-term split between unified audit storage and Sharing-specific audit storage.

The architectural decision is:

- Unify audit persistence.
- Preserve domain-specific Sharing interfaces.

That means `UnifiedAuditService` becomes the long-term source of truth for audit persistence, while Sharing keeps a domain-facing service and its existing admin audit API shape.

For practicality, Sharing audit events will use shared unified-audit storage semantics even when the general audit mode is `per_user`, because Sharing admin audit is inherently cross-user and must remain queryable without fan-out across every user DB.

## Why This Exists

The current system has two audit tracks:

- Unified audit for most backend domains through `UnifiedAuditService`
- A separate Sharing audit path backed by `share_audit_log` in the AuthNZ database

The split appears historical rather than fundamental. The Sharing feature was originally designed as an AuthNZ-domain package with its own repo-managed tables:

- `shared_workspaces`
- `share_tokens`
- `share_audit_log`
- `sharing_config`

That gave Sharing a local audit trail and simple owner-based filtering, but it also created duplicated audit storage, duplicated query behavior, separate migration burden, and future semantic drift risk.

There is no strong evidence in the current code or docs that Sharing requires permanently separate audit persistence semantics. For long-term pragmatism and stability, one persistence system is preferable.

## Goals

- Fix the populated legacy `audit_events` migration failure.
- Fix the eviction shutdown path that leaves pending stop tasks behind.
- Keep the already-in-progress chain-hash durability fixes and buffered count/export visibility fixes.
- Move Sharing audit persistence onto `UnifiedAuditService`.
- Preserve `/api/v1/sharing/admin/audit` as a compatibility API.
- Migrate existing `share_audit_log` history into unified audit.
- Define a clear mandatory-vs-best-effort audit policy boundary for the code touched in this tranche.

## Non-Goals

- Rewriting every audit caller across the repo to use one shared helper.
- Replacing `/api/v1/sharing/admin/audit` with the generic `/api/v1/audit/*` model.
- Removing all legacy Sharing audit structures in the same tranche.
- Broad audit taxonomy cleanup outside the touched domains.
- Expanding generic audit surfaces to aggregate Sharing events under `per_user` mode in this tranche.

## Confirmed Problems Being Addressed

### 1. Populated Legacy Audit Migration Fails

The legacy `audit_events` migration recreates the table with `chain_hash` but does not always provide that binding during row copy. Any populated legacy table can fail during initialization.

Desired outcome:

- Legacy populated audit DBs migrate successfully.
- Fresh writes after migration continue to work with valid chain hashes.
- No partial silent migration success.

### 2. Eviction Shutdown Is Not Reliably Drained

The audit DI layer schedules cross-loop service stop work in a fire-and-forget fashion. In practice, this can leave pending tasks during teardown and reduce confidence that the final flush completed before loop shutdown.

Desired outcome:

- Evicted or removed audit services shut down through tracked, drainable work.
- Application shutdown remains the single global owner of audit-service teardown.
- Adapter-local exit handling avoids duplicate global ownership.

### 3. Sharing Uses Separate Audit Persistence

Sharing currently writes audit events into `share_audit_log` via `SharedWorkspaceRepo`, while the rest of the backend uses unified audit. This is operationally inconsistent and increases long-term maintenance cost.

Desired outcome:

- Sharing events persist in unified audit.
- Existing Sharing admin audit consumers continue to work.
- Historical Sharing audit data is migrated, not abandoned.

## Architectural Decision

### Decision

Unify audit persistence in `UnifiedAuditService`, but keep domain-specific adapters and compatibility projections where they provide value.

### Rationale

- One persistence layer reduces storage drift, migration burden, retention divergence, and operator confusion.
- Sharing still benefits from a domain service that speaks in Sharing concepts such as `share_id`, `token_id`, `owner_user_id`, and `actor_user_id`.
- Preserving the Sharing admin audit API avoids unnecessary breakage while still consolidating persistence.
- Sharing admin audit is cross-user by design, so it needs shared queryability even if the rest of audit storage remains per-user.

### Consequences

Positive:

- One audit source of truth
- Shared durability semantics
- Simpler long-term export and compliance story
- Less duplicated audit schema maintenance

Tradeoffs:

- Requires a compatibility mapping layer for Sharing admin audit reads
- Requires historical migration from `share_audit_log`
- Requires explicit cutover sequencing
- Requires a narrow shared-storage exception for Sharing audit when global audit mode is `per_user`

## Scope Of Changes

### A. Audit Core Hardening

Touched area:

- `tldw_Server_API/app/core/Audit/unified_audit_service.py`

Changes:

- Fix populated legacy-table migration so copied rows always satisfy the unified insert contract, including `chain_hash`.
- Keep chain-head advancement commit-bound so failed flushes do not advance in-memory chain state.
- Keep buffered `count_events()` and `export_events()` behavior aligned with durable visibility expectations.

Constraints:

- Migration failures must be loud and reversible.
- No silent row dropping.
- Chain verification must remain valid after normal flush and fallback replay paths.

### B. Audit Lifecycle And Shutdown Hardening

Touched areas:

- `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- `tldw_Server_API/app/core/Embeddings/audit_adapter.py`
- `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- `tldw_Server_API/app/main.py`

Changes:

- Make scheduled stop work tracked and drainable instead of purely fire-and-forget.
- Keep app shutdown as the single global owner of full audit-service teardown.
- Remove Embeddings adapter ownership of full global async audit shutdown at `atexit`.
- Keep adapter exit logic limited to local sync-loop cleanup and other local best-effort cleanup.

Constraints:

- Do not introduce blocking behavior that can deadlock loop shutdown.
- Cross-loop shutdown must still be safe when owner loops are closed or no longer running.

### C. Sharing Audit Persistence Unification

Touched areas:

- `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- `tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py`
- Supporting migration utility/code near the Audit/Sharing boundary

Changes:

- Stop writing new Sharing audit events to `share_audit_log`.
- Write Sharing audit events through `UnifiedAuditService`.
- Use shared unified-audit storage semantics for Sharing events so admin reads stay practical.
- Preserve the Sharing domain-facing service so call sites stay semantically clear.
- Preserve `/api/v1/sharing/admin/audit` by projecting unified audit rows back into the existing Sharing response schema.

### D. Dedicated Sharing Audit Boundary

Touched areas:

- `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- Supporting DI or factory helpers near the Audit and Sharing boundary

Changes:

- Introduce a dedicated Sharing audit service or factory boundary that always resolves the correct unified-audit backend for Sharing events.
- Do not expose the shared-storage exception as a generic option that unrelated callers can accidentally reuse.
- Keep the exception local to Sharing audit orchestration.

Constraints:

- General audit callers must continue to follow normal `per_user` or `shared` storage rules.
- Sharing-specific storage resolution must be explicit and isolated.

## Storage Decision For Sharing Audit

Sharing audit uses unified audit persistence, but it does not follow the general per-user storage split.

Decision:

- If audit storage mode is `shared`, Sharing uses the normal shared audit DB.
- If audit storage mode is `per_user`, Sharing still writes to the shared unified audit DB path for Sharing events.

Reason:

- `/api/v1/sharing/admin/audit` is an admin cross-user query surface.
- Reconstructing it from per-user DB fan-out would be operationally fragile and much more expensive.
- Sharing is already a cross-user domain, so a shared audit backing is the pragmatic long-term choice.

Implementation boundary:

- This shared-storage exception must be hidden behind a dedicated Sharing audit resolver or factory.
- No unrelated audit caller should opt into it accidentally.

## Sharing Event Mapping

### Source Of Truth

All new Sharing audit persistence will use unified audit.

### Owner And Actor Semantics

Sharing audit has two distinct user identities:

- owner: the user who owns the shared resource
- actor: the user or system principal that performed the action

This tranche makes those roles explicit instead of overloading one field.

Rules:

- `tenant_user_id` in shared unified-audit storage represents `owner_user_id` for Sharing events.
- `context.user_id` represents the actor or request user when present.
- If no actor exists for a system-generated event, `context.user_id` may be omitted or set to a system identity according to the surrounding code path.
- `metadata.owner_user_id` is retained for compatibility and migration clarity.
- `metadata.actor_user_id` is retained for compatibility projection.

This preserves owner-based queryability without changing the general meaning of `context.user_id` for the rest of unified audit.

### Event Representation

For Sharing events:

- `event_type`: preserve existing domain event strings such as `share.created`, `token.used`, `share.workspace_deleted`
- `resource_type`: unchanged
- `resource_id`: unchanged
- `action`: by default mirror the Sharing event string unless a stronger existing normalized action is already required
- `tenant_user_id`: owner user for shared-storage queryability
- `context.user_id`: actor user when present
- `metadata.actor_user_id`: preserve actor when present
- `metadata.share_id`: preserve share identifier when present
- `metadata.token_id`: preserve token identifier when present
- `metadata.owner_user_id`: preserve owner explicitly for compatibility and migration clarity
- `context.ip_address`: preserve IP when present
- `context.user_agent`: preserve user agent when present

### Compatibility Id

The Sharing admin audit response requires a stable integer `id`.

This design will preserve one explicitly:

- Historical migrated rows will store the legacy `share_audit_log.id` as `legacy_share_audit_id`.
- New Sharing events will receive a stable compatibility id from a dedicated monotonic sequence stored in the shared unified audit DB.
- The sequence floor will be advanced to at least the maximum migrated legacy id before new writes are accepted.
- `/api/v1/sharing/admin/audit` will return that stable compatibility id in the existing `id` field.

This id must not be synthesized at read time from ordering or pagination position.

### Compatibility Id Transaction Rules

The compatibility-id mechanism must be transactional.

Rules:

- Compatibility-id allocation and unified audit event insertion must occur in the same database transaction.
- A transaction that allocates an id but fails before the event is committed must not produce a visible partial event.
- If migration is rerun after new Sharing writes already exist, the effective sequence floor becomes the maximum of:
  - the highest migrated legacy compatibility id
  - the current stored sequence value
  - the highest compatibility id already present in unified audit
- Sequence advancement must be monotonic and non-destructive.

## Sharing Admin Audit Compatibility

### API Contract

The existing endpoint remains:

- `GET /api/v1/sharing/admin/audit`

The response shape remains:

- `AuditLogResponse`
- `AuditEventResponse`

### Read Path

The endpoint will query unified audit filtered to Sharing events and map each unified row into the existing Sharing schema.

Sharing events are identified by event-type namespace:

- `share.*`
- `token.*`

Mapped fields:

- `id`: stable compatibility id
- `event_type`: original Sharing event string
- `actor_user_id`: from metadata and compatibility projection
- `resource_type`: from unified audit
- `resource_id`: from unified audit
- `owner_user_id`: from `tenant_user_id`, with metadata retained only for compatibility verification
- `share_id`: from metadata
- `token_id`: from metadata
- `metadata`: compatibility-safe metadata payload
- `ip_address`: from context
- `user_agent`: from context
- `created_at`: from audit timestamp

### Filtering

The endpoint must preserve current practical filtering semantics for:

- `owner_user_id`
- `resource_type`
- `resource_id`
- `limit`
- `offset`

`owner_user_id` remains first-class and queryable through `tenant_user_id`; it must not rely on opaque metadata scanning.

## Historical Sharing Audit Migration

### Requirement

Existing `share_audit_log` rows must be migrated into unified audit in this tranche.

### Migration Properties

- Idempotent
- Explicitly invoked, not hidden inside request handling
- Preserves compatibility id
- Preserves historical timestamps
- Preserves owner, actor, share, token, metadata, IP, and user agent
- Safe to rerun without duplicate unified events

### Event Identity

Historical rows will receive stable unified audit event ids derived from legacy Sharing row identity. The derivation must be deterministic so reruns skip already-migrated rows.

### Sequence Handling

Migration must also initialize the shared unified-audit compatibility-id sequence so that:

- all migrated legacy ids are reserved
- new Sharing events allocate ids strictly above the migrated maximum
- reruns after fresh unified Sharing writes do not regress the stored sequence floor

### Cutover Sequence

1. Add unified Sharing writer and compatibility reader.
2. Add idempotent Sharing history migration.
3. Run migration and verify counts and sample reads.
4. Advance the compatibility-id sequence ceiling transactionally.
5. Retire legacy writes to `share_audit_log`.
6. Treat unified audit as authoritative for `/sharing/admin/audit`.

The system must not rely on dual-read indefinitely.

## Durability Policy Boundary

This tranche defines and applies a bounded durability policy rather than refactoring every caller.

### Mandatory Audit In This Tranche

- Sharing lifecycle events
- Share token lifecycle and use events
- Security violations
- Permission-denied style security outcomes in touched code
- Admin destructive operations in touched code

### Best-Effort Audit Remains Acceptable

- Routine API request telemetry
- Routine API response telemetry
- Observational metrics-style audit emissions that do not represent compliance-critical state changes

### Why This Boundary

It reduces operational ambiguity where it matters most without turning this tranche into a repo-wide audit rewrite.

## Category And Severity Policy For Sharing Events

The mapping must be explicit.

### Default Categories

- `share.created`, `share.updated`, `share.revoked`, `share.cloned`, `share.workspace_deleted` -> `DATA_MODIFICATION`
- `share.accessed` -> `DATA_ACCESS`
- `token.created`, `token.revoked` -> `DATA_MODIFICATION`
- `token.used` -> `DATA_ACCESS`
- `token.password_verified` -> `SECURITY`
- `token.password_failed` -> `SECURITY`

### Default Severities

- Normal successful lifecycle events -> `INFO`
- Expected but security-relevant failures such as password failure or denied access -> `WARNING`
- Confirmed security-violation style misuse in touched code -> `ERROR` or `CRITICAL` according to existing unified policy thresholds

If an existing Sharing behavior clearly depends on a different severity for a specific event, the implementation may specialize that event, but the above becomes the default contract.

## Generic Audit Visibility Decision

This tranche makes the source-of-truth decision explicit but keeps generic audit visibility bounded.

Decision:

- `/api/v1/sharing/admin/audit` is the required operator-facing compatibility surface for Sharing audit in all modes.
- When the global audit mode is `shared`, Sharing events may naturally be visible through generic unified audit surfaces.
- When the global audit mode is `per_user`, this tranche does not require `/api/v1/audit/*` to aggregate or expose Sharing events from the shared Sharing-audit path.

Reason:

- The primary compatibility obligation is the Sharing admin audit API.
- Expanding generic audit aggregation under `per_user` mode would increase scope and create a second cross-surface compatibility contract.
- The source-of-truth decision still holds because Sharing persistence lives in unified audit, even if generic audit surfaces do not expose it in every mode yet.

## Error Handling

### Audit Core

- Legacy audit migration failures must raise and roll back when possible.
- Failed flushes must not advance the persisted chain head.

### Sharing Migration

- Migration failures must fail loudly and report enough context to identify the offending source rows.
- Partial migration must remain rerunnable.

### Compatibility Endpoint

- `/sharing/admin/audit` must fail rather than return misleading partial compatibility data if projection fails.

### Mandatory Sharing Writes

- Where Sharing behavior is designated mandatory-audit in this tranche, write failures should not be silently discarded.
- If an existing endpoint contract explicitly guarantees non-failure on audit write issues, that contract must be revisited carefully and changed only with matching tests.

## Testing Strategy

### Audit Core Tests

Add or keep tests covering:

- Populated legacy `audit_events` migration
- Fresh writes after migration with valid `chain_hash`
- Failed flush does not advance chain head
- Fallback replay restores valid chain hashes
- Buffered count and export flush semantics

### Lifecycle Tests

Add tests covering:

- Eviction-stop work is tracked and drained
- No pending shutdown task leak in the running-owner-loop path
- Embeddings adapter exit behavior does not assume ownership of global audit shutdown

### Sharing Tests

Add tests covering:

- New Sharing events write only to unified audit
- Sharing writes use the shared unified-audit path regardless of global per-user mode
- Owner and actor identities remain distinct in stored Sharing events
- Legacy `share_audit_log` rows migrate into unified audit
- Migration is idempotent
- `/sharing/admin/audit` preserves the current response contract using unified-backed data
- `owner_user_id` filtering remains correct after unification
- Compatibility id remains stable for both migrated and new rows
- Compatibility-id allocation is atomic with event insertion

### Verification Commands

Before implementation is considered complete, run focused tests for:

- `tldw_Server_API/tests/Audit/`
- `tldw_Server_API/tests/Sharing/`
- `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`
- Any directly touched AuthNZ and endpoint tests needed by the Sharing compatibility path

Run Bandit on the touched scope in the project virtual environment before completion.

## Rollout And Stability Notes

- The compatibility endpoint should switch reads only after the unified writer and migration path exist.
- Legacy Sharing audit structures may remain temporarily for rollback safety, but they should no longer be the primary source of truth after cutover.
- The implementation must avoid a prolonged dual-read operational model.

## Success Criteria

- Populated legacy unified audit DBs initialize successfully.
- Eviction and shutdown no longer emit pending-task teardown warnings in the covered regression path.
- New Sharing audit events are persisted in unified audit.
- Sharing admin audit remains queryable without per-user DB fan-out.
- Existing Sharing admin audit consumers continue to receive the same response shape.
- Historical Sharing audit rows are preserved through migration.
- The touched modules have a documented and test-backed durability boundary.
