# Audit Unification Roadmap Design

Date: 2026-04-29
Status: Proposed for approval
Scope: Remaining split, legacy, compatibility, and logger-only audit paths
Related: GitHub issue `rmusser01/tldw_server#1053`

## Summary

The Audit strict-remediation tranche made selected high-value paths durable and moved several domains toward `UnifiedAuditService`. It intentionally did not finish the broader audit architecture cleanup.

This roadmap covers that remaining cleanup. The target architecture is:

- `UnifiedAuditService` is the source of truth for durable audit persistence.
- Domain services may keep narrow adapters when they preserve useful domain language or API compatibility.
- Legacy audit tables are compatibility projections, import sources, or temporary mirrors, not co-primary stores.
- Logger-only events are diagnostics unless they are backed by a unified audit event.
- Each domain has an explicit mandatory, durable-intent, or best-effort audit contract.

## Goals

- Inventory remaining non-unified audit paths and classify them as `migrate`, `keep-with-boundary`, or `deprecate`.
- Define a repo-wide policy for mandatory versus best-effort audit behavior.
- Normalize required audit context enough that migrated paths can be queried consistently.
- Stage cutovers so compatibility APIs keep working while the source of truth moves to unified audit.
- Reduce hidden audit failures and duplicated persistence logic.
- Explicitly deprecate at least one representative legacy path in public developer docs.

## Non-Goals

- Re-implement every migration in this design PR.
- Break existing admin APIs without a replacement query path.
- Force every low-value telemetry or metrics event to be fail-closed.
- Perform broad taxonomy renames without compatibility mapping and tests.
- Remove legacy tables before historical import and read-projection work exists.

## Architectural Decisions

### 1. Unified Audit Is Authoritative

New durable audit behavior should write through `UnifiedAuditService` or a domain adapter that delegates to it. Legacy tables may remain while admin APIs, reports, or migration tools still need them, but new code must not treat those legacy tables as the authoritative audit record once a path has been moved.

### 2. Compatibility Is One-Way

Compatibility mirrors are allowed only from unified audit to legacy shape or as temporary best-effort mirrors during a deprecation window. A legacy write path must not be required for success after the same operation has a mandatory unified audit contract.

### 3. Audit Strictness Is Surface-Specific

The right contract depends on the parent operation:

- `mandatory`: the operation may not report success until the unified audit event has satisfied that surface's explicit durability contract, usually primary-store persistence through `flush(raise_on_failure=True)`.
- `durable-intent`: the source transaction records audit intent, and a worker projects it to unified audit.
- `best-effort`: failure is logged and does not change the parent operation result.
- `diagnostic-only`: logs may exist for debugging, but they are not audit evidence.

### 4. Logger-Only "Audit" Is Not Audit Persistence

Loguru sinks filtered by `extra={"audit": True}` are useful operational diagnostics, but they are not queryable, tenant-aware, or governed by unified retention/export policy. These paths must either emit unified audit events or be renamed/documented as diagnostic security logs.

## Inventory And Decisions

| Area | Current behavior | Decision | Rationale | First migration step |
| --- | --- | --- | --- | --- |
| API-key management history | `create`, `create_virtual`, `rotate`, and `revoke` use mandatory unified audit, then best-effort mirror to `api_key_audit_log`. Admin key history still reads `api_key_audit_log`. | Deprecate legacy source; migrate reads. | Unified audit is already the mandatory store. The legacy table is now a compatibility mirror and should not remain the admin source of truth. | Add a unified-audit projection for `get_api_key_audit_log`, backfill/import legacy rows, then disable legacy mirroring by default after a deprecation window. |
| Sharing audit | `ShareAuditService` writes through `UnifiedShareAuditWriter` by default, with a repo-backed legacy path still available for injected compatibility. `share_audit_log` exists for old history/import. | Keep with boundary, then deprecate direct legacy writes. | Sharing needs a domain-facing API and compatibility projection, but persistence has already moved to unified audit for the normal path. | Restrict direct repo writes to tests/import tools, document the compatibility boundary, and add guardrails against new production callers. |
| Jobs audit bridge | `submit_job_audit_event` queues selected lifecycle events to a side-channel worker when `JOBS_AUDIT_ENABLED` is enabled. Failures are best-effort. | Migrate to durable intent. | Job lifecycle audit should not depend on an optional side-channel queue for events operators treat as durable. Immediate synchronous unified writes are also not the right contract for every worker edge. | Introduce a job audit outbox or durable intent table at source transaction boundaries, starting with `job.created`, then project lifecycle events to unified audit. |
| MCP Unified core | MCP config can write a file sink for records carrying `extra={"audit": True}`. Multiple auth, RBAC, protocol, server, guard, and endpoint paths emit logger-only audit records. | Migrate. | These events are security-relevant but bypass unified audit storage, tenant context, export, and retention. | Add an MCP unified audit adapter backed by `UnifiedAuditService`; keep `MCP_AUDIT_LOG_FILE` as diagnostic output only. |
| MCP Hub management | `emit_mcp_hub_audit` emits unified audit events for hub mutations and flushes best-effort. | Keep with boundary; review strictness by action. | It already uses unified audit, but sensitive credential or shared workspace mutations may need stricter contracts than general hub edits. | Classify hub actions into mandatory and best-effort groups; make credential grant/revocation fail closed if audit cannot persist. |
| ACP session audit logger | `AuditLogger` buffers session events to an injected persistence callback and swallows write failures after logging. | Keep with boundary, then migrate selected security events. | ACP event streams may include high-volume session diagnostics that should not all be mandatory audit. Security or workspace mutation events need unified coverage. | Define which ACP events are audit evidence, add a unified sink for those, and keep bulk trace events separate. |
| Embeddings audit adapter | Adapter writes to unified audit and has mandatory flush behavior for selected operations. | Keep with boundary. | This is already unified through a domain adapter. The adapter is useful for sync/async boundary handling. | Document adapter ownership and ensure no separate embeddings audit store is introduced. |
| Evaluations audit adapter | Evaluation run creation and selected actions use mandatory unified audit helpers. | Keep with boundary. | This matches strict-remediation decisions and avoids duplicating audit persistence in evaluation tables. | Keep tests enforcing mandatory write ordering and use the adapter as the pattern for similar domains. |
| AuthNZ `audit_logs` and `audit_log` tables | Monitoring, registration, admin system views, and historical AuthNZ flows still write/read legacy operational audit tables. | Migrate with compatibility projection. | Some rows are operational metrics, not compliance audit. Others are security events that should become unified audit evidence. | Split metric rows from security/account rows, project admin views from unified audit where possible, and preserve old tables as historical import sources. |
| Billing audit log | `billing_audit_log` repo methods still exist, while SQLite migration 034 is a compatibility no-op for the retired public schema. Budget updates use mandatory unified audit. | Deprecate or remove after usage audit. | The migration already marks the public billing audit schema retired, but repo methods can still imply a live legacy store. | Grep production call sites, remove dead repo audit methods if unused, or wrap any live billing mutation with unified audit. |
| Admin action helpers | Shared admin helper `_emit_admin_audit_event` writes unified audit best-effort. Some dedicated account and budget helpers are stricter. | Migrate by action policy. | Admin reads or low-risk operational actions can stay best-effort, but destructive admin actions need a clearer failure boundary. | Classify admin actions and convert destructive/security mutations to mandatory or durable-intent helpers. |
| Chatbooks, Workflows, Sandbox, and similar endpoint-local try/except audit calls | Many endpoints catch audit failures and continue after logging warnings. | Migrate by policy. | Some events are telemetry; others cover export, import, cleanup, path traversal, or data deletion where best-effort may be too weak. | Build a small audit contract matrix for these endpoint families, then replace ad hoc try/except blocks with shared mandatory or best-effort helpers. |

Representative files checked for this inventory:

- `tldw_Server_API/app/core/AuthNZ/api_key_audit.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/services/admin_api_keys_service.py`
- `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- `tldw_Server_API/app/core/Sharing/unified_share_audit.py`
- `tldw_Server_API/app/core/Jobs/audit_bridge.py`
- `tldw_Server_API/app/core/MCP_unified/config.py`
- `tldw_Server_API/app/services/mcp_hub_service.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/audit_logger.py`
- `tldw_Server_API/app/core/AuthNZ/repos/billing_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/monitoring_repo.py`
- `tldw_Server_API/app/services/admin_audit_service.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- `tldw_Server_API/app/services/budget_audit_service.py`

## Mandatory Versus Best-Effort Policy

### Mandatory

Use mandatory audit when the event is the audit trail for a security decision, privilege change, credential lifecycle event, tenant/data boundary change, or destructive data operation.

Initial mandatory categories:

- API-key create, virtual-create, rotate, revoke, and ownership changes.
- Password, MFA, session revocation, registration-code, role, org, team, and permission mutations.
- Admin destructive actions such as user disable/delete, data deletion, retention override, and privileged export.
- Budget and billing configuration mutations that affect quotas or money movement.
- Chat moderation enforcement where the system blocks or modifies content for policy reasons.
- Evaluation run creation and state transitions that external systems may observe.
- Credential grants/revocations and shared workspace trust changes in MCP Hub.

### Durable Intent

Use durable intent when the source operation is asynchronous or lifecycle-heavy and cannot reasonably await direct unified audit persistence for every edge.

Initial durable-intent categories:

- Jobs lifecycle, starting with `job.created` and expanding to acquisition, completion, failure, cancellation, quarantine, and SLA breach.
- Long-running imports/exports where source transaction state already owns recovery.
- Scheduler-driven or worker-driven operations that need replay after process crash.

### Best-Effort

Use best-effort only when audit loss does not undermine the correctness of the parent operation or compliance story.

Allowed best-effort categories:

- Read-only admin views and low-risk UI actions.
- Derived metrics and health counters.
- Duplicate compatibility mirrors after unified audit has already persisted.
- Diagnostic breadcrumbs that are clearly not the authoritative audit record.

### Diagnostic-Only

Logger-only paths must be named and documented as diagnostic security logs, not audit evidence, unless they also emit unified audit events.

## Required Context For Migrated Events

Migrated audit events should carry these fields when available:

- `context_user_id`: actor user id, or `system`/anonymous semantics through shared audit rules.
- `tenant_user_id`: tenant owner in shared storage.
- `context_request_id` and `context_correlation_id`: request and trace linkage.
- `context_ip_address` and `context_user_agent`: request-origin metadata for user/API calls.
- `context_endpoint` and `context_method`: route and method for HTTP-origin events.
- `resource_type` and `resource_id`: stable audited object identity.
- `action`: domain action in dot-separated form, such as `api_key.rotate`.
- `result`: `success`, `failure`, or `error`.
- `metadata.actor_id`, `metadata.target_user_id`, `metadata.org_id`, `metadata.team_id` when relevant.
- `metadata.source_system`: legacy/source adapter identity during migration.
- `metadata.compatibility_id` or `metadata.legacy_*_id`: stable legacy projection identity when preserving old API shapes.

## Staged Implementation Plan

### Stage 0: Roadmap And Representative Deprecation

Deliverables:

- Add this design doc.
- Document that `api_key_audit_log` is deprecated as an authoritative source.
- State that API-key management unified audit is mandatory and legacy mirroring is best-effort compatibility.

Success criteria:

- Issue `#1053` has an inventory, decisions, and staged plan to review.
- `Docs/Audit/README.md` contains the representative legacy-path deprecation.

### Stage 1: API-Key Legacy Read Cutover

Goal:

- Make API-key admin history read from unified audit while preserving the old response shape.

Implementation notes:

- Add a projection helper that maps unified `api_key.*` events to `APIKeyAuditEntry`.
- Import or backfill `api_key_audit_log` rows into unified audit with stable legacy metadata.
- Keep legacy mirror writes behind a compatibility flag during the transition.
- Add tests proving mandatory unified writes still fail closed while legacy mirror failures do not block success.

Exit criteria:

- Admin API-key audit history no longer depends on `api_key_audit_log` for new rows.
- Legacy table is documented as read/import compatibility only.

### Stage 2: Jobs Durable Audit Intent

Goal:

- Replace the optional side-channel bridge as the source of truth for audited job lifecycle events.

Implementation notes:

- Add a durable job audit intent/outbox record at source transaction boundaries.
- Start with `job.created`, then add acquisition, completion, failure, cancellation, quarantine, and SLA breach.
- Keep `JOBS_AUDIT_ENABLED` bridge as a temporary projection transport only if needed.
- Add replay/idempotency tests for worker crash and retry behavior.

Exit criteria:

- Operators can rely on durable job audit intent even when the projection worker is delayed or restarts.

### Stage 3: MCP Unified Audit Adapter

Goal:

- Move MCP Unified logger-only audit events into unified audit.

Implementation notes:

- Add a small `McpUnifiedAuditSink` or adapter that normalizes auth, RBAC, guard, protocol, and endpoint events.
- Carry client id, subject, tool, permission, route, result, and peer IP in metadata/context.
- Preserve `MCP_AUDIT_LOG_FILE` as optional diagnostic output, not audit persistence.
- Add tests that logger-only emission is no longer the sole audit path for security events.

Exit criteria:

- MCP security-relevant events are queryable through unified audit export/count APIs.

### Stage 4: Legacy Store Compatibility Sweep

Goal:

- Finish domain-local audit store classification and compatibility migration.

Implementation notes:

- Split AuthNZ `audit_logs` metrics from security/account events.
- Remove or wrap live billing audit repo methods after production call-site verification.
- Lock Sharing legacy writes to import/test paths and document the projection contract.
- Add ACP unified sink for security/workspace mutation events while keeping trace-style session events separate.

Exit criteria:

- No legacy table remains undocumented as source, projection, import source, or retired schema.

### Stage 5: Guardrails

Goal:

- Prevent reintroducing split audit paths.

Implementation notes:

- Add lint/regression tests that flag new `extra={"audit": True}` logger-only security events without a unified sink.
- Add tests or docs checks for direct writes to deprecated tables from production code.
- Add a small helper library for mandatory, durable-intent, and best-effort event emission so endpoint-local try/except blocks are not repeatedly hand-rolled.

Exit criteria:

- New audit-affecting changes have a clear contract and use a known helper or documented adapter.

## Representative Deprecation

Effective with this roadmap, `api_key_audit_log` is deprecated as an authoritative audit source.

The current contract is:

- Unified audit is mandatory for API-key create, virtual-create, rotate, and revoke.
- `api_key_audit_log` is a best-effort compatibility mirror while admin reads and historical imports are migrated.
- New API-key management behavior must not depend on `api_key_audit_log` for success.
- Future API-key audit history APIs should project from unified audit and only consult the legacy table for historical compatibility during migration.

This is the first representative legacy path for issue `#1053` because it already has a mandatory unified write path and therefore can be cut over with limited behavioral risk.

## Testing Strategy For Implementation PRs

- Unit tests for each domain adapter mapping to unified audit fields.
- Failure-injection tests for mandatory flush behavior.
- Crash/replay tests for durable-intent outboxes.
- Compatibility projection tests that compare legacy API shapes against unified rows.
- Regression tests that no production code writes directly to deprecated audit tables outside import or compatibility modules.
- Export/count tests confirming migrated events are visible through unified audit APIs.

## Open Questions For Approval

- What deprecation window should legacy compatibility mirrors use before they are disabled by default?
- Should `MCP_AUDIT_LOG_FILE` be renamed to make diagnostic-only semantics explicit?
- Should AuthNZ operational metric rows remain in `audit_logs` permanently, or move to a metrics-specific table during the compatibility sweep?
- Should admin destructive actions all become mandatory in one tranche, or be migrated endpoint family by endpoint family?
