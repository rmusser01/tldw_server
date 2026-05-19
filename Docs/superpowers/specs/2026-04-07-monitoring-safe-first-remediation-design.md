# Monitoring Safe-First Remediation Design

Date: 2026-04-07
Topic: Non-breaking remediation of reviewed Monitoring backend defects and gaps
Status: Drafted for user review

## Summary

This spec covers a safe-first remediation batch for the Monitoring backend.

The batch is intentionally non-breaking at the public API contract level. It
fixes internal correctness bugs that are clearly wrong today, adds regression
coverage around reviewed behaviors, tightens internal invariants where that can
be done without changing route contracts, aligns documentation with current
backend behavior, and creates GitHub follow-up issues for the contract changes
that should not be shipped silently in a compatibility-focused pass.

## Goals

- Fix correctness defects that are safe to correct without changing public API
  shapes or permission boundaries.
- Add focused regression coverage for the highest-value reviewed risks and
  current behavior seams.
- Make preserved-but-surprising behavior explicit in tests and documentation.
- Tighten internal enforcement around reviewed monitoring invariants where the
  enforcement does not break the current contract.
- Create one umbrella GitHub issue that tracks deferred contract changes and
  links the major follow-up items.

## Scope

In scope:

- [`tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py)
- [`tldw_Server_API/app/core/Monitoring/notification_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Monitoring/notification_service.py)
- [`tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py)
- [`tldw_Server_API/app/api/v1/endpoints/monitoring.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/monitoring.py)
- [`tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py)
- [`tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py)
- monitoring-related definitions in
  [`tldw_Server_API/app/api/v1/schemas/admin_schemas.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/schemas/admin_schemas.py)
- [`tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py)
- monitoring tests under
  [`tldw_Server_API/tests/Monitoring/`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Monitoring)
- focused admin monitoring tests:
  [`tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py)
  [`tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Admin/test_admin_monitoring_api.py)
  [`tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py)
  [`tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py)
- monitoring-related permission and backend-selection tests:
  [`tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py)
  [`tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py)
  [`tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py)
- monitoring product documentation:
  [`Docs/Product/Completed/Topic_Monitoring_Watchlists.md`](/Users/appledev/Documents/GitHub/tldw_server/Docs/Product/Completed/Topic_Monitoring_Watchlists.md)
- topic-monitoring and notification documentation in
  [`tldw_Server_API/app/core/Monitoring/README.md`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Monitoring/README.md)

Out of scope:

- breaking public route redesign
- changing response shapes for monitoring alert mutations
- forbidding overlay-only identities at runtime in this batch
- changing admin/public authorization boundaries in this batch
- Guardian/self-monitoring remediation
- claims-monitoring remediation
- repo-wide observability cleanup outside the monitoring module boundary

## Locked Decisions

- This batch is safe-first and non-breaking for current monitoring clients.
- Internal correctness defects may be fixed even if they alter buggy runtime
  behavior, provided they do not change the documented external API contract.
- Reviewed contract flaws that require public behavior changes will be deferred
  to follow-up GitHub issues instead of being changed silently here.
- Runtime enforcement added in this batch must not reject requests that are
  currently accepted by the public monitoring/admin contracts.
- If stronger invariant checks are added, they must be limited to:
  - regression tests
  - non-throwing runtime diagnostics
  - helper-level assertions that do not change route success paths
- Tests will explicitly encode both:
  - behavior we are intentionally preserving for compatibility
  - behavior that is currently defective and should change now
- One umbrella GitHub issue will be created for deferred monitoring follow-up,
  with linked sub-items for the major contract changes we are not shipping in
  this pass.
- GitHub follow-up creation must have an execution fallback:
  - use `gh` only if it is available and authenticated in the execution
    environment
  - otherwise generate issue-ready markdown/text artifacts and stop short of
    claiming that remote GitHub issues were created

## Approaches Considered

### 1. Minimal Coverage Pass

Add tests and documentation only, with almost no code changes.

Pros:

- smallest behavioral risk
- easiest to land in a dirty wider workspace

Cons:

- leaves known internal defects in place
- defers too much value from the review

### 2. Full Contract Remediation

Fix the underlying lifecycle, overlay, and notification contract problems now.

Pros:

- addresses the deepest issues directly
- leaves fewer deferred follow-ups

Cons:

- incompatible with the approved safe-first boundary
- too likely to break current clients or operator workflows

### 3. Recommended: Safe-First Stabilization Plus Internal Enforcement

Fix internal correctness bugs, add regression coverage, add non-breaking
guardrails, clarify docs, and defer contract changes through GitHub tracking.

Pros:

- delivers meaningful defect reduction without stealth API breakage
- makes current behavior explicit instead of accidental
- creates a clean bridge to a stricter follow-up phase

Cons:

- preserves some known design debt temporarily
- requires disciplined distinction between defects and deferred redesign

## Design

### 1. Topic Monitoring Correctness

The `reload()` path in
[`topic_monitoring_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py)
will be corrected so that runtime dedupe tuning is refreshed from the current
config/environment in the same pass that already refreshes enablement,
max-scan-chars, paths, and watchlists.

This is treated as a direct bugfix, not a contract redesign:

- `reload()` should re-read dedupe window seconds
- `reload()` should re-read simhash distance
- existing dedupe state reset behavior should remain in place
- no route contract changes are needed

Regression coverage will prove:

- a service constructed with one dedupe configuration can be reloaded after env
  changes
- the effective dedupe settings change after reload
- existing reload path semantics for watchlists/paths still hold

### 2. Notification Hardening Without Contract Expansion

The notification layer keeps its current outward behavior but becomes safer and
more explicit.

Safe fixes allowed in this batch:

- ensure retry-exhaustion from webhook/email worker helpers is handled as a
  controlled best-effort failure instead of leaking through “safe” wrappers
- add tests that document the current generic-notification and digest-mode
  behavior
- keep digest behavior non-delivering in this batch if that is the current
  contract, but document it clearly instead of leaving it implicit
- clarify that `notify_generic()` and digest paths do not currently share the
  full topic-alert delivery path
- preserve current Guardian-facing behavior for shared
  `NotificationService` helpers; this batch may harden those helpers but does
  not redesign Guardian dispatch semantics

Not allowed in this batch:

- silently turning digest mode into a new real delivery feature
- introducing new outbound notification channels or changing route responses

This track is about honesty and internal robustness, not feature expansion.

### 3. Public Alert Lifecycle Clarification

The reviewed public monitoring lifecycle ambiguity is preserved as an external
contract for this batch, but it becomes explicit in tests and documentation.

The remediation work will:

- add route-level or integration-level tests that lock in current `read`,
  `acknowledge`, and `dismiss` state effects
- make it explicit that the current mutation responses are minimal and that the
  authoritative merged state is observed by re-reading/listing alerts
- avoid redesigning response shapes or route semantics in this batch

This keeps compatibility while preventing the current lifecycle behavior from
remaining undocumented folklore.

### 4. Overlay/Admin Seam Enforcement

Overlay-only identities remain allowed in this batch, but their behavior should
stop being accidental.

The remediation work will:

- add tests that make the current overlay-only identity behavior explicit
- add runtime-safe consistency checks that surface invariant drift without
  rejecting currently accepted requests in production
- add stronger assertions in test coverage where we want failures to be loud
- keep merge-time and visibility invariants under closer test coverage

This does not ban overlay-only state. It makes the current seam legible and
prepares the stricter contract work for follow-up.

### 5. Docs And Follow-Up Tracking

The completed monitoring product doc will be updated to match backend reality.

Specifically:

- the completed product doc and the core Monitoring README must stop describing
  current notification behavior in contradictory ways
- notification behavior must stop describing webhook/email as mere placeholders
  if the backend already attempts them
- current local-first and best-effort caveats should be stated plainly
- current lifecycle and delivery limits preserved by this batch should be
  described without pretending they are more complete than they are

GitHub tracking:

- create one umbrella issue for deferred monitoring follow-up
- link major deferred work items under that umbrella, including:
  - stricter overlay identity validation or a first-class overlay-only contract
  - public alert lifecycle/response redesign
  - true digest delivery semantics if desired
  - stronger admin/public permission model clarification if needed
- if the environment cannot create GitHub issues directly, generate the issue
  title/body text in-repo or in turn output so the follow-up work is still
  captured without falsely reporting remote issue creation

## Testing Strategy

This work will be implemented with strict TDD.

Required failing tests first:

- `reload()` refreshes dedupe window and simhash settings
- webhook/email “safe” wrappers swallow retry exhaustion as best-effort failure
- `notify_generic()` current channel behavior is explicit and covered
- digest-mode buffering/flush semantics are explicit and covered
- current public alert lifecycle behavior is explicitly covered
- current overlay-only identity behavior is explicitly covered
- documentation-sensitive monitoring behavior has regression coverage where
  practical so docs are less likely to drift again

Execution tiers:

- narrow unit tests for the direct bugfixes and helper behavior
- monitoring/admin integration tests for preserved lifecycle and seam behavior
- targeted reruns of the reviewed monitoring/admin/auth slices after changes
- PostgreSQL monitoring parity tests only if the fixture environment is
  available during implementation; otherwise the follow-up caveat remains

## Risks

- The surrounding workspace is dirty outside the monitoring scope, especially in
  AuthNZ-adjacent areas. The implementation plan should keep the changed file
  set narrow and avoid unnecessary coupling.
- Internal assertions must not turn preserved current behavior into accidental
  breaking changes.
- Documentation updates must describe current stable behavior, not aspirational
  behavior intended for later follow-up.

## Success Criteria

This remediation is successful if:

- the stale dedupe-reload bug is fixed with regression coverage
- notification helper safety gaps are covered and hardened without expanding the
  public contract
- current public lifecycle behavior and overlay-only identity behavior are
  explicitly covered by tests rather than left implicit
- monitoring docs match the actual backend behavior shipped after this batch
- deferred contract follow-up is captured either in one remote umbrella GitHub
  issue or in issue-ready text artifacts when direct issue creation is
  unavailable
- the existing monitoring/admin/auth verification slices still pass after the
  remediation work
