# Claims_Extraction Hardening and Refactor Design

## Goal

Harden the validated Claims_Extraction review findings with focused regression tests, then prepare a staged refactor plan that reduces module coupling without changing the public API surface.

Backlog task: TASK-9934.

## Validated Findings

The hardening pass addresses these current-code findings:

- Rebuild workers soft-delete existing claims before replacement storage succeeds.
- Shared claims exception tuples include `asyncio.CancelledError`, allowing cancellation to be swallowed.
- Ingestion LLM timeout uses a `ThreadPoolExecutor` context that still waits for stuck workers during shutdown.
- Runtime config resolvers bound minimum values but not the same maximum values enforced by the settings API.
- Review notification HTML interpolates claim text and metadata without escaping.
- Claims dashboard analytics accepts an owner scope but does not apply it consistently.
- FVA adjudication score metrics run in the branch where no adjudication exists.
- Notification webhook retries spawn daemon threads without local backpressure.

## Hardening Design

The fix pass stays narrow and behavior-focused. It adds failing regression tests before production changes and avoids broad file moves.

Rebuild replacement will fail closed. If non-empty extraction produces zero stored claims, the rebuild task will raise and be counted as failed instead of deleting active claims and logging success. The rebuild path must not rely on the permissive ingestion behavior where `store_claims` logs storage errors and returns `0`; it should use a strict replacement helper or explicitly raise before any old active claims are committed as deleted. Normal ingestion-time `store_claims` semantics should remain compatible for existing callers.

Where possible, delete and insert should run under one Media DB transaction or a DB helper that preserves old active claims until replacement storage is confirmed. If the delete must happen before insert inside a transaction, any insert failure or zero-insert result for non-empty extraction must roll back the delete.

Cancellation handling will remove `asyncio.CancelledError` from Claims_Extraction noncritical exception tuples. Async paths that encounter cancellation will re-raise it instead of falling back to heuristic extraction, default verification, or empty analytics payloads.

The LLM timeout path will avoid `ThreadPoolExecutor` context-manager shutdown semantics after timeout. The implementation should either use a provider/client timeout or explicitly shut down the executor with `wait=False` and `cancel_futures=True` so the caller returns promptly. It must also avoid replacing one hang with unbounded abandoned work; any executor fallback should be bounded, record timeout metrics once, and leave the caller free to continue.

Runtime safety will enforce a maximum context window and extraction pass count in `runtime_config`, matching the settings service caps. This protects direct config and environment paths, not only API updates.

Review notification email HTML will escape all interpolated values with `html.escape`. Plain text bodies remain readable and unescaped.

Dashboard analytics will apply owner scope consistently to claims, review-log, per-media, and orphan-cluster queries. The owner-scope SQL should be centralized so backend-specific table casing and parameter placeholders stay correct. In single-user SQLite paths this should preserve current results; in shared/admin paths it prevents ambiguous cross-owner totals.

FVA metrics will record adjudication scores only after adjudication exists. The no-anti-context branch will only count wasted falsification.

Notification dispatch will use a small bounded local dispatcher for the immediate fix. Saturated-queue behavior must be explicit: record/log the dispatch failure and avoid silent drops or unbounded thread creation. Full Jobs/Scheduler migration is deferred to the refactor plan because it changes operational contracts and persistence expectations.

## Refactor Direction

The follow-up refactor should split by responsibility while keeping imports and endpoint behavior stable:

- `claims_analytics_service.py`: dashboard aggregation, status trends, latency, per-media stats, cluster analytics, and owner-scope SQL helpers.
- `claims_notification_dispatcher.py`: review and alert notification dispatch, bounded retry execution, and delivery helpers.
- `claims_rebuild_orchestrator.py`: rebuild task orchestration and atomic replacement semantics.
- `claims_runtime_limits.py` or expanded `runtime_config.py`: shared bounds and config normalization for extraction runtime settings.

`claims_service.py` should remain the public service facade during the first refactor stage. Endpoint modules should not need new imports until the extracted helpers are stable.

## Testing Design

Regression tests should extend existing Claims tests rather than create a new harness:

- Rebuild tests in `tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py`.
- Runtime config bounds in `tldw_Server_API/tests/Claims/test_claims_runtime_config.py`.
- Review notification escaping and dispatcher behavior in `tldw_Server_API/tests/Claims/test_claims_review_notifications.py`.
- Dashboard owner scoping in `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`.
- FVA metric branch behavior in `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py`.
- Cancellation propagation tests near the claims engine/service paths they exercise.
- Timeout behavior with a fake executor or provider call that proves the function returns after timeout without waiting for the worker body to finish.
- At least one SQLite-backed rebuild regression that proves old active claims remain active when replacement storage fails.
- Notification dispatcher tests for normal dispatch and saturated-queue behavior.

Touched code must run targeted pytest checks, Bandit on the touched Claims_Extraction scope, and any narrower tests needed by changed DB helpers.

## Out of Scope

This pass does not rename public API endpoints, change claim schemas, rewrite extraction strategies, or migrate notification delivery to Jobs. Those are follow-up refactor tasks once the hardened behavior is verified.

## Spec Review

- Placeholder scan: no placeholders remain.
- Consistency check: the hardening design and testing design cover each validated finding.
- Scope check: the immediate fix pass is separate from the larger refactor.
- Ambiguity check: notification delivery uses a bounded local dispatcher now; Jobs migration is deferred.
- Follow-up review: rebuild strictness, timeout worker bounds, analytics SQL scoping, and notification saturation behavior are specified explicitly.
