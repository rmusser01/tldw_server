# Audit Module Code Guide

This guide helps project developers understand the Audit module’s architecture, what it contains, how it works, and how to work with it across the codebase.

## Overview

- Purpose: Provide consistent, durable audit logging across AuthNZ, RAG, Chat, Evals, Workflows, and API surfaces.
- Core module: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- Dependency injection: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- REST endpoints: `tldw_Server_API/app/api/v1/endpoints/audit.py` (admin‑only export and count)
- Storage: SQLite per user (default) with WAL; background flush + retention cleanup.
- Tests: `tldw_Server_API/tests/Audit/` (PII, risk scoring, service behavior, endpoints)

## Code Map

- Service + types
  - `tldw_Server_API/app/core/Audit/unified_audit_service.py`
    - `AuditEventCategory`, `AuditEventType`, `AuditSeverity`
    - `AuditContext`, `AuditEvent`
    - `PIIDetector`, `RiskScorer`
    - `UnifiedAuditService` (async service: buffer, flush, query, export, cleanup)
    - `audit_operation` context manager
    - Deprecated globals for back-compat guidance

- DI and lifecycle (per-user services)
  - `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
    - `get_audit_service_for_user(...)` FastAPI dependency
    - LRU caching of user‑scoped instances with graceful shutdown
    - `shutdown_user_audit_service(...)`, `shutdown_all_audit_services(...)`
    - DB path resolution via `tldw_Server_API/app/core/DB_Management/db_path_utils.py`

- API endpoints (admin only)
  - `tldw_Server_API/app/api/v1/endpoints/audit.py`
    - `GET /api/v1/audit/export` → JSON/JSONL/CSV (streaming for JSON/JSONL)
    - `GET /api/v1/audit/count` → count for pagination UIs

- Tests and examples
  - `tldw_Server_API/tests/Audit/test_unified_audit_service.py`
  - `tldw_Server_API/tests/Audit/test_audit_export_endpoint.py`
  - `tldw_Server_API/tests/Audit/test_pii_pattern_groups.py`, `test_audit_pii_overrides.py`, `test_risk_settings_overrides.py`

## Architecture

- Async, buffered writes with periodic flush; per‑user SQLite DBs by default.
- Unified schema across modules with rich context and metadata.
- PII detection and redaction applied to metadata and selected string fields.
- If metadata isn’t JSON-serializable, the service stores a redacted representation as `{ "redacted_text": "..." }`.
- Heuristic risk scoring; high‑risk events trigger immediate flush.
- Background tasks: periodic flush and daily cleanup (disabled in test mode).
- Strict write support: callers can force durability with `flush(raise_on_failure=True)` and surface
  `MandatoryAuditWriteError` for fail-closed security or state-changing paths.

### Key Types

- `AuditContext`: request/session context (request ID, correlation ID, session ID, user ID, IP, UA, endpoint, method).
- `AuditEvent`: dataclass that normalizes metadata to JSON and flattens `context_*` fields for storage.
- `AuditEventCategory` / `AuditEventType` / `AuditSeverity`: enums that standardize classification and severity.

### Service Responsibilities

- Schema management and connection pooling (`aiosqlite`) with WAL PRAGMAs.
- Buffering + batch insert with retries and backoff.
- Risk scoring (`RiskScorer`) and PII detection (`PIIDetector`).
- Query helpers, count, export (JSON/JSONL/CSV), and simple daily stats aggregation.
- Retention cleanup (delete events older than `retention_days`) and daily stats pruning.
- Fallback durability: if flush repeatedly fails and the buffer overflows, surplus events are appended to a fallback JSONL file adjacent to the audit DB (per‑user under the audit directory when using DI; `./Databases/` when using the default constructor).

## Data Model & Storage

When using DI (single or multi-user), the audit DB path is per-user under `<USER_DB_BASE_DIR>/<user_id>/audit/unified_audit.db` unless `AUDIT_STORAGE_MODE=shared` is enabled. In shared mode, all audit events are stored in a single DB at `AUDIT_SHARED_DB_PATH` (default `Databases/audit_shared.db`) with tenant scoping. If you construct the service directly without DI, the default path is `./Databases/unified_audit.db` (or the shared path when shared mode is enabled).

Tables:
- `audit_events` (primary table):
  - Core: `event_id` (PK), `timestamp` (ISO8601), `category`, `event_type`, `severity`, `result`, `error_message`
  - Shared mode: `tenant_user_id` (reserved `system` tenant for anonymous/system events)
  - Context: `context_request_id`, `context_correlation_id`, `context_session_id`, `context_user_id`, `context_api_key_hash`, `context_ip_address`, `context_user_agent`, `context_endpoint`, `context_method`
  - Details: `resource_type`, `resource_id`, `action`
  - Metrics: `duration_ms`, `tokens_used`, `estimated_cost`, `result_count`
  - Risk & compliance: `risk_score`, `pii_detected`, `compliance_flags` (JSON text)
  - Metadata: `metadata` (JSON text)

- `audit_daily_stats` (aggregates by UTC date + category): totals, failures, high‑risk counts, token/cost sums, average duration. In shared mode this table is tenant-scoped with `tenant_user_id` in the primary key.

Indexes (selected): `timestamp`, `context_user_id`, `context_request_id`, `context_correlation_id`, `event_type`, `category`, `severity`, `risk_score`, `context_ip_address`, `context_session_id`, `context_endpoint`, `context_user_agent`, plus resource/action indexes.

Shared audit DB schema versioning uses SQLite `PRAGMA user_version` (currently `1`).

Note: a legacy migration file exists for evaluation DB instrumentation (`tldw_Server_API/app/core/DB_Management/migrations_v6_audit_logging.py`). The unified audit DB schema is created by `UnifiedAuditService._init_database()` and is separate from that migration path.

## Sharing Audit

- Source of truth: Sharing audit events now persist in unified audit through `tldw_Server_API/app/core/Sharing/unified_share_audit.py`.
- Service boundary: `tldw_Server_API/app/core/Sharing/share_audit_service.py` remains the Sharing-facing API, but its default path writes and reads through the unified Sharing audit boundary instead of `share_audit_log`.
- Storage rule: even when general audit storage mode is `per_user`, Sharing audit still uses the shared audit DB so `GET /api/v1/sharing/admin/audit` remains a practical cross-user query surface.
- Compatibility surface: `GET /api/v1/sharing/admin/audit` is the required operator-facing endpoint for Sharing audit history. Generic `GET /api/v1/audit/*` surfaces may include Sharing naturally in shared mode, but they are not required to aggregate Sharing events in `per_user` mode.
- Historical backfill: `tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py` migrates legacy `share_audit_log` rows idempotently, preserves historical timestamps, and reserves legacy integer ids as the compatibility ids returned by the Sharing admin audit API.

## AuthNZ API-Key Audit

- Mandatory unified audit now applies to API-key management actions in `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`:
  create, rotate, revoke, and virtual-key creation.
- Helper boundary: `tldw_Server_API/app/core/AuthNZ/api_key_audit.py` writes those events through an isolated
  audit service instance, flushes with `raise_on_failure=True`, and stops that isolated service immediately after
  the write. This avoids unrelated buffered events in a cached service causing false failures for strict AuthNZ flows.
- Actor/target context: strict API-key audit metadata always includes `target_user_id`, and admin-triggered actions
  also include actor metadata (`actor_user_id`, `actor_subject`, `actor_kind`, `actor_roles`) when available.
- Failure contract: HTTP surfaces that depend on these mandatory writes should translate `MandatoryAuditWriteError`
  to `503 Service Unavailable`. Auth state changes must not commit successfully if the mandatory audit write fails.
- Compatibility boundary: legacy `api_key_audit_log` rows remain a best-effort mirror for compatibility during the
  transition, but unified audit is now the required source of truth for those management events.
- Best-effort exception: API-key usage/touch auditing remains best-effort so an audit outage does not become an
  authentication outage.

## Configuration

Environment and settings (read from `app.core.config.settings` and env):
- `AUDIT_HIGH_RISK_SCORE` (env): high‑risk threshold (default 70).
- `AUDIT_STORAGE_MODE` (settings/env): `per_user` (default) or `shared`.
- `AUDIT_SHARED_DB_PATH` (settings/env): shared audit DB path.
- `AUDIT_STORAGE_ROLLBACK` (settings/env): force per-user behavior when true.
- `AUDIT_ETL_USER_SUBPATH` (settings/env): optional subpath appended to `USER_DB_BASE_DIR` for migration discovery.
- `AUDIT_ACTION_RISK_BONUS` (settings dict): `{ action_label: 0..100 }` risk bonuses.
- `AUDIT_HIGH_RISK_OPERATIONS` (settings list/CSV): extra action substrings treated as high‑risk.
- `AUDIT_SUSPICIOUS_THRESHOLDS` (settings dict): `{ failed_auth: int, data_export: int, after_hours: bool, ... }`.
- `AUDIT_PII_USE_RAG_PATTERNS` (settings bool): merge RAG patterns into `PIIDetector`.
- `AUDIT_PII_PATTERNS` (settings dict): override/add regex groups.
- `AUDIT_PII_SCAN_FIELDS` (settings list/CSV): extra event fields to scan (default includes `action`, `resource_id`, `error_message`, `context_user_agent`). Use `context_*` for context fields.
- `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`): per-user DB root directory; defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.

Constructor overrides (per instance): `db_path`, `retention_days`, `buffer_size`, `flush_interval`, `enable_pii_detection`, `enable_risk_scoring`, `max_db_mb`.

## Migration Utility

Use the one-time migration helper to merge per-user audit DBs plus the legacy `Databases/unified_audit.db` into the shared audit DB:

```bash
python -m tldw_Server_API.app.core.Audit.audit_shared_migration \
  --shared-db Databases/audit_shared.db \
  --user-db-base Databases/user_databases \
  --default-db Databases/unified_audit.db
```

Discovery scans `USER_DB_BASE_DIR/*/audit/unified_audit.db` and `USER_DB_BASE_DIR/<AUDIT_ETL_USER_SUBPATH>/*/audit/unified_audit.db` when configured. The migration is idempotent, tracks per-source checkpoints in the shared DB to resume large runs, and skips locked/corrupt sources with a warning.

## Using the Service in Endpoints (DI)

Preferred pattern is dependency injection; do not use global singletons.

```python
from fastapi import Depends
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType, AuditContext, UnifiedAuditService
from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user

@router.post("/example")
async def handler(audit_service: UnifiedAuditService = Depends(get_audit_service_for_user)):
    ctx = AuditContext(user_id="42", endpoint="/api/v1/example", method="POST")
    await audit_service.log_event(
        event_type=AuditEventType.API_REQUEST,
        context=ctx,
        action="example.create",
        metadata={"foo": "bar"},
    )
    # ...handle request...
    await audit_service.log_event(
        event_type=AuditEventType.API_RESPONSE,
        context=ctx,
        action="example.create",
        result="success",
    )
    return {"ok": True}
```

Tips:
- Always set `AuditContext.user_id`, and when in HTTP, set `endpoint` and `method`.
- For best-effort streaming paths, schedule audit writes with `asyncio.create_task(...)` to avoid blocking.
- For mandatory audit paths, await the write and `flush(raise_on_failure=True)` before reporting success.
- For timing an operation, prefer `audit_operation(...)` context manager (it logs success/failure and duration).

## Programmatic API (service methods)

- `await initialize()` → prepares schema, opens pool (unless in test mode), starts background tasks (unless in test mode).
- `await log_event(...)` → append to buffer; applies PII detection and risk scoring; high‑risk triggers ad‑hoc flush.
- `await log_login(user_id, username, ip, user_agent, success, session_id)` → convenience wrapper.
- `await flush()` → batch insert buffered events.
- `await query_events(...)` → returns list of dicts; decode JSON fields via `decode_row_fields(row)` if desired.
- `await count_events(...)` → integer count for UI pagination.
- `await export_events(...)` → JSON/JSONL/CSV content, or streaming generator, or writes to file.
- `await cleanup_old_logs()` → deletes older than `retention_days`; prunes stats.
- `await stop()` → cancels background tasks, flushes pending buffers, closes pool.
- `get_statistics()` → in-memory counters and configuration snapshot.
- `get_security_summary(hours=24)` → aggregated security stats (high-risk counts, failures, unique users, top failing IPs).

Context manager:

```python
from tldw_Server_API.app.core.Audit.unified_audit_service import audit_operation, AuditEventType
ctx = AuditContext(user_id="42")
async with audit_operation(service, AuditEventType.DATA_READ, ctx,
                           start_event_type=AuditEventType.API_REQUEST,
                           completed_event_type=AuditEventType.API_RESPONSE,
                           resource_type="document", resource_id="abc"):
    # work
    ...
```

## REST: Export and Count

- `GET /api/v1/audit/export` (admin)
  - `format`: `json` (default), `jsonl`, `csv`
  - Filtering: `start_time`, `end_time` (ISO8601, `Z` supported), `event_type`, `category`, `min_risk_score`, `user_id`, `request_id`, `correlation_id`, `ip_address`, `session_id`, `endpoint`, `method`
  - Output: direct content; streaming `true` supports JSON/JSONL; CSV uses fixed header schema
  - Programmatic CSV export supports incremental streaming when `file_path` is provided; HTTP streaming is only supported for JSON/JSONL.
  - `filename` is sanitized and normalized per format

- `GET /api/v1/audit/count` (admin)
  - Same filters as export (minus output controls)
  - Returns `{ "count": int }`

## Background Tasks & Shutdown

- Background tasks: periodic flush (`flush_interval`) and daily cleanup. Disabled in test mode to avoid busy loops.
- DI layer caches per‑user services (LRU). On app shutdown, call `shutdown_all_audit_services()` (see `Audit_DB_Deps.py`).
- `shutdown_all_audit_services()` returns an `AuditShutdownSummary` and is best-effort by default; pass
  `raise_on_error=True` for targeted callers or tests that need shutdown failures surfaced.
- The service enforces owner‑loop shutdown; DI helpers schedule safe cross‑loop shutdown.
- `UnifiedAuditService.stop()` attempts a final durable spill of buffered events to the fallback queue before raising
  if the last flush still cannot be persisted cleanly.

## Performance & Scaling

- Batch insert with `executemany()`; WAL mode; indexed queries for common filters.
- High‑risk events and buffer thresholds trigger ad‑hoc flushes (tracked futures are awaited during shutdown).
- Export supports streaming JSON/JSONL and chunked CSV writes for large datasets.
- Tests validate throughput (e.g., 1k events under a few seconds) and indexed query times.

## Testing

- Run: `python -m pytest tldw_Server_API/tests/Audit -v`
- Notable tests:
  - `test_unified_audit_service.py`: PII, risk scoring, buffering/flush, queries, export, cleanup, context manager
  - `test_audit_export_endpoint.py`: endpoint auth, headers/filenames, streaming
  - Overrides: PII and risk settings tests ensure settings‑driven behavior remains stable
- In tests, background tasks are disabled (test mode), so explicitly call `flush()` when needed.

## Extending & Customizing

- Add new `AuditEventType` values for new domains; keep them namespaced (e.g., `workflows.job.started`).
- If a new domain doesn’t match existing prefix mappings, extend `_determine_category(...)` in the service accordingly.
- Add convenience wrappers (similar to `log_login`) for repetitive patterns in your module.
- PII: extend `AUDIT_PII_PATTERNS` and/or `AUDIT_PII_SCAN_FIELDS`; consider consolidating shared PII utilities across modules if needed.
- Risk: tune `AUDIT_ACTION_RISK_BONUS`, `AUDIT_HIGH_RISK_OPERATIONS`, `AUDIT_SUSPICIOUS_THRESHOLDS` in settings.

## Troubleshooting

- DB locked / flush failures: service retries with backoff; worst‑case, events spill to a fallback JSONL file adjacent to the audit DB (per‑user under the audit directory when using DI; `./Databases/` when using the default constructor). Check logs for `flush_failures` and the persisted queue file.
- Missing events: ensure `await flush()` is called before shutdown; prefer DI and `await stop()` on app shutdown.
- Mandatory write failures: check for `MandatoryAuditWriteError` and confirm the caller is using an isolated or
  otherwise clean audit service boundary before the strict `flush(raise_on_failure=True)` call.
- PII not redacting: verify patterns and `AUDIT_PII_*` settings; confirm fields are strings or in `metadata`.
- Slow queries: verify indexes were created (service creates on init); filter narrowing with indexed fields (`timestamp`, `context_*`, `event_type`, `category`).
- Admin requirement: export/count endpoints enforce admin via claim-first dependencies (`require_roles("admin")` / `require_permissions(...)`); override those deps in tests accordingly.

## Security Notes

- Avoid logging secrets. PII redaction is a safety net, not a substitute for disciplined metadata.
- `pii_detected` flag is set and `compliance_flags` include `pii_detected` when redaction occurs.
- Consider alerting on `AuditEventType.SECURITY_VIOLATION`, high `risk_score`, or repeated failures in your monitoring.

---

See also:
- Core README: `tldw_Server_API/app/core/Audit/README.md`
- Audit docs: `Docs/Audit/README.md`
- DB paths: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- DI layer: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- Endpoints: `tldw_Server_API/app/api/v1/endpoints/audit.py`
