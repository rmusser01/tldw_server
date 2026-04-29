# Unified Audit Service - Developer Guide

This page documents the unified Audit module: event schema, categories, risk scoring, storage, exports, configuration, and recommended usage patterns.

## Overview

- Purpose: Provide consistent, durable audit logging across AuthNZ, RAG, Chat, Evals, and system components.
- Module: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- Dependency injection: `get_audit_service_for_user` in `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- Storage: SQLite (per-user by default) with optional shared storage mode, WAL mode, and background flush/cleanup tasks.

## Event Schema (selected fields)

- Core: `event_id`, `timestamp` (ISO8601), `category`, `event_type`, `severity`, `result`, `error_message`
- Shared mode: `tenant_user_id` (reserved `system` tenant for anonymous/system events)
- Context: `context_request_id`, `context_correlation_id`, `context_session_id`, `context_user_id`, `context_ip_address`, `context_user_agent`, `context_endpoint`, `context_method`
- Details: `resource_type`, `resource_id`, `action`
- Metrics: `duration_ms`, `tokens_used`, `estimated_cost`, `result_count`
- Risk & Compliance: `risk_score`, `pii_detected`, `compliance_flags` (JSON array)
- Metadata: `metadata` (JSON object)

Notes:
- Timestamps are stored as ISO8601 strings; SQLite queries use lexicographic ordering which is correct for ISO8601.
- `metadata` and `compliance_flags` are stored as JSON-encoded text (never NULL in latest code). Consumers can decode via `decode_row_fields(row)`.

## Categories & Event Types

- Automatic mappings are derived from `event_type` prefixes:
  - `auth_*` → AUTHENTICATION
  - `user_*` → AUTHORIZATION
  - `data_read` → DATA_ACCESS
  - `data_write|update|delete|import|export` → DATA_MODIFICATION
  - `rag_*` → RAG, `eval_*` → EVALUATION, `api_*` → API_CALL, `security_*` → SECURITY, `system_*` → SYSTEM
- Explicit, non-prefixed mappings:
  - `permission.denied`, `suspicious.activity` → SECURITY
  - `pii.detected` → COMPLIANCE

## Risk Scoring & Thresholds

- Risk scores calculated from event type, failures, PII, time of day, weekend, etc.
- High-risk threshold: `AUDIT_HIGH_RISK_SCORE` env var (default 70)
  - Used for risk counters and to trigger immediate (ad-hoc) flushes

## Storage & Maintenance

- SQLite PRAGMAs: WAL, NORMAL sync, temp in memory, foreign keys ON
- Auto-vacuum: `auto_vacuum=INCREMENTAL`; cleanup triggers `PRAGMA incremental_vacuum`
- Optional `max_db_mb` parameter logs a warning if file size exceeds configured limit
- Fallback durability: if repeated flush failures cause the buffer to overflow, dropped events are appended to a fallback JSONL file adjacent to the audit DB (per-user under the audit directory when using DI; `./Databases/` when using the default constructor)

### Shared Storage Mode

- Enable with `AUDIT_STORAGE_MODE=shared` (config/env); default is `per_user`.
- Set shared DB path via `AUDIT_SHARED_DB_PATH` (default `Databases/audit_shared.db`).
- Set `AUDIT_STORAGE_ROLLBACK=true` to force `per_user` behavior even when shared is configured.
- Shared mode enforces `tenant_user_id` on writes; non-admin export/count requests are forced to their own tenant.
- `tenant_user_id=system` is reserved for anonymous/system events and visible only to admins.
- Per-user mode remains supported during the deprecation window; shared mode is recommended when ready.

### Indexes

- Common filters: `timestamp`, `context_user_id`, `context_request_id`, `context_correlation_id`, `event_type`, `category`, `severity`, `risk_score`
- Additional: `context_ip_address`, `context_session_id`, `context_endpoint`, `context_user_agent`
- Query ordering: `ORDER BY timestamp DESC, event_id DESC` for deterministic pagination

## PII Detection & Redaction

- PII is scanned in:
  - `metadata` (recursively, preserving structure)
  - Selected strings: `action`, `resource_id`, `error_message`, and `context_user_agent`
- Redaction placeholders: `[EMAIL_REDACTED]`, `[API_KEY_REDACTED]`, `[CREDIT_CARD_REDACTED]`, etc.
- When PII is detected, `pii_detected=true` and `compliance_flags` includes `pii_detected`.
- If metadata isn’t JSON-serializable, the service stores a redacted representation as `{ "redacted_text": "..." }`.

## Programmatic Usage

### Initialize (usually via DI)

When using dependency injection (the recommended approach in multi-user production), the service is initialized with a per-user path under `<USER_DB_BASE_DIR>/<user_id>/audit/` unless `AUDIT_STORAGE_MODE=shared` is enabled. In shared mode, all audit events are written to the shared audit DB path and tagged with `tenant_user_id`.

For single-user setups, development, or testing, you may instantiate the service directly:

```python
from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService
svc = UnifiedAuditService(db_path="./Databases/unified_audit.db")  # Direct instantiation (non-DI)
await svc.initialize()
```

### Log an event

```python
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType
ctx = AuditContext(user_id="42", request_id="req-1")
await svc.log_event(
    event_type=AuditEventType.DATA_READ,
    context=ctx,
    resource_type="document",
    resource_id="doc-123",
    metadata={"source": "api"},
)
```

### Context-managed operations (start/end)

```python
from tldw_Server_API.app.core.Audit.unified_audit_service import audit_operation, AuditEventType
ctx = AuditContext(user_id="42")
async with audit_operation(
    svc,
    AuditEventType.DATA_READ,  # logical operation type
    ctx,
    start_event_type=AuditEventType.API_REQUEST,
    completed_event_type=AuditEventType.API_RESPONSE,
    resource_type="doc",
    resource_id="abc",
):
    ...  # work
```

### Query & Count

```python
rows = await svc.query_events(user_id="42", limit=100, offset=0)
count = await svc.count_events(user_id="42")
# Decode JSON fields if needed
rows = [svc.decode_row_fields(r) for r in rows]
```

## Exporting Audit Logs

### REST Endpoint

- Path: `GET /api/v1/audit/export`
- Permissions: `system.logs` (admin)
- Query parameters:
  - `format`: `json` (default), `jsonl` (NDJSON), or `csv`
  - `start_time`, `end_time`: ISO8601 (accepts trailing `Z`)
  - `event_type`, `category`: comma-separated enum names or values
  - `min_risk_score`, `user_id`, `request_id`, `correlation_id`
  - `ip_address`, `session_id`, `endpoint`, `method`: additional filters by context fields
  - `filename`: suggested download name (server sanitizes and normalizes extension)
  - `stream`: `true` for JSON/JSONL/CSV streaming responses
  - `max_rows`: hard cap on number of rows to export (streaming stops when reached)
- CSV exports use a fixed header schema for consistent columns.
- JSONL exports (NDJSON) are useful for very large datasets and simple line-by-line processing.

Examples:

```bash
# JSON (non-streaming)
curl -H "X-API-KEY: $KEY" \
  "http://127.0.0.1:8000/api/v1/audit/export?format=json&user_id=42" -OJ

# JSON streaming
curl -H "X-API-KEY: $KEY" \
  "http://127.0.0.1:8000/api/v1/audit/export?format=json&stream=true&min_risk_score=70" -OJ

# CSV (fixed headers)
curl -H "X-API-KEY: $KEY" \
  "http://127.0.0.1:8000/api/v1/audit/export?format=csv&user_id=42" -OJ

# JSONL (NDJSON)
curl -H "X-API-KEY: $KEY" \
  "http://127.0.0.1:8000/api/v1/audit/export?format=jsonl&stream=true&user_id=42" -OJ
```

### Programmatic Export

```python
# In-memory JSON or CSV content
content = await svc.export_events(user_id="42", format="json", max_rows=1000)

# Streaming to file - CSV
count = await svc.export_events(user_id="42", format="csv", file_path="/tmp/audit.csv", chunk_size=5000)

# Streaming to file - JSON
count = await svc.export_events(user_id="42", format="json", file_path="/tmp/audit.json", chunk_size=5000)

# Streaming JSON generator to client code
agen = await svc.export_events(user_id="42", format="json", stream=True, chunk_size=5000)
async for chunk in agen:
    ...  # write to socket / file
```
Note: HTTP streaming applies to JSON/JSONL/CSV. Programmatic CSV export streams when `file_path` is provided.

## Configuration

- `AUDIT_HIGH_RISK_SCORE` (env): Risk threshold for high-risk classification; default `70`.
- PII configuration (tunable; optional):
  - `AUDIT_PII_USE_RAG_PATTERNS`: `true|false` (default `false`). When enabled, merges Audit PII patterns with those from RAG security filters for consistency.
  - `AUDIT_PII_PATTERNS`: dict of pattern overrides, e.g. `{ "email": "...regex...", "my_secret": ["regex1", "regex2"] }`.
  - `AUDIT_PII_SCAN_FIELDS`: comma-separated string or list of additional string fields to scan/redact beyond metadata (defaults include: `action`, `resource_id`, `error_message`, `context_user_agent`). Use `context_*` prefix for context fields, e.g. `context_endpoint`.
- Retention: `retention_days` (constructor) controls cleanup.
- Flush/cleanup: `buffer_size`, `flush_interval` tune background activity.
- Database location:
  - Per-user mode: `<USER_DB_BASE_DIR>/<user_id>/audit/unified_audit.db`.
  - Shared mode: `AUDIT_SHARED_DB_PATH` (default `Databases/audit_shared.db`).
  - If you construct the service directly without DI, the default `db_path` is `./Databases/unified_audit.db` (or shared path when `AUDIT_STORAGE_MODE=shared`).
- `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`): per-user DB root directory; defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
- `AUDIT_ETL_USER_SUBPATH` (settings/env): optional subpath appended to `USER_DB_BASE_DIR` for migration discovery (e.g., `nested_users`).

## Migration Utility

Use the shared DB migration tool to merge per-user audit DBs (plus legacy `Databases/unified_audit.db`) into the shared DB:

```bash
python -m tldw_Server_API.app.core.Audit.audit_shared_migration \
  --shared-db Databases/audit_shared.db \
  --user-db-base Databases/user_databases \
  --default-db Databases/unified_audit.db
```

Discovery scans `USER_DB_BASE_DIR/*/audit/unified_audit.db`, plus `USER_DB_BASE_DIR/<AUDIT_ETL_USER_SUBPATH>/*/audit/unified_audit.db` when configured, and `Databases/unified_audit.db` if present. The migration is idempotent (deduped by `event_id`), preserves timestamps and metadata, and resumes using per-source checkpoints stored in the shared DB. Locked/corrupt sources are skipped with warnings and reported in the summary logs.

## Legacy Compatibility And Deprecation

Unified audit is the authoritative audit persistence layer for new durable audit work. Domain-specific audit tables may remain temporarily as compatibility projections, historical import sources, or best-effort mirrors, but new code should not treat them as co-primary stores once the same domain has a unified audit contract.

`api_key_audit_log` is deprecated as an authoritative source for API-key management history. API-key create, virtual-create, rotate, and revoke operations now use unified audit as the mandatory source of truth; the legacy table is only a best-effort compatibility mirror while admin reads and historical imports are migrated. New API-key management code must not require a successful `api_key_audit_log` write after the unified audit event has persisted.

`share_audit_log` is also a compatibility source/projection for Sharing audit history. Normal Sharing audit persistence should go through the Sharing unified-audit adapter, with direct legacy writes limited to import, migration, or narrowly scoped compatibility tests.

Logger-only records, including records emitted with `extra={"audit": True}`, are diagnostic logs unless they are also backed by a `UnifiedAuditService` event. Security-relevant logger-only paths should be migrated to unified audit or renamed so they are not mistaken for durable audit evidence.

## Operational Notes

- Deterministic ordering with `timestamp DESC, event_id DESC` supports paginated UIs.
- Fixed CSV headers simplify downstream ingestion.
- Fallback queue (`audit_fallback_queue.jsonl`) is written adjacent to the audit DB and captures dropped events on persistent DB write failures.
- Shared audit DB schema version is tracked via SQLite `PRAGMA user_version`.
- PII scanning can be tuned centrally in future; currently Audit and RAG use separate detectors.
- Shutdown: the service enforces owner-loop shutdown; DI helpers schedule safe cross-loop shutdown.

### Utilities

- `get_statistics()` returns in-memory counters and configuration snapshot.
- `get_security_summary(hours=24)` returns aggregated security stats (high-risk counts, failures, unique users, top failing IPs).

---

For questions or contributions, see `README.md` and the tests under `tldw_Server_API/tests/Audit/` for example usage.
