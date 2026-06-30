# Audit Module

## 1. Current Feature Set

- Unified, async audit logging across AuthNZ, RAG, Evaluations, Workflows, and public APIs
- Common schema with event categories/types, severity, context, risk score, and optional PII redaction
- Buffered writes with size/interval/high‑risk flush triggers, daily aggregates and retention cleanup
- CSV/JSON/JSONL export with optional streaming for large datasets; count endpoint for pagination
- Per‑user or shared audit databases; fallback JSONL queue for resilience on flush failures
- Test‑friendly behavior (background loops disabled in TEST_MODE)

## 2. Technical Details of Features

### Purpose
The audit module provides a single, async-friendly service for capturing security,
compliance, and operational events across the tldw_server backend. It unifies audit
logging for AuthNZ, RAG, Evaluations, Workflows, and API surfaces, enforcing a
common schema, risk scoring, and optional PII redaction before data is persisted.

### Key Components
- `AuditEventCategory` / `AuditEventType` / `AuditSeverity` - canonical enums that
  describe high-level categories, fine-grained event IDs, and severity levels.
- `AuditContext` - request/session metadata (IDs, IP, UA, method, endpoint, etc.)
  automatically folded into stored events using a `context_*` column prefix.
- `AuditEvent` - dataclass representing a single entry; handles JSON encoding,
  metadata storage, and conversion to DB rows.
- `PIIDetector` - configurable regex-driven detector with integration hooks into
  the RAG PII patterns; can redact strings, nested dicts, or lists prior to storage.
- `RiskScorer` - heuristic scorer that weights event type, result, metadata and
  volume to produce a `0..100` score; high-risk events trigger immediate flushes.
- `UnifiedAuditService` - the async facade that buffers, flushes, exports, and
  rotates audit data. It owns lifecycle management, schema creation, and stats.

### Storage & Schema
- Default DB lives at `Databases/unified_audit.db` (configurable via constructor).
- Schema consists of `audit_events` and `audit_daily_stats`. The service applies
  WAL/JOURNAL pragmas for better concurrency and creates indexes on category,
  event_type, resource metadata, timestamps, and risk scores.
- ISO8601 timestamps allow lexicographic queries across SQLite and Postgres alike.
- Daily aggregates track volume, failures, cost/tokens, and average latency.
- On repeated flush failures, surplus events are persisted to
  `Databases/audit_fallback_queue.jsonl` for later replay.

### Runtime Model
- **Buffer & flush**: Events are appended to an in-memory buffer protected by an
  `asyncio.Lock`. Flushes happen when the buffer reaches `buffer_size`, when a
  high-risk event (>=`AUDIT_HIGH_RISK_SCORE`, default `70`) arrives, or on the
  timed flush loop (`flush_interval` seconds).
- **Background tasks**: `start_background_tasks()` spins up the periodic flush and
  cleanup loops. In test mode (`TEST_MODE`/`TLDW_TEST_MODE` env or
  `PYTEST_CURRENT_TEST`), the loops stay disabled so tests can drive `flush()` manually.
- **Cleanup**: `cleanup_old_logs()` enforces the retention window (`retention_days`,
  default 90 days) and prunes the daily stats table.
- **PII handling**: When enabled, metadata and selected string fields are scanned
  and redacted using placeholder markers; matching events receive a
  `pii_detected` compliance flag.
- **Risk**: High scores increment `stats["high_risk_events"]` and emit warnings via
  `loguru`, making it easy to pipe alerts into structured logging or metrics.

### Typical Usage
```python
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    UnifiedAuditService, AuditEventType, AuditContext
)

audit = UnifiedAuditService()

async def start_app():
    await audit.initialize()
    await audit.start_background_tasks()

async def shutdown_app():
    await audit.stop()

async def handle_request(user_id: int, payload: dict):
    ctx = AuditContext(
        user_id=str(user_id),
        endpoint="/api/v1/rag/search",
        method="POST",
    )
    await audit.log_event(
        event_type=AuditEventType.RAG_SEARCH,
        context=ctx,
        resource_type="rag_pipeline",
        action="retrieval",
        metadata={"query": payload["query"], "filters": list(payload.get("tags", []))}
    )
```

### Authentication Helpers
`UnifiedAuditService.log_login()` wraps `log_event()` for login attempts, filling
contextual fields automatically. Similar helpers can be added for domain-specific
events-mirror the pattern so tests can stub them easily.

### Export & Reporting
The service exposes `query_events(...)`, `count_events(...)`, `export_events(...)`
(CSV/JSONL), `get_security_summary(...)`, and other query helpers used by the API endpoints in
`app/api/v1/endpoints/audit.py`. Keep export code async (it uses the same pooled
connection and locks) to avoid blocking the event loop.

### Configuration Surface
| Setting | Description |
| --- | --- |
| `AUDIT_HIGH_RISK_SCORE` | Threshold for immediate flush/log alerts (default `70`). |
| `AUDIT_PII_USE_RAG_PATTERNS` | Merge RAG security filter regexes into the PII detector. |
| `AUDIT_PII_PATTERNS` | Dict of overrides or new patterns (`{label: regex or [regex...]}`). |
| `AUDIT_PII_SCAN_FIELDS` | Comma-separated string/list of additional event fields to inspect. |
| `AUDIT_MAX_DB_MB` | Optional soft cap (MB); currently logs a warning during cleanup when the audit DB exceeds the limit. |
| `TEST_MODE` / `TLDW_TEST_MODE` | Disable background loops and reduce I/O for tests. |

Constructor kwargs (`retention_days`, `buffer_size`, `flush_interval`, `enable_pii_detection`,
`enable_risk_scoring`, `db_path`) override global settings per instance.

## 3. Developer Guide for Contributors

### Related Endpoints
- Router (mounted under `/api/v1`):
  - Export audit events `/audit/export`: tldw_Server_API/app/api/v1/endpoints/audit.py:104
  - Count audit events `/audit/count`: tldw_Server_API/app/api/v1/endpoints/audit.py:217

### Related Schemas
- The public endpoints are query-driven and stream JSON/CSV; core types defined in service:
  - tldw_Server_API/app/core/Audit/unified_audit_service.py:1

### Related Tests (selection)
- Postgres membership flows produce audit events: tldw_Server_API/tests/AuthNZ_Postgres/test_admin_membership_audit_team_pg.py:12
- Admin audit log access and listing: tldw_Server_API/tests/AuthNZ/integration/test_auth_comprehensive.py:475
- Health checks summarize audit risk: tldw_Server_API/tests/Health/test_security_health_thresholds.py:21

### Integration Guidelines
1. **Per-user instances**: AuthNZ uses per-user caches keyed by user ID; reuse that
   pattern when per-tenant isolation is required. Avoid global singletons unless
   you truly need cross-tenant aggregation.
2. **Context discipline**: Always populate `AuditContext.user_id`, `endpoint`, and
   `method` when called from HTTP handlers. If the caller is a background job,
   prefer machine identifiers (`user_id="system"`, `api_key_hash`, etc.).
3. **Metadata hygiene**: Keep metadata JSON serializable. For large payloads, prefer
   summaries or IDs; only opt into redaction when necessary because it adds CPU
   cost. Redaction is recursive but still string-based.
4. **Async boundaries**: `log_event()` is async; never call it from sync contexts
   without wrapping in `asyncio.create_task()` or `asyncio.run_coroutine_threadsafe`.
5. **Flushing during shutdown**: Call `await audit.stop()` inside FastAPI shutdown
   hooks or worker termination handlers to flush the buffer and close the pool.

### Testing & Tooling
- Unit and integration tests live under `tldw_Server_API/tests/Audit/` plus modules
  that depend on auditing (AuthNZ, Embeddings, etc.). Run
  `python -m pytest tldw_Server_API/tests/Audit -v` after modifying core logic.
- Tests rely on the built-in test mode behaviour; avoid adding long-running async
  loops or thread sleeps without gating them behind the `_test_mode` check.
- When adding new risk heuristics or PII patterns, extend the corresponding tests
  to lock behaviour and prevent regressions.

### Extensibility Tips
- Add new `AuditEventType` values for clearly defined actions; keep namespaced
  (e.g., `workflows.job.started`) to avoid collisions.
- Prefer deriving categories automatically in `_determine_category()`; add to the
  mapping when the event type belongs to a new domain.
- Expose convenience wrappers alongside `log_login()` when a module produces
  repeating event templates. It keeps call sites short and reduces mistakes.
- For external sinks (SIEM, queue streams, etc.), hook into `flush()` by extending
  the service or by subscribing to the fallback JSONL queue in a background worker.

### Contribution Checklist
1. Update or add tests in `tests/Audit` (and dependent modules) for behaviour changes.
2. Document new settings or event types in this README and in `Docs/` if they affect
   end users or operators.
3. Keep log messages free of secrets; rely on the PII detector when in doubt.
4. Validate migrations or schema changes against an existing production DB copy;
   audit data is compliance-critical and must never be silently dropped.
5. Coordinate with the security reviewers if you change risk scoring thresholds or
   redaction logic-they feed into automated alerting.

With these guardrails, contributors can evolve the audit subsystem confidently
while maintaining the guarantees expected by downstream modules and operators.
