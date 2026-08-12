# Claims Jobs Stage 2A Analytics Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Claims analytics export execution onto the shared Jobs control plane behind an opt-in producer flag while preserving the existing synchronous API fallback.

**Architecture:** Claims remains responsible for authorization, normalized export requests, artifacts, rendering, reconciliation, and downloads. Jobs remains responsible for durable execution lifecycle, retries, leases, cancellation, quarantine, status, and administration; the only Jobs additions are scoped internal read helpers. Both synchronous requests and WorkerSDK handlers call the same bounded renderer against an owner-scoped Media DB snapshot.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, shared Media DB runtime, Jobs `JobManager` and `WorkerSDK`, pytest, Hypothesis, Ruff, Bandit.

---

## Source Specification

- Approved design: `Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md`
- Planning task: `TASK-12990`
- Predecessor: `TASK-12989`
- Stage 1 implementation reference: `Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md`

## Scope Guardrails

- Do not add a Claims queue, lease loop, retry counter, scheduler, or queue-control endpoint.
- Do not add review-metrics aggregation or cluster-rebuild work.
- Do not require `CLAIMS_JOBS_WORKER_ENABLED` in the producer path; dedicated workers are supported.
- Do not place filters, event content, rendered exports, paths, or secrets in Jobs payloads or results.
- Do not return HTTP 503 after `JobManager.create_job()` has returned an accepted row.
- Keep `ready` monotonic. A late worker or admin retry must not overwrite a ready artifact.
- Keep request-level idempotency out of scope. Jobs idempotency covers one server-generated export artifact only.

## File Responsibility Map

**Create**

- `tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py`: export request normalization, deterministic rendering, size and CSV safety, artifact lifecycle, read-only Jobs hydration, reconciliation, and lifecycle-aware cleanup.
- `tldw_Server_API/app/core/DB_Management/migrations/024_claims_analytics_export_jobs.sql`: SQLite upgrade from schema v23 to v24.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_claims_analytics_export_jobs.py`: PostgreSQL v24 migration body.
- `tldw_Server_API/tests/Claims/test_claims_analytics_exports.py`: normalization, rendering, limits, lifecycle, reconciliation, and cleanup tests.
- `tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py`: synchronous and asynchronous API behavior, owner scope, download responses, and OpenAPI checks.
- `tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py`: API-to-Jobs-to-WorkerSDK-to-download test.
- `tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py`: ready-terminal state property tests.
- `tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py`: SQLite export persistence and bounded event-page tests.
- `tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py`: scoped active/archive Jobs reads and exact batch-group lookup for SQLite.
- `tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py`: PostgreSQL parity for the internal Jobs reads.

**Modify**

- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`: bump Media DB schema to v24, update fresh SQLite export schema, import and bind runtime operations, and bind the PostgreSQL migration.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_claims_collection_structures.py`: update fresh PostgreSQL export schema.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`: register PostgreSQL migration v24.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/__init__.py`: export the v24 migration body.
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py`: owner-scoped CRUD, conditional transitions, exact deletion, and maintenance reads.
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py`: add deterministic, bounded keyset pages for export scans.
- `tldw_Server_API/app/core/Jobs/manager.py`: add internal ID-batch and exact batch-group reads with optional owner/domain/type/archive scope.
- `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`: add the analytics-export job type and strict ID-only payload validator.
- `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`: add producer flag and enqueue helper without a post-accept refresh.
- `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`: dispatch analytics-export jobs through the owner-scoped Media DB factory.
- `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`: retain authorization and target-database orchestration while delegating export behavior.
- `tldw_Server_API/app/api/v1/schemas/claims_schemas.py`: add artifact and read-only Job metadata.
- `tldw_Server_API/app/api/v1/endpoints/claims.py`: document and return dynamic 200/202 create responses, owner-scoped downloads, 409 responses, and safe CSV headers.
- `tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py`: extend payload contract coverage.
- `tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py`: extend flag and enqueue coverage.
- `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`: extend worker dispatch and failure classification coverage.
- `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`: preserve existing synchronous regression behavior.
- `tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py`: replace age-only cleanup expectations with terminal-aware behavior.
- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`: verify fresh SQLite v24 schema and v23-to-v24 migration.
- `tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py`: verify PostgreSQL v24 migration.
- `tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py`: verify fresh PostgreSQL columns and index.
- `tldw_Server_API/tests/Services/test_openapi_contracts.py`: assert create 200/202 and download 200/409 documentation.
- `tldw_Server_API/Config_Files/.env.example`: document producer, limits, retry, grace, and retention settings.
- `Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md`: document synchronous fallback, Jobs lifecycle projection, safe downloads, cleanup, rollout, and rollback.

## Task 1: Add Media DB Schema v24

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/migrations/024_claims_analytics_export_jobs.sql`
- Create: `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_claims_analytics_export_jobs.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py:513`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py:1151`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_claims_collection_structures.py:456`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/__init__.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py`

- [ ] **Step 1: Write failing fresh-schema and upgrade tests**

Add tests that assert schema version 24 and the three additive columns plus the `job_id` index:

```python
def _claims_export_columns(db: MediaDatabase) -> set[str]:
    return {
        str(row[1])
        for row in db.get_connection()
        .execute("PRAGMA table_info(claims_analytics_exports)")
        .fetchall()
    }


def test_fresh_sqlite_bootstrap_includes_claims_export_job_fields() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="claims-export-v24")
    try:
        assert db._CURRENT_SCHEMA_VERSION == 24
        assert {"job_id", "error_code", "snapshot_at"}.issubset(
            _claims_export_columns(db)
        )
        indexes = {
            str(row[1])
            for row in db.get_connection()
            .execute("PRAGMA index_list(claims_analytics_exports)")
            .fetchall()
        }
        assert "idx_claims_analytics_exports_job_id" in indexes
    finally:
        db.close_connection()
```

For the on-disk upgrade test, create only `schema_version(version=23)` and the current v23 `claims_analytics_exports` table, open `MediaDatabase`, then assert version 24, preserved rows, new nullable columns, and the index. Extend the PostgreSQL migration test to set `schema_version=23`, drop the three columns and index, call `_initialize_schema()`, and assert the restored v24 shape.

- [ ] **Step 2: Run the migration tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py \
  -k 'claims_export_job_fields or claims_analytics_export' -v
```

Expected: FAIL because schema version 24, fields, migration, or index do not exist. Run the PostgreSQL test with the repository fixture; the only accepted skip is the fixture reporting PostgreSQL unavailable.

- [ ] **Step 3: Add the SQLite migration**

Create `024_claims_analytics_export_jobs.sql` with exactly:

```sql
-- version: 24
-- description: Link Claims analytics export artifacts to shared Jobs lifecycle

BEGIN TRANSACTION;

ALTER TABLE claims_analytics_exports ADD COLUMN job_id INTEGER;
ALTER TABLE claims_analytics_exports ADD COLUMN error_code TEXT;
ALTER TABLE claims_analytics_exports ADD COLUMN snapshot_at TEXT;

CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_job_id
    ON claims_analytics_exports(job_id);

UPDATE schema_version SET version = 24;

COMMIT;
```

Bump `MediaDatabase._CURRENT_SCHEMA_VERSION` to 24 and add the same nullable fields and index to the fresh SQLite schema block.

- [ ] **Step 4: Add and register the PostgreSQL migration**

Create the v24 migration body with this public surface and statements:

```python
def run_postgres_migrate_to_v24(db: PostgresClaimsAnalyticsExportJobsBody, conn: Any) -> None:
    backend = db.backend
    ident = backend.escape_identifier
    statements = (
        f"ALTER TABLE {ident('claims_analytics_exports')} "
        f"ADD COLUMN IF NOT EXISTS {ident('job_id')} BIGINT",
        f"ALTER TABLE {ident('claims_analytics_exports')} "
        f"ADD COLUMN IF NOT EXISTS {ident('error_code')} TEXT",
        f"ALTER TABLE {ident('claims_analytics_exports')} "
        f"ADD COLUMN IF NOT EXISTS {ident('snapshot_at')} TIMESTAMPTZ",
        f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_analytics_exports_job_id')} "
        f"ON {ident('claims_analytics_exports')} ({ident('job_id')})",
    )
    for statement in statements:
        backend.execute(statement, connection=conn)
```

Add `_postgres_migrate_to_v24` to `SupportsPostgresMigrations`, map version 24 in `build_postgres_migration_map`, import and bind the body in `media_database_impl.py`, and update the fresh PostgreSQL table definition.

- [ ] **Step 5: Run migration tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_support.py \
  -k 'claims_export or migration_map or current_schema' -v
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py \
  -k 'v24' -v
```

Expected: PASS, with only fixture-declared PostgreSQL skips.

```bash
git add tldw_Server_API/app/core/DB_Management tldw_Server_API/tests/DB_Management
git commit -m "feat: add Claims export Jobs schema fields"
```

## Task 2: Add Owner-Scoped Artifact Operations and Bounded Event Pages

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py:2001`
- Create: `tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py`

- [ ] **Step 1: Write failing persistence and transition tests**

Cover these database contracts:

```python
def test_export_get_requires_matching_owner(media_db: MediaDatabase) -> None:
    media_db.create_claims_analytics_export(
        export_id="a" * 32,
        user_id="1",
        format="json",
        status="queued",
        filters_json="{}",
        pagination_json='{"limit":10,"offset":0}',
        snapshot_at="2026-08-08T12:00:00.000Z",
    )
    assert media_db.get_claims_analytics_export("a" * 32, user_id="1")
    assert media_db.get_claims_analytics_export("a" * 32, user_id="2") == {}


def test_ready_export_rejects_late_state_changes(media_db: MediaDatabase) -> None:
    export_id = "b" * 32
    _seed_export(media_db, export_id=export_id, status="processing")
    assert media_db.mark_claims_analytics_export_ready(
        export_id=export_id,
        user_id="1",
        payload_json='{"events":[]}',
        payload_csv=None,
    ) is True
    assert media_db.transition_claims_analytics_export_status(
        export_id=export_id,
        user_id="1",
        from_statuses=("ready",),
        to_status="failed",
        error_code="late_failure",
    ) is False
```

Also test: queued-to-processing, processing-to-failed, failed-to-processing, idempotent attachment of the same `job_id`, rejection of a different `job_id`, list/count fields, maintenance ordering, exact owner-scoped deletion, and cleanup based on `updated_at` rather than `created_at`.

- [ ] **Step 2: Write a failing bounded event-page test**

Insert rows with equal timestamps and assert keyset pagination has no gaps or duplicates:

```python
first = media_db.list_claims_monitoring_events_page(
    user_id="1",
    end_time="2026-08-08T12:00:00.000Z",
    limit=2,
)
second = media_db.list_claims_monitoring_events_page(
    user_id="1",
    end_time="2026-08-08T12:00:00.000Z",
    after_created_at=first[-1]["created_at"],
    after_id=int(first[-1]["id"]),
    limit=2,
)
assert [row["id"] for row in first + second] == sorted(
    row["id"] for row in first + second
)
```

- [ ] **Step 3: Run focused tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py -v
```

Expected: FAIL because the new fields and methods are not exposed.

- [ ] **Step 4: Implement strict artifact operations**

Keep every read and write scoped by both export ID and owner. Expose and bind these exact methods from `claims_analytics_export_ops.py`:

- `create_claims_analytics_export(self, *, export_id: str, user_id: str, format: str, status: str, payload_json: str | None = None, payload_csv: str | None = None, filters_json: str | None = None, pagination_json: str | None = None, error_message: str | None = None, job_id: int | None = None, error_code: str | None = None, snapshot_at: str | None = None) -> dict[str, Any]`
- `get_claims_analytics_export(self, export_id: str, *, user_id: str) -> dict[str, Any]`
- `attach_claims_analytics_export_job(self, *, export_id: str, user_id: str, job_id: int) -> bool`
- `transition_claims_analytics_export_status(self, *, export_id: str, user_id: str, from_statuses: tuple[str, ...], to_status: str, error_code: str | None = None, error_message: str | None = None) -> bool`
- `mark_claims_analytics_export_ready(self, *, export_id: str, user_id: str, payload_json: str | None, payload_csv: str | None) -> bool`
- `list_claims_analytics_exports_for_maintenance(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]`
- `delete_claims_analytics_exports(self, *, user_id: str, export_ids: list[str], updated_before: str) -> int`

`transition_claims_analytics_export_status` must validate the transition against:

```python
ALLOWED_EXPORT_TRANSITIONS = {
    ("queued", "processing"),
    ("queued", "failed"),
    ("processing", "ready"),
    ("processing", "failed"),
    ("failed", "processing"),
    ("ready", "ready"),
}
```

The SQL must include `WHERE export_id = ? AND user_id = ?` followed by a generated, parameterized `status IN` placeholder list. `mark_claims_analytics_export_ready` writes exactly one of `payload_json` or `payload_csv`, clears safe error fields, updates `updated_at`, and only succeeds from `processing`. All list projections include `job_id`, `error_code`, and `snapshot_at`; payload bodies remain excluded from list results.

Task 12 batch 1 extends the persisted artifact projection with nullable internal
`snapshot_event_id`. Capture the owner's non-negative event-ID high-water when
creating either sync or async artifacts; retries and synchronous rendering pass
it to monitoring-event scans as `id <= snapshot_event_id`. Legacy null rows
retain time-cutoff-only behavior. Keep this field out of Jobs contracts and
public lifecycle responses, order export history by `created_at DESC, export_id
DESC`, and maintain `claims_monitoring_events(user_id, created_at, id)` in fresh
SQLite/PostgreSQL schemas and both v24 upgrade paths.

- [ ] **Step 5: Implement bounded monitoring-event pages**

Add:

```python
def list_claims_monitoring_events_page(
    self,
    *,
    user_id: str,
    event_type: str | None = None,
    severity: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    after_created_at: Any = None,
    after_id: int | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    # Query is owner scoped and ends with:
    # ORDER BY created_at ASC, id ASC LIMIT ?
```

When a cursor is present, append:

```sql
AND (created_at > ? OR (created_at = ? AND id > ?))
```

Require `after_created_at` and `after_id` together. Clamp page size to 1 through 1000. Continue using `?` placeholders because the Media DB execution layer performs PostgreSQL conversion.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py
git commit -m "feat: harden Claims analytics export persistence"
```

## Task 3: Add Scoped Jobs Read Helpers

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py:2499`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py`

- [ ] **Step 1: Write failing SQLite tests**

Create jobs for two domains and owners, archive one terminal Claims job, and assert:

```python
rows = manager.get_jobs_by_ids(
    [claims_active_id, claims_archived_id, foreign_owner_id, other_domain_id],
    domain="claims",
    owner_user_id="1",
    include_archived=True,
)
assert set(rows) == {claims_active_id, claims_archived_id}
assert rows[claims_active_id]["archived"] is False
assert rows[claims_archived_id]["archived"] is True
```

Test duplicate IDs, more than one chunk, rejected booleans/zero/negative/non-integral values, decrypted payload/result parity, active-row preference if an ID appears in both stores, and an empty input that returns `{}` without opening a query.

Add exact batch-group tests:

```python
found = manager.find_job_by_batch_group(
    batch_group=f"claims-analytics-export:{export_id}",
    domain="claims",
    owner_user_id="1",
    job_type="claims_generate_analytics_export",
    include_archived=True,
)
assert int(found["id"]) == expected_job_id
```

Assert another owner, another domain, another type, or a prefix-only batch-group does not match.

- [ ] **Step 2: Add equivalent PostgreSQL tests**

Use `jobs_pg_dsn` and `JobManager(None, backend="postgres", db_url=jobs_pg_dsn)`. Mark the module `pytestmark = pytest.mark.pg_jobs`. Test the same owner/domain/archive behavior and a list large enough to exercise chunking.

- [ ] **Step 3: Run the tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py -v
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py -v
```

Expected: FAIL because the read helpers do not exist. PostgreSQL may skip only through the shared fixture.

- [ ] **Step 4: Implement `get_jobs_by_ids`**

Add this exact public signature to `JobManager`:

```python
def get_jobs_by_ids(
    self,
    job_ids: list[int],
    *,
    domain: str | None = None,
    owner_user_id: str | None = None,
    include_archived: bool = False,
) -> dict[int, dict[str, Any]]:
```

Normalize unique positive integer IDs, rejecting booleans and non-integral values with `BadRequestError`. Chunk SQLite at 400 IDs and PostgreSQL at 1000 IDs. Add `domain` and `owner_user_id` predicates inside every active and archive query. Normalize active payload/result with the same decrypt path as `get_job`; normalize archived rows with `_normalize_archived_job_row`. Set `archived=False` on active rows and never replace an active row with an archive row.

- [ ] **Step 5: Implement exact batch-group lookup**

Add:

```python
def find_job_by_batch_group(
    self,
    *,
    batch_group: str,
    domain: str,
    owner_user_id: str,
    job_type: str,
    include_archived: bool = False,
) -> dict[str, Any] | None:
```

Use equality predicates for all four fields and `ORDER BY id DESC LIMIT 1` for active rows. If absent and `include_archived=True`, query `jobs_archive` with the same predicates and `ORDER BY archive_id DESC LIMIT 1`. Do not expose a new HTTP endpoint.

- [ ] **Step 6: Run Jobs parity tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py -v
python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py -q
```

Expected: PASS, with fixture-declared PostgreSQL skips only.

```bash
git add \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py
git commit -m "feat: add scoped Jobs batch reads"
```

## Task 4: Build Deterministic Export Normalization and Rendering

**Files:**
- Create: `tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py`
- Create: `tldw_Server_API/tests/Claims/test_claims_analytics_exports.py`

- [ ] **Step 1: Write failing normalization tests**

Test JSON/CSV format normalization, ISO-8601 UTC conversion, fixed clock injection, effective end cutoff, invalid ranges, pagination clamping, canonical owner IDs, and removal of `workspace_id`:

```python
normalized = normalize_export_request(
    {
        "format": "json",
        "filters": {
            "workspace_id": "7",
            "start_time": "2026-08-01T01:00:00-07:00",
            "end_time": "2026-08-10T00:00:00Z",
        },
        "pagination": {"limit": 20_000, "offset": -4},
    },
    owner_user_id="7",
    now=datetime(2026, 8, 8, 12, tzinfo=timezone.utc),
)
assert normalized["filters"] == {
    "start_time": "2026-08-01T08:00:00.000Z",
    "end_time": "2026-08-08T12:00:00.000Z",
}
assert normalized["pagination"] == {"limit": 10_000, "offset": 0}
assert normalized["snapshot_at"] == "2026-08-08T12:00:00.000Z"
```

Naive ISO timestamps are interpreted as UTC for backward compatibility. Unsupported formats, malformed timestamps, and start-after-end raise `ClaimsAnalyticsExportError` with a stable non-retryable code.

- [ ] **Step 2: Write failing rendering and safety tests**

Cover:

- Deterministic `(created_at, id)` ordering across equal timestamps.
- Provider/model filtering while scanning bounded pages.
- Fixed snapshot results when newer rows are inserted between attempts.
- Equivalent JSON content for synchronous and worker calls using the same normalized request.
- CSV Unicode, delimiter, quote, and newline handling.
- Formula protection for strings beginning with `=`, `+`, `-`, `@`, tab, or carriage return.
- UTF-8 byte enforcement at exactly the limit and one byte above it.
- A compact result that contains only export ID, format, event count, and size.

Use this assertion for dangerous CSV cells:

```python
assert spreadsheet_safe("=SUM(A1:A2)") == "'=SUM(A1:A2)"
assert spreadsheet_safe("+1") == "'+1"
assert spreadsheet_safe("-1") == "'-1"
assert spreadsheet_safe("@cmd") == "'@cmd"
assert spreadsheet_safe("\tcmd") == "'\tcmd"
assert spreadsheet_safe("\rcmd") == "'\rcmd"
assert spreadsheet_safe("safe") == "safe"
```

- [ ] **Step 3: Run the tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  -k 'normalize or render or csv or size or snapshot' -v
```

Expected: FAIL because the domain module does not exist.

- [ ] **Step 4: Create the domain types and normalization helpers**

Define:

```python
DEFAULT_EXPORT_MAX_BYTES = 10_485_760
DEFAULT_EXPORT_ORPHAN_GRACE_SEC = 300
EXPORT_SCAN_PAGE_SIZE = 1000
EXPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CSV_COLUMNS = ("id", "event_type", "severity", "created_at", "payload_json")


class ClaimsAnalyticsExportError(RuntimeError):
    def __init__(
        self,
        public_message: str,
        *,
        code: str,
        retryable: bool = False,
        http_status: int = 400,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.code = code
        self.retryable = retryable
        self.http_status = http_status
```

Expose `normalize_export_request(payload, *, owner_user_id, now=None)`, `validate_export_id(value)`, `export_max_bytes(settings_obj=None)`, and `orphan_grace_seconds(settings_obj=None)`. Persist only known filters: `event_type`, `severity`, `provider`, `model`, `start_time`, and effective `end_time`.

- [ ] **Step 5: Implement bounded deterministic rendering**

Expose:

```python
def render_export(
    db: Any,
    *,
    owner_user_id: str,
    format: str,
    filters: dict[str, Any],
    pagination: dict[str, int],
    snapshot_at: str,
    max_bytes: int,
) -> dict[str, Any]:
```

Scan `list_claims_monitoring_events_page()` in pages of 1000. Apply
provider/model filters in parameterized database predicates, increment a total
match counter from constant-size metadata rows, and retain only matches in
`[offset, offset + limit)`. Continue JSON's bounded scan to calculate the stable
`pagination.total`; stop CSV once its selected window is complete because CSV
does not expose a total. Never retain unrelated pages. Metadata pages omit
variable-width `event_type`, `severity`, and payload text. Load them together
only for selected owner-scoped rows through a query that caps their combined raw
source before JSON parsing at six times the builder's decreasing remaining-byte
budget plus a fixed 64 KiB formatting allowance, then canonicalizes payload JSON
with compact separators, `ensure_ascii=False`, and strict finite numbers before
applying the exact UTF-8 limit. Provider/model filters match JSON strings only
on both database backends. Render JSON with compact deterministic separators and
UTF-8 preservation:

```python
payload_text = json.dumps(
    {"events": events, "filters": filters, "pagination": pagination_meta},
    ensure_ascii=False,
    separators=(",", ":"),
)
```

Render CSV through `csv.writer(io.StringIO(newline=""), lineterminator="\r\n")`, applying `spreadsheet_safe()` to every string cell. Measure `len(payload_text.encode("utf-8"))`; raise `claims_export_too_large` with HTTP 413 when above the limit. Return:

```python
{
    "payload_json": payload_text if format == "json" else None,
    "payload_csv": payload_text if format == "csv" else None,
    "format": format,
    "event_count": len(events),
    "size_bytes": size_bytes,
}
```

- [ ] **Step 6: Run rendering tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  -k 'normalize or render or csv or size or snapshot' -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py
git commit -m "feat: add bounded Claims analytics export renderer"
```

## Task 5: Add Retry-Safe Artifact Lifecycle, Hydration, and Reconciliation

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_analytics_exports.py`
- Create: `tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py`

- [ ] **Step 1: Write failing artifact lifecycle tests**

Cover queued creation, synchronous ready creation, missing-job repair, already-ready skip, failed retry, late-ready race, compact results, safe failure persistence, and transient DB classification:

```python
result = process_export_artifact(
    db,
    owner_user_id="1",
    export_id=export_id,
    job_id=42,
)
assert result == {
    "outcome": "ok",
    "export_id": export_id,
    "format": "json",
    "event_count": 1,
    "size_bytes": len(stored_payload.encode("utf-8")),
}
assert db.get_claims_analytics_export(export_id, user_id="1")["status"] == "ready"
```

Assert a ready artifact returns:

```python
{
    "outcome": "skipped",
    "reason": "already_ready",
    "export_id": export_id,
}
```

- [ ] **Step 2: Write failing hydration, reconciliation, and cleanup tests**

Use a fake `JobManager` that records calls. Assert ID hydration performs one `get_jobs_by_ids` call for the page, always supplies `domain="claims"` and owner, and returns `job_status=None` on lookup failure.

For a queued artifact without `job_id`, test:

- Before 300 seconds: unchanged and no orphan failure.
- Exact active or archived batch group: attach matching Job ID.
- Prefix-only or wrong-owner match: ignored.
- Jobs lookup exception: unchanged.
- Successful active/archive lookup with no match after grace: failed with `claims_export_enqueue_failed`.

For cleanup, prove old ready rows and terminal failed rows can be removed, while queued, processing, retrying, or uncertain rows are preserved. Base age on `updated_at`.

- [ ] **Step 3: Write the ready-terminal property test**

Create `tests/Claims/property/__init__.py` only if the package needs it, then add:

```python
@given(st.lists(st.sampled_from(["processing", "failed", "ready"]), max_size=30))
def test_ready_state_is_monotonic(candidate_states: list[str]) -> None:
    state = "queued"
    for candidate in candidate_states:
        state = apply_export_transition(state, candidate)
        if state == "ready":
            assert apply_export_transition(state, "processing") == "ready"
            assert apply_export_transition(state, "failed") == "ready"
```

The tested pure `apply_export_transition(current, requested)` helper must mirror the database transition table and return the unchanged current status for rejected transitions.

- [ ] **Step 4: Run the tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py \
  -k 'artifact or reconcile or hydrate or cleanup or monotonic or retry' -v
```

Expected: FAIL because lifecycle orchestration is absent.

- [ ] **Step 5: Implement artifact creation and processing**

Expose these exact functions:

- `create_queued_artifact(db: Any, *, owner_user_id: str, normalized: dict[str, Any]) -> dict[str, Any]`
- `create_ready_artifact(db: Any, *, owner_user_id: str, normalized: dict[str, Any]) -> dict[str, Any]`
- `process_export_artifact(db: Any, *, owner_user_id: str, export_id: str, job_id: int) -> dict[str, Any]`

`create_queued_artifact` generates `uuid4().hex`, stores normalized filters/pagination and `snapshot_at`, and returns the row. `create_ready_artifact` creates the row as `processing`, calls `render_export`, and conditionally marks it ready so synchronous and worker paths use identical rendering. If rendering exceeds the byte limit, store `claims_export_too_large` and re-raise the domain error for HTTP 413.

`process_export_artifact` loads by owner and export ID, repairs a missing `job_id`, transitions `queued|failed -> processing`, reloads and validates persisted JSON, renders, and conditionally marks ready. If another attempt wins, reload and return `already_ready`. Persist only `error.code` and `error.public_message`; never persist `str(raw_exception)`.

- [ ] **Step 6: Implement read-only Jobs hydration and conservative reconciliation**

Expose these exact functions:

- `hydrate_job_statuses(rows: list[dict[str, Any]], *, owner_user_id: str, job_manager: JobManager) -> dict[int, str | None]`
- `reconcile_export_artifacts(db: Any, *, owner_user_id: str, job_manager: JobManager, now: datetime | None = None, limit: int = 100) -> dict[str, int]`
- `cleanup_export_artifacts(db: Any, *, owner_user_id: str, job_manager: JobManager, now: datetime | None = None, retention_hours: float = 24, limit: int = 100) -> int`

Hydration catches Jobs availability errors and returns null projections without mutating artifact state. Reconciliation uses exact `claims-analytics-export:{export_id}` batch groups and `include_archived=True`. It only marks a proven orphan after grace and a successful no-match result. Cleanup accepts terminal Jobs statuses `completed`, `failed`, `cancelled`, and `quarantined`; it skips non-ready uncertainty.

Use independent bounded reconciliation pages for queued artifacts missing a Job
ID and queued/processing artifacts with an attached Job ID. The latter uses
read-only exact Jobs projection and marks the artifact failed only when Jobs'
shared terminal-status classifier confirms a terminal Job. Failed artifacts
without a Job ID, regardless of error code, require retention plus grace and an
exact archived-aware Jobs absence proof before cleanup.

Rotate each reconciliation page by export ID on a bounded maintenance interval
so unchanged active rows cannot starve later candidates. Strict serialization
must reject non-finite JSON values. Persisted cancelled/quarantined Claims codes
remain in the public safe-code allowlist even when Jobs projection is absent.

- [ ] **Step 7: Run lifecycle tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/property \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py
git commit -m "feat: add retry-safe Claims export artifacts"
```

## Task 6: Add the Analytics Export Jobs Contract and Producer

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py`

- [ ] **Step 1: Write failing payload contract tests**

Add acceptance for exactly:

```python
{
    "version": 1,
    "owner_user_id": "123",
    "export_id": "0123456789abcdef0123456789abcdef",
}
```

Reject uppercase, hyphenated, short, long, and non-hex IDs; unsupported versions; non-canonical owners; filters, pagination, payload bodies, `workspace_id`, database paths, credentials, and every unknown key. Assert stable `claims_export_invalid_payload` or existing owner/version codes as specified by the design.

- [ ] **Step 2: Write failing producer tests**

Assert the complete create call:

```python
assert fake.created[0] == {
    "domain": "claims",
    "queue": "default",
    "job_type": "claims_generate_analytics_export",
    "payload": {
        "version": 1,
        "owner_user_id": "123",
        "export_id": export_id,
    },
    "owner_user_id": "123",
    "priority": 5,
    "max_retries": 3,
    "batch_group": f"claims-analytics-export:{export_id}",
    "idempotency_key": f"claims:analytics_export:123:{export_id}",
}
```

Test the full flag matrix: asynchronous production is enabled only if both `CLAIMS_JOBS_ENABLED` and `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` are true. Test negative/invalid retry values fall back to three.

Use a fake manager whose `get_job()` raises and assert analytics enqueue still succeeds without calling it. This protects the rule that no fallible refresh occurs after Jobs acceptance.

- [ ] **Step 3: Run the tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py \
  -k 'analytics_export' -v
```

Expected: FAIL because the new contract and helper do not exist.

- [ ] **Step 4: Implement the strict contract**

Add:

```python
CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE = "claims_generate_analytics_export"
CLAIMS_ANALYTICS_EXPORT_PAYLOAD_KEYS = {"version", "owner_user_id", "export_id"}
CLAIMS_ANALYTICS_EXPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def validate_analytics_export_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    version = _version(payload)
    owner_user_id = _owner_user_id(payload.get("owner_user_id"))
    _reject_unknown_keys(payload, CLAIMS_ANALYTICS_EXPORT_PAYLOAD_KEYS)
    export_id = str(payload.get("export_id") or "")
    if CLAIMS_ANALYTICS_EXPORT_ID_RE.fullmatch(export_id) is None:
        raise ClaimsJobError(
            "claims analytics export payload has invalid export_id",
            retryable=False,
            failure_code="claims_export_invalid_payload",
        )
    return {"version": version, "owner_user_id": owner_user_id, "export_id": export_id}
```

Add export-body keys such as `filters`, `pagination`, `events`, `payload_json`, and `payload_csv` to the sensitive/disallowed coverage so failures happen before any logging or persistence.

- [ ] **Step 5: Implement the flag and enqueue helper**

Add:

```python
def claims_analytics_export_jobs_enabled(
    settings_obj: Mapping[str, Any] | None = None,
) -> bool:
    return claims_jobs_enabled(settings_obj) and _truthy(
        _setting_value("CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED", False, settings_obj)
    )


def enqueue_claims_analytics_export(
    *,
    owner_user_id: str,
    export_id: str,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_analytics_export_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "export_id": export_id,
        }
    )
    manager = _manager(job_manager)
    return manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries(
            "CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT", 3, settings_obj
        ),
        batch_group=f"claims-analytics-export:{payload['export_id']}",
        idempotency_key=(
            f"claims:analytics_export:{payload['owner_user_id']}:"
            f"{payload['export_id']}"
        ),
    )
```

Do not call `_refresh()` from this helper.

- [ ] **Step 6: Run contract tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py
git commit -m "feat: add Claims analytics export job contract"
```

## Task 7: Dispatch Export Jobs Through the Existing Claims Worker

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`

- [ ] **Step 1: Write failing handler tests**

Test owner equality, backend-aware DB creation, attached Job repair, success metadata, already-ready skip, missing/wrong-owner artifact, malformed persisted request, oversized output, serialization failure, and transient database errors.

Use a fake managed database context and assert the handler calls:

```python
process_export_artifact(
    db,
    owner_user_id="7",
    export_id=export_id,
    job_id=81,
)
```

Assert deterministic failures are `ClaimsJobError(retryable=False)` with their stable domain codes. Assert a temporary `sqlite3.OperationalError("database is locked")` is converted to `ClaimsJobError(retryable=True, failure_code="claims_export_storage_unavailable")` without the raw exception text appearing in the public message.

- [ ] **Step 2: Run handler tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py \
  -k 'analytics_export' -v
```

Expected: FAIL because dispatch is absent.

- [ ] **Step 3: Add the worker dispatch branch**

In `process_claims_job`, add the analytics branch before the unsupported-type error:

```python
if job_type == CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE:
    payload = validate_analytics_export_payload(_payload(job))
    owner_user_id = _assert_owner(job, payload["owner_user_id"])
    job_id = _positive_job_id(job.get("id"))
    return await asyncio.to_thread(
        _process_analytics_export,
        owner_user_id=owner_user_id,
        export_id=payload["export_id"],
        job_id=job_id,
    )
```

Implement `_process_analytics_export` with this exact context-manager call:

```python
with managed_media_database(
    client_id="claims_jobs_worker",
    db_path=_db_path(owner_user_id),
    initialize=False,
    suppress_init_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
    suppress_close_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
) as db:
    return process_export_artifact(
        db,
        owner_user_id=owner_user_id,
        export_id=export_id,
        job_id=job_id,
    )
```

The shared factory will select PostgreSQL when configured; the owner ID still scopes every artifact and event query.

Catch `ClaimsAnalyticsExportError` and copy only its public message, code, and retryable flag into `ClaimsJobError`. Catch the project’s concrete transient database/storage exception tuple separately and raise `claims_export_storage_unavailable` with `retryable=True`. Do not classify validation, JSON decoding, or byte-limit errors as transient.

- [ ] **Step 4: Run handler and worker-service regressions**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py \
  tldw_Server_API/tests/Services/test_claims_jobs_worker.py -v
```

Expected: PASS. No changes are required in `app/services/claims_jobs_worker.py`.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py
git commit -m "feat: handle Claims analytics export jobs"
```

## Task 8: Refactor Create API Into Shared Sync and Async Paths

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py:3389`
- Modify: `tldw_Server_API/app/api/v1/schemas/claims_schemas.py:409`
- Modify: `tldw_Server_API/app/api/v1/endpoints/claims.py:561`
- Create: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`
- Modify: `tldw_Server_API/tests/Services/test_openapi_contracts.py`

- [ ] **Step 1: Write failing synchronous create tests**

With both producer flags false, assert HTTP 200, `status="ready"`, null `job_id` and `job_status`, a fixed `snapshot_at`, correct normalized output, and no enqueue call. Assert oversized synchronous output returns 413 with `claims_export_too_large` and leaves a failed artifact with no payload.

- [ ] **Step 2: Write failing asynchronous create tests**

With both producer flags true, assert HTTP 202, queued artifact, exact Job ID/status, no inline event query/render, and persisted normalized request. Add `test_enqueue_failure_marks_artifact_failed_and_returns_503`, which stores `claims_export_enqueue_failed`, and `test_attach_failure_after_jobs_acceptance_still_returns_202`, which returns the accepted Job ID and leaves reconciliation/worker repair possible.

Test the flag matrix where either flag false remains synchronous. Name the interruption tests `test_enqueue_failure_marks_artifact_failed_and_returns_503` and `test_attach_failure_after_jobs_acceptance_still_returns_202`. Do not include `CLAIMS_JOBS_WORKER_ENABLED` in any producer decision.

- [ ] **Step 3: Write failing owner-routing tests**

For per-user SQLite, a platform admin creating with `filters.workspace_id="2"` must write to user 2’s database and return a download URL ending in `?workspace_id=2`. A non-platform-admin caller receives 403. Add a PostgreSQL fake/backend test that confirms the shared DB is retained but every operation receives owner `"2"`.

- [ ] **Step 4: Write failing OpenAPI tests**

Assert POST `/api/v1/claims/analytics/export` documents both 200 and 202 with `ClaimsAnalyticsExportResponse`. Assert fields are nullable/additive rather than separate incompatible models.

- [ ] **Step 5: Run API tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'claims_analytics_export' -v
```

Expected: FAIL because create is always synchronous and response fields are absent.

- [ ] **Step 6: Extend the response schemas**

Add these fields to `ClaimsAnalyticsExportResponse`:

```python
job_id: int | None = None
job_status: str | None = None
error_code: str | None = None
snapshot_at: str | None = None
```

Add the same fields to `ClaimsAnalyticsExportListItem`. Keep `status` as artifact status. Keep `error_message` nullable for backward-compatible safe messages, but never populate it from raw Jobs or exception text.

- [ ] **Step 7: Refactor service orchestration**

Make `export_claims_analytics(*, payload: dict[str, Any], principal: AuthPrincipal, current_user: User, db: MediaDatabase) -> tuple[dict[str, Any], int]` return `(body, http_status)`. Implement the orchestration in this fixed order:

1. Authorize Claims administration.
2. Resolve `(owner_user_id, workspace_id)` through a helper that validates a canonical positive integer and requires platform-admin Claims permission for a different owner.
3. Normalize the request through `claims_analytics_exports.normalize_export_request(payload, owner_user_id=owner_user_id)`.
4. Open the target DB through `_resolve_media_db`; SQLite uses the owner path and PostgreSQL retains the shared backend.
5. Run bounded reconciliation and cleanup in a catch-and-log best-effort block.
6. When export Jobs are disabled, call `create_ready_artifact` and return the projected response with status 200.
7. When enabled, call `create_queued_artifact`, then `enqueue_claims_analytics_export(owner_user_id=owner_user_id, export_id=row["export_id"])`.
8. Catch `Exception` at the enqueue compensation boundary with `# noqa: BLE001`, mark the row failed using only `claims_export_enqueue_failed`, and raise HTTP 503 with a stable public detail object.
9. After `create_job` returns, parse its positive Job ID. Attempt `attach_claims_analytics_export_job`; on a storage exception log only export ID, Job ID, operation, and exception type, then continue.
10. Return the queued artifact, accepted Job ID/status, and HTTP 202.

Use `_resolve_media_db` or a narrow wrapper around it so SQLite opens the target user path and PostgreSQL keeps the shared backend. Validate target IDs as canonical positive integers before path routing.

- [ ] **Step 8: Return dynamic status codes from the endpoint**

Inject FastAPI `Response` and document 202:

```python
@router.post(
    "/analytics/export",
    response_model=ClaimsAnalyticsExportResponse,
    responses={202: {"model": ClaimsAnalyticsExportResponse}},
)
def export_claims_analytics(
    payload: ClaimsAnalyticsExportRequest,
    response: Response,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    body, response_status = claims_service.export_claims_analytics(
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )
    response.status_code = response_status
    return body
```

Remove export-specific CSV/IO code and imports from `claims_service.py`; retain `uuid4` where used by FVA code.

- [ ] **Step 9: Run create/API regressions and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'claims_analytics_export or dashboard_analytics_and_export' -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/api/v1/schemas/claims_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
git commit -m "feat: queue Claims analytics exports through Jobs"
```

## Task 9: Harden List, Download, Reconciliation, and Retention APIs

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py:3501`
- Modify: `tldw_Server_API/app/api/v1/endpoints/claims.py:577`
- Modify: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py`
- Modify: `tldw_Server_API/tests/Services/test_openapi_contracts.py`

- [ ] **Step 1: Write failing list and hydration tests**

Assert list results are owner scoped, retain artifact `status`, add nullable `job_status`, and batch all linked IDs through one scoped Jobs call. When Jobs is unavailable, return rows with null Job statuses. Keep status filtering artifact-only.

For platform-admin cross-user lists, assert the correct SQLite database is opened and every returned `download_url` includes the canonical owner query, such as `?workspace_id=2`. A non-admin cross-user request remains 403.

- [ ] **Step 2: Write failing download tests**

Cover:

- Ready JSON: `application/json` and payload body.
- Ready CSV: `text/csv; charset=utf-8`, `X-Content-Type-Options: nosniff`, and a filename derived only from the validated ID, such as `Content-Disposition: attachment; filename="claims-analytics-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.csv"`.
- Ready artifact while Jobs lookup fails: still HTTP 200.
- Queued/processing/retrying: HTTP 409 with `claims_export_not_ready`.
- Cancelled/quarantined projection: HTTP 409 with the matching stable code.
- Failed artifact: HTTP 409 with stored safe code or `claims_export_failed`.
- Missing and wrong-owner IDs: indistinguishable HTTP 404.
- Malformed export ID: HTTP 404, not a reflected filename or query.
- Platform-admin cross-user download through `workspace_id`; non-admin denial.

- [ ] **Step 3: Write failing lifecycle-aware cleanup tests**

Update the old cleanup test so age alone cannot delete queued/processing/uncertain failed rows. Assert terminal rows are deleted only after `updated_at` crosses retention and a successful Jobs lookup establishes eligibility. Include pruned Job behavior and failed rows without Jobs after orphan grace.

- [ ] **Step 4: Run API tests and confirm red failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'list or download or cleanup or conflict or claims_analytics_export' -v
```

Expected: FAIL because current downloads return empty success for non-ready rows and reads are not fully owner scoped.

- [ ] **Step 5: Delegate list behavior to the domain module**

In `list_claims_analytics_exports`, resolve the target Media DB before reading rows, run bounded best-effort reconciliation/cleanup, and call `hydrate_job_statuses`. Parse only persisted normalized filter/pagination JSON. Do not expose Jobs error strings or results.

Build cross-user URLs only from canonical owner and export ID:

```python
def export_download_url(export_id: str, workspace_id: str | None = None) -> str:
    validated = validate_export_id(export_id)
    base = f"/api/v1/claims/analytics/export/{validated}"
    return f"{base}?workspace_id={workspace_id}" if workspace_id else base
```

- [ ] **Step 6: Make downloads owner-scoped and readiness-aware**

Add `workspace_id: Optional[str] = None` to the download endpoint and service. Query `get_claims_analytics_export(export_id, user_id=owner_user_id)` directly; never fetch globally and authorize afterward.

Return the domain result as a FastAPI `Response`:

```python
if result["format"] == "csv":
    return Response(
        content=result["payload_csv"],
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="claims-analytics-{result["export_id"]}.csv"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )
return Response(
    content=result["payload_json"],
    media_type="application/json",
    headers={"X-Content-Type-Options": "nosniff"},
)
```

For non-ready artifacts, raise:

```python
HTTPException(
    status_code=409,
    detail={
        "code": public_code,
        "status": artifact_status,
        "job_status": job_status,
    },
)
```

Document 409 in OpenAPI. Do not create Claims cancellation, retry, pause, quarantine, or drain endpoints.

- [ ] **Step 7: Run list/download/cleanup tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py -v
```

Expected: PASS.

```bash
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
git commit -m "fix: enforce Claims export lifecycle and owner scope"
```

## Task 10: Add WorkerSDK End-to-End and PostgreSQL Parity Coverage

**Files:**
- Create: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py`

- [ ] **Step 1: Write the local API-to-worker end-to-end test**

Use a temporary Jobs SQLite DB and owner Media DB. Enable producer flags, POST the export, then run one bounded WorkerSDK iteration:

```python
sdk = WorkerSDK(
    manager,
    WorkerConfig(
        domain="claims",
        queue="default",
        worker_id="claims-export-e2e",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
    ),
)

async def on_completed(_job: dict[str, Any], _result: dict[str, Any]) -> None:
    sdk.stop()

await asyncio.wait_for(
    sdk.run(handler=process_claims_job, on_completed=on_completed),
    timeout=2,
)
```

Assert POST returned 202, Jobs completed, artifact is ready, list projects `job_status="completed"`, and download returns the expected deterministic payload. Do not start a daemon or sleep waiting for background polling.

- [ ] **Step 2: Add a retry-recovery integration test**

Inject one retryable storage failure after the artifact reaches processing. Assert WorkerSDK requeues/fails according to its configured retry behavior, a second handler attempt transitions failed back to processing, and the eventual ready artifact is not overwritten by a late failing attempt.

- [ ] **Step 3: Add PostgreSQL operation parity tests**

Using the repository PostgreSQL Media DB fixture or backend factory, exercise:

- Create/get/list/count with explicit owners.
- Cross-owner denial.
- Conditional transitions and ready monotonicity.
- Bounded event pages with equal timestamps.
- Job ID attachment and v24 fields.
- Exact deletion based on `updated_at`.

Record a skip only when the shared fixture reports PostgreSQL unavailable. Do not create a separate database bootstrap path.

- [ ] **Step 4: Run integration coverage**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py -v
```

Expected: PASS, with fixture-declared PostgreSQL skips only.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py
git commit -m "test: cover Claims export Jobs end to end"
```

## Task 11: Document Configuration, Rollout, and Operator Boundaries

**Files:**
- Modify: `tldw_Server_API/Config_Files/.env.example`
- Modify: `Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md`

- [ ] **Step 1: Update environment configuration documentation**

Add a Claims Jobs section with these defaults:

```dotenv
# Route Claims background producers through shared Jobs. Existing Stage 1 switch.
#CLAIMS_JOBS_ENABLED=false
# Start the Claims WorkerSDK service in this process; dedicated workers may enable this separately.
#CLAIMS_JOBS_WORKER_ENABLED=false
#CLAIMS_JOBS_QUEUE=default

# Stage 2A opt-in producer. Requires CLAIMS_JOBS_ENABLED=true.
#CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED=false
#CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT=3
#CLAIMS_ANALYTICS_EXPORT_MAX_BYTES=10485760
#CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC=300
#CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS=24
```

State that producer and worker flags are independent and producer-first rollback is required.

- [ ] **Step 2: Correct the product documentation**

Document:

- HTTP 200 ready response when either producer flag is disabled.
- HTTP 202 queued response with `job_id`, `job_status`, and `snapshot_at` when enabled.
- Artifact status versus Jobs lifecycle status.
- 409 for non-ready/failed downloads and 404 for missing/wrong-owner exports.
- 10,000-row and 10 MiB limits.
- CSV formula protection and attachment headers.
- Owner-scoped cross-user URLs.
- Request-time bounded reconciliation/cleanup rather than a Claims scheduler.
- Existing Jobs admin endpoints as the only pause/cancel/retry/quarantine/drain controls.
- Rollout order: schema/handler, workers, producer canary.
- Rollback order: producer off, workers continue draining.

Remove the inaccurate statement that cleanup is performed by a scheduled Claims job.

- [ ] **Step 3: Run documentation checks and commit**

Run:

```bash
rg -n 'CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED|producer|rollback|HTTP 202|HTTP 409' \
  tldw_Server_API/Config_Files/.env.example \
  Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md
git diff --check
```

Expected: every new setting and rollout rule is present; `git diff --check` exits 0.

```bash
git add \
  tldw_Server_API/Config_Files/.env.example \
  Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md
git commit -m "docs: describe Claims export Jobs rollout"
```

## Task 12: Run Final Review and Verification Gates

**Files:**
- Modify only files required to fix findings from this task’s checks.

- [ ] **Step 1: Run the focused Claims and Jobs suites**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py \
  tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py \
  tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py -v
```

Expected: PASS, with only shared-fixture PostgreSQL skips.

- [ ] **Step 2: Run Stage 1 and lifecycle regressions**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Services/test_claims_jobs_worker.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py -q
```

Expected: PASS, with only shared-fixture PostgreSQL skips.

- [ ] **Step 3: Run formatting, compile, and security checks**

Run:

```bash
source .venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py \
  tldw_Server_API/app/api/v1/schemas/claims_schemas.py
python -m compileall -q \
  tldw_Server_API/app/core/Claims_Extraction \
  tldw_Server_API/app/core/Jobs \
  tldw_Server_API/app/core/DB_Management/media_db \
  tldw_Server_API/app/api/v1/endpoints/claims.py \
  tldw_Server_API/app/api/v1/schemas/claims_schemas.py
python -m bandit -r \
  tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  -f json -o /tmp/bandit_claims_jobs_stage2a.json
git diff --check
```

Expected: all commands exit 0 and Bandit reports no new findings in touched code.

- [ ] **Step 4: Perform the boundary and privacy audit**

Run:

```bash
rg -n 'pause|resume|drain|cancel|quarantine|retry_count|lease' \
  tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py
rg -n 'filters|pagination|payload_json|payload_csv|workspace_id|db_path|event' \
  tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py
```

Expected: the first command finds no Claims-owned queue controls or lifecycle management. The second finds only rejection lists, validators, and test-safe names; the enqueued analytics payload remains exactly version, owner ID, and export ID.

- [ ] **Step 5: Review the complete diff against the specification**

Check every acceptance criterion in the approved specification against a passing test. Confirm:

- 200 fallback and 202 Jobs mode.
- Separate artifact and Job statuses.
- No 503 after accepted Jobs creation.
- Owner scope in artifact, event, Jobs, API, and DB routing.
- Fixed snapshot and deterministic ordering.
- Row, memory-page, and byte bounds.
- CSV formula and filename safety.
- Ready monotonicity and retry recovery.
- Active/archive hydration and exact orphan reconciliation.
- Terminal-aware retention.
- No review-metrics, cluster-rebuild, scheduler, or Claims queue-control changes.

- [ ] **Step 6: Update Backlog and commit verification fixes**

Record test counts, PostgreSQL fixture skips, Ruff/compile/Bandit results, touched files, and any known environment limitation in the implementation Backlog task created for execution. Commit only if verification required code or documentation fixes:

After inspecting `git status --short` and confirming every tracked modification belongs to Stage 2A, run:

```bash
git add -u
git commit -m "fix: address Claims export verification findings"
```

If no files changed, do not create an empty commit.

## Execution Notes

- Create a separate Backlog implementation task before editing source files; `TASK-12990` tracks this plan artifact only.
- Execute in an isolated worktree based on the latest `origin/dev`.
- Rebase before opening the PR, rerun the focused and security gates after the rebase, and address review comments by verifying each finding before editing.
- The AI-authored PR is not merge-ready until the human requester supplies the required `Change summary` explaining what changed and why these implementation choices were made.
- Roll out handlers and schema first, then workers, then the producer flag. Roll back by disabling the producer first and leaving workers running to drain accepted exports.

## Task 12 Review Hardening Follow-up (2026-08-11)

**Goal:** Close the validated migration-ordering, mutable-rendering, PostgreSQL
commit-ordering, and high-water index findings without changing schema version 24
or public API/Jobs contracts.

### Stage 1: Regression tests (RED)

- [x] Extend SQLite v22-to-current and current-v24 bootstrap tests to cover a
  missing monitoring-event table during migration 024 and a missing
  `snapshot_event_id` column on an already-current export table.
- [x] Add retry-after-delivery JSON determinism coverage and remove assertions
  that treat mutable `delivered_at` as export content.
- [x] Add PostgreSQL transaction-fake coverage proving shared lock-before-insert
  and exclusive lock-before-high-water ordering, plus an official-fixture
  concurrency test.
- [x] Add SQLite/PostgreSQL `(user_id, id)` index parity and SQLite query-plan
  coverage while retaining `(user_id, created_at, id)`.
- [x] Run only the new focused test nodes and record the expected failures before
  production edits.

### Stage 2: Minimal implementation (GREEN)

- [x] Remove monitoring-event DDL from SQLite migration 024 and let the
  post-migration Claims extension ensure both event indexes.
- [x] Introspect `claims_analytics_exports` in SQLite Claims-extension bootstrap
  and add nullable `snapshot_event_id` when missing.
- [x] Exclude `delivered_at` from JSON export event projections and document the
  immutable exported event fields.
- [x] For PostgreSQL only, wrap monitoring-event insert and owner high-water in
  Media DB transactions; acquire a shared transaction advisory lock before the
  insert and an exclusive transaction advisory lock before `MAX(id)`.
- [x] Add `(user_id, id)` to SQLite/PostgreSQL fresh and repair schema paths and
  the PostgreSQL v24 migration body.

### Stage 3: Focused verification and commit

- [x] Run Claims export, monitoring-event DB, SQLite schema/migration, PostgreSQL
  structure/migration, API, and worker suites. Accept only skips emitted by the
  official PostgreSQL fixture.
- [x] Run Ruff, compileall, Bandit on production touched scope, and
  `git diff --check`; inspect SQL placeholders, transaction boundaries, and
  API/Jobs privacy.
- [x] Stage only follow-up files and commit separately with
  `fix: harden Claims export snapshot fence`.

## Task 12 Bounded Rendering Follow-up (2026-08-11)

**Goal:** Enforce the serialized byte limit before payload pages or complete
outputs accumulate, and apply orphan grace after retention for pruned Jobs.

### Stage 1: Regression tests (RED)

- [x] Add JSON/CSV exact-boundary, single-oversized-event, and cumulative-row
  overflow tests.
- [x] Add SQLite/PostgreSQL metadata-page and owner-scoped bounded-payload tests.
- [x] Add failed-artifact cleanup cases between `max(retention, grace)` and
  `retention + grace`, at the exact sum, and after the sum.

### Stage 2: Minimal implementation (GREEN)

- [x] Apply payload-derived filters in parameterized database predicates,
  return only constant-size event metadata from keyset pages, and load one
  owner-scoped payload only when the database proves it fits the builder's
  current remaining-byte budget.
- [x] Incrementally serialize byte-counted JSON event and CSV row chunks without
  retaining decoded event lists or materializing an unchecked final payload.
- [x] Require retention plus orphan grace when an attached Job is proven absent.
- [x] Document environment precedence, fractional retention, producer/worker
  byte-limit parity, and the bounded rendering mechanism.

### Stage 3: Verification and review

- [x] Run rendering, cleanup, monitoring-event, API, worker, and database suites:
  final bounded rendering/cleanup/database run passed 284 tests with 13
  official PostgreSQL-unreachable fixture skips; API/worker integration passed
  60 tests.
- [x] Run Ruff and `git diff --check` on the exact touched scope.
- [x] Run compileall and Bandit on production touched scope; Bandit reported
  zero findings across 3,724 lines of production code.
- [x] Complete fresh specification and quality reviews. Specification re-review
  approved the corrected batch. Three bounded quality-review subagent attempts
  did not return; the local code-review checklist found no additional issue.
- [x] Commit the corrected batch separately as `d88a27a4bb`.

Review correction: both reviewers validated that raw payload size was being
applied before provider/model filtering and pagination. RED: six JSON/CSV cases
for oversized nonmatching rows, oversized off-page rows, and whitespace-heavy
selected JSON. GREEN: provider/model filters are applied by the database;
constant-size scan rows contain no payload-derived values; only selected rows
use the owner-scoped bounded payload read. PostgreSQL uses schema-installed JSON
helpers compatible with the documented PostgreSQL 13+ baseline instead of
newer `IS JSON` or `json_serialize` syntax (7 focused rendering tests passed).

Second review correction: selected payload reads now receive the builder's
decreasing remaining-byte budget instead of the original export limit. SQLite
and PostgreSQL provider/model predicates now share an explicit string-only JSON
contract so booleans, numbers, nulls, arrays, and objects cannot produce
backend-specific matches. RED: three focused tests; GREEN: three focused tests.

## Task 12 Jobs Acceptance And Identity Follow-up (2026-08-11)

**Goal:** Preserve durable Jobs admission across ambiguous enqueue exceptions
and prevent reused numeric Job IDs from projecting or authorizing unrelated
lifecycle state.

### Stage 1: Regression tests (RED)

- [x] Cover enqueue exception outcomes for exact durable admission, proven
  absence, Jobs lookup outage, and malformed/mismatched lookup results.
- [x] Cover row-specific hydration when an unrelated active Job shadows the
  exact archived export Job under the same numeric ID.
- [x] Cover cleanup fallback for exact terminal, exact active, proven absent,
  and uncertain Jobs states after retention plus orphan grace.

### Stage 2: Minimal implementation (GREEN)

- [x] On enqueue exceptions, perform one exact owner/domain/type/batch read;
  compensate and return 503 only when that read proves absence. Preserve 202
  for exact durable admission or uncertain Jobs state and never retry enqueue.
- [x] Validate exact export Job identity after the bounded numeric-ID batch
  read, use one archived-aware exact fallback per mismatch, and key hydrated
  status by export identity.
- [x] Apply the same exact identity and conservative fallback contract to
  failed-artifact cleanup without adding Claims lifecycle controls.

### Stage 3: Verification and review

- [x] Run focused API, hydration, reconciliation, cleanup, Jobs regression,
  Ruff, compileall, Bandit, and diff checks.
- [x] Complete fresh specification and quality reviews. Specification review
  approved. Quality review found one terminal cleanup gap for enqueue-failed
  artifacts without an attached ID; focused RED was 4 failed/1 passed and GREEN
  was 5 passed, the full cleanup suite passed 43 tests, and re-review approved.
- [x] Commit separately as `e1b03907c0`.

Initial TDD evidence: enqueue regressions failed 5 tests before the fix;
hydration/cleanup regressions failed 7 tests with one incidental pass. GREEN
focused runs passed 5 enqueue tests and 8 identity/cleanup tests. The full API
and cleanup suites passed 100 tests, and the broader Claims producer, handler,
worker, and Jobs batch-read matrix passed 396 tests with six official
PostgreSQL-unreachable fixture skips.

After the quality-review correction, the final API/cleanup run passed 105
tests. Ruff, compileall, Bandit (zero findings), and `git diff --check` passed
on the final production and test scope.

## Task 12 Final Review Corrections (2026-08-11)

**Goal:** Close the independently validated canonical-Unicode sizing, blank
scalar-filter parity, CSV over-scan, synchronous-failure retention, attached
terminal-Job reconciliation, and PostgreSQL v24 migration-ordering findings.

### Stage 1: Verified regressions (RED)

- [x] Add escaped-Unicode exact-boundary tests against real SQLite rendering and
  the official PostgreSQL fixture.
- [x] Add blank scalar-filter parity and JSON-versus-CSV scan-count tests.
- [x] Add failed artifacts without Jobs for non-enqueue error codes, attached
  active/terminal Job reconciliation, and Jobs-owned terminal classification.
- [x] Add a partial-v23 PostgreSQL migration test where the monitoring-event
  extension table is absent.

### Stage 2: Minimal corrections (GREEN)

- [x] Canonicalize bounded payload JSON before exact UTF-8 sizing while keeping
  source reads within a six-times constant-factor cap.
- [x] Apply present empty scalar filters and stop CSV after its selected window;
  retain JSON's exact-total scan.
- [x] Reconcile attached artifacts through exact read-only Jobs projections and
  move the terminal-status contract into Jobs without adding Claims controls.
- [x] Apply retention plus grace and exact absence checks to all failed artifacts
  without Job IDs.
- [x] Defer PostgreSQL monitoring-event index creation to the extension repair
  path that owns creation of the optional table.

### Stage 3: Final verification and independent review

- [ ] Run the complete Stage 2A, Stage 1/lifecycle, migration/schema, Ruff,
  compileall, Bandit, boundary, and diff gates.
- [ ] Complete fresh specification and quality reviews and address every
  validated finding.
- [ ] Update TASK-12993 with final evidence and commit the aligned records.

Quality review correction: a fresh audit found and validated four additional
issues. A full page of unchanged active artifacts could starve later terminal
artifacts; a valid large exponent could round to infinity and emit invalid JSON;
the database normalized unbounded raw JSON before applying its compact-byte
cap; and persisted cancelled/quarantined codes were omitted from the safe
download allowlist. RED was 6 failed/3 passed. GREEN uses rotating independently
bounded reconciliation pages, a raw pre-parse cap, `allow_nan=False`, and the
existing stable terminal codes; the focused run passed 9 tests and the affected
matrix passed 197 tests with 4 official PostgreSQL skips before the one expected
query-shape assertion was updated. The complete cleanup suite then passed 52
tests.

Specification re-review correction: metadata pages still returned unrestricted
`event_type` and `severity` text, and product documentation omitted the raw
preparse cap and full failed-artifact grace rule. RED was 2 focused failures.
GREEN keeps metadata pages fixed-width, hydrates all selected variable text in
one owner-scoped remaining-budget query, and aligns product/design guidance.

Final quality-audit correction: four additional findings were independently
validated. Cleanup could delete a failed artifact after observing a terminal Job
that Jobs could subsequently retry; provider/model predicates could parse an
unrestricted raw payload; maintenance scans lacked indexes matching their
filters and ordering; and owner validation accepted digit strings beyond the
routable signed-64-bit range. Each correction followed RED/GREEN TDD:

- Cleanup now preserves failed artifacts while any exact active or archived Job
  exists and deletes only after retention plus grace and proven exact absence;
  the full cleanup suite passed 56 tests.
- Provider/model scans measure raw payload bytes before JSON evaluation, return
  only a fixed oversized marker, and fail closed with
  `claims_export_too_large`; renderer/database coverage passed 263 tests with 4
  official PostgreSQL-unavailable skips.
- Fresh, migration, and current-v24 repair paths now install
  `(user_id, status, export_id)` and
  `(user_id, status, updated_at, export_id)` indexes; query-plan and schema
  coverage passed 73 tests with 3 official PostgreSQL-unavailable skips.
- Contracts, handlers, rendering, and API routing now accept owner IDs only in
  the canonical range 1 through 9,223,372,036,854,775,807; affected suites
  passed 377 tests.

Fresh quality re-review then validated two final resource-bound gaps. Analytics
filter and timestamp strings could be persisted without character ceilings,
and very large Python integer owners could raise the interpreter's digit-limit
`ValueError` before stable Claims error translation. RED reproduced all 16
cases. GREEN centralizes schema/core string limits, range-checks integer owners
before conversion in contracts, handlers, and API routing, and passes the four
affected suites with 393 tests.
