# Audit Hardening And Sharing Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed Audit reliability defects, harden shutdown ownership, and migrate Sharing audit persistence onto unified audit without breaking the existing Sharing admin audit API.

**Architecture:** Keep the Audit core fixes local to `UnifiedAuditService` and the DI lifecycle layer. Introduce a dedicated Sharing audit boundary that always writes to shared unified-audit storage, preserves owner-vs-actor semantics, allocates stable compatibility ids transactionally, and projects unified rows back into the existing `/api/v1/sharing/admin/audit` contract. Migrate historical `share_audit_log` rows with an idempotent backfill instead of dual-running two audit stores.

**Tech Stack:** Python, FastAPI, aiosqlite, SQLite schema migration helpers, pytest, loguru, Bandit

---

## File Map

### Core Audit

- Modify: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
  - Fix populated legacy-table migration row copying.
  - Keep chain-hash advancement commit-bound.
  - Preserve buffered count/export visibility behavior.
- Modify: `tldw_Server_API/tests/Audit/test_unified_audit_service.py`
  - Add red tests for populated legacy migration and keep chain-hash regressions covered.

### Audit Lifecycle

- Modify: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
  - Track scheduled stop work and drain it safely.
- Modify: `tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py`
  - Add regression coverage for the running-owner-loop stop path.
- Modify: `tldw_Server_API/tests/Audit/test_audit_db_deps.py`
  - Cover stop-drain bookkeeping if a focused unit seam is added.

### Adapter Ownership Cleanup

- Modify: `tldw_Server_API/app/core/Embeddings/audit_adapter.py`
  - Remove global async shutdown ownership from `atexit`.
  - Expose local cleanup only.
- Modify: `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
  - Align public cleanup shape with Embeddings adapter.
- Modify: `tldw_Server_API/app/main.py`
  - Keep app shutdown as single global audit-service owner and explicitly clean adapter-local loops.
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py`

### Sharing Audit Unification

- Create: `tldw_Server_API/app/core/Sharing/unified_share_audit.py`
  - Dedicated Sharing audit resolver/factory.
  - Shared unified-audit service resolution.
  - Compatibility-id allocation helpers.
  - Event payload mapping with separate owner-vs-actor semantics.
- Modify: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
  - Accept custom namespaced string event types for Sharing.
  - Add an explicit tenant override seam for shared-mode writes used by the Sharing boundary.
- Modify: `tldw_Server_API/app/core/Sharing/share_audit_service.py`
  - Route writes and reads through the new unified boundary.
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
  - Keep `/sharing/admin/audit` stable while switching its backing store.
- Modify: `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
  - Only if field docs need clarification; avoid schema shape changes.
- Modify: `tldw_Server_API/tests/Sharing/test_share_audit_service.py`
- Modify: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
- Create: `tldw_Server_API/tests/Sharing/test_unified_share_audit.py`

### Historical Sharing Migration

- Create: `tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py`
  - Idempotent backfill from `share_audit_log` into unified audit.
  - Sequence-floor initialization for compatibility ids.
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py`
  - Add a focused read helper for legacy share audit rows if direct repo access simplifies migration.
- Create: `tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py`

### Verification

- Modify: `Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md`
  - Document Sharing audit unification and bounded generic-audit visibility decision.

### Notes

- Do not remove `share_audit_log` schema or legacy repo methods in this tranche.
- Do not expand generic `/api/v1/audit/*` aggregation to pull Sharing events under `per_user` mode in this tranche.

### Task 1: Lock The Core Audit Regressions

**Files:**
- Modify: `tldw_Server_API/tests/Audit/test_unified_audit_service.py`
- Modify: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- Test: `tldw_Server_API/tests/Audit/test_unified_audit_service.py`

- [ ] **Step 1: Write the failing regression tests**

```python
@pytest.mark.asyncio
async def test_legacy_migration_populated_rows_preserves_chain_hash_binding(tmp_path):
    db_path = tmp_path / "legacy_populated_audit.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO audit_events (event_id, timestamp, event_type, severity, user_id, action, outcome, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy-1", "2026-01-01T00:00:00+00:00", "data.read", "info", "17", "read", "success", "{}"),
        )
        conn.commit()

    service = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=AuditContext(user_id="17"),
            action="post_migration_read",
            resource_id="doc-1",
        )
        await service.flush(raise_on_failure=True)
    finally:
        await service.stop()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action = ?",
            ("post_migration_read",),
        ).fetchone()
    assert row[0] == 1
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audit/test_unified_audit_service.py -k legacy_migration_populated_rows_preserves_chain_hash_binding`
Expected: FAIL with a `ProgrammingError` about missing `:chain_hash` or another migration-time binding failure.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Audit/unified_audit_service.py
record = {
    "event_id": str(data.get("event_id") or uuid4()),
    "timestamp": _coerce_timestamp(data.get("timestamp")),
    "category": _infer_category(event_type_val),
    "event_type": str(event_type_val),
    "severity": _normalize_severity(data.get("severity")),
    "context_request_id": data.get("context_request_id"),
    "context_correlation_id": data.get("context_correlation_id"),
    "context_session_id": data.get("context_session_id") or data.get("session_id"),
    "context_user_id": context_user_id,
    "context_api_key_hash": data.get("context_api_key_hash"),
    "context_ip_address": data.get("context_ip_address") or data.get("ip_address"),
    "context_user_agent": data.get("context_user_agent") or data.get("user_agent"),
    "context_endpoint": data.get("context_endpoint") or data.get("endpoint"),
    "context_method": data.get("context_method") or data.get("method"),
    "resource_type": data.get("resource_type"),
    "resource_id": data.get("resource_id"),
    "action": data.get("action"),
    "result": data.get("result") or data.get("outcome") or "success",
    "error_message": data.get("error_message") or data.get("details"),
    "duration_ms": data.get("duration_ms"),
    "tokens_used": data.get("tokens_used"),
    "estimated_cost": data.get("estimated_cost"),
    "result_count": data.get("result_count"),
    "risk_score": data.get("risk_score") or 0,
    "pii_detected": bool(data.get("pii_detected") or False),
    "compliance_flags": _json_text(data.get("compliance_flags"), default="[]"),
    "metadata": _json_text(data.get("metadata"), default="{}"),
    "chain_hash": str(data.get("chain_hash") or ""),
}
```

- [ ] **Step 4: Run the focused core audit tests**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audit/test_unified_audit_service.py -k "legacy_migration or failed_flush or replay_fallback_queue or count_events_flushes_buffered_events or export_events_flushes_buffered_events"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Audit/unified_audit_service.py tldw_Server_API/tests/Audit/test_unified_audit_service.py
git commit -m "fix: harden audit migration and chain regressions"
```

### Task 2: Drain Scheduled Audit Stops Safely

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- Modify: `tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py`
- Modify: `tldw_Server_API/tests/Audit/test_audit_db_deps.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py`

- [ ] **Step 1: Write the failing lifecycle regression test**

```python
@pytest.mark.asyncio
async def test_schedule_stop_tracks_owner_loop_future_until_drained(monkeypatch, tmp_path):
    service = UnifiedAuditService(
        db_path=str(tmp_path / "audit.db"),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)

    owner_loop = asyncio.get_running_loop()
    service._owner_loop = owner_loop

    drained = asyncio.Event()
    original_stop = service.stop

    async def _stop():
        try:
            await original_stop()
        finally:
            drained.set()

    monkeypatch.setattr(service, "stop", _stop)

    audit_deps._schedule_service_stop(123, service, "test")
    await audit_deps._drain_scheduled_audit_stops(timeout=1.0)
    assert drained.is_set()
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py -k tracks_owner_loop_future_until_drained`
Expected: FAIL because `_drain_scheduled_audit_stops` does not exist or pending stop work is not tracked.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py
_scheduled_stop_futures: set[concurrent.futures.Future[Any]] = set()
_scheduled_stop_lock = threading.Lock()


def _track_stop_future(future: concurrent.futures.Future[Any]) -> None:
    with _scheduled_stop_lock:
        _scheduled_stop_futures.add(future)

    def _cleanup(done_future: concurrent.futures.Future[Any]) -> None:
        with _scheduled_stop_lock:
            _scheduled_stop_futures.discard(done_future)

    future.add_done_callback(_cleanup)


async def _drain_scheduled_audit_stops(timeout: float) -> None:
    with _scheduled_stop_lock:
        futures = list(_scheduled_stop_futures)
    if not futures:
        return
    await asyncio.wait_for(
        asyncio.gather(*(asyncio.wrap_future(fut) for fut in futures), return_exceptions=True),
        timeout=timeout,
    )

# inside _schedule_service_stop(...)
future = asyncio.run_coroutine_threadsafe(_stop(), owner_loop)
_track_stop_future(future)
```

- [ ] **Step 4: Run the lifecycle tests**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py tldw_Server_API/tests/Audit/test_audit_db_deps.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py tldw_Server_API/tests/Audit/test_audit_db_deps.py
git commit -m "fix: drain scheduled audit stop work"
```

### Task 3: Make Adapter Cleanup Local And Explicit

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/audit_adapter.py`
- Modify: `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`

- [ ] **Step 1: Write the failing adapter-cleanup regression tests**

```python
def test_embeddings_atexit_only_stops_local_loop(monkeypatch):
    calls = {"global": 0, "local": 0}

    async def _global_shutdown():
        calls["global"] += 1

    def _local_shutdown():
        calls["local"] += 1

    monkeypatch.setattr(emb_adapter, "shutdown_all_audit_services", _global_shutdown)
    monkeypatch.setattr(emb_adapter, "_stop_sync_loop", _local_shutdown)

    emb_adapter._shutdown_on_exit()

    assert calls == {"global": 0, "local": 1}
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py -k atexit_only_stops_local_loop`
Expected: FAIL because the current Embeddings `atexit` handler still attempts full async shutdown ownership.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Embeddings/audit_adapter.py

def shutdown_local_audit_adapter_loop() -> None:
    _stop_sync_loop()


def _shutdown_on_exit() -> None:
    with contextlib.suppress(Exception):
        shutdown_local_audit_adapter_loop()

# tldw_Server_API/app/core/Evaluations/audit_adapter.py

def shutdown_local_evaluations_audit_loop() -> None:
    _stop_sync_loop()

# tldw_Server_API/app/main.py
await shutdown_all_audit_services()
shutdown_local_audit_adapter_loop()
shutdown_local_evaluations_audit_loop()
```

- [ ] **Step 4: Run the adapter and shutdown-focused tests**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py tldw_Server_API/tests/Audit/test_audit_service_init_race.py`
Expected: PASS with no `Task was destroyed but it is pending!` warning in the Embeddings adapter suite.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Embeddings/audit_adapter.py tldw_Server_API/app/core/Evaluations/audit_adapter.py tldw_Server_API/app/main.py tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py
git commit -m "fix: align audit adapter shutdown ownership"
```

### Task 4: Introduce The Dedicated Sharing Audit Boundary

**Files:**
- Modify: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- Create: `tldw_Server_API/app/core/Sharing/unified_share_audit.py`
- Modify: `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- Modify: `tldw_Server_API/tests/Sharing/test_share_audit_service.py`
- Create: `tldw_Server_API/tests/Sharing/test_unified_share_audit.py`
- Test: `tldw_Server_API/tests/Sharing/test_share_audit_service.py`

- [ ] **Step 1: Write the failing Sharing boundary tests**

```python
@pytest.mark.asyncio
async def test_share_audit_writer_keeps_owner_and_actor_distinct(tmp_path):
    writer = UnifiedShareAuditWriter(db_path=str(tmp_path / "audit_shared.db"))
    await writer.initialize()
    try:
        await writer.log_event(
            event_type="share.created",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=7,
            actor_user_id=11,
            share_id=42,
            metadata={"scope_type": "team"},
        )
        rows = await writer.query_events(owner_user_id=7)
    finally:
        await writer.stop()

    assert rows[0]["owner_user_id"] == 7
    assert rows[0]["actor_user_id"] == 11
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_share_audit_service.py tldw_Server_API/tests/Sharing/test_unified_share_audit.py`
Expected: FAIL because the dedicated writer/projection boundary does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Audit/unified_audit_service.py
@dataclass
class AuditEvent:
    event_type: AuditEventType | str = AuditEventType.SYSTEM_START


def _event_type_value(event_type: AuditEventType | str) -> str:
    return event_type.value if isinstance(event_type, AuditEventType) else str(event_type)


async def log_event(
    self,
    event_type: AuditEventType | str,
    context: AuditContext | None = None,
    *,
    category: AuditEventCategory | None = None,
    severity: AuditSeverity | None = None,
    tenant_user_id_override: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    metadata: dict[str, Any] | None = None,
    result: str = "success",
) -> str:
    if category is None:
        category = self._determine_category(event_type)
    if severity is None:
        severity = self._determine_severity(event_type, result)
    tenant_user_id = None
    if self._shared_mode:
        tenant_user_id = self._resolve_tenant_id_for_write(
            raw_tenant=tenant_user_id_override,
            context_user_id=(context.user_id if context else None),
            event_type=event_type,
            category=category,
        )
    ...

# tldw_Server_API/app/core/Sharing/unified_share_audit.py
class UnifiedShareAuditWriter:
    def __init__(self, db_path: str | None = None) -> None:
        self._service = UnifiedAuditService(
            db_path=db_path or str(DatabasePaths.get_shared_audit_db_path()),
            storage_mode="shared",
        )

    async def log_event(...):
        compatibility_id = await self._allocate_compatibility_id()
        ctx = AuditContext(
            user_id=str(actor_user_id) if actor_user_id is not None else None,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        await self._service.log_event(
            event_type=event_type,
            category=category,
            severity=severity,
            context=ctx,
            tenant_user_id_override=str(owner_user_id),
            resource_type=resource_type,
            resource_id=resource_id,
            action=event_type,
            metadata={
                **(metadata or {}),
                "owner_user_id": owner_user_id,
                "actor_user_id": actor_user_id,
                "share_id": share_id,
                "token_id": token_id,
                "compatibility_id": compatibility_id,
            },
        )
```

- [ ] **Step 4: Run the focused Sharing unit tests**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_share_audit_service.py tldw_Server_API/tests/Sharing/test_unified_share_audit.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Audit/unified_audit_service.py tldw_Server_API/app/core/Sharing/unified_share_audit.py tldw_Server_API/app/core/Sharing/share_audit_service.py tldw_Server_API/tests/Sharing/test_share_audit_service.py tldw_Server_API/tests/Sharing/test_unified_share_audit.py
git commit -m "feat: add unified sharing audit boundary"
```

### Task 5: Cut The Sharing Endpoint Over To Unified Audit

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
- Modify: `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- Test: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

- [ ] **Step 1: Write the failing compatibility endpoint tests**

```python
def test_admin_audit_log_returns_unified_backed_rows(client, mock_repo):
    client.post("/api/v1/sharing/workspaces/ws-1/share", json={
        "share_scope_type": "team",
        "share_scope_id": 10,
    })

    resp = client.get("/api/v1/sharing/admin/audit")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] >= 1
    assert isinstance(payload["events"][0]["id"], int)
    assert payload["events"][0]["event_type"].startswith("share.")
```

- [ ] **Step 2: Run the focused endpoint tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -k admin_audit_log_returns_unified_backed_rows`
Expected: FAIL because the endpoint is still backed by legacy repo audit rows.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/endpoints/sharing.py

def _get_audit_service():
    from tldw_Server_API.app.core.Sharing.share_audit_service import ShareAuditService
    return ShareAuditService()

@router.get("/admin/audit", ...)
async def admin_audit_log(...):
    audit = _get_audit_service()
    events = await audit.query(
        owner_user_id=owner_user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        limit=limit,
        offset=offset,
    )
    return AuditLogResponse(events=[AuditEventResponse(**event) for event in events], total=len(events))
```

- [ ] **Step 4: Run the Sharing endpoint suite**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/core/Sharing/share_audit_service.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py
git commit -m "feat: back sharing admin audit with unified audit"
```

### Task 6: Backfill Historical Sharing Audit Rows

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py`
- Create: `tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py`
- Test: `tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py`

- [ ] **Step 1: Write the failing migration tests**

```python
@pytest.mark.asyncio
async def test_share_audit_backfill_is_idempotent(tmp_path, repo):
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        actor_user_id=2,
        share_id=5,
        metadata={"scope_type": "team"},
    )

    report1 = await migrate_share_audit_log_to_unified_audit(repo=repo, shared_audit_db_path=tmp_path / "audit_shared.db")
    report2 = await migrate_share_audit_log_to_unified_audit(repo=repo, shared_audit_db_path=tmp_path / "audit_shared.db")

    assert report1.inserted == 1
    assert report2.inserted == 0
```

- [ ] **Step 2: Run the focused migration test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py`
Expected: FAIL because the migration utility does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py
@dataclass
class ShareAuditMigrationReport:
    inserted: int
    skipped: int
    max_legacy_id: int


async def migrate_share_audit_log_to_unified_audit(*, repo: SharedWorkspaceRepo, shared_audit_db_path: Path) -> ShareAuditMigrationReport:
    rows = await repo.list_legacy_share_audit_rows(limit=10_000, offset=0)
    writer = UnifiedShareAuditWriter(db_path=str(shared_audit_db_path))
    await writer.initialize()
    try:
        inserted = 0
        skipped = 0
        max_legacy_id = 0
        for row in rows:
            max_legacy_id = max(max_legacy_id, int(row["id"]))
            created = await writer.migrate_legacy_row(row)
            if created:
                inserted += 1
            else:
                skipped += 1
        await writer.bump_compatibility_floor(max_legacy_id)
        return ShareAuditMigrationReport(inserted=inserted, skipped=skipped, max_legacy_id=max_legacy_id)
    finally:
        await writer.stop()
```

- [ ] **Step 4: Run the migration and Sharing suites**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py tldw_Server_API/tests/Sharing/test_share_audit_service.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sharing/share_audit_unified_migration.py tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py
git commit -m "feat: migrate legacy sharing audit into unified audit"
```

### Task 7: Final Verification, Security Scan, And Docs

**Files:**
- Modify: `Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md`
- Modify: `Docs/superpowers/specs/2026-04-07-audit-hardening-and-sharing-unification-design.md` only if implementation forces an explicit spec correction
- Test: touched Audit, Sharing, Embeddings, and Evaluations suites

- [ ] **Step 1: Update the audit guide for the new Sharing boundary**

```markdown
## Sharing Audit

Sharing audit persistence now uses unified audit as the source of truth.
The `/api/v1/sharing/admin/audit` endpoint remains the stable operator-facing
compatibility surface. Under global `per_user` audit mode, Sharing audit still
uses the shared unified-audit path because the endpoint is cross-user by design.
Owner identity is stored in `tenant_user_id`; actor identity remains in
`context_user_id` and compatibility metadata.
```

- [ ] **Step 2: Run the full focused verification suite**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audit/test_unified_audit_service.py tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py tldw_Server_API/tests/Audit/test_audit_db_deps.py tldw_Server_API/tests/Sharing/test_share_audit_service.py tldw_Server_API/tests/Sharing/test_unified_share_audit.py tldw_Server_API/tests/Sharing/test_share_audit_unified_migration.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py`
Expected: PASS

- [ ] **Step 3: Run Bandit on the touched scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Audit tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py tldw_Server_API/app/core/Embeddings/audit_adapter.py tldw_Server_API/app/core/Evaluations/audit_adapter.py tldw_Server_API/app/core/Sharing tldw_Server_API/app/api/v1/endpoints/sharing.py -f json -o /tmp/bandit_audit_sharing_unification.json`
Expected: JSON report written to `/tmp/bandit_audit_sharing_unification.json` with no new high-signal findings in changed code.

- [ ] **Step 4: Review git diff and commit the wrap-up docs/test updates**

```bash
git add Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md
git commit -m "docs: document unified sharing audit behavior"
```

## Self-Review

### Spec Coverage

- Core audit migration failure: covered by Task 1.
- Chain-head commit-bound behavior and buffered count/export visibility: covered by Task 1.
- Eviction shutdown drainability: covered by Task 2.
- Adapter-local cleanup and app-owned global shutdown: covered by Task 3.
- Dedicated Sharing audit boundary: covered by Task 4.
- Stable Sharing compatibility id and endpoint preservation: covered by Tasks 4 and 5.
- Historical `share_audit_log` migration and sequence-floor handling: covered by Task 6.
- Bounded generic-audit visibility decision: preserved by Task 5 and documented in Task 7.
- Security verification and touched-scope docs: covered by Task 7.

### Placeholder Scan

- No `TODO`, `TBD`, or cross-task “same as above” placeholders remain.
- Each code-touching task includes the concrete files, test commands, and a minimal target code sketch.

### Type Consistency

- The plan consistently uses `UnifiedShareAuditWriter`, `migrate_share_audit_log_to_unified_audit`, `compatibility_id`, `owner_user_id`, `actor_user_id`, and `tenant_user_id`.
- The compatibility endpoint continues returning `AuditLogResponse` and `AuditEventResponse`.
