# API Boundary Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved API boundary remediation series without changing external HTTP contracts.

**Architecture:** Keep public routes as thin adapters and move durable behavior behind the owning router, Media DB, Jobs, document workspace repository, and Prototype Workspace service APIs. Each stage is independently testable and should be committed separately. Worker lifecycle state consolidation is explicitly out of scope.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL-aware repository helpers, pytest/httpx/TestClient, Loguru, Bandit.

---

## Source Spec

- Design: `Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md`
- Backlog design task: `TASK-500`
- Backlog plan task: `TASK-501`

## File Structure

Planned files by responsibility:

- `tldw_Server_API/app/api/v1/router_groups/selection.py`
  - New small helper for selecting canonical `RouterSpec` objects by `name` and applying explicit overrides.
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
  - Derive minimal-test specs from `iter_core_router_specs`, `iter_content_router_specs`, and `iter_admin_router_specs` where a production spec exists.
- `tldw_Server_API/tests/Services/test_router_groups_contract.py`
  - Add route metadata preservation tests and adjust minimal router expectations.
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py`
  - New user-facing media item update operation that owns Media update transaction details.
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
  - Bind the new Media DB update helper to `MediaDatabase`.
- `tldw_Server_API/app/api/v1/endpoints/media/item.py`
  - Replace endpoint-owned update SQL/private helper calls with the public Media DB operation.
- `tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py`
  - New focused Media DB unit tests for update invariants.
- `tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py`
  - Update endpoint tests to assert delegation and preserved error mapping.
- `tldw_Server_API/app/core/Jobs/manager.py`
  - Expand `JobManager.list_job_events_after` filters and normalized event shape.
- `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py`
  - Migrate event list/SSE reads to `JobManager.list_job_events_after`.
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
  - Migrate ingest events stream reads to `JobManager.list_job_events_after`.
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`
  - Migrate audio jobs event reads only if the touched path currently reads `job_events` directly.
- `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py`
  - Migrate event-related reads only if they are direct `job_events` reads; leave unrelated queue/status aggregation for a later task.
- `tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py`
  - Add JobManager filter and event shape coverage.
- `tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py`
  - Preserve Jobs admin SSE behavior.
- `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py`
  - Preserve media ingest SSE behavior.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/document_workspace_schema.py`
  - New schema helper for reading progress, annotations, and parsed-reference cache tables.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
  - Call the document workspace schema helper during SQLite post-core bootstrap.
- `tldw_Server_API/app/core/DB_Management/media_db/repositories/document_workspace_repository.py`
  - New repository for reading progress, annotations, and parsed reference cache.
- `tldw_Server_API/app/core/DB_Management/media_db/repositories/__init__.py`
  - Export the document workspace repository if local package convention requires it.
- `tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py`
  - Delegate DB reads/writes to repository methods and remove lazy DDL.
- `tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py`
  - Delegate annotation storage to repository methods and remove lazy DDL.
- `tldw_Server_API/app/api/v1/endpoints/media/document_references.py`
  - Delegate parsed-reference cache storage to repository methods and remove lazy DDL.
- `tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py`
  - New repository/schema tests.
- `tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py`
  - Preserve endpoint error sanitization behavior.
- `tldw_Server_API/tests/Media/test_document_references.py`
  - Preserve parsed reference cache behavior.
- `tldw_Server_API/app/core/Prototype_Workspaces/service.py`
  - Add public `review_promotion_request(...)` service method.
- `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
  - Replace private service/repo access with the public review method.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py`
  - Add service-level reject/authorization/state transition tests.
- `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`
  - Add endpoint reject/non-promoter regressions and preserve approval behavior.

## Stage 1: Router Spec Metadata Single Source

**Status:** Not Started

### Task 1: Add RouterSpec Selection Helpers

**Files:**
- Create: `tldw_Server_API/app/api/v1/router_groups/selection.py`
- Modify: `tldw_Server_API/tests/Services/test_router_groups_contract.py`

- [ ] **Step 1: Write failing helper tests**

Add tests near existing router group contract tests:

```python
def test_router_spec_selection_preserves_policy_metadata() -> None:
    from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
    from tldw_Server_API.app.api.v1.router_groups.selection import select_router_specs_by_name

    selected = select_router_specs_by_name(iter_core_router_specs(), ("health", "auth"))

    by_name = {spec.name: spec for spec in selected}
    assert by_name["health"].route_key == "health"
    assert by_name["health"].default_stable is True
    assert by_name["auth"].route_key == "auth"
    assert by_name["auth"].tags == ("authentication",)
```

Add an override-specific test:

```python
def test_router_spec_selection_allows_explicit_overrides() -> None:
    from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
    from tldw_Server_API.app.api.v1.router_groups.selection import RouterSpecOverride, select_router_specs_by_name

    selected = select_router_specs_by_name(
        iter_core_router_specs(),
        ("auth",),
        overrides={"auth": RouterSpecOverride(tags=("minimal-auth",))},
    )

    assert selected[0].route_key == "auth"
    assert selected[0].tags == ("minimal-auth",)
```

- [ ] **Step 2: Run helper tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py::test_router_spec_selection_preserves_policy_metadata tldw_Server_API/tests/Services/test_router_groups_contract.py::test_router_spec_selection_allows_explicit_overrides -q
```

Expected: FAIL because `router_groups.selection` does not exist.

- [ ] **Step 3: Implement helper module**

Implement the helper with a frozen override dataclass and `dataclasses.replace`:

```python
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


@dataclass(frozen=True)
class RouterSpecOverride:
    prefix: str | None = None
    tags: tuple[str, ...] | None = None
    route_key: str | None = None
    default_stable: bool | None = None
    name: str | None = None
    skip_context: str | None = None


def select_router_specs_by_name(
    specs: Iterable[RouterSpec],
    names: Sequence[str],
    *,
    overrides: dict[str, RouterSpecOverride] | None = None,
) -> list[RouterSpec]:
    by_name = {spec.name: spec for spec in specs}
    selected: list[RouterSpec] = []
    overrides = overrides or {}
    for name in names:
        source = by_name[name]
        override = overrides.get(name)
        if override is None:
            selected.append(source)
            continue
        selected.append(
            replace(
                source,
                **{
                    key: value
                    for key, value in {
                        "prefix": override.prefix,
                        "tags": override.tags,
                        "route_key": override.route_key,
                        "default_stable": override.default_stable,
                        "name": override.name,
                        "skip_context": override.skip_context,
                    }.items()
                    if value is not None
                },
            )
        )
    return selected
```

- [ ] **Step 4: Run helper tests to verify GREEN**

Run the same focused pytest command. Expected: PASS.

### Task 2: Refactor Minimal Always-Included Router Specs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/tests/Services/test_router_groups_contract.py`

- [ ] **Step 1: Write failing minimal metadata tests**

Add tests that prove the current drift:

```python
def test_minimal_test_router_specs_preserve_production_route_keys() -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_test_router_specs

    specs = list(iter_minimal_test_router_specs())
    by_name = {spec.name: spec for spec in specs}

    assert by_name["health"].route_key == "health"
    assert by_name["auth"].route_key == "auth"
    assert by_name["research"].route_key == "research"
    assert by_name["paper_search"].route_key == "paper-search"
    assert by_name["workspaces"].route_key == "workspaces"
```

Add a route policy test by monkeypatching `route_enabled` and registering only one minimal spec:

```python
def test_minimal_router_specs_participate_in_route_policy(monkeypatch) -> None:
    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1 import router_registry
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_test_router_specs

    calls: list[tuple[str, bool]] = []

    def deny_health(route_key: str, *, default_stable: bool = True) -> bool:
        calls.append((route_key, default_stable))
        return route_key != "health"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        deny_health,
        raising=True,
    )
    health_spec = next(spec for spec in iter_minimal_test_router_specs() if spec.name == "health")

    assert router_registry.register_router_specs(FastAPI(), (health_spec,)) == 0
    assert calls == [("health", True)]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py::test_minimal_test_router_specs_preserve_production_route_keys tldw_Server_API/tests/Services/test_router_groups_contract.py::test_minimal_router_specs_participate_in_route_policy -q
```

Expected: FAIL because current minimal specs omit several `route_key` values.

- [ ] **Step 3: Replace hardcoded production duplicates**

In `minimal.py`, import:

```python
from tldw_Server_API.app.api.v1.router_groups.admin import iter_admin_router_specs
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_groups.selection import select_router_specs_by_name
```

Build `iter_minimal_test_router_specs()` from canonical specs:

```python
minimal_names = (
    "health",
    "auth",
    "research",
    "research_runs",
    "paper_search",
    "chat",
    "chat_loop",
    "conversations_alias",
    "characters",
    "character_memory",
    "character_chat_sessions",
    "character_messages",
    "workspace_migrations",
    "workspaces",
)
canonical_specs = (
    *iter_core_router_specs(),
    *iter_content_router_specs(),
    *iter_admin_router_specs(),
)
specs.extend(select_router_specs_by_name(canonical_specs, minimal_names))
```

Keep minimal-only specs local and explicit. Do not refactor optional minimal routes in the same step unless tests show they are direct production duplicates with missing metadata.

- [ ] **Step 4: Update existing contract expectations**

Adjust any assertions in `test_router_groups_contract.py` that expected missing `route_key` values from minimal specs. The new expectation is production parity unless the test documents an intentional minimal-only override.

- [ ] **Step 5: Run focused router tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Router smoke check**

Run:

```bash
rg -n "route_key == \"\"" tldw_Server_API/tests/Services/test_router_groups_contract.py
rg -n "ImportedRouterSpec\\(" tldw_Server_API/app/api/v1/router_groups/minimal.py
```

Expected: any remaining empty route-key expectations or local imported specs are intentional and documented in nearby tests/comments.

- [ ] **Step 7: Commit Stage 1**

```bash
git add tldw_Server_API/app/api/v1/router_groups/selection.py tldw_Server_API/app/api/v1/router_groups/minimal.py tldw_Server_API/tests/Services/test_router_groups_contract.py
git commit -m "refactor: derive minimal router metadata from canonical specs"
```

## Stage 2: Media DB Update Ownership

**Status:** Not Started

### Task 3: Add Media DB User-Facing Update Operation

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Create: `tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py`

- [ ] **Step 1: Write failing Media DB unit tests**

Add tests for these cases:

- missing media raises `InputError`
- optimistic rowcount zero raises `ConflictError`
- metadata-only update increments `version`, logs sync, updates FTS when title changes
- changed content updates `content`, `content_hash`, `chunking_status='pending'`, `vector_processing=0`, creates a document version, logs sync, refreshes FTS, and returns effect metadata
- identical non-null content still creates a document version but does not reset content/vector flags

Use `SimpleNamespace` fakes similar to `tldw_Server_API/tests/DB_Management/test_media_db_synced_document_update_ops.py`.

Expected helper contract:

```python
result = media_item_update_ops.apply_media_item_update(
    db,
    media_id=9,
    fields={"title": "New title", "content": "new body"},
    prompt="prompt",
    analysis_content="analysis",
)

assert result["media_id"] == 9
assert result["new_media_version"] == 2
assert result["content_changed"] is True
assert result["document_version_uuid"] == "dv-uuid"
assert result["invalidate_rag"] is True
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py -q
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement `apply_media_item_update`**

Follow the transaction structure in `synced_document_update_ops.apply_synced_document_content_update`, but support user-facing partial fields:

```python
def apply_media_item_update(
    self: Any,
    *,
    media_id: int,
    fields: dict[str, Any],
    prompt: str | None = None,
    analysis_content: str | None = None,
) -> dict[str, Any]:
    """Apply a user-facing media item update and return side-effect metadata."""
```

Required behavior:

- reject empty `fields` with `InputError`; no-op remains endpoint-owned because it returns current details
- fetch `id`, `uuid`, `title`, `content`, `content_hash`, `version` where `deleted = 0 AND is_trash = 0`
- use optimistic `WHERE id = ? AND version = ?`
- for changed content set `content`, `content_hash`, `chunking_status = 'pending'`, `vector_processing = 0`
- for identical non-null content do not update content/hash but do call `create_document_version`
- call `_update_fts_media` when title or actual content changes
- call `_log_sync_event` with the updated Media row and document version metadata
- after transaction, mark collection highlights stale for DB-local hooks and return explicit invalidation flags for endpoint/user-scoped work

- [ ] **Step 4: Bind helper to `MediaDatabase`**

In `media_database_impl.py`, import the function and bind:

```python
from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_item_update_ops import (
    apply_media_item_update,
)

MediaDatabase.apply_media_item_update = apply_media_item_update
```

Add a rebind test, mirroring existing rebind tests:

```python
def test_apply_media_item_update_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_item_update_ops import apply_media_item_update

    assert MediaDatabase.apply_media_item_update is apply_media_item_update
```

- [ ] **Step 5: Run Media DB unit tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/DB_Management/test_media_db_synced_document_update_ops.py -q
```

Expected: PASS.

### Task 4: Thin `update_media_item` Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/item.py`
- Modify: `tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py`
- Modify if needed: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`

- [ ] **Step 1: Write failing endpoint delegation test**

In `test_media_item_endpoint_error_mapping.py`, extend the fake DB with `apply_media_item_update` and assert endpoint delegation:

```python
class _DelegatingMediaUpdateDb:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_media_item_update(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"media_id": kwargs["media_id"], "invalidate_rag": True}
```

Monkeypatch `get_full_media_details_rich` to return a valid detail payload and `invalidate_rag_caches` to record calls. Assert:

- `apply_media_item_update` receives `fields=payload.model_dump(exclude_unset=True, exclude={"prompt", "analysis"})`
- `prompt` and `analysis_content` are passed separately
- private DB helpers are not invoked by the endpoint
- RAG invalidation happens only from returned effect metadata

- [ ] **Step 2: Run endpoint test to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py::test_update_media_item_delegates_to_media_db_update_operation -q
```

Expected: FAIL because the endpoint still owns the transaction and private helper calls.

- [ ] **Step 3: Replace endpoint transaction with helper call**

Keep the no-field branch as-is. For non-empty payloads:

```python
fields = payload.model_dump(exclude_unset=True, exclude={"prompt", "analysis"})
effects = db.apply_media_item_update(
    media_id=media_id,
    fields=fields,
    prompt=payload.prompt,
    analysis_content=payload.analysis,
)
if effects.get("invalidate_rag", True):
    invalidate_rag_caches(current_user, media_id=media_id)
details = get_full_media_details_rich(...)
return MediaDetailResponse(**details)
```

Endpoint must continue mapping `ConflictError`, `InputError`, and `DatabaseError` through `map_db_error_to_http`.

- [ ] **Step 4: Run focused media tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py::TestMediaListDetailEndpoints::test_update_media_item_title tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py::TestMediaListDetailEndpoints::test_update_media_item_nonexistent -q
```

Expected: PASS.

- [ ] **Step 5: Smoke check endpoint no longer owns private persistence calls**

Run:

```bash
rg -n "_update_fts_media|_log_sync_event|UPDATE Media SET|CREATE TABLE" tldw_Server_API/app/api/v1/endpoints/media/item.py
```

Expected: no matches in `update_media_item`.

- [ ] **Step 6: Commit Stage 2**

```bash
git add tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py tldw_Server_API/app/api/v1/endpoints/media/item.py tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py
git commit -m "refactor: move media item updates into Media DB"
```

## Stage 3: Jobs Event Query Ownership

**Status:** Not Started

### Task 5: Expand `JobManager.list_job_events_after`

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py`
- Modify if Postgres fixtures are available: `tldw_Server_API/tests/Jobs/test_jobs_events_outbox_postgres.py`

- [ ] **Step 1: Write failing JobManager filter tests**

Add SQLite tests that create jobs in different domains/queues/job types/owners and assert:

```python
events = jm.list_job_events_after(
    after_id=0,
    limit=10,
    domain="media_ingest",
    queue="default",
    job_type="download",
    owner_user_id="u1",
    event_types=("job.created",),
)

assert all(event["domain"] == "media_ingest" for event in events)
assert all(event["queue"] == "default" for event in events)
assert all(event["job_type"] == "download" for event in events)
assert all(event["owner_user_id"] == "u1" for event in events)
assert set(events[0]) >= {
    "id",
    "event_type",
    "attrs_json",
    "job_id",
    "domain",
    "queue",
    "job_type",
    "owner_user_id",
    "request_id",
    "trace_id",
    "created_at",
}
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py -q
```

Expected: FAIL because the new filters are unsupported.

- [ ] **Step 3: Implement filter expansion**

Update signature:

```python
def list_job_events_after(
    self,
    *,
    after_id: int = 0,
    limit: int = 100,
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    job_id: int | None = None,
    owner_user_id: str | None = None,
    event_types: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
```

Implementation requirements:

- bound `limit` to `1..1000`
- normalize `after_id` to non-negative int
- build backend-specific placeholders
- keep one selected column order for SQLite and PostgreSQL
- return dictionaries with the canonical raw storage keys
- do not parse `attrs_json` in `JobManager`

- [ ] **Step 4: Run JobManager event tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py -q
```

Expected: PASS.

### Task 6: Migrate Event Endpoints To Public JobManager API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Modify if needed: `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`
- Modify if needed: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py`

- [ ] **Step 1: Add endpoint regression tests**

Pin the list endpoint response shape:

```python
response = client.get("/api/v1/jobs/events", params={"after_id": 0, "domain": "d"})
assert response.status_code == 200
body = response.json()
assert body[0]["attrs"] == {}
assert "attrs_json" not in body[0]
```

For SSE tests, assert payload still has parsed `attrs`:

```python
assert event["event"] == "job"
assert isinstance(event["data"]["attrs"], dict)
```

- [ ] **Step 2: Run endpoint tests to verify current baseline**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py -q
```

Expected: existing tests may pass before refactor; new targeted assertions should fail only if current shape differs from the documented contract. If a new assertion reveals an existing contract mismatch, stop and document before changing behavior.

- [ ] **Step 3: Replace private event SQL in Jobs admin**

In `jobs_admin.py`:

- `list_job_events` calls `jm.list_job_events_after(...)`
- `stream_job_events` producer polls `jm.list_job_events_after(...)`
- endpoint parses `attrs_json` into response/SSE `attrs`
- keep `_enforce_domain_scope_unified` and `_set_pg_rls_for_user` in endpoint layer

- [ ] **Step 4: Replace private event SQL in media ingest SSE**

In `media/ingest_jobs.py`, poll:

```python
rows = jm.list_job_events_after(
    after_id=after_id,
    limit=500,
    domain="media_ingest",
    owner_user_id=owner_filter,
)
```

Then keep the existing `tracked_job_ids` filtering and SSE payload shape.

- [ ] **Step 5: Check audio/prompt-studio event paths**

Run:

```bash
rg -n "job_events|jm\\._connect|jm\\._pg_cursor" tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py
```

Only migrate direct `job_events` reads in this stage. Do not rewrite unrelated queue/status aggregation SQL from `prompt_studio_status.py`.

- [ ] **Step 6: Run focused Jobs/API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py -q
```

Expected: PASS.

- [ ] **Step 7: Smoke check direct event storage access is gone from targeted endpoints**

Run:

```bash
rg -n "jm\\._connect|jm\\._pg_cursor|FROM job_events|job_events WHERE" tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py
```

Expected: no matches for event-read code in targeted endpoints. If prompt-studio still has non-event `jobs` table aggregation, leave it and document why.

- [ ] **Step 8: Commit Stage 3**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py
git commit -m "refactor: read job events through JobManager"
```

## Stage 4: Document Workspace Repository And Migrations

**Status:** Not Started

### Task 7: Move Document Workspace Schema To Media DB Bootstrap

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/media_db/schema/document_workspace_schema.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
- Modify if PostgreSQL support is required by current Media DB bootstrap tests: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`

- [ ] **Step 1: Write failing schema tests**

Add tests proving bootstrap creates:

- `document_reading_progress`
- `document_annotations`
- `idx_annotations_media_user`
- `document_parsed_references_cache`
- `idx_doc_refs_cache_lookup`

Add an old-schema upgrade test that creates the tables without `cfi`, `percentage`, or `chapter_title`, runs the schema helper, and asserts missing columns are added idempotently.

- [ ] **Step 2: Run schema tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py -q
```

Expected: FAIL until the new helper is wired.

- [ ] **Step 3: Implement schema helper**

Create functions:

```python
def ensure_sqlite_document_workspace_schema(conn: Any) -> None:
    """Ensure document workspace tables and indexes exist for SQLite."""

def ensure_postgres_document_workspace_schema(conn: Any) -> None:
    """Ensure document workspace tables and indexes exist for PostgreSQL."""
```

SQLite helper owns the DDL currently embedded in:

- `_ensure_progress_table`
- `_ensure_annotations_table`
- `_ensure_parsed_references_cache_table`

Prefer constant SQL strings and small `_sqlite_columns(conn, table_name)` helper for idempotent column additions.

- [ ] **Step 4: Wire SQLite bootstrap**

Call `ensure_sqlite_document_workspace_schema(conn)` from `ensure_sqlite_post_core_structures` after existing content/collection structures.

- [ ] **Step 5: Run schema tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py -q
```

Expected: PASS.

### Task 8: Add Document Workspace Repository

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/media_db/repositories/document_workspace_repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/repositories/__init__.py`
- Create: `tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py`

- [ ] **Step 1: Write repository tests**

Cover:

- `get_reading_progress`
- `upsert_reading_progress`
- `delete_reading_progress`
- `list_annotations`
- `create_annotation`
- `update_annotation`
- `sync_annotations`
- `soft_delete_annotation`
- `get_parsed_references_cache`
- `upsert_parsed_references_cache`

Use a real temporary SQLite `MediaDatabase` where practical so schema bootstrap is also exercised.

- [ ] **Step 2: Run repository tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py -q
```

Expected: FAIL because the repository does not exist.

- [ ] **Step 3: Implement repository module**

Use a small class:

```python
class DocumentWorkspaceRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    @classmethod
    def from_media_db(cls, db: Any) -> "DocumentWorkspaceRepository":
        return cls(db)
```

Repository methods own SQL and return plain dictionaries. Endpoints keep Pydantic response shaping.

- [ ] **Step 4: Run repository tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py -q
```

Expected: PASS.

### Task 9: Thin Document Workspace Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_references.py`
- Modify: `tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py`
- Modify: `tldw_Server_API/tests/Media/test_document_references.py`

- [ ] **Step 1: Add endpoint delegation tests**

For each endpoint module, monkeypatch repository construction and assert the endpoint calls repository methods instead of running `_ensure_*_table`.

For document references, preserve cache semantics:

```python
cached = repo.get_parsed_references_cache(...)
assert cached == (["[1] Cached"], 1)
```

- [ ] **Step 2: Run endpoint tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py tldw_Server_API/tests/Media/test_document_references.py -q
```

Expected: new delegation tests fail until endpoints use the repository.

- [ ] **Step 3: Replace lazy DDL helpers**

Remove endpoint calls to:

- `_ensure_progress_table`
- `_ensure_annotations_table`
- `_ensure_parsed_references_cache_table`

Do not keep these helpers as endpoint wrappers. Use:

```python
repo = DocumentWorkspaceRepository.from_media_db(db)
```

Keep media existence checks in endpoints unless repository methods already require media existence and return explicit domain errors.

- [ ] **Step 4: Preserve row-to-schema mapping in endpoints**

Keep response conversion helpers such as `_row_to_response` in endpoint modules if they are only Pydantic/transport mapping. Repository should return storage rows, not response models.

- [ ] **Step 5: Run focused document workspace tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py tldw_Server_API/tests/Media/test_document_references.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke check endpoint DDL is gone**

Run:

```bash
rg -n "CREATE TABLE IF NOT EXISTS|ALTER TABLE|PRAGMA table_info|_ensure_.*table" tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py tldw_Server_API/app/api/v1/endpoints/media/document_references.py
```

Expected: no matches.

- [ ] **Step 7: Commit Stage 4**

```bash
git add tldw_Server_API/app/core/DB_Management/media_db/schema/document_workspace_schema.py tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py tldw_Server_API/app/core/DB_Management/media_db/repositories/document_workspace_repository.py tldw_Server_API/app/core/DB_Management/media_db/repositories/__init__.py tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py tldw_Server_API/app/api/v1/endpoints/media/document_references.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py tldw_Server_API/tests/Media/test_document_references.py
git commit -m "refactor: move document workspace storage into Media DB"
```

## Stage 5: Prototype Promotion Review Ownership

**Status:** Not Started

### Task 10: Add Public Promotion Review Service Method

**Files:**
- Modify: `tldw_Server_API/app/core/Prototype_Workspaces/service.py`
- Modify: `tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py`

- [ ] **Step 1: Write failing service tests**

Add tests for:

- owner reject updates request to `rejected`
- designated promoter reject updates request to `rejected`
- non-promoter raises `PermissionError`
- missing promotion request raises `ValueError`
- approve delegates to existing promotion behavior and returns promoted/stale/failed shapes unchanged

Expected service call:

```python
result = await passing_promotion_service.review_promotion_request(
    promotion_request_id=promotion_request["id"],
    reviewer_user_id=1,
    decision="reject",
    review_notes="Not ready",
)

assert result["status"] == "rejected"
assert result["details"] == {"review_notes": "Not ready"}
```

- [ ] **Step 2: Run service tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q
```

Expected: FAIL because `review_promotion_request` does not exist.

- [ ] **Step 3: Implement `review_promotion_request`**

Add public method:

```python
async def review_promotion_request(
    self,
    *,
    promotion_request_id: str,
    reviewer_user_id: int,
    decision: str,
    review_notes: str | None = None,
    review_baseline_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Review a promotion request through one service-owned boundary."""
```

Implementation requirements:

- load promotion request
- load workspace
- apply `_is_promoter` internally
- for `decision == "reject"`, call `repo.update_promotion_request(...)` and return the current rejected response shape
- for `decision == "approve"`, call `promote_candidate(...)` with the resolved workspace/candidate/request values
- keep error types simple: `ValueError` for missing/not found/invalid state, `PermissionError` for forbidden

- [ ] **Step 4: Run service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q
```

Expected: PASS.

### Task 11: Thin Prototype Promotion Review Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
- Modify: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`

- [ ] **Step 1: Add endpoint regression tests**

Add endpoint tests for:

- reject returns status `rejected`, existing workspace/candidate IDs, current canonical snapshot, and `details.review_notes`
- non-promoter review returns `403`
- missing request still returns `404`

- [ ] **Step 2: Run endpoint tests to verify RED/baseline**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py::TestPrototypeWorkspaceEndpoints::test_owner_can_review_promotion_request tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py::TestPrototypeWorkspaceEndpoints::test_designated_promoter_can_review_promotion_request tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py::TestPrototypeWorkspaceEndpoints::test_stale_promotion_response_shape -q
```

Then run the new reject/non-promoter tests. Expected: existing approval tests pass; new test expectations may fail until endpoint delegates.

- [ ] **Step 3: Replace endpoint split logic with service call**

Endpoint should become:

```python
try:
    result = await service.review_promotion_request(
        promotion_request_id=promotion_request_id,
        reviewer_user_id=_coerce_user_id(user),
        decision=body.decision,
        review_notes=body.review_notes,
        review_baseline_snapshot_id=body.review_baseline_snapshot_id,
    )
except PermissionError as exc:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Reviewer does not have promotion permissions") from exc
except ValueError as exc:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
return PrototypePromotionReviewResponse.model_validate(result)
```

Keep existing client-visible details where tests pin them. If a `ValueError` message should not be exposed directly, map it to the current endpoint detail string in the except block.

- [ ] **Step 4: Run prototype tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q
```

Expected: PASS.

- [ ] **Step 5: Smoke check endpoint no longer reaches private/repo transition APIs**

Run:

```bash
rg -n "_is_promoter|repo\\.update_promotion_request" tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py
```

Expected: no matches.

- [ ] **Step 6: Commit Stage 5**

```bash
git add tldw_Server_API/app/core/Prototype_Workspaces/service.py tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py
git commit -m "refactor: review prototype promotions through service"
```

## Final Verification

**Status:** Not Started

- [ ] **Step 1: Run full focused backend suite for touched areas**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py \
  tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py \
  tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py \
  tldw_Server_API/tests/Media/test_document_references.py \
  tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py \
  tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py \
  -q
```

Expected: PASS or documented environment-specific skips only.

- [ ] **Step 2: Run smoke checks for removed boundary leaks**

```bash
rg -n "_update_fts_media|_log_sync_event|UPDATE Media SET" tldw_Server_API/app/api/v1/endpoints/media/item.py
rg -n "jm\\._connect|jm\\._pg_cursor|FROM job_events|job_events WHERE" tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py
rg -n "CREATE TABLE IF NOT EXISTS|ALTER TABLE|PRAGMA table_info|_ensure_.*table" tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py tldw_Server_API/app/api/v1/endpoints/media/document_references.py
rg -n "_is_promoter|repo\\.update_promotion_request" tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py
```

Expected: no matches in the targeted endpoint-owned logic. Document any intentional remaining matches.

- [ ] **Step 3: Run Bandit on touched production paths**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/router_groups \
  tldw_Server_API/app/api/v1/endpoints/media/item.py \
  tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py \
  tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py \
  tldw_Server_API/app/api/v1/endpoints/media/document_references.py \
  tldw_Server_API/app/api/v1/endpoints/jobs_admin.py \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
  tldw_Server_API/app/core/DB_Management/media_db \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Prototype_Workspaces/service.py \
  tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py \
  -f json -o /tmp/bandit_api_boundary_remediation.json
```

Expected: no new high/medium findings in changed code. If baseline findings appear outside changed lines, record them in the final task summary.

- [ ] **Step 4: Run diff hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Staged files should match the current task or final commit scope only.

- [ ] **Step 5: Final commit if needed**

If final verification required small fixes after the per-stage commits:

```bash
git add <only final fix files>
git commit -m "test: verify api boundary remediation"
```

## Rollback And Risk Notes

- Stage 1 risk is route gating drift in minimal-test mode. Keep route inclusion tests before refactoring.
- Stage 2 risk is subtle Media update behavior change. Preserve identical-content document-version creation unless tests and the user explicitly approve a behavior change.
- Stage 3 risk is SSE payload drift. Keep `attrs_json` raw in `JobManager`, parse `attrs` only in endpoint response mapping.
- Stage 4 risk is old per-user DB compatibility. Schema helper must be idempotent and should not run ad hoc DDL from request handlers.
- Stage 5 risk is endpoint error mapping drift. Pin endpoint response status/details before moving logic.

## Execution Handoff

Recommended execution mode is subagent-driven development, one fresh worker per stage, with local review between stages. Inline execution is also viable if you prefer a single session with checkpoints after each commit.
