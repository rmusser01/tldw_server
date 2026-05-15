# First-Class Watchlists Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class Watchlist container contract while preserving existing source, monitor, run, item, and output workflows.

**Architecture:** Introduce a persisted `watchlists` container inside the existing Watchlists DB, associate existing source catalog entries via a join table, and add `watchlist_id` to jobs/monitors so runs, items, and outputs can derive scope without duplicating the whole data model. Keep current endpoints backward compatible by defaulting unscoped operations into a migrated default Watchlist, then add Watchlist-aware filters and a scoped frontend shell around the existing child tabs.

**Tech Stack:** FastAPI, Pydantic, per-user WatchlistsDatabase with SQLite/Postgres backend abstraction, Collections outputs metadata, Next.js route shim, shared React UI package, Zustand, Ant Design, Vitest, pytest, Bandit.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Tracking task for this plan: `TASK-350`
- Design task: `TASK-349`

## Product Decisions For Stage 1

- A Watchlist is a project-like container, not an alias for a job/monitor.
- Stage 1 introduces the container and scoping contract only. It does not implement semantic content alerts, CTI entity extraction, novelty scoring, or defensible report-builder UI.
- Sources remain a per-user URL catalog. Use a `watchlist_sources` join table so a source can be reused by more than one Watchlist without changing the existing `UNIQUE(user_id, url)` source constraint.
- Jobs/Monitors belong to exactly one Watchlist in Stage 1 via `scrape_jobs.watchlist_id`.
- Runs and scraped items inherit Watchlist scope from their job. Add direct `scraped_items.watchlist_id` only if a focused performance test proves the join is too expensive.
- Generated outputs should record `watchlist_id` in metadata and should remain filterable by Watchlist through job IDs so older outputs from migrated jobs remain visible.
- Existing API calls without `watchlist_id` remain valid and use the default migrated Watchlist for creates.
- Current `/api/v1/watchlists/{watchlist_id}/clusters` uses a job ID despite the path name. Do not expand cluster semantics in Stage 1. Add regression coverage so first-class Watchlist CRUD does not break this legacy route.
- Paused/archived status is container metadata in Stage 1. Do not cascade pause/archive into child jobs until a later scheduler/lifecycle task defines the semantics.

## Current Evidence To Preserve

- WebUI route: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Main page: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Store: `apps/packages/ui/src/store/watchlists.tsx`
- Types: `apps/packages/ui/src/types/watchlists.ts`
- Service: `apps/packages/ui/src/services/watchlists.ts`
- Overview service: `apps/packages/ui/src/services/watchlists-overview.ts`
- Backend schemas: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Backend router: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Persistence: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Existing API tests: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
- Existing route/deep-link tests: `apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx`

## Implementation Boundaries

- Do not rename `/watchlists`.
- Do not remove existing Feeds, Monitors, Activity, Articles/Items, Reports/Outputs, Templates, or Settings tabs in Stage 1.
- Do not migrate the separate admin Topic Monitoring Watchlist model in Stage 1.
- Do not build content-match alerts in Stage 1; only reserve the future relationship in the object model where needed.
- Do not change source URL uniqueness in Stage 1.
- Do not change scheduler behavior for paused/archived Watchlists in Stage 1.
- Do not introduce a new frontend state library or design system.
- Do not use browser Computer Use; browser verification should use CDP/Playwright when implementation reaches rendered UI QA.

## Proposed File Responsibilities

Backend:

- `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - Add `WatchlistRow` dataclass.
  - Add `watchlists` and `watchlist_sources` schema for SQLite and Postgres.
  - Add idempotent migrations/backfills.
  - Add CRUD, soft delete/restore, default Watchlist, membership, and filtered list helpers.
  - Add optional `watchlist_id` filtering to source/job/run/item helpers.
- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add Watchlist request/response schemas.
  - Add `watchlist_id` fields or filters to source/job/output request/response schemas where needed.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add top-level Watchlist CRUD endpoints.
  - Wire default Watchlist behavior for existing create paths.
  - Add `watchlist_id` filters to child list/create paths.
  - Preserve route ordering so static paths like `/sources` are not shadowed by `/{watchlist_id}`.
  - Add `watchlist_id` metadata on new outputs and filter output lists by Watchlist.
- `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
  - Add a narrow `job_ids` filter to `list_output_artifacts` if needed for Watchlist-scoped reports. Prefer filtering by job IDs over metadata-only filtering so pre-Stage-1 outputs remain visible after job backfill.

Frontend:

- `apps/packages/ui/src/types/watchlists.ts`
  - Add `WatchlistContainer`, create/update payloads, lifecycle/status/priority/domain types.
  - Add `watchlist_id` and `watchlist_ids` fields where API responses expose them.
- `apps/packages/ui/src/services/watchlists.ts`
  - Add Watchlist CRUD service functions.
  - Add `watchlist_id` query support to source/job/run/item/output fetchers.
- `apps/packages/ui/src/services/watchlists-overview.ts`
  - Accept `watchlist_id` and scope aggregate calls.
- `apps/packages/ui/src/store/watchlists.tsx`
  - Add selected Watchlist state and list/loading/error state.
- `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - Add a Watchlist selector/overview shell above existing child views.
  - Pass selected Watchlist scope into child tabs.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Scope overview data and creation entry points to the selected Watchlist.
- Child tabs:
  - `SourcesTab/SourcesTab.tsx`
  - `JobsTab/JobsTab.tsx`
  - `RunsTab/RunsTab.tsx`
  - `ItemsTab/ItemsTab.tsx`
  - `OutputsTab/OutputsTab.tsx`
  - Add scoped fetch parameters without redesigning the tabs.
- Locale/copy:
  - `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - `apps/packages/ui/src/public/_locales/en/watchlists.json` if the repo still mirrors extension locale files manually.

Tests:

- Backend:
  - Create: `tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py`
  - Create: `tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py`
  - Modify/extend: `tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py`
  - Modify/extend: `tldw_Server_API/tests/Watchlists/test_runs_list_global.py`
  - Modify/extend: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
- Frontend:
  - Create: `apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts`
  - Create: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
  - Modify/extend: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts`
  - Modify/extend: `apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx`

## Backlog Task Map For Implementation

Create implementation tasks before code changes. Recommended split:

- Stage 1A: Watchlists DB container and migration default.
- Stage 1B: Watchlist CRUD API and child endpoint scoping.
- Stage 1C: Frontend types/services/store and Watchlist selector shell.
- Stage 1D: Scoped child-tab integration and extension-sized smoke QA.

Each implementation task should reference this plan and the design spec. If the first implementer chooses a single PR, keep commits aligned with the task groups above.

## Task 0: Baseline And Task Setup

**Files:**
- Reference: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Reference: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md`
- Reference: `backlog/tasks/task-350 - Plan-Stage-1-first-class-Watchlist-container-implementation.md`

- [ ] **Step 1: Create implementation Backlog tasks**

Use Backlog MCP to create the four implementation tasks listed above, or create a single implementation task if doing one focused PR. Include the implementation boundaries and the known route-conflict risk in every task description.

- [ ] **Step 2: Capture current route/API baseline**

Run:

```bash
rg -n "@router\\.(get|post|patch|delete).*watchlists|/{watchlist_id}/clusters|def list_sources|def list_jobs|def list_outputs" tldw_Server_API/app/api/v1/endpoints/watchlists.py
rg -n "WatchlistTab|fetchWatchlistSources|fetchWatchlistJobs|WatchlistsPlaygroundPage" apps/packages/ui/src
```

Expected: current sources/jobs/runs/items/outputs endpoints and the legacy `/{watchlist_id}/clusters` route are identified before route edits.

- [ ] **Step 3: Run current focused baseline tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q
bunx vitest run apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx
```

Expected: establish current pass/fail baseline. If any fail for unrelated repo state, record exact failures in the implementation task before editing.

- [ ] **Step 4: Commit baseline task records if they changed**

Run:

```bash
git add backlog/tasks/<implementation-task-files>
git commit -m "chore: task first-class watchlists stage 1"
```

Expected: only task records are committed.

## Task 1: Watchlists DB Contract And Migration Default

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Create: `tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py`

- [ ] **Step 1: Write failing DB contract tests**

Add tests for:

```python
def test_default_watchlist_created_once_and_backfills_jobs_and_sources(tmp_path):
    db = WatchlistsDatabase(user_id=123)
    source = db.create_source(name="Feed", url="https://example.com/rss.xml", source_type="rss")
    job = db.create_job(
        name="Daily",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone=None,
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    default = db.ensure_default_watchlist()
    assert default.name == "Imported Watchlist"
    assert default.domain == "general"
    assert default.status == "active"

    assert db.ensure_default_watchlist().id == default.id
    assert db.list_watchlist_sources(default.id, limit=50, offset=0)[1] == 1
    assert db.get_job(job.id).watchlist_id == default.id
```

Also test:
- create/list/get/update/archive/delete/restore Watchlist.
- deleted Watchlists are excluded by default.
- restored Watchlists keep source memberships and job associations.
- source URL uniqueness is unchanged.
- Postgres DDL text includes equivalent columns/tables.

Expected failure: `WatchlistRow` and helper methods do not exist.

- [ ] **Step 2: Add dataclass and schema**

Add a `WatchlistRow` dataclass:

```python
@dataclass
class WatchlistRow:
    id: int
    user_id: str
    name: str
    description: str | None
    objective: str | None
    domain: str
    status: str
    priority: str
    tags_json: str | None
    created_at: str
    updated_at: str
    archived_at: str | None = None
    deleted_at: str | None = None
    restore_expires_at: str | None = None
```

Add SQLite and Postgres DDL:

```sql
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    objective TEXT,
    domain TEXT NOT NULL DEFAULT 'general',
    status TEXT NOT NULL DEFAULT 'active',
    priority TEXT NOT NULL DEFAULT 'medium',
    tags_json TEXT,
    archived_at TEXT,
    deleted_at TEXT,
    restore_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_watchlists_user_status ON watchlists(user_id, status);
CREATE INDEX IF NOT EXISTS idx_watchlists_user_deleted ON watchlists(user_id, deleted_at);

CREATE TABLE IF NOT EXISTS watchlist_sources (
    watchlist_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (watchlist_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_watchlist_sources_watchlist ON watchlist_sources(watchlist_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_sources_source ON watchlist_sources(source_id);
```

Use `BIGSERIAL`/`BIGINT` equivalents in the Postgres branch.

Add nullable `watchlist_id` to `scrape_jobs` with an index:

```sql
ALTER TABLE scrape_jobs ADD COLUMN watchlist_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_jobs_user_watchlist ON scrape_jobs(user_id, watchlist_id);
```

Use existing idempotent column-add helpers/patterns in the file.

- [ ] **Step 3: Add default/backfill helpers**

Implement:

```python
def ensure_default_watchlist(self) -> WatchlistRow: ...
def backfill_default_watchlist_scope(self, watchlist_id: int | None = None) -> None: ...
```

Rules:
- One default imported Watchlist per user.
- Existing jobs with `watchlist_id IS NULL` get the default Watchlist ID.
- Existing sources are inserted into `watchlist_sources` for the default Watchlist.
- Re-running is idempotent.
- New database initialization should call this lazily from public APIs, not during every `ensure_schema()` if that causes request-time write noise.

- [ ] **Step 4: Add Watchlist CRUD helpers**

Implement:

```python
def create_watchlist(...): ...
def get_watchlist(watchlist_id: int, include_deleted: bool = False): ...
def list_watchlists(...): ...
def update_watchlist(watchlist_id: int, fields: dict[str, Any]): ...
def delete_watchlist(watchlist_id: int, restore_window_seconds: int = ...): ...
def restore_watchlist(watchlist_id: int): ...
def add_source_to_watchlist(watchlist_id: int, source_id: int): ...
def list_watchlist_sources(watchlist_id: int, limit: int, offset: int): ...
```

Validation:
- `domain`: `cti_osint`, `news`, `general`.
- `status`: `active`, `paused`, `archived`.
- `priority`: `low`, `medium`, `high`, `critical`.
- Tag storage uses `tags_json` and preserves simple string tags.
- Delete sets `deleted_at` and `restore_expires_at`; it does not remove child sources/jobs.

- [ ] **Step 5: Add source/job filters**

Extend existing DB helpers:

```python
def create_source(..., watchlist_id: int | None = None) -> SourceRow
def list_sources(..., watchlist_id: int | None = None) -> tuple[list[SourceRow], int]
def create_job(..., watchlist_id: int | None = None) -> JobRow
def list_jobs(..., watchlist_id: int | None = None) -> tuple[list[JobRow], int]
def list_runs(..., watchlist_id: int | None = None) -> tuple[list[RunRow], int]
def list_items(..., watchlist_id: int | None = None) -> tuple[list[ScrapedItemRow], int]
```

Expected behavior:
- If a create path omits `watchlist_id`, attach to `ensure_default_watchlist()`.
- `list_sources(watchlist_id=...)` uses `watchlist_sources`.
- `list_jobs(watchlist_id=...)` filters `scrape_jobs.watchlist_id`.
- `list_runs` and `list_items` filter through joined jobs.

- [ ] **Step 6: Run DB tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py -q
```

Expected: new DB contract tests pass.

- [ ] **Step 7: Commit DB contract**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py
git commit -m "feat: add watchlist container persistence"
```

## Task 2: Watchlist Schemas And CRUD API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Create: `tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

```python
def test_watchlist_crud_and_default_scope(client_with_user):
    created = client_with_user.post("/api/v1/watchlists", json={
        "name": "Ransomware Healthcare Watch",
        "description": "Track hospital impact",
        "objective": "Find new ransomware reports affecting hospitals in Germany",
        "domain": "cti_osint",
        "priority": "high",
        "tags": ["ransomware", "healthcare"],
    })
    assert created.status_code == 201
    watchlist = created.json()
    assert watchlist["domain"] == "cti_osint"
    assert watchlist["status"] == "active"

    listed = client_with_user.get("/api/v1/watchlists")
    assert any(item["id"] == watchlist["id"] for item in listed.json()["items"])
```

Also cover:
- PATCH lifecycle/status fields.
- DELETE returns restore window.
- POST restore returns the row.
- existing `POST /sources` without `watchlist_id` attaches source to the default Watchlist.
- `POST /jobs` without `watchlist_id` attaches job to the default Watchlist.
- `GET /sources?watchlist_id=...` filters.
- `GET /jobs?watchlist_id=...` filters.
- `/api/v1/watchlists/sources` and `/api/v1/watchlists/{job_id}/clusters` do not regress.

Expected failure: schemas and endpoints do not exist.

- [ ] **Step 2: Add Pydantic schemas**

Add:

```python
WatchlistDomain = Literal["cti_osint", "news", "general"]
WatchlistStatus = Literal["active", "paused", "archived"]
WatchlistPriority = Literal["low", "medium", "high", "critical"]

class WatchlistCreateRequest(BaseModel): ...
class WatchlistUpdateRequest(BaseModel): ...
class WatchlistContainer(BaseModel): ...
class WatchlistsListResponse(BaseModel): ...
class WatchlistDeleteResponse(ReversibleDeleteResponse): ...
```

Extend:
- `SourceCreateRequest` with optional `watchlist_id`.
- `Source` with `watchlist_ids: list[int] = []`.
- `JobCreateRequest` and `Job` with optional `watchlist_id`.

Keep optional fields backward compatible.

- [ ] **Step 3: Add CRUD endpoints with safe route ordering**

Add fixed root endpoints:

```python
@router.get("", response_model=WatchlistsListResponse)
@router.post("", response_model=WatchlistContainer, status_code=201)
@router.get("/{watchlist_id}", response_model=WatchlistContainer)
@router.patch("/{watchlist_id}", response_model=WatchlistContainer)
@router.delete("/{watchlist_id}", response_model=WatchlistDeleteResponse)
@router.post("/{watchlist_id}/restore", response_model=WatchlistContainer)
```

Route-order warning:
- Add static child routes before dynamic root `/{watchlist_id}` routes, or add regression tests that prove `/sources`, `/jobs`, `/runs`, `/items`, `/outputs`, `/settings`, and `/templates` are not shadowed.
- Keep `/{watchlist_id}/clusters` behavior stable in Stage 1 even though the path name is misleading.

- [ ] **Step 4: Wire source and job create/list scoping**

Update:
- `create_source`
- `list_sources`
- `bulk_create_sources`
- `import_sources_opml`
- `create_job`
- `list_jobs`
- `list_runs_global`
- `list_scraped_items`
- smart counts if it reuses `list_items`

Rules:
- If create payload omits `watchlist_id`, use default Watchlist.
- If list payload has `watchlist_id`, filter to that Watchlist.
- If a requested Watchlist is deleted/not found, return `404 watchlist_not_found`.
- If a source is idempotently reused by URL, still add it to the requested Watchlist.

- [ ] **Step 5: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q
```

Expected: new CRUD/scoping tests and existing API tests pass.

- [ ] **Step 6: Commit API contract**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py
git commit -m "feat: expose first-class watchlist API"
```

## Task 3: Output Provenance And Watchlist-Scoped Output Listing

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify if needed: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
- Create or extend: `tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py`

- [ ] **Step 1: Write failing output provenance tests**

Test:
- Creating an output from a run whose job belongs to Watchlist A stores `metadata.watchlist_id == A`.
- `GET /outputs?watchlist_id=A` returns that output.
- `GET /outputs?watchlist_id=B` does not return it.
- Outputs created before metadata backfill are still returned if their `job_id` belongs to A.

Expected failure: outputs do not carry/filter by Watchlist.

- [ ] **Step 2: Add output metadata**

In `create_output`, after loading `job`, resolve `job.watchlist_id` and add:

```python
metadata["watchlist_id"] = int(job.watchlist_id) if job.watchlist_id is not None else None
```

If `job.watchlist_id` is missing on an older DB, call default backfill and reload the job before creating output.

- [ ] **Step 3: Add robust output filtering**

Preferred approach:
- Add optional `job_ids: list[int] | None` to `CollectionsDatabase.list_output_artifacts`.
- In `list_outputs`, when `watchlist_id` is supplied, fetch job IDs for that Watchlist and pass `job_ids`.
- If there are no job IDs, return an empty paginated result.

Avoid metadata-only filtering as the primary path because old outputs may not include `watchlist_id` until regenerated.

- [ ] **Step 4: Run output tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py -k "output or provenance" -q
```

Expected: output scoping tests pass.

- [ ] **Step 5: Commit output provenance**

Run:

```bash
git add tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py
git commit -m "feat: scope watchlist outputs by container"
```

## Task 4: Frontend Types, Services, And Store

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists-overview.ts`
- Modify: `apps/packages/ui/src/store/watchlists.tsx`
- Create: `apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts`

- [ ] **Step 1: Write failing service tests**

Test:

```ts
it("creates and fetches watchlists through the expected API paths", async () => {
  await createWatchlist({ name: "Healthcare ransomware", domain: "cti_osint" })
  expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: "/api/v1/watchlists",
    method: "POST"
  }))

  await fetchWatchlistSources({ watchlist_id: 42 })
  expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: expect.stringContaining("/api/v1/watchlists/sources?watchlist_id=42")
  }))
})
```

Expected failure: types and services are missing.

- [ ] **Step 2: Add frontend types**

Add:

```ts
export type WatchlistDomain = "cti_osint" | "news" | "general"
export type WatchlistStatus = "active" | "paused" | "archived"
export type WatchlistPriority = "low" | "medium" | "high" | "critical"

export interface WatchlistContainer { ... }
export interface WatchlistCreate { ... }
export interface WatchlistUpdate { ... }
```

Extend:
- `WatchlistSource` with `watchlist_ids?: number[]`.
- `WatchlistSourceCreate` with `watchlist_id?: number`.
- `WatchlistJob` and `WatchlistJobCreate` with `watchlist_id?: number`.
- fetch params with `watchlist_id?: number`.

- [ ] **Step 3: Add service methods**

Add:

```ts
fetchWatchlists()
getWatchlist(id)
createWatchlist(payload)
updateWatchlist(id, payload)
deleteWatchlist(id)
restoreWatchlist(id)
```

Extend query builders for source/job/run/item/output fetches.

- [ ] **Step 4: Scope overview service**

Allow:

```ts
fetchWatchlistsOverviewData({ watchlist_id })
```

Each aggregate call should pass the selected Watchlist ID.

- [ ] **Step 5: Add store state**

Add:

```ts
watchlists: WatchlistContainer[]
watchlistsLoading: boolean
watchlistsError: string | null
selectedWatchlistId: number | null
```

Add actions:

```ts
setWatchlists
setWatchlistsLoading
setWatchlistsError
setSelectedWatchlistId
addWatchlist
updateWatchlistInList
removeWatchlist
```

- [ ] **Step 6: Run frontend service/store tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts
```

Expected: frontend type/service tests pass.

- [ ] **Step 7: Commit frontend contract**

Run:

```bash
git add apps/packages/ui/src/types/watchlists.ts apps/packages/ui/src/services/watchlists.ts apps/packages/ui/src/services/watchlists-overview.ts apps/packages/ui/src/store/watchlists.tsx apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts
git commit -m "feat: add watchlist container client contract"
```

## Task 5: Watchlist Selector Shell And Scoped Child Tabs

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify if used by extension bundle: `apps/packages/ui/src/public/_locales/en/watchlists.json`
- Create: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx`

- [ ] **Step 1: Write failing shell tests**

Cover:
- Page loads Watchlists list and selects the first/default Watchlist.
- User can switch selected Watchlist.
- Child fetches include `watchlist_id`.
- Existing `?tab=feeds`, `?tab=articles`, `?job_id=...`, `?run_id=...` aliases still set the expected tab/state.
- Empty list offers a create Watchlist action.

Expected failure: selected Watchlist state does not exist.

- [ ] **Step 2: Add lightweight shell**

Add a Watchlist selector area above existing tabs:

- Selected Watchlist name.
- Status/priority tags.
- Objective line if present.
- Create/Edit action entry points can be simple modals or drawer placeholders wired to API.
- Keep existing progressive tab layout under the selector.

Do not redesign all tabs in this PR.

- [ ] **Step 3: Thread selected Watchlist into child tabs**

Each tab should read `selectedWatchlistId` from store and include it in fetch params:

- Sources: `fetchWatchlistSources({ watchlist_id })`.
- Jobs: `fetchWatchlistJobs({ watchlist_id })`.
- Runs: `fetchWatchlistRuns({ watchlist_id })`.
- Items: `fetchScrapedItems({ watchlist_id })` and smart counts.
- Outputs: `fetchWatchlistOutputs({ watchlist_id })`.
- Overview: `fetchWatchlistsOverviewData({ watchlist_id })`.

If no selected Watchlist exists yet, show a Watchlist-level empty state rather than loading all child tabs.

- [ ] **Step 4: Create/edit/delete UI minimal path**

Provide minimum viable controls:

- Create Watchlist.
- Edit Watchlist metadata.
- Archive/pause status update.
- Delete/restore can be API-covered in Stage 1; UI can expose delete only if restore copy is clear.

Use existing Ant Design form/modal patterns.

- [ ] **Step 5: Add copy**

Add concise labels:

- "Watchlist"
- "Objective"
- "Tracking container"
- "Active"
- "Paused"
- "Archived"
- "Create Watchlist"

Do not introduce CTI-heavy copy into the generic default state yet. CTI/news presets are Stage 2.

- [ ] **Step 6: Run UI tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx
```

Expected: shell and route state tests pass.

- [ ] **Step 7: Commit frontend shell**

Run:

```bash
git add apps/packages/ui/src/components/Option/Watchlists apps/packages/ui/src/assets/locale/en/watchlists.json apps/packages/ui/src/public/_locales/en/watchlists.json apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx
git commit -m "feat: add watchlist container shell"
```

## Task 6: Focused Integration And Regression Verification

**Files:**
- Modify only if tests reveal scoped issues:
  - `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - `apps/packages/ui/src/components/Option/Watchlists/*`
- Test files touched in previous tasks.

- [ ] **Step 1: Run backend focused suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_runs_list_global.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py \
  tldw_Server_API/tests/Watchlists/test_preview_endpoint.py \
  -q
```

Expected: focused backend Watchlists suite passes.

- [ ] **Step 2: Run frontend focused suite**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts \
  apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-terminology-contract.test.ts \
  apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx
```

Expected: focused frontend Watchlists suite passes.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/core/DB_Management/Collections_DB.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  -f json -o /tmp/bandit_watchlists_stage1.json
```

Expected: no new findings in touched code. If existing baseline findings appear, document them with file/line and why they are pre-existing or out of scope.

- [ ] **Step 4: Run a browser/CDP smoke if UI was materially changed**

Start WebUI in the project-approved way for the current branch. Then use CDP/Playwright, not Computer Use, to verify:

- Desktop `/watchlists` shows selected Watchlist.
- Extension-sized viewport shows the selector without clipped controls.
- Switching Watchlists changes child list requests.
- Existing tab deep links still work.

Expected: page renders nonblank, controls are reachable, no obvious route errors.

- [ ] **Step 5: Commit verification fixes**

If verification required fixes:

```bash
git add <fixed-files>
git commit -m "fix: harden watchlist container integration"
```

## Task 7: Docs, Task Finalization, And PR Packaging

**Files:**
- Modify: `Docs/API-related/Watchlists_API.md`
- Modify if published docs are maintained in parallel: `Docs/Published/API-related/Watchlists_API.md`
- Modify: relevant Backlog task files through MCP only.

- [ ] **Step 1: Update API docs**

Document:

- Watchlist CRUD endpoints.
- Default migrated Watchlist behavior.
- `watchlist_id` on source/job create and child list filters.
- Output provenance behavior.
- Explicit note that content-match alerts are future Stage 3 work.

- [ ] **Step 2: Run docs hygiene**

Run:

```bash
git diff --check -- Docs/API-related/Watchlists_API.md Docs/Published/API-related/Watchlists_API.md
```

Expected: no trailing whitespace or patch hygiene warnings.

- [ ] **Step 3: Final task updates**

Use Backlog MCP:

- Mark implementation task acceptance criteria complete.
- Record verification commands and results.
- Record Bandit result path.
- Record browser/CDP evidence location if generated.
- Add final summary.

- [ ] **Step 4: Final status check**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: only intended Stage 1 files are modified/staged/committed. Pre-existing unrelated worktree changes must remain untouched.

## Rollout Gates

- Existing `POST /sources`, `POST /jobs`, `GET /runs`, `GET /items`, and `GET /outputs` remain compatible without `watchlist_id`.
- Existing unscoped data appears under the default migrated Watchlist.
- New Watchlist CRUD does not shadow static routes.
- Legacy job-cluster subscription route remains stable.
- A selected Watchlist scopes Sources, Monitors, Activity, Items, and Reports.
- Output provenance includes Watchlist scope for new outputs.
- Extension-sized UI can select/create/manage a Watchlist without clipped primary controls.
- No content-match alert UI ships in Stage 1.

## Known Risks And Follow-Up Questions

- Route naming conflict: `/{watchlist_id}/clusters` currently means job clusters. Stage 1 should preserve it and plan a later route rename/deprecation.
- Source membership model: join table supports reuse, but child tabs may still mentally treat sources as owned by one Watchlist. Copy must be clear enough for Stage 1.
- Output filtering: robust Watchlist-scoped output listing likely needs `job_ids` filtering in Collections DB.
- Container pause semantics: Stage 1 does not pause schedules. A later lifecycle task must define whether pausing a Watchlist disables active child jobs.
- Default backfill timing: eager writes during schema ensure can create noise. Prefer lazy default creation on API operations and explicit test coverage for idempotency.
- Postgres parity: DDL and tests must cover Postgres shape even if focused local tests use SQLite.

## Full Verification Command Set

Run before declaring implementation complete:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_runs_list_global.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py \
  -q

bunx vitest run \
  apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts \
  apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx

source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/core/DB_Management/Collections_DB.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  -f json -o /tmp/bandit_watchlists_stage1.json

git diff --check
```

Expected: all focused tests pass, no new Bandit findings in touched code, and diff hygiene is clean.

## Execution Handoff

Recommended execution mode: subagent-driven development if the user explicitly authorizes subagents; otherwise execute inline with superpowers:executing-plans and checkpoint after each task group.

First implementation slice should be Task 1 plus its tests only. Do not start frontend shell work until backend CRUD/scoping tests pass.
