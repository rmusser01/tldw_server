# Bulk Conference Ingest Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable WebUI and extension workflow for ingesting, organizing, tracking, recovering, and reviewing a many-video conference playlist as one coherent collection.

**Architecture:** Extend the existing shared Quick Ingest, media ingest jobs, Collections DB item layer, Media review, Knowledge QA, and extension sidepanel surfaces instead of creating a separate conference-ingest product. The backend owns playlist metadata preflight, duplicate lookup, durable collection/run state, and scoped retrieval enforcement; the shared frontend owns the preflight/review UI, batch metadata editing, run progress, and WebUI/extension handoff.

**Tech Stack:** FastAPI, Pydantic, yt-dlp metadata extraction, existing Jobs/JobManager media ingest jobs, SQLite/Postgres-backed `CollectionsDatabase`, Next.js/shared React UI under `apps/packages/ui`, Ant Design, Zustand/session stores, WXT/browser-extension runtime messaging, Vitest, React Testing Library, pytest, Playwright, Backlog.md.

---

## Inputs

- Design spec: `Docs/superpowers/specs/2026-05-16-bulk-conference-ingest-workflow-design.md`
- Backlog task: `TASK-399`
- Current media ingest jobs API docs: `Docs/API-related/Media_Ingest_Jobs_API.md`
- Existing quick-ingest resume plan for adjacent context: `Docs/superpowers/plans/2026-03-24-quick-ingest-resume-and-e2e-implementation-plan.md`
- Existing quick-ingest UX remediation plan for active wizard paths: `Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md`

## Scope Rules

- Preserve ordinary one-file and one-URL Quick Ingest behavior in every stage.
- Keep shared WebUI/extension behavior under `apps/packages/ui/src` unless platform-specific extension APIs are required.
- Do not parse playlists in the frontend. The server provides metadata-only preflight.
- Do not depend on `media:collections:v1` localStorage as durable source of truth.
- Do not expose scoped Knowledge QA until backend retrieval is actually constrained by media IDs or collection ID.
- Do not store browser cookies, auth headers, downloaded video/audio, or secrets in preflight records or job payloads.
- Treat worker availability separately from endpoint existence.
- Each task below is intended as a reviewable PR or small sequence of commits.

## Program File Map

### Backend playlist preflight and capabilities

- Create: `tldw_Server_API/app/api/v1/schemas/media_playlist_preflight.py`
  - Pydantic request/response models for metadata-only playlist preflight.
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py`
  - Pure playlist URL classification, metadata normalization, duplicate-in-batch detection, and yt-dlp metadata extraction wrappers.
- Create: `tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py`
  - `POST /api/v1/media/playlists/preflight`, owner-scoped read-only endpoint.
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
  - Append `playlist_preflight` to `_MEDIA_ENDPOINT_MODULES`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`
  - Add granular capability keys for playlist preflight, media jobs endpoint, worker availability, job SSE, durable media collections, and scoped Knowledge QA.
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py`
- Test: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`

### Backend durable conference collections and run binding

- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
  - Prefer a small extension of the existing `content_items` layer with collection/group identity only if the inventory proves it fits; otherwise add a narrow media collection table here.
- Create if needed: `tldw_Server_API/app/api/v1/schemas/media_collections.py`
  - Stable conference collection, collection item, run, and retry request/response models.
- Create if needed: `tldw_Server_API/app/api/v1/endpoints/media/collections.py`
  - Media collection list/get/create/update and collection item status endpoints under `/api/v1/media/collections`.
- Modify if using a new media subrouter: `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
  - Append `collections` to `_MEDIA_ENDPOINT_MODULES`.
- Modify if reusing `/api/v1/items`: `tldw_Server_API/app/api/v1/endpoints/items.py`
  - Add collection filters/metadata only where needed; do not make `/items` own run orchestration.
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
  - Accept optional planned collection item IDs/idempotency keys and surface them in job payload/status.
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
  - Resolve planned collection items to completed/skipped/failed/cancelled status after job terminal state.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
  - Preserve collection metadata in the existing `sync_media_add_results_to_collections` path for synchronous fallback.
- Test: `tldw_Server_API/tests/Collections/test_conference_media_collections.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_conference_collection.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_conference_collection.py`

### Backend scoped Knowledge QA

- Read first: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`, `tldw_Server_API/app/api/v1/endpoints/rag_health.py`, `tldw_Server_API/app/core/RAG/`, and `tldw_Server_API/tests/RAG/test_rag_selection_filters.py`
- Modify only after confirming current contract: RAG search/generation schema and endpoint files that already support selection/media filters.
- Test: `tldw_Server_API/tests/RAG/test_conference_collection_scope.py`

### Shared frontend services, types, and capability detection

- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
  - Add granular booleans:
    - `hasMediaPlaylistPreflight`
    - `hasMediaIngestJobs`
    - `hasMediaIngestJobEvents`
    - `hasMediaIngestWorker`
    - `hasDurableMediaCollections`
    - `hasKnowledgeQaMediaScope`
- Modify: `apps/packages/ui/src/services/tldw/domains/media.ts`
  - Add `preflightPlaylist`, collection create/get/update, run status/retry/cancel methods if the API lives under `/media`.
- Modify if collection APIs remain under `/items`: `apps/packages/ui/src/services/tldw/domains/collections.ts`
  - Add typed wrappers for conference collection and collection item operations.
- Create: `apps/packages/ui/src/services/tldw/playlist-preflight.ts`
  - Pure client normalizers for server preflight payloads and UI state.
- Create: `apps/packages/ui/src/services/tldw/conference-collections.ts`
  - Pure helpers for inherited metadata, planned item payloads, retryability, grouped result counts.
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
  - Carry collection/run metadata, planned item IDs, idempotency keys, and retry metadata through direct WebUI submission.
- Modify: `apps/packages/ui/src/entries/background.ts`
  - Mirror shared metadata submission for extension-runtime batch handling.
- Test: `apps/packages/ui/src/services/__tests__/server-capabilities.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

### Shared Quick Ingest UI

- Modify: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
  - Add playlist preflight, batch metadata, collection/run, item override, and retry status types.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
  - Detect playlist-capable URLs and offer preflight before adding a single opaque row.
- Create: `apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx`
  - Expanded playlist preview with count, duplicate state, selection, warnings, and partial/expired states.
- Create: `apps/packages/ui/src/components/Common/QuickIngest/BatchMetadataPanel.tsx`
  - Conference-level metadata fields and shared tags.
- Create: `apps/packages/ui/src/components/Common/QuickIngest/ItemMetadataTable.tsx`
  - Per-item title/speaker/date/track/tag overrides and selection.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`
  - Persist playlist/batch/collection/run state in wizard/session state.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`
  - Show conference metadata and selected item count before submission.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`
  - Show durable/degraded mode, collection/run counts, cancel, retry-all, and export-failed affordances as stages become available.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`
  - Group results and provide collection handoff.
- Test: existing Quick Ingest tests under `apps/packages/ui/src/components/Common/QuickIngest/__tests__/`.

### Media/collection review and Knowledge QA UI

- Read first: `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts`
  - Current local collection key: `media:collections:v1`.
- Modify or create in Review area after route inventory:
  - Collection route/page component for conference collection detail.
  - Talk list, status badges, next/previous navigation, compare selected, scoped QA CTA.
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
  - Accept collection/media-ID scope only after backend support is present.
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/` or closest existing KnowledgeQA test folder.
- Test: `apps/packages/ui/src/components/Review/__tests__/conference-collection-review.test.tsx`

### Extension handoff

- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
  - Add or refine playlist-aware quick action when active-tab context exists.
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - Pass playlist URL/context into the shared Quick Ingest modal via `requestQuickIngestOpen`.
- Modify: `apps/packages/ui/src/utils/quick-ingest-open.ts`
  - Add typed `detail` payload for playlist preflight seed.
- Modify: `apps/packages/ui/src/entries/background.ts`
  - Add active-tab URL/context handoff if not already available.
- Test: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/sidepanel-chat.*.test.tsx`
- Test: extension/WXT tests as available under `apps/tldw-frontend/extension/__tests__` or `apps/extension`.

### Browser and e2e QA

- Modify/add: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Modify/add: `tldw_Server_API/tests/frontend_e2e/test_quick_ingest_media_workflow.py`
- Add fixtures: mocked 34-item playlist metadata, mocked jobs event stream, duplicate/failure permutations.

## Task 0: Contract Inventory And Slice Boundaries

**Files:**
- Create: `Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md`
- Read: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Read: `tldw_Server_API/app/api/v1/endpoints/items.py`
- Read: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Read: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Read: `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts`
- Read: `apps/packages/ui/src/services/tldw/domains/collections.ts`
- Read: `apps/packages/ui/src/services/tldw/domains/media.ts`

- [ ] **Step 1: Write the inventory artifact**

Create `Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md` with:

```markdown
# Bulk Conference Contract Inventory

Date: 2026-05-16
Backlog: TASK-399

## Candidate Stores

| Candidate | Supports stable collection ID | Supports ordered membership | Supports planned items | Supports media resolution | Collision risk | Decision |
|---|---:|---:|---:|---:|---|---|

## Selected Contract

## Rejected Alternatives

## API Placement

## Migration/Bridge Notes
```

- [ ] **Step 2: Verify the artifact names a selected source of truth**

Run:

```bash
rg -n "Selected Contract|Rejected Alternatives|API Placement" Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md
```

Expected: all headings present, with a concrete selection before Task 2 starts.

- [ ] **Step 3: Commit**

```bash
git add Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md
git commit -m "docs: inventory bulk conference collection contracts"
```

## Task 1: Playlist Preflight And Basic Dedupe

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/media_playlist_preflight.py`
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/media.ts`
- Create: `apps/packages/ui/src/services/tldw/playlist-preflight.ts`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
- Create: `apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py`
- Test: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
- Test: `apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts`

- [ ] **Step 1: Write failing backend classification tests**

Add tests like:

```python
def test_youtube_watch_list_url_detects_playlist_context():
    parsed = classify_playlist_url(
        "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
    )

    assert parsed.source_kind == "youtube_watch_playlist"
    assert parsed.playlist_id == "PL0065D9B288E6804B"
    assert parsed.video_id == "PrNmmN6qBiw"
```

Also test:

```python
def test_preflight_duplicate_in_batch_uses_normalized_source_id():
    items = normalize_preflight_items([
        {"source_url": "https://youtu.be/abc123", "title": "A"},
        {"source_url": "https://www.youtube.com/watch?v=abc123", "title": "A duplicate"},
    ])

    assert [item.duplicate_status for item in items] == [
        "new",
        "duplicate_in_batch",
    ]
```

- [ ] **Step 2: Run failing backend tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  -v
```

Expected: FAIL because the new module does not exist.

- [ ] **Step 3: Implement pure preflight helpers**

Implement `playlist_preflight.py` with small pure functions first:

```python
@dataclass(frozen=True)
class PlaylistUrlClassification:
    source_kind: str
    playlist_id: str | None
    video_id: str | None
    normalized_source_id: str | None


def classify_playlist_url(raw_url: str) -> PlaylistUrlClassification:
    return _classify_youtube_playlist_url(raw_url)


def normalize_preflight_items(raw_items: list[dict[str, Any]]) -> list[PlaylistPreflightItemData]:
    return _mark_duplicate_source_ids(_coerce_preflight_items(raw_items))
```

Keep yt-dlp access behind an injectable function so tests can mock metadata extraction without network.

- [ ] **Step 4: Write failing endpoint read-only tests**

Use dependency overrides and a fake extractor. Assert:

```python
def test_playlist_preflight_does_not_create_jobs_or_media(client, monkeypatch):
    response = client.post(
        "/api/v1/media/playlists/preflight",
        json={"url": "https://www.youtube.com/playlist?list=PLx"},
    )

    assert response.status_code == 200
    assert response.json()["item_count"] == 2
    assert fake_job_manager.created_jobs == []
    assert fake_media_db.created_rows == []
```

Add explicit tests for partial/timeout/unsupported URL responses.

- [ ] **Step 5: Implement endpoint and schemas**

Add request/response models:

```python
class PlaylistPreflightRequest(BaseModel):
    url: HttpUrl
    max_items: int = Field(default=100, ge=1, le=500)
    timeout_seconds: int = Field(default=20, ge=1, le=60)


class PlaylistPreflightItem(BaseModel):
    source_url: str
    normalized_source_id: str | None = None
    position: int | None = None
    title: str | None = None
    duration_seconds: int | None = None
    thumbnail_url: str | None = None
    channel_or_uploader: str | None = None
    published_at: str | None = None
    existing_media_id: int | None = None
    duplicate_status: Literal["new", "existing", "duplicate_in_batch", "unknown"]
    selected: bool = True
    warnings: list[str] = Field(default_factory=list)
```

Wire the router in `media/__init__.py`.

- [ ] **Step 6: Add granular capabilities**

In `config_info.py`, expose at least:

```python
caps["hasMediaPlaylistPreflight"] = bool(config_mod.route_enabled("media", default_stable=True))
caps["hasMediaIngestJobs"] = bool(config_mod.route_enabled("media", default_stable=True))
caps["hasMediaIngestJobEvents"] = bool(config_mod.route_enabled("media", default_stable=True))
caps["hasMediaIngestWorker"] = worker_path_enabled(
    "MEDIA_INGEST_JOBS_WORKER_ENABLED",
    "media-ingest-jobs",
    default_stable=False,
    test_mode=False,
)
```

In `server-capabilities.ts`, add matching fields, fallback spec paths, OpenAPI detection, and docs-info gate application.

- [ ] **Step 7: Add frontend normalizer and preflight client**

Add `preflightPlaylist` to `domains/media.ts` and normalize payloads in `playlist-preflight.ts`:

```ts
export const normalizePlaylistPreflight = (payload: unknown): PlaylistPreflight => ({
  preflightId: String((payload as any)?.preflight_id || ""),
  sourceUrl: String((payload as any)?.source_url || ""),
  status: normalizePreflightStatus((payload as any)?.status),
  items: Array.isArray((payload as any)?.items)
    ? (payload as any).items.map(normalizePlaylistPreflightItem)
    : [],
  warnings: normalizeStringList((payload as any)?.warnings)
})
```

- [ ] **Step 8: Add Quick Ingest preflight UI**

In `AddContentStep.tsx`, when a URL is playlist-capable and `hasMediaPlaylistPreflight` is true, call preflight and render `PlaylistPreflightPanel`. The panel must support loading, ready, partial, failed, and expired states, plus deselection.

- [ ] **Step 9: Run focused verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py \
  -v

bunx vitest run \
  apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts

git diff --check
```

Expected: all focused tests pass; diff check has no whitespace errors.

- [ ] **Step 10: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py \
     tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py \
  -f json -o /tmp/bandit_bulk_conference_playlist_preflight.json
```

Expected: no new high/medium findings in touched code.

- [ ] **Step 11: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/media_playlist_preflight.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py \
  tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py \
  tldw_Server_API/app/api/v1/endpoints/media/__init__.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py \
  apps/packages/ui/src/services/tldw/server-capabilities.ts \
  apps/packages/ui/src/services/tldw/domains/media.ts \
  apps/packages/ui/src/services/tldw/playlist-preflight.ts \
  apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/types.ts \
  apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts
git commit -m "feat: add playlist preflight for quick ingest"
```

## Task 2: Durable Conference Collection Contract

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Create if selected by inventory: `tldw_Server_API/app/api/v1/schemas/media_collections.py`
- Create if selected by inventory: `tldw_Server_API/app/api/v1/endpoints/media/collections.py`
- Modify if using media subrouter: `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
- Modify if extending unified items: `tldw_Server_API/app/api/v1/endpoints/items.py`
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/media.ts`
- Create: `apps/packages/ui/src/services/tldw/conference-collections.ts`
- Test: `tldw_Server_API/tests/Collections/test_conference_media_collections.py`
- Test: `apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts`

- [ ] **Step 1: Write failing collection contract tests**

Start from the selected contract in Task 0. Minimum backend tests:

```python
def test_conference_collection_persists_planned_and_resolved_items(collections_db):
    collection = collections_db.create_media_collection(
        name="PyCon 2026",
        kind="conference",
        metadata={"conference_name": "PyCon", "event_year": "2026"},
    )
    planned = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url="https://www.youtube.com/watch?v=a",
        normalized_source_id="youtube:a",
        status="planned",
        metadata={"speaker": "Ada"},
    )

    collections_db.resolve_media_collection_item(planned.id, media_id=123, status="completed")

    loaded = collections_db.get_media_collection(collection.id)
    assert loaded.items[0].media_id == 123
    assert loaded.items[0].metadata["speaker"] == "Ada"
```

Add tests that planned/source items do not overwrite unrelated `content_items` with matching URL/hash unless the selected contract intentionally reuses those rows with a safe namespace.

- [ ] **Step 2: Run failing collection tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Collections/test_conference_media_collections.py \
  -v
```

Expected: FAIL due missing collection APIs/storage helpers.

- [ ] **Step 3: Implement the selected durable storage contract**

If reusing `CollectionsDatabase`, prefer narrow methods rather than exposing raw SQL:

```python
def create_media_collection(
    self,
    *,
    name: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> MediaCollectionRow:
    """Create a stable, user-owned media collection."""


def add_media_collection_item(
    self,
    *,
    collection_id: int,
    source_url: str,
    normalized_source_id: str | None,
    status: str,
    position: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MediaCollectionItemRow:
    """Add an unresolved planned/source item to a media collection."""


def update_media_collection_item_status(
    self,
    item_id: int,
    *,
    status: str,
    error_summary: str | None = None,
) -> MediaCollectionItemRow:
    """Update processing status without losing source metadata."""


def resolve_media_collection_item(
    self,
    item_id: int,
    *,
    media_id: int,
    status: str = "completed",
) -> MediaCollectionItemRow:
    """Resolve a planned item to an existing or newly created media row."""


def get_media_collection(self, collection_id: int) -> MediaCollectionRow:
    """Return collection metadata plus ordered membership."""


def list_media_collections(
    self,
    *,
    kind: str | None = None,
    page: int = 1,
    size: int = 20,
) -> tuple[list[MediaCollectionRow], int]:
    """List durable media collections for the current user."""
```

Supported statuses:

```python
CONFERENCE_ITEM_STATUSES = {
    "planned",
    "processing",
    "completed",
    "skipped_existing",
    "submit_failed",
    "failed",
    "cancelled",
}
```

- [ ] **Step 4: Implement API and frontend client**

Expose stable collection operations. If using `/api/v1/media/collections`, add:

```text
POST /api/v1/media/collections
GET /api/v1/media/collections
GET /api/v1/media/collections/{collection_id}
PATCH /api/v1/media/collections/{collection_id}
POST /api/v1/media/collections/{collection_id}/items
PATCH /api/v1/media/collections/{collection_id}/items/{item_id}
```

Add typed client wrappers and normalize collection item status/counts in `conference-collections.ts`.

- [ ] **Step 5: Preserve localStorage collection boundary**

Read `useMediaSelection.ts` and make one explicit choice:

- migrate existing local collections to durable collections,
- bridge by showing local collections separately,
- or leave local collections as local-only and label them as such.

Record the choice in the inventory artifact.

- [ ] **Step 6: Run focused verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Collections/test_conference_media_collections.py \
  -v

bunx vitest run \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts \
  apps/packages/ui/src/services/__tests__/server-capabilities.test.ts

git diff --check
```

- [ ] **Step 7: Run Bandit**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/core/DB_Management/Collections_DB.py \
     tldw_Server_API/app/api/v1/endpoints/media/collections.py \
  -f json -o /tmp/bandit_bulk_conference_collections.json
```

Expected: no new high/medium findings in touched code.

- [ ] **Step 8: Commit**

```bash
git add \
  Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md \
  tldw_Server_API/app/core/DB_Management/Collections_DB.py \
  tldw_Server_API/app/api/v1/schemas/media_collections.py \
  tldw_Server_API/app/api/v1/endpoints/media/collections.py \
  tldw_Server_API/app/api/v1/endpoints/media/__init__.py \
  tldw_Server_API/tests/Collections/test_conference_media_collections.py \
  apps/packages/ui/src/services/tldw/server-capabilities.ts \
  apps/packages/ui/src/services/tldw/domains/media.ts \
  apps/packages/ui/src/services/tldw/conference-collections.ts \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts
git commit -m "feat: add durable conference media collections"
```

## Task 3: Batch Metadata In Quick Ingest

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Create: `apps/packages/ui/src/components/Common/QuickIngest/BatchMetadataPanel.tsx`
- Create: `apps/packages/ui/src/components/Common/QuickIngest/ItemMetadataTable.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`
- Modify: `apps/packages/ui/src/services/tldw/conference-collections.ts`
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

- [ ] **Step 1: Write failing UI tests for inherited metadata**

Add a Quick Ingest integration test:

```tsx
await user.type(screen.getByLabelText(/Conference name/i), "Strange Loop")
await user.type(screen.getByLabelText(/Event year/i), "2012")
await user.type(screen.getByLabelText(/Shared tags/i), "conference, clojure")

await user.click(screen.getByRole("button", { name: /Review/i }))

expect(screen.getByText(/Strange Loop/i)).toBeInTheDocument()
expect(screen.getByText(/34 selected/i)).toBeInTheDocument()
```

Add an item override test for one speaker/title.

- [ ] **Step 2: Run failing UI tests**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
```

Expected: FAIL because batch metadata controls and persisted session fields do not exist.

- [ ] **Step 3: Add metadata types and helpers**

Add types:

```ts
export type ConferenceBatchMetadata = {
  collectionName: string
  conferenceName?: string
  eventDate?: string
  eventYear?: string
  sharedTags: string[]
  sourcePlaylistUrl?: string
}

export type ConferenceItemMetadataOverride = {
  title?: string
  speaker?: string
  talkDate?: string
  track?: string
  tags?: string[]
  selected: boolean
}
```

Add pure merge helpers in `conference-collections.ts`.

- [ ] **Step 4: Implement batch metadata panel and item table**

Use compact forms and progressive disclosure. Batch fields are visible; per-item overrides live in a table/drawer and are optional.

- [ ] **Step 5: Submit metadata before or atomically with jobs**

Update quick-ingest submission so selected preflight items create planned collection items before jobs are submitted. If the API supports atomic create+submit, use that. Otherwise:

1. Create collection.
2. Create planned selected items.
3. Submit jobs with planned item IDs/idempotency keys.
4. Mark item `submit_failed` if submission fails.

- [ ] **Step 6: Mirror extension runtime payload**

Apply the same fields in `apps/packages/ui/src/entries/background.ts` so extension-runtime Quick Ingest does not drop collection metadata.

- [ ] **Step 7: Run verification**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts \
  apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts

git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add \
  apps/packages/ui/src/components/Common/QuickIngest/types.ts \
  apps/packages/ui/src/components/Common/QuickIngest/BatchMetadataPanel.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/ItemMetadataTable.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx \
  apps/packages/ui/src/services/tldw/conference-collections.ts \
  apps/packages/ui/src/services/tldw/quick-ingest-batch.ts \
  apps/packages/ui/src/entries/background.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
git commit -m "feat: add conference metadata to quick ingest"
```

## Task 4: Jobs-Backed Ingest Run Tracking

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `apps/packages/ui/src/services/tldw/ingest-jobs-orchestrator.ts`
- Modify: `apps/packages/ui/src/services/tldw/ingest-job-results.ts`
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
- Modify: `apps/packages/ui/src/store/quick-ingest-session.ts`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

- [ ] **Step 1: Write failing backend tests for planned item binding**

Test that job payload/status includes planned item ID:

```python
def test_submit_jobs_preserves_collection_item_binding(client):
    response = client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": ["https://www.youtube.com/watch?v=a"],
            "collection_id": "conf_1",
            "planned_item_ids": json.dumps(["item_1"]),
            "idempotency_keys": json.dumps(["conf_1:item_1:0"]),
        },
    )

    job = response.json()["jobs"][0]
    assert job["status"] == "queued"
```

Then fetch job and assert payload-derived `collection_id` and `planned_item_id` appear in status.

- [ ] **Step 2: Run failing backend tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  -v
```

- [ ] **Step 3: Extend job payload contract**

Accept optional collection/run form fields in `AddMediaForm` or endpoint-specific parsing, validate lengths match URL count, and write:

```python
payload.update({
    "collection_id": collection_id,
    "planned_item_id": planned_item_id,
    "idempotency_key": idempotency_key,
})
```

Do not put secrets or cookies in the job payload.

- [ ] **Step 4: Resolve item statuses in worker**

On terminal result:

- `completed` with media ID -> collection item `completed`
- completed duplicate/skipped existing -> `skipped_existing` if user opted in
- exception/failure -> `failed`
- cancellation -> `cancelled`

For job creation failures in the submit endpoint, mark planned item `submit_failed` with source URL/error.

- [ ] **Step 5: Update frontend tracking**

Extend `PersistedQuickIngestTracking` with collection/run IDs and planned item mapping. Ensure refresh restore can show:

- queued/running/completed/failed/cancelled counts
- durable mode vs synchronous fallback
- retry all retryable failures
- export failed URLs

- [ ] **Step 6: Run verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py \
  -v

bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts

git diff --check
```

- [ ] **Step 7: Bandit**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
     tldw_Server_API/app/services/media_ingest_jobs_worker.py \
     tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_bulk_conference_jobs.json
```

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  apps/packages/ui/src/services/tldw/ingest-jobs-orchestrator.ts \
  apps/packages/ui/src/services/tldw/ingest-job-results.ts \
  apps/packages/ui/src/services/tldw/quick-ingest-batch.ts \
  apps/packages/ui/src/store/quick-ingest-session.ts \
  apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
git commit -m "feat: track conference ingest runs through media jobs"
```

## Task 5: Results And Collection Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ResultsListItem.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Modify: `apps/packages/ui/src/services/tldw/conference-collections.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`

- [ ] **Step 1: Write failing grouped-results test**

```tsx
render(<WizardResultsStep results={[
  completedResult,
  skippedExistingResult,
  submitFailedResult,
  failedResult,
  cancelledResult,
]} collectionId="conf_1" />)

expect(screen.getByText(/Succeeded/i)).toBeInTheDocument()
expect(screen.getByText(/Skipped existing/i)).toBeInTheDocument()
expect(screen.getByText(/Submit failed/i)).toBeInTheDocument()
expect(screen.getByRole("button", { name: /Open collection/i })).toBeEnabled()
```

- [ ] **Step 2: Run failing UI tests**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
```

- [ ] **Step 3: Implement grouped result model and CTAs**

Primary CTA: open conference collection.

Secondary CTAs:

- retry all retryable failures
- export failed URLs
- review failed items
- ingest more
- ask this collection only when `hasKnowledgeQaMediaScope` is true and collection readiness is nonzero

- [ ] **Step 4: Separate submit failures from processing failures**

Use distinct outcome/status copy:

```ts
const RESULT_GROUP_LABELS = {
  submit_failed: "Not submitted",
  failed: "Failed during processing",
}
```

Export should include source URL, title, collection item ID, status, error summary, and retry attempt.

- [ ] **Step 5: Run verification**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts

git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add \
  apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/ResultsListItem.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/types.ts \
  apps/packages/ui/src/services/tldw/conference-collections.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
git commit -m "feat: hand off bulk ingest results to collections"
```

## Task 6: Conference Collection Review And Scoped QA

**Files:**
- Modify/create after route inventory: `apps/packages/ui/src/components/Review/ConferenceCollectionReview.tsx`
- Modify/create after route inventory: `apps/packages/ui/src/components/Review/__tests__/conference-collection-review.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Modify backend RAG files identified by current scope support inventory.
- Test: `tldw_Server_API/tests/RAG/test_conference_collection_scope.py`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/conference-scope.test.tsx`

- [ ] **Step 1: Inventory current Knowledge QA/RAG scope contract**

Search:

```bash
rg -n "media_ids|source_ids|filters|selection|rag/search|KnowledgeQA" \
  tldw_Server_API/app/api/v1/endpoints \
  tldw_Server_API/app/core/RAG \
  apps/packages/ui/src/components/Option/KnowledgeQA \
  apps/packages/ui/src/services/tldw
```

Record the actual contract in Task 0's inventory or a new short note before modifying RAG.

- [ ] **Step 2: Write failing backend scoped retrieval test**

```python
def test_conference_collection_scope_limits_rag_to_ready_media(client, seeded_collection):
    response = client.post(
        "/api/v1/rag/search",
        json={
            "query": "What did the keynote say about macros?",
            "collection_id": seeded_collection.id,
        },
    )

    assert response.status_code == 200
    assert {hit["media_id"] for hit in response.json()["results"]} <= seeded_collection.ready_media_ids
```

- [ ] **Step 3: Implement or reuse backend-enforced scope**

If existing RAG selection filters support media IDs, map collection ID to ready media IDs server-side. If not, add a minimal request field and retrieval filter. Do not rely on client-only filtering.

- [ ] **Step 4: Write failing review UI test**

Assert talk list, transcript readiness counts, next/previous navigation, compare selected, and disabled QA when no items are ready.

- [ ] **Step 5: Implement collection review UI**

Minimum V1:

- ordered talk list
- status badges for planned/processing/completed/skipped/submit_failed/failed/cancelled
- transcript/summary readiness
- previous/next talk navigation
- selected-talk comparison using metadata plus available summaries/excerpts
- scoped QA CTA with readiness copy

- [ ] **Step 6: Run verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/RAG/test_conference_collection_scope.py \
  -v

bunx vitest run \
  apps/packages/ui/src/components/Review/__tests__/conference-collection-review.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/conference-scope.test.tsx

git diff --check
```

- [ ] **Step 7: Bandit**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/api/v1/endpoints \
     tldw_Server_API/app/core/RAG \
  -f json -o /tmp/bandit_bulk_conference_scoped_rag.json
```

Review only findings in touched RAG/API files.

- [ ] **Step 8: Commit**

```bash
git add \
  tldw_Server_API/tests/RAG/test_conference_collection_scope.py \
  apps/packages/ui/src/components/Review/ConferenceCollectionReview.tsx \
  apps/packages/ui/src/components/Review/__tests__/conference-collection-review.test.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/conference-scope.test.tsx \
  apps/packages/ui/src/services/tldw/domains/chat-rag.ts
git commit -m "feat: review conference collections with scoped qa"
```

## Task 7: Extension Playlist Capture

**Files:**
- Modify: `apps/packages/ui/src/utils/quick-ingest-open.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Test: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts`
- Test as available: `apps/tldw-frontend/extension/__tests__/*`

- [ ] **Step 1: Write failing context-handoff tests**

Add a test that `requestQuickIngestOpen` accepts:

```ts
requestQuickIngestOpen({
  source: "extension_active_tab",
  url: "https://www.youtube.com/watch?v=a&list=PLx",
  sourceKind: "youtube_watch_playlist",
  action: "playlist_preflight",
})
```

And that Sidepanel passes this detail to the modal/open request.

- [ ] **Step 2: Run failing tests**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx \
  apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts
```

- [ ] **Step 3: Add typed Quick Ingest open detail**

In `quick-ingest-open.ts`:

```ts
export type QuickIngestOpenDetail =
  | { source: "manual"; action?: "normal" }
  | {
      source: "extension_active_tab"
      url: string
      sourceKind?: "youtube_playlist" | "youtube_watch_playlist" | "unknown"
      action: "playlist_preflight"
    }
```

- [ ] **Step 4: Detect active-tab playlist context**

Use active-tab URL or existing sidepanel context, not content-script parsing. Show "Import playlist to tldw" only when the URL has a YouTube playlist/list context and extension readiness allows it.

- [ ] **Step 5: Seed shared preflight**

Quick Ingest should consume the open detail and start the same preflight path as paste-from-WebUI.

- [ ] **Step 6: Run verification**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx

git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add \
  apps/packages/ui/src/utils/quick-ingest-open.ts \
  apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/form.tsx \
  apps/packages/ui/src/entries/background.ts \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx
git commit -m "feat: add extension playlist quick ingest handoff"
```

## Task 8: Duplicate And Failure Recovery

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `apps/packages/ui/src/services/tldw/ingest-job-results.ts`
- Modify: `apps/packages/ui/src/services/tldw/conference-collections.ts`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Test: `apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx`

- [ ] **Step 1: Write failing duplicate policy tests**

Policies:

- skip
- overwrite
- update metadata only
- include existing in collection

Assert each policy produces expected planned item status and job submission behavior.

- [ ] **Step 2: Write failing failure taxonomy tests**

Classify conservative failure types:

```ts
expect(classifyConferenceIngestFailure("Private video")).toBe("auth_required")
expect(classifyConferenceIngestFailure("HTTP Error 404")).toBe("unavailable")
expect(classifyConferenceIngestFailure("timed out")).toBe("timeout")
```

- [ ] **Step 3: Implement duplicate policy UI and payloads**

Expose policy choice in preflight/results only when duplicates exist. Default should avoid surprise overwrite.

- [ ] **Step 4: Implement selected-subset retry**

Retry selected applies only to retryable `submit_failed`, `failed`, or `cancelled` items. It must use collection item ID plus retry attempt/idempotency key and must skip completed items.

- [ ] **Step 5: Run verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  -v

bunx vitest run \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx

git diff --check
```

- [ ] **Step 6: Bandit**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py \
     tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
     tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  -f json -o /tmp/bandit_bulk_conference_recovery.json
```

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  apps/packages/ui/src/services/tldw/ingest-job-results.ts \
  apps/packages/ui/src/services/tldw/conference-collections.ts \
  apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx
git commit -m "feat: recover duplicate and failed playlist items"
```

## Task 9: Notifications, Full-Path QA, And Documentation

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx`
- Modify/add: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Modify/add: `tldw_Server_API/tests/frontend_e2e/test_quick_ingest_media_workflow.py`
- Create: `Docs/User_Guides/Bulk_Conference_Playlist_Ingest.md`
- Modify if needed: `Docs/API-related/Media_Ingest_Jobs_API.md`
- Test fixtures as needed under existing frontend/backend test fixture directories.

- [ ] **Step 1: Add mocked 34-item playlist fixture**

Create a deterministic fixture with 34 metadata-only items, duplicates, and failure permutations. Do not depend on real YouTube or downloads in automated tests.

- [ ] **Step 2: Write full-path WebUI test**

Test:

1. paste playlist URL
2. preflight expands to 34 items
3. deselect one item
4. set conference metadata once
5. submit mocked jobs
6. receive mixed mocked events
7. open collection
8. see readiness counts

- [ ] **Step 3: Write extension handoff test**

Assert active-tab playlist context opens the same preflight state as WebUI paste.

- [ ] **Step 4: Add completion notification**

Use existing WebUI/extension notification/message patterns. Notification should include collection name and mixed success counts; it should not claim all searchable until readiness counts confirm it.

- [ ] **Step 5: Write user documentation**

Document:

- playlist preflight
- conference metadata
- durable vs degraded mode
- failure export/retry
- collection review
- scoped Knowledge QA readiness

- [ ] **Step 6: Run full verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py \
  tldw_Server_API/tests/Collections/test_conference_media_collections.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/RAG/test_conference_collection_scope.py \
  -v

bunx vitest run \
  apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts \
  apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx

npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts

git diff --check
```

- [ ] **Step 7: Final Bandit touched backend sweep**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/api/v1/endpoints/media \
     tldw_Server_API/app/core/Ingestion_Media_Processing/Video \
     tldw_Server_API/app/services/media_ingest_jobs_worker.py \
     tldw_Server_API/app/core/DB_Management/Collections_DB.py \
  -f json -o /tmp/bandit_bulk_conference_final.json
```

- [ ] **Step 8: Commit**

```bash
git add \
  apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx \
  apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts \
  tldw_Server_API/tests/frontend_e2e/test_quick_ingest_media_workflow.py \
  Docs/User_Guides/Bulk_Conference_Playlist_Ingest.md \
  Docs/API-related/Media_Ingest_Jobs_API.md
git commit -m "test: verify bulk conference ingest workflow"
```

## Cross-Stage Review Checklist

- [ ] Preflight remains read-only: no jobs, no media rows, no collection mutation.
- [ ] Server capabilities distinguish endpoint presence, worker availability, SSE, durable collection support, playlist preflight, and scoped QA.
- [ ] Collection identity is stable and not tag-only.
- [ ] Planned, processing, completed, skipped_existing, `submit_failed`, failed, and cancelled item states are represented.
- [ ] Submit failures keep source URL, metadata, error, and export/retry path.
- [ ] Retry is idempotent by collection item and retry attempt.
- [ ] Synchronous fallback preserves metadata and clearly communicates weaker recovery.
- [ ] Extension capture hands off to shared preflight and does not duplicate playlist parsing.
- [ ] Scoped QA is backend-enforced and shows ready/not-ready counts.
- [ ] One-off Quick Ingest remains unchanged.

## Execution Notes

- The writing-plans skill recommends an external plan-document-reviewer subagent. In this environment, subagents are only allowed when the user explicitly asks for delegation, so this plan should receive a local self-review or a user-requested delegated review before implementation.
- Before implementation, update `TASK-399` or create follow-on Backlog tasks per stage so code edits are tracked according to the repo instructions.
- If a stage exceeds the file scope above, split it before coding rather than broadening the PR.
