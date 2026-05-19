# Browser Extension Web Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a V1 Evernote-style browser web clipper that captures page/article/text/screenshot content, files it to a canonical note with optional workspace placement, and supports safe OCR/VLM enrichment plus analyze-now handoff.

**Architecture:** Keep the extension responsible for capture and review UX, but move canonical save semantics into a focused backend `web_clipper` service so idempotency, attachment naming, workspace placement, and conflict-safe enrichment rules live in one place. Persist clip-specific structured state in a note-sidecar table plus a workspace-placement table, then expose a typed extension client and a dedicated sidepanel clipper route that reuses the existing background, notes, screenshot, and sidepanel infrastructure.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes SQLite/PostgreSQL DB layer, existing Notes/Workspaces APIs, React, TypeScript, WXT browser extension runtime, TanStack Query, Ant Design, Vitest, React Testing Library, pytest, Playwright where practical, Bandit

---

## Scope Check

This spec spans backend persistence, extension capture, and sidepanel filing UX, but they are one coherent feature rather than independent subprojects. The plan keeps them in one implementation track because nothing user-visible ships correctly without all three layers.

## File Structure

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: add clipper sidecar tables, idempotent placement helpers, deterministic attachment naming helpers, and cleanup behavior tied to canonical note lifecycle.
- `tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py`
  Purpose: define typed request/response models for capture bundles, save results, placements, enrichment payloads, and save outcome states.
- `tldw_Server_API/app/core/WebClipper/service.py`
  Purpose: implement the canonical save saga, stage idempotency, workspace placement creation, content-budget truncation, and conflict-safe enrichment persistence.
- `tldw_Server_API/app/api/v1/endpoints/web_clipper.py`
  Purpose: expose the clipper save/status/enrichment routes without bloating the already large `notes.py` surface.
- `tldw_Server_API/app/main.py`
  Purpose: register the new clipper router behind the API v1 prefix.
- `tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py`
  Purpose: verify sidecar schema, placement persistence, idempotent retries, and delete/cleanup rules.
- `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
  Purpose: verify staged save semantics, content-budget truncation, idempotent behavior, and enrichment version-guard logic without HTTP.
- `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
  Purpose: verify save, retry, workspace placement, and enrichment flows through the real API.
- `tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py`
  Purpose: extend coverage for the clipper’s workspace note placement assumptions if the API contract needs adjustment.
- `apps/packages/ui/src/services/tldw/domains/web-clipper.ts`
  Purpose: define typed extension client methods for clipper save/status/enrichment operations.
- `apps/packages/ui/src/services/tldw/domains/index.ts`
  Purpose: export the new web clipper domain methods.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  Purpose: expose high-level `saveWebClip`, `getWebClipStatus`, and `persistWebClipEnrichment` helpers to the extension UI.
- `apps/packages/ui/src/services/web-clipper/types.ts`
  Purpose: centralize clip draft, capture adapter, save request, save result, and enrichment types on the frontend.
- `apps/packages/ui/src/services/web-clipper/draft-builder.ts`
  Purpose: normalize bookmark/article/full-page/selection/screenshot captures into one `ClipDraft` contract.
- `apps/packages/ui/src/services/web-clipper/content-extract.ts`
  Purpose: isolate Readability/main-content/full-page extraction plus fallback resolution for page-based clip types.
- `apps/packages/ui/src/services/web-clipper/pending-draft.ts`
  Purpose: manage pending clip draft handoff between background and sidepanel, similar to existing companion-capture storage patterns.
- `apps/packages/ui/src/services/web-clipper/save-runtime.ts`
  Purpose: orchestrate the review-sheet save call, status mapping, retry affordances, and open-after-save navigation.
- `apps/packages/ui/src/services/web-clipper/enrichment.ts`
  Purpose: run OCR/VLM follow-up calls, enforce inline budget rules client-side where useful, and persist enrichment results back through the clipper API.
- `apps/packages/ui/src/entries/web-clipper.content.ts`
  Purpose: provide DOM extraction, selection access, and visible-region overlay selection for the clipper on supported pages.
- `apps/packages/ui/src/entries/background.ts`
  Purpose: add clipper launch message handling, context menu hooks, and background-to-sidepanel routing.
- `apps/packages/ui/src/entries/shared/background-init.ts`
  Purpose: register the clipper context menu item and any capability-gated launch affordances.
- `apps/packages/ui/src/libs/get-screenshot.ts`
  Purpose: keep visible screenshot capture reusable and extend it only as needed for clipper draft generation.
- `apps/packages/ui/src/routes/sidepanel-clipper.tsx`
  Purpose: mount the dedicated clipper review route inside the sidepanel shell.
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
  Purpose: register the new sidepanel clipper route.
- `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
  Purpose: render the main review sheet and coordinate clip type selection, preview, filing controls, and save actions.
- `apps/packages/ui/src/components/Sidepanel/Clipper/ClipPreview.tsx`
  Purpose: render content/image previews and actual-fallback messaging.
- `apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx`
  Purpose: isolate note folder selection, workspace selection, and destination-specific validation.
- `apps/packages/ui/src/components/Sidepanel/Clipper/ClipEnhancementFields.tsx`
  Purpose: isolate OCR/VLM toggles, privacy disclosure, and pending-enrichment status.
- `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`
  Purpose: verify review-sheet validation, partial-success states, and open-after-save routing.
- `apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts`
  Purpose: verify frontend client request serialization and status mapping.
- `apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts`
  Purpose: verify deterministic draft normalization and capture fallback labeling.
- `apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts`
  Purpose: verify context menu launch, background routing, and restricted-page fallback behavior.
- `apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx`
  Purpose: verify route registration and draft hydration behavior.
- `apps/packages/ui/src/public/_locales/en/messages.json`
  Purpose: add English context-menu and launch feedback strings for clipper entry points.
- `apps/packages/ui/src/public/_locales/en/sidepanel.json`
  Purpose: add English review-sheet strings for the clipper route.
- `apps/packages/ui/src/assets/locale/en/sidepanel.json`
  Purpose: keep the source sidepanel locale bundle aligned with the runtime locale output.
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  Purpose: whitelist the new clipper API paths for frontend typed request helpers.
- `apps/packages/ui/src/services/tldw/server-capabilities.ts`
  Purpose: surface whether the connected server exposes the clipper API and degrade UI cleanly if it does not.

## Stages

### Stage 1: Backend Clipper Persistence And Save Contract

**Goal:** Land the sidecar storage, placement persistence, and backend save contract needed to make clip saves deterministic and idempotent.

**Success Criteria:** The API can create one canonical note per `clip_id`, dedupe retries, create zero-or-one workspace placement, and persist clip metadata without using free-form markdown as the source of truth.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -v`

**Status:** Complete

### Stage 2: Extension Capture Launchers And Draft Collection

**Goal:** Add clipper launch entry points and normalize capture output into one frontend draft model.

**Success Criteria:** Toolbar/context-menu/sidepanel entry can hydrate a `ClipDraft` for the confirmed clip types and label any fallback path actually used.

**Tests:** `bunx vitest run apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts`

**Status:** Not Started

### Stage 3: Sidepanel Review Sheet And Save/Open Flow

**Goal:** Add the Evernote-style review sheet inside the sidepanel and connect it to the new save API.

**Success Criteria:** Users can review/edit/title/tag/file a clip to `Note`, `Workspace`, or `Both`, see partial-success outcomes, and open the appropriate destination after save.

**Tests:** `bunx vitest run apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

**Status:** Not Started

### Stage 4: Enrichment And Analyze-Now

**Goal:** Add optional OCR/VLM enrichment plus analyze-now handoff without overwriting user-authored note content.

**Success Criteria:** OCR/VLM requests persist structured results, inline summaries respect the content budget, and version mismatches avoid destructive writeback.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py -k enrichment -v`, `bunx vitest run apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

**Status:** Not Started

### Stage 5: Verification, Security, And Handoff

**Goal:** Prove the feature works across backend + extension boundaries and does not introduce security regressions in touched backend code.

**Success Criteria:** Targeted backend/frontend test suites pass and Bandit reports no new findings in the touched Python scope.

**Tests:** `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/web_clipper.py tldw_Server_API/app/core/WebClipper/service.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_web_clipper.json`

**Status:** Not Started

## Task 1: Add Clipper Sidecar Storage And Backend Schemas

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py`

- [x] **Step 1: Write the failing DB and schema tests**

Add tests that prove:

1. the DB creates `note_clipper_documents`
2. the DB creates `note_clipper_workspace_placements`
3. one canonical clip row can be fetched by `clip_id` and by `note_id`
4. a `(clip_id, workspace_id)` retry upserts instead of duplicating placement rows
5. deleting the canonical note removes or invalidates clipper sidecar rows consistently

Use concrete assertions like:

```python
note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-123")
db.upsert_note_clipper_document(
    clip_id="clip-123",
    note_id=note_id,
    clip_type="article",
    source_url="https://example.com/story",
    source_title="Example Story",
    capture_metadata={"fallback_path": ["article"]},
    enrichments={"ocr": {"status": "pending"}},
)
db.upsert_note_clipper_workspace_placement(
    clip_id="clip-123",
    workspace_id="ws-1",
    workspace_note_id=42,
    source_note_id=note_id,
)

clip_doc = db.get_note_clipper_document_by_clip_id("clip-123")
placements = db.list_note_clipper_workspace_placements("clip-123")

assert clip_doc["note_id"] == note_id
assert placements == [{
    "clip_id": "clip-123",
    "workspace_id": "ws-1",
    "workspace_note_id": 42,
    "source_note_id": note_id,
}]
```

- [x] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py -v
```

Expected: FAIL because the clipper sidecar schema and helpers do not exist yet.

- [x] **Step 3: Add the minimal storage and typed schema support**

Implement the boring persistence layer first:

- add `note_clipper_documents` keyed by `clip_id` with `note_id`, `clip_type`, `source_url`, `source_title`, `capture_metadata_json`, `analysis_json`, `content_budget_json`, timestamps, and source note version fields
- add `note_clipper_workspace_placements` keyed by `clip_id + workspace_id`
- add helper methods such as:

```python
def upsert_note_clipper_document(..., conn=None) -> dict[str, Any]:
    ...

def get_note_clipper_document_by_clip_id(self, clip_id: str) -> dict[str, Any] | None:
    ...

def upsert_note_clipper_workspace_placement(..., conn=None) -> dict[str, Any]:
    ...
```

- keep this data in sidecar tables instead of broadening the plain `notes` table contract prematurely
- define typed Pydantic schemas for save requests/results, enrichment payloads, and outcome states in `web_clipper_schemas.py`

- [x] **Step 4: Re-run the targeted backend tests**

Run the pytest command from Step 2.

Expected: PASS for sidecar creation, placement upsert, and cleanup behavior.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py
git commit -m "feat: add web clipper sidecar storage"
```

## Task 2: Add The Backend Save Saga And Enrichment API

**Files:**
- Create: `tldw_Server_API/app/core/WebClipper/service.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/web_clipper.py`
- Modify: `tldw_Server_API/app/main.py`
- Test: `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`

- [x] **Step 1: Write the failing service and API tests**

Add tests that prove:

1. `POST /api/v1/web-clipper/save` creates one canonical note using `clip_id` as the note ID
2. retrying the same request reuses the note instead of creating a duplicate
3. attachment slots are deterministic and duplicate retries do not create duplicate attachment records
4. workspace placement creation is idempotent by `(clip_id, workspace_id)`
5. save outcomes are classified as `saved`, `saved_with_warnings`, `partially_saved`, or `failed`
6. enrichment persistence refuses destructive inline writeback when `source_note_version` is stale
7. visible-body truncation follows the numeric content budget from the spec

Use API-level assertions like:

```python
response = client.post("/api/v1/web-clipper/save", json={
    "clip_id": "clip-123",
    "clip_type": "article",
    "destination_mode": "both",
    "note": {"title": "Example Story", "folder_id": None, "keywords": ["example"]},
    "workspace": {"workspace_id": "ws-1"},
    "content": {"visible_body": "Alpha", "full_extract": "Alpha"},
    "attachments": [],
    "enhancements": {"run_ocr": False, "run_vlm": False},
})
assert response.status_code == 200
assert response.json()["status"] == "saved"
assert response.json()["note"]["id"] == "clip-123"
```

- [x] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -v
```

Expected: FAIL because the web clipper service and routes do not exist yet.

- [x] **Step 3: Implement the minimal backend orchestration**

Create a focused backend service rather than spreading save semantics across `notes.py` and frontend saga code:

- add `WebClipperService.save_clip(...)` that:
  - creates or reuses the canonical note via `db.add_note(..., note_id=clip_id)`
  - builds the visible note body using the spec content-budget thresholds
  - applies keyword/folder filing
  - stores or reuses attachments under deterministic clipper slot names
  - creates or reuses workspace placement when requested
  - persists sidecar state and stage results for retries
- add `WebClipperService.persist_enrichment(...)` that:
  - stores structured OCR/VLM output
  - writes to the machine-managed inline section only when `source_note_version` still matches
  - leaves structured metadata intact and reports a non-destructive conflict when the note version changed
- expose routes such as:

```python
@router.post("/save", response_model=WebClipperSaveResponse)
async def save_web_clip(...): ...

@router.get("/{clip_id}", response_model=WebClipperStatusResponse)
async def get_web_clip_status(...): ...

@router.post("/{clip_id}/enrichments", response_model=WebClipperEnrichmentResponse)
async def persist_web_clip_enrichment(...): ...
```

- keep route logic thin; put branching and idempotency in `service.py`

- [x] **Step 4: Re-run the targeted backend tests**

Run the pytest command from Step 2.

Expected: PASS for canonical note creation, retry convergence, placement idempotency, and version-safe enrichment persistence.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/WebClipper/service.py \
  tldw_Server_API/app/api/v1/endpoints/web_clipper.py \
  tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py
git commit -m "feat: add web clipper save api"
```

## Task 3: Add Frontend Client Methods And Save Runtime

**Files:**
- Create: `apps/packages/ui/src/services/tldw/domains/web-clipper.ts`
- Create: `apps/packages/ui/src/services/web-clipper/types.ts`
- Create: `apps/packages/ui/src/services/web-clipper/save-runtime.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Test: `apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts`

- [ ] **Step 1: Write the failing frontend client/runtime tests**

Add tests that prove:

1. the new client posts to `/api/v1/web-clipper/save`
2. status polling hits `/api/v1/web-clipper/{clip_id}`
3. enrichment persistence posts the expected payload including `source_note_version`
4. the runtime maps backend states to UI-friendly banners without losing warning details

Use concrete expectations like:

```ts
await tldwClient.saveWebClip({
  clip_id: "clip-123",
  clip_type: "article",
  destination_mode: "note",
})

expect(bgRequestMock).toHaveBeenCalledWith(
  expect.objectContaining({
    path: "/api/v1/web-clipper/save",
    method: "POST",
  })
)
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts
```

Expected: FAIL because the web clipper domain/client methods do not exist yet.

- [ ] **Step 3: Implement the typed client and runtime helpers**

Add focused helpers:

```ts
export interface WebClipperMethods {
  saveWebClip(payload: WebClipSaveRequest): Promise<WebClipSaveResponse>
  getWebClipStatus(clipId: string): Promise<WebClipStatusResponse>
  persistWebClipEnrichment(
    clipId: string,
    payload: WebClipEnrichmentRequest
  ): Promise<WebClipEnrichmentResponse>
}
```

and a runtime helper like:

```ts
export const classifyWebClipSave = (result: WebClipSaveResponse) => {
  if (result.status === "saved") return { tone: "success" as const }
  if (result.status === "saved_with_warnings") return { tone: "warning" as const }
  if (result.status === "partially_saved") return { tone: "warning" as const }
  return { tone: "error" as const }
}
```

Also:

- register the new paths in `openapi-guard.ts`
- expose a capability bit in `server-capabilities.ts`
- keep request/response typing in one frontend `types.ts` file rather than scattering `any`

- [ ] **Step 4: Re-run the targeted frontend tests**

Run the Vitest command from Step 2.

Expected: PASS for client request shape and status classification.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/domains/web-clipper.ts \
  apps/packages/ui/src/services/web-clipper/types.ts \
  apps/packages/ui/src/services/web-clipper/save-runtime.ts \
  apps/packages/ui/src/services/tldw/domains/index.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/services/tldw/server-capabilities.ts \
  apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts
git commit -m "feat: add web clipper client runtime"
```

## Task 4: Add Capture Adapters And Background Launchers

**Files:**
- Create: `apps/packages/ui/src/entries/web-clipper.content.ts`
- Create: `apps/packages/ui/src/services/web-clipper/content-extract.ts`
- Create: `apps/packages/ui/src/services/web-clipper/draft-builder.ts`
- Create: `apps/packages/ui/src/services/web-clipper/pending-draft.ts`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Modify: `apps/packages/ui/src/entries/shared/background-init.ts`
- Modify: `apps/packages/ui/src/libs/get-screenshot.ts`
- Modify: `apps/packages/ui/src/public/_locales/en/messages.json`
- Test: `apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts`
- Test: `apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts`

- [ ] **Step 1: Write the failing capture and background tests**

Add tests that prove:

1. the clipper context-menu item opens the sidepanel clipper route instead of the generic note-save route
2. bookmark/article/full-page/selection captures normalize into one `ClipDraft`
3. requested clip types record the actual fallback used
4. screenshot capture uses the existing visible-tab helper
5. restricted/internal pages fail with a user-visible explanation instead of a silent error

Use draft assertions like:

```ts
const draft = buildClipDraft({
  requestedType: "article",
  pageUrl: "https://example.com/story",
  pageTitle: "Story",
  extracted: {
    articleText: "",
    fullPageText: "Fallback body",
  },
})

expect(draft.clipType).toBe("article")
expect(draft.captureMetadata.fallbackPath).toEqual(["article", "full_page"])
expect(draft.visibleBody).toContain("Fallback body")
```

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts \
  apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts
```

Expected: FAIL because the draft builder, pending draft handoff, and clipper launch hooks do not exist yet.

- [ ] **Step 3: Implement the capture primitives and launch wiring**

Implement the capture layer in focused modules:

- `web-clipper.content.ts`
  - extract selection text
  - run article/full-page extraction
  - support visible-region selection in V1 only
- `content-extract.ts`
  - wrap Readability/main-content/full-page fallback logic
- `draft-builder.ts`
  - normalize all capture modes into one `ClipDraft`
- `pending-draft.ts`
  - persist the captured draft so the sidepanel route can pick it up
- extend `background-init.ts` and `background.ts`
  - add a dedicated clipper context menu item
  - route launch to the sidepanel clipper path
  - keep existing `save to notes` and `save to companion` flows intact

- [ ] **Step 4: Re-run the targeted frontend tests**

Run the Vitest command from Step 2.

Expected: PASS for launch routing, fallback labeling, and restricted-page behavior.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/entries/web-clipper.content.ts \
  apps/packages/ui/src/services/web-clipper/content-extract.ts \
  apps/packages/ui/src/services/web-clipper/draft-builder.ts \
  apps/packages/ui/src/services/web-clipper/pending-draft.ts \
  apps/packages/ui/src/entries/background.ts \
  apps/packages/ui/src/entries/shared/background-init.ts \
  apps/packages/ui/src/libs/get-screenshot.ts \
  apps/packages/ui/src/public/_locales/en/messages.json \
  apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts \
  apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts
git commit -m "feat: add web clipper capture launchers"
```

## Task 5: Add The Sidepanel Review Sheet And Save/Open UX

**Files:**
- Create: `apps/packages/ui/src/routes/sidepanel-clipper.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Clipper/ClipPreview.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Clipper/ClipEnhancementFields.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`
- Modify: `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- Test: `apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx`
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [ ] **Step 1: Write the failing route and review-sheet tests**

Add tests that prove:

1. the sidepanel clipper route hydrates the pending draft
2. the review sheet shows title/comment/tags/destination controls
3. `Workspace` and `Both` require a selected workspace before save
4. save-state banners reflect `saved`, `saved_with_warnings`, and `partially_saved`
5. `Save and open` routes to notes by default and to workspace when that is the only destination

Use UI assertions like:

```tsx
render(<SidepanelClipperRoute />)

expect(screen.getByLabelText("Title")).toHaveValue("Example Story")
expect(screen.getByRole("button", { name: "Save clip" })).toBeEnabled()
expect(screen.getByLabelText("Run OCR")).not.toBeChecked()
```

- [ ] **Step 2: Run the targeted route/UI tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Expected: FAIL because the sidepanel clipper route and review-sheet components do not exist yet.

- [ ] **Step 3: Implement the review sheet and save/open behavior**

Build a compact filing-first surface:

- `sidepanel-clipper.tsx`
  - fetches the pending draft
  - guards against missing drafts and unsupported capability states
- `WebClipperPanel.tsx`
  - owns form state, preview, save actions, and banner rendering
- `ClipDestinationFields.tsx`
  - validates folder/workspace inputs by destination mode
- `ClipEnhancementFields.tsx`
  - renders OCR/VLM toggles and privacy disclosure
- route registration
  - add `/clipper` to `sidepanel-route-registry.tsx`

Keep the note/workspace surfaces equal in the UI, but do not expose the internal canonical-note implementation detail.

- [ ] **Step 4: Re-run the targeted route/UI tests**

Run the Vitest command from Step 2.

Expected: PASS for route hydration, validation, and open-after-save routing.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/sidepanel-clipper.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/ClipPreview.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/ClipEnhancementFields.tsx \
  apps/packages/ui/src/routes/sidepanel-route-registry.tsx \
  apps/packages/ui/src/public/_locales/en/sidepanel.json \
  apps/packages/ui/src/assets/locale/en/sidepanel.json \
  apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
git commit -m "feat: add sidepanel web clipper review sheet"
```

## Task 6: Add OCR/VLM Enrichment And Analyze-Now Handoff

**Files:**
- Create: `apps/packages/ui/src/services/web-clipper/enrichment.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Modify: `apps/packages/ui/src/services/web-clipper/save-runtime.ts`
- Modify: `tldw_Server_API/app/core/WebClipper/service.py`
- Test: `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [ ] **Step 1: Write the failing enrichment/conflict tests**

Add tests that prove:

1. OCR/VLM persistence stores structured results even when inline note update is skipped
2. inline summaries respect the 12,000 / 1,500 / 1,000 / 2,500 character limits from the spec
3. a note version mismatch reports a non-destructive conflict instead of overwriting user edits
4. `Analyze now` hands the captured image/text context into the existing sidepanel chat/vision path

Use service-level assertions like:

```python
result = service.persist_enrichment(
    clip_id="clip-123",
    enrichment_type="vlm",
    source_note_version=1,
    inline_summary="A" * 5000,
    structured_payload={"raw": "B" * 9000},
)
assert result.inline_applied is True
assert len(result.inline_summary) <= 1000
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py -k enrichment -v
bunx vitest run apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Expected: FAIL because enrichment persistence and analyze-now handoff are not implemented yet.

- [ ] **Step 3: Implement the minimal safe enrichment flow**

Implement:

- `enrichment.ts`
  - call the chosen OCR/VLM backend path
  - send structured results back through `persistWebClipEnrichment`
- backend service
  - clamp inline summaries to the spec budget
  - update only the machine-managed section when note version still matches
  - return `inline_applied: false` plus conflict metadata on mismatch
- review sheet
  - surface pending/completed/conflict states without blocking the original save
- analyze-now
  - route the user into the existing sidepanel chat/vision surface with the clip’s screenshot plus page text context

- [ ] **Step 4: Re-run the targeted tests**

Run the commands from Step 2.

Expected: PASS for budget enforcement, conflict-safe writeback, and analyze-now handoff.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/web-clipper/enrichment.ts \
  apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx \
  apps/packages/ui/src/services/web-clipper/save-runtime.ts \
  tldw_Server_API/app/core/WebClipper/service.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py \
  apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
git commit -m "feat: add web clipper enrichment flow"
```

## Task 7: Verification, Security, And Release Readiness

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Modify: `apps/packages/ui/src/entries/shared/background-init.ts`
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
- Test: `apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts`
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [ ] **Step 1: Run the full targeted backend and frontend suites**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py -v

bunx vitest run apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts \
  apps/packages/ui/src/services/web-clipper/__tests__/draft-builder.test.ts \
  apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts \
  apps/packages/ui/src/routes/__tests__/sidepanel-clipper.test.tsx \
  apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on the touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/web_clipper.py \
  tldw_Server_API/app/core/WebClipper/service.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_web_clipper.json
```

Expected: zero new findings in the touched scope.

- [ ] **Step 3: Fix any final drift in capability guards and copy**

Before calling the feature done:

- confirm the UI hides or disables clipper entry when `/api/v1/web-clipper/save` is not advertised
- confirm restricted-page messaging is localized and user-readable
- confirm `Save and open` still works when only the canonical note was created

- [ ] **Step 4: Commit**

```bash
git add apps/packages/ui/src/services/tldw/server-capabilities.ts \
  apps/packages/ui/src/entries/shared/background-init.ts
git commit -m "chore: finalize web clipper rollout guards"
```
