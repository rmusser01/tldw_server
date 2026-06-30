# Writing Playground Manuscript Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable manuscript-owned annotations to the shared Writing Playground, including saved-scene range comments, inspector management, desktop margin comments, selected-text AI critique, Jobs-backed scene review, and suggested-fix handoff into the existing revision queue.

**Architecture:** Store annotations in the per-user ChaChaNotes manuscript schema and expose them through `ManuscriptDBHelper` plus `/api/v1/writing/manuscripts` endpoints. Keep frontend state in focused Writing Playground annotation modules, require saved-scene binding before range comments, and render the margin rail only when the editor adapter can provide reliable DOM coordinates. AI review uses a small Writing core service for prompt, parsing, validation, and duplicate suppression; scene review is queued through the existing Jobs manager patterns.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL ChaChaNotes migrations, `ManuscriptDBHelper`, Jobs `JobManager`/`WorkerSDK`, React, TypeScript, Ant Design, TanStack Query, Zustand, TipTap/ProseMirror, Vitest, Testing Library, Playwright, pytest, Bandit.

---

## Source Documents

- Design spec: `Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md`
- Design task: `backlog/tasks/task-607 - Design-Writing-Playground-manuscript-annotations.md`
- Planning task: `backlog/tasks/task-2400 - Plan-Writing-Playground-manuscript-annotations-implementation.md`

## Scope Check

This plan covers one end-to-end feature slice with staged, independently testable milestones:

- backend schema, helper methods, and CRUD/list endpoints
- frontend saved-scene binding prerequisite
- inspector-based annotation creation and management
- desktop margin rail for measurable editor ranges
- selected-text AI review
- Jobs-backed scene review
- suggested-fix handoff into the existing revision queue

Before executing Task 1, create a separate Backlog.md implementation task for the code work and record this plan path on that task. Keep `TASK-2400` as the planning task only.

Use @superpowers:test-driven-development for implementation tasks and @superpowers:verification-before-completion before claiming the implementation is complete.

## Cross-Cutting Implementation Contracts

- Persisted and API text offsets are Unicode code-point offsets, matching Python string indexing. Browser and ProseMirror selection helpers must convert DOM UTF-16 code-unit positions to code-point offsets before calling backend APIs, and convert back only when restoring editor selections. Anchor tests must include astral symbols before and inside selections.
- AI review requests use `provider` and `model`, matching existing Writing Playground analysis contracts. Do not introduce a second provider field for annotation review unless the existing manuscript analysis contract is changed first.
- Jobs-backed scene review passes `owner_user_id` as top-level Jobs metadata from the authenticated request user. The Jobs payload must never carry user ids, scene text, selected text, annotation body, suggested fix, or raw model output.
- The scene-review worker loads the per-user ChaChaNotes DB from the Jobs row `owner_user_id` via `get_chacha_db_for_user_id(...)` before constructing `ManuscriptDBHelper`.
- The in-process worker route key is `writing`; the worker env flag is `WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED`.
- Application logs must not include manuscript text, selected text, raw model output, annotation body, or suggested fix. Sync-log payloads remain database records for sync/export and should be bounded to the annotation fields required by the schema contract.

## File Structure

### Backend Files

- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Bump `_CURRENT_SCHEMA_VERSION` from 50 to 51.
  - Add SQLite migration SQL for `manuscript_annotations`, indexes, sync-log triggers, and `PRAGMA user_version`/`db_schema_version` update.
  - Add PostgreSQL migration SQL for the same table and indexes.
  - Add SQLite and PostgreSQL migration routing from version 50 to 51.
- Modify `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
  - Add annotation row constants, create/get/list/update/soft-delete helper methods.
  - Add scene range validation and side-effect-free anchor derivation helpers.
  - Add duplicate suppression helpers for AI review.
- Create `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
  - Pure annotation categories/status/source constants.
  - Anchor fingerprint, prefix/suffix, and reattachment algorithms.
  - Structured AI prompt builders and response parsers for selected-text and scene review.
- Create `tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py`
  - Jobs payload builder, enqueue helper, and worker-facing processor for scene review jobs.
- Create `tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py`
  - WorkerSDK loop that acquires `writing_scene_annotation_review` Jobs, resolves the per-user DB from top-level `owner_user_id`, and calls the Writing annotation job processor.
- Modify `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`
  - Register the in-process writing annotation review Jobs worker behind `WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED`, matching the workspace file-inventory worker pattern.
- Modify `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
  - Add annotation request/response/list/job schemas and validators.
- Modify `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
  - Add annotation CRUD/list/get endpoints.
  - Add selected-text review endpoint.
  - Add scene review enqueue endpoint.
  - Reuse the existing provider/model validation pattern from analysis endpoints.

### Backend Tests

- Create `tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py`
  - Tests pure anchor derivation and reattachment.
- Create `tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py`
  - Tests schema, helper CRUD, ownership, optimistic locking, sync logs, and duplicate suppression.
- Create `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`
  - Tests endpoint contracts, pagination, filters, conflicts, and provider validation.
- Create `tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py`
  - Tests Jobs payloads, idempotency keys, enqueue failures, and worker processor behavior.
- Create `tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py`
  - Tests WorkerSDK payload validation, missing/invalid owner handling, per-user DB loading, retry classification, and sanitized logging.
- Modify `tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py`
  - Update exact worker spec names/order, factory delegation, route predicate calls, handle fields, and `worker_inventory` pass-through for the new writing worker.

### Frontend Files

- Modify `apps/packages/ui/src/services/writing-playground.ts`
  - Add annotation request/response types and service methods.
- Modify `apps/packages/ui/src/store/writing-playground.tsx`
  - Add narrow state only if active annotation id or scene binding state must be shared across panels.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-types.ts`
  - Shared annotation domain, filter, anchor, review, and UI state types.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-anchor-utils.ts`
  - Pure client helpers for fingerprints, prefix/suffix capture, selection validation, sorting, rail collision layout, and dirty-scene guards.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useActiveManuscriptScene.ts`
  - Loads active saved scenes, binds editor state to saved scene content/version, tracks dirty scene state, and saves scene edits.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingAnnotations.ts`
  - Owns query/mutation orchestration for annotations and exposes stable UI actions.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/hooks/index.ts`
  - Export the new scene-binding and annotation hooks when they are consumed through the existing hook barrel.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx`
  - Inspector tab with filters, create forms, status transitions, review actions, and fallback surface.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationList.tsx`
  - Reusable list and row rendering for inspector/drawer.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx`
  - Compact card for margin rail and active expanded card.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationMarginRail.tsx`
  - Desktop rail, derived layout, focus sync, collision handling, and fallback suppression.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts`
  - Add `"annotations"` to `InspectorTabKey`.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`
  - Add the Annotations tab and panel slot.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/writing-editor-adapter.ts`
  - Extend the adapter with optional DOM range measurement methods.
  - Implement TipTap coordinate measurement using ProseMirror positions.
  - Keep textarea measurement unavailable unless a tested real-coordinate implementation is added.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`
  - Pass editor container/coordinate hooks through the adapter.
  - Provide stable ids for highlighted ranges when implemented.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
  - Wire saved-scene binding, annotation hook, Annotations inspector tab, margin rail, and suggested-fix revision handoff.
- Modify `apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts`
  - Guard that shared annotation surfaces remain in the shared component used by WebUI and extension.

### Frontend Tests

- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx`
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts`
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx`
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx`
- Modify `apps/packages/ui/src/services/__tests__/writing-playground.snapshot.test.ts` or add `apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts`

## Implementation Tasks

### Task 1: Add Pure Annotation Types And Anchor Algorithms

**Files:**
- Create: `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py`

- [ ] **Step 1: Write failing anchor tests**

Cover exact attachment, unique selected-text reattachment, ambiguous selected-text matches, prefix/suffix local reattachment, needs-review fallback, and non-scene notes.
Include cases where text before and inside the selected range contains astral Unicode symbols so code-point offsets are verified independent of browser UTF-16 selection offsets.

```python
from tldw_Server_API.app.core.Writing.manuscript_annotations import (
    build_scene_anchor,
    derive_scene_anchor_status,
)

def test_exact_range_attaches_when_scene_version_and_text_match():
    text = "Alpha beta gamma"
    anchor = build_scene_anchor(text, start=6, end=10, scene_version=3)

    status = derive_scene_anchor_status(anchor, text, current_scene_version=3)

    assert status["anchor_status"] == "attached"
    assert status["derived_start"] == 6
    assert status["derived_end"] == 10

def test_unique_selected_text_reattaches_after_prefix_insert():
    original = "Alpha beta gamma"
    current = "Intro. Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=3)

    status = derive_scene_anchor_status(anchor, current, current_scene_version=4)

    assert status["anchor_status"] == "reattached"
    assert current[status["derived_start"]:status["derived_end"]] == "beta"

def test_ambiguous_selected_text_requires_review():
    anchor = build_scene_anchor("Alpha beta gamma", start=6, end=10, scene_version=3)

    status = derive_scene_anchor_status(anchor, "beta Alpha beta gamma", current_scene_version=4)

    assert status["anchor_status"] == "needs_review"
    assert status["derived_start"] is None
    assert status["derived_end"] is None
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q
```

Expected: FAIL because `manuscript_annotations.py` does not exist.

- [ ] **Step 3: Implement constants and pure anchor helpers**

Add:

```python
VALID_ANNOTATION_STATUSES = ("open", "resolved")
VALID_ANNOTATION_SOURCES = ("user", "ai_selected_text", "ai_scene_review")
VALID_ANNOTATION_CATEGORIES = (
    "style",
    "clarity",
    "pacing",
    "continuity",
    "character",
    "worldbuilding",
    "structure",
    "research",
    "other",
)
VALID_TARGET_TYPES = ("scene", "chapter", "project")
ANCHOR_CONTEXT_CHARS = 240

def build_scene_anchor(text: str, *, start: int, end: int, scene_version: int) -> dict[str, object]:
    """Build persisted range-anchor metadata from saved scene text."""
    normalized_start, normalized_end = _validate_range(text, start, end)
    selected_text = text[normalized_start:normalized_end]
    return {
        "scene_version": int(scene_version),
        "anchor_start": normalized_start,
        "anchor_end": normalized_end,
        "selected_text": selected_text,
        "document_fingerprint": create_document_fingerprint(text),
        "anchor_prefix": text[max(0, normalized_start - ANCHOR_CONTEXT_CHARS):normalized_start],
        "anchor_suffix": text[normalized_end:normalized_end + ANCHOR_CONTEXT_CHARS],
        "anchor_status": "attached",
    }
```

Keep `derive_scene_anchor_status()` side-effect free. It must return derived fields but must not mutate database rows.
All helper range validation and returned offsets use Unicode code-point indexes. Do not encode offsets as bytes or UTF-16 code units.

- [ ] **Step 4: Pass the anchor tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Writing/manuscript_annotations.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py
git commit -m "feat: add manuscript annotation anchor helpers"
```

### Task 2: Add Annotation Schema, Migrations, And DB Helper Methods

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py`

- [ ] **Step 1: Write failing DB tests**

Cover:

- fresh DB creates `manuscript_annotations`
- migrating an existing SQLite v50 DB registers and applies `_migrate_from_v50_to_v51`, then leaves schema version 51 with annotation table/indexes/triggers present
- PostgreSQL migration routing registers `_MIGRATION_SQL_V50_TO_V51_POSTGRES`, applies through `_apply_postgres_migration_script(..., expected_version=51)`, and contains PostgreSQL-compatible syntax; use an available Postgres fixture when present, otherwise unit-test the routing and migration script contract directly
- create/get scene range annotation validates saved scene version and selected text
- chapter/project note creation has no range offsets and returns `scene_level`
- update/delete use optimistic locking
- list filters by target/status/category/source
- derived `anchor_status` changes after scene text edits without mutating the row
- sync log records create/update/delete for annotations

Example test skeleton:

```python
import json

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper

@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "annotations.db"), client_id="test_client")
    try:
        yield ManuscriptDBHelper(db)
    finally:
        db.close_connection()

def _scene(mdb: ManuscriptDBHelper):
    project_id = mdb.create_project("Novel")
    chapter_id = mdb.create_chapter(project_id, "Chapter 1")
    scene_id = mdb.create_scene(
        chapter_id,
        project_id,
        title="Scene",
        content_plain="Alpha beta gamma",
    )
    scene = mdb.get_scene(scene_id)
    return project_id, chapter_id, scene_id, scene

def test_create_scene_range_annotation_attaches_to_saved_scene(mdb):
    project_id, _chapter_id, scene_id, scene = _scene(mdb)

    annotation_id = mdb.create_annotation(
        project_id=project_id,
        target_type="scene",
        target_id=scene_id,
        category="clarity",
        source="user",
        body="This line is unclear.",
        scene_version=scene["version"],
        anchor_start=6,
        anchor_end=10,
        selected_text="beta",
    )

    annotation = mdb.get_annotation(annotation_id)

    assert annotation["anchor_status"] == "attached"
    assert annotation["derived_start"] == 6
    assert annotation["derived_end"] == 10
    assert annotation["version"] == 1

def test_update_annotation_uses_version_conflict(mdb):
    project_id, _chapter_id, scene_id, scene = _scene(mdb)
    annotation_id = mdb.create_annotation(
        project_id=project_id,
        target_type="scene",
        target_id=scene_id,
        category="clarity",
        source="user",
        body="Body",
        scene_version=scene["version"],
        anchor_start=0,
        anchor_end=5,
        selected_text="Alpha",
    )

    with pytest.raises(ConflictError):
        mdb.update_annotation(annotation_id, {"body": "New"}, expected_version=99)
```

- [ ] **Step 2: Run the failing DB tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py -q
```

Expected: FAIL because the schema and helper methods do not exist.

- [ ] **Step 3: Add schema version 51**

Add the SQLite table and indexes near the manuscript schema area:

```sql
CREATE TABLE IF NOT EXISTS manuscript_annotations (
  id                   TEXT PRIMARY KEY,
  project_id           TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  target_type          TEXT NOT NULL CHECK(target_type IN ('scene','chapter','project')),
  target_id            TEXT NOT NULL,
  status               TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved')),
  category             TEXT NOT NULL CHECK(category IN ('style','clarity','pacing','continuity','character','worldbuilding','structure','research','other')),
  tags_json            TEXT NOT NULL DEFAULT '[]',
  source               TEXT NOT NULL CHECK(source IN ('user','ai_selected_text','ai_scene_review')),
  body                 TEXT NOT NULL,
  suggested_fix        TEXT,
  followup_note        TEXT,
  metadata_json        TEXT NOT NULL DEFAULT '{}',
  scene_version        INTEGER,
  anchor_start         INTEGER,
  anchor_end           INTEGER,
  selected_text        TEXT,
  document_fingerprint TEXT,
  anchor_prefix        TEXT,
  anchor_suffix        TEXT,
  anchor_status        TEXT NOT NULL DEFAULT 'scene_level',
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted              BOOLEAN NOT NULL DEFAULT 0,
  client_id            TEXT NOT NULL,
  version              INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mann_project_target
  ON manuscript_annotations(project_id, target_type, target_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_project_status
  ON manuscript_annotations(project_id, status, last_modified);
CREATE INDEX IF NOT EXISTS idx_mann_source
  ON manuscript_annotations(project_id, source, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_deleted
  ON manuscript_annotations(deleted);
```

Add sync triggers for create/update/delete/undelete mirroring nearby manuscript entity trigger style. Include payload fields needed for sync/export and never omit anchor fields from the payload.

- [ ] **Step 4: Add PostgreSQL migration SQL**

Add `_MIGRATION_SQL_V50_TO_V51_POSTGRES` with PostgreSQL-compatible checks, indexes, and schema-version update. Follow the existing `_apply_postgres_migration_script(..., expected_version=...)` pattern.

- [ ] **Step 5: Wire SQLite and PostgreSQL migration routing**

Update:

- `_CURRENT_SCHEMA_VERSION = 51`
- SQLite migration function `_migrate_from_v50_to_v51`
- `_sqlite_linear_migration_steps()` to include `(50, "_migrate_from_v50_to_v51")`
- normal SQLite migration routing after `_migrate_from_v49_to_v50`
- PostgreSQL bootstrap and normal migration routing from version 50 to 51

- [ ] **Step 6: Implement helper methods**

In `ManuscriptDBHelper`, add methods:

```python
def create_annotation(..., annotation_id: str | None = None) -> str: ...
def get_annotation(self, annotation_id: str) -> dict[str, Any] | None: ...
def list_annotations(self, project_id: str, *, target_type: str | None = None, target_id: str | None = None, status: str | None = None, category: str | None = None, source: str | None = None, anchor_status: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]: ...
def update_annotation(self, annotation_id: str, updates: dict[str, Any], expected_version: int) -> None: ...
def soft_delete_annotation(self, annotation_id: str, expected_version: int) -> None: ...
def suppress_duplicate_annotation_candidates(...): ...
```

Implementation constraints:

- Validate target ownership in one transaction.
- For scene range annotations, validate current scene version, range bounds, and selected text against `manuscript_scenes.content_plain`.
- For chapter/project notes, reject range fields and set `anchor_status` to `scene_level`.
- Derive anchor status on read/list using current saved scene text.
- Only allow `anchor_status` filter when bounded by `target_type="scene"` and `target_id`, or when the candidate set is under a documented cap.
- Do not write refreshed offsets during normal read/list.

- [ ] **Step 7: Pass DB tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q
```

Expected: PASS.
If the Postgres fixture is unavailable, the SQL-routing/unit test for `_MIGRATION_SQL_V50_TO_V51_POSTGRES` must still run and pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py
git commit -m "feat: persist manuscript annotations"
```

### Task 3: Add Annotation API Schemas And CRUD Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- `POST /api/v1/writing/manuscripts/annotations` creates a manual scene range annotation.
- `GET /api/v1/writing/manuscripts/annotations/{annotation_id}` returns derived anchor state.
- `GET /api/v1/writing/manuscripts/projects/{project_id}/annotations` returns pagination aliases and `pagination`.
- `PATCH /api/v1/writing/manuscripts/annotations/{annotation_id}` requires `expected-version`.
- `DELETE /api/v1/writing/manuscripts/annotations/{annotation_id}` soft deletes with `expected-version`.
- Broad `anchor_status` filtering is rejected unless bounded.
- Soft-deleted targets return not found.
- Manual range annotation offsets are interpreted as Unicode code-point offsets and validated correctly when scene text contains astral symbols.

- [ ] **Step 2: Run the failing API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q
```

Expected: FAIL because schemas/endpoints do not exist.

- [ ] **Step 3: Add Pydantic schemas**

Add schema classes near the manuscript analysis/citation models:

```python
AnnotationTargetType = Literal["scene", "chapter", "project"]
AnnotationStatus = Literal["open", "resolved"]
AnnotationSource = Literal["user", "ai_selected_text", "ai_scene_review"]
AnnotationCategory = Literal[
    "style",
    "clarity",
    "pacing",
    "continuity",
    "character",
    "worldbuilding",
    "structure",
    "research",
    "other",
]
AnnotationAnchorStatus = Literal["attached", "reattached", "needs_review", "scene_level"]

class ManuscriptAnnotationCreate(BaseModel):
    target_type: AnnotationTargetType
    target_id: str
    category: AnnotationCategory
    body: str = Field(..., min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=10)
    suggested_fix: str | None = Field(None, max_length=8000)
    followup_note: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scene_version: int | None = None
    start: int | None = Field(None, ge=0)
    end: int | None = Field(None, ge=0)
    selected_text: str | None = Field(None, max_length=12000)
```

Add validators to require range fields for scene range comments and to reject range fields for chapter/project notes. Keep expected versions in headers for update/delete.

- [ ] **Step 4: Add endpoints**

Add routes under `writing_manuscripts.py`:

- `GET /projects/{project_id}/annotations`
- `POST /annotations`
- `GET /annotations/{annotation_id}`
- `PATCH /annotations/{annotation_id}`
- `DELETE /annotations/{annotation_id}`

Use scopes:

- `writing.manuscripts.annotations.list`
- `writing.manuscripts.annotations.create`
- `writing.manuscripts.annotations.update`
- `writing.manuscripts.annotations.delete`

Follow `_handle_db_errors()` and the existing `expected_version: int = Header(...)` convention.

- [ ] **Step 5: Pass API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Writing/test_writing_error_mapping.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py
git commit -m "feat: expose manuscript annotation endpoints"
```

### Task 4: Add Selected-Text AI Review Endpoint

**Files:**
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`

- [ ] **Step 1: Write failing selected-text review tests**

Test:

- Missing provider/model returns a validation error.
- Unknown provider/model follows the analysis endpoint validation behavior.
- Scene version mismatch returns conflict.
- Range text mismatch returns conflict.
- Valid JSON model output persists one `source="ai_selected_text"` annotation.
- Unparseable output returns diagnostics and creates no annotation.
- Request schema uses `provider` and `model`, matching `ManuscriptAnalysisRequest` and the frontend writing service contract.

Patch `tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async` with `AsyncMock`, following `test_manuscript_analysis_integration.py`.

- [ ] **Step 2: Run failing tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q
```

Expected: FAIL for the new review endpoint.

- [ ] **Step 3: Implement prompt and parser helpers**

Add functions:

```python
def build_selected_text_review_prompt(*, scene_text: str, selected_text: str, category_hints: list[str], instruction: str | None) -> list[dict[str, str]]: ...

def parse_annotation_review_response(raw_text: str) -> list[dict[str, str]]: ...
```

Validation rules:

- exactly one annotation for selected-text review
- category in fixed set
- non-empty body under 2000 chars
- optional `suggested_fix` under 8000 chars
- no partial persistence on parse failure

- [ ] **Step 4: Add endpoint**

Add:

```python
@router.post(
    "/scenes/{scene_id}/annotations/review-selection",
    response_model=ManuscriptAnnotationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def review_selected_text_annotation(...):
    ...
```

Use `rbac_rate_limit("writing.manuscripts.annotations.review")` plus runtime rate limit scope `"writing.manuscripts.annotations.review"`.

- [ ] **Step 5: Pass selected-text review tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Writing/manuscript_annotations.py tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py
git commit -m "feat: add selected text annotation review"
```

### Task 5: Add Jobs-Backed Scene Review

**Files:**
- Create: `tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py`
- Create: `tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`

- [ ] **Step 1: Write failing Jobs helper tests**

Mirror `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py` with a recording job manager.

Expected helper behavior:

- domain: `"writing"`
- queue: environment override `WRITING_ANNOTATION_REVIEW_JOBS_QUEUE`, default `"default"`
- job type: `"writing_scene_annotation_review"`
- payload includes `scene_id`, `scene_version`, `project_id`, `max_comments`, category filters, provider, model, and no raw manuscript text
- `owner_user_id` is passed to `JobManager.create_job` as top-level Jobs metadata and is not duplicated in the payload
- idempotency key includes scene id, scene version, review focus, provider, model, and max comments

- [ ] **Step 2: Implement enqueue helper**

Add:

```python
WRITING_JOBS_DOMAIN = "writing"
WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE = "writing_scene_annotation_review"

def writing_annotation_review_jobs_queue() -> str: ...

def build_scene_annotation_review_job_payload(...) -> dict[str, Any]: ...

def enqueue_scene_annotation_review_job(*, job_manager: JobManager, owner_user_id: str, ...) -> dict[str, Any]: ...
```

Do not put user ids, scene text, selected text, annotation body, suggested fix, or raw model output in the Jobs payload. The worker must load saved scene text by id/version after resolving the per-user DB from the top-level Jobs row `owner_user_id`.

- [ ] **Step 3: Add API endpoint tests**

Test `POST /scenes/{scene_id}/annotations/review-scene`:

- rejects missing provider/model
- rejects bad version
- rejects `max_comments > 10`
- returns a job response when enqueue succeeds
- returns a sanitized 503/500-style error when Jobs are unavailable, following the selected project convention
- depends on `get_request_user` and passes `str(current_user.id)` as the enqueue helper `owner_user_id`

- [ ] **Step 4: Implement API endpoint**

Inject Jobs through `try_get_job_manager` or `get_job_manager` based on whether failure should block the request. Since scene review is explicitly Jobs-backed, fail the request when Jobs is unavailable and return a clear error. Use `current_user: User = Depends(get_request_user)` and pass `owner_user_id=str(current_user.id)` into `enqueue_scene_annotation_review_job(...)`.

- [ ] **Step 5: Implement worker-facing processor**

Add a processor that accepts a resolved per-user manuscript helper and sanitized job payload:

```python
async def process_scene_annotation_review_job(*, manuscript_db: ManuscriptDBHelper, job_payload: dict[str, Any], job_manager: JobManager | None = None) -> dict[str, Any]:
    ...
```

Behavior:

- Load current saved scene and verify version still matches payload.
- Call scene-review prompt helper.
- Parse bounded annotation results.
- Anchor each result by unambiguous quote/range.
- Suppress duplicates against existing open annotations.
- Persist up to `max_comments`.
- Return `{"created_annotation_ids": [...], "diagnostics": [...]}`.

- [ ] **Step 6: Add and register the service worker**

Create `tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py` by following the `workspace_file_inventory_jobs_worker.py` shape:

- build `WorkerConfig` for domain `"writing"` and `writing_annotation_review_jobs_queue()`
- acquire only `writing_scene_annotation_review` jobs
- consume the full Jobs row, require non-empty `owner_user_id`, and treat missing or invalid owner metadata as non-retryable
- call `get_chacha_db_for_user_id(owner_user_id, client_id=f"writing-annotation-review-worker-{owner_user_id}")` before constructing `ManuscriptDBHelper`
- call `process_scene_annotation_review_job(...)`
- complete successful jobs with created annotation ids and diagnostics
- fail invalid payloads as non-retryable
- fail provider/runtime failures as retryable only when the exception is actually retryable
- never log manuscript text, selected text, raw model output, annotation bodies, or suggested fixes

Modify `startup_primary_jobs_pollers.py`:

- add `writing_annotation_review_jobs_stop_event` and `writing_annotation_review_jobs_task` to `PrimaryJobsPollerHandles`
- add a `provide_primary_jobs_worker_specs()` entry named `"writing_annotation_review_jobs_task"` behind `WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED`
- use `route_enabled_predicate("WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED", "writing")` in the declarative worker spec
- start the worker in `start_primary_jobs_pollers()` with the same `worker_inventory.register_custom(...)` and `register_owned_job_poller(...)` branches used by `_start_workspace_file_inventory_jobs_worker`
- use `should_start_worker("WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED", "writing_annotation_review_jobs_task")` in the legacy startup branch

Update `tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py`:

- add `"writing_annotation_review_jobs_task"` to the expected spec names in the exact order chosen in `provide_primary_jobs_worker_specs()`
- assert factory delegation to `_run_writing_annotation_review_jobs_worker_service`
- assert route predicate calls include `("writing",)` when `WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ENABLED=true`
- assert handles include `writing_annotation_review_jobs_stop_event` and `writing_annotation_review_jobs_task`
- assert `worker_inventory.register_custom(...)` receives the writing worker metadata in startup paths that use custom inventory registration

- [ ] **Step 7: Pass Jobs tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py tldw_Server_API/app/services/startup_primary_jobs_pollers.py tldw_Server_API/app/core/Writing/manuscript_annotations.py tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py
git commit -m "feat: queue manuscript scene annotation review"
```

### Task 6: Add Frontend Annotation Service Types And Client Methods

**Files:**
- Modify: `apps/packages/ui/src/services/writing-playground.ts`
- Create: `apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts`

- [ ] **Step 1: Write failing service tests**

Assert exact paths/methods:

- `listManuscriptAnnotations(projectId, filters)`
- `createManuscriptAnnotation(input)`
- `getManuscriptAnnotation(annotationId)`
- `updateManuscriptAnnotation(annotationId, input, version)`
- `deleteManuscriptAnnotation(annotationId, version)`
- `reviewManuscriptSelection(sceneId, input)`
- `reviewManuscriptScene(sceneId, input)`
- review request payloads use `provider` and `model`, with no alternate provider field

- [ ] **Step 2: Run failing service tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts
```

Expected: FAIL because methods do not exist.

- [ ] **Step 3: Implement service types and methods**

Use explicit response types, not `Record<string, unknown>` for annotation contracts. Use `buildExpectedVersionHeaders(version)` for update/delete.

- [ ] **Step 4: Pass service tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/writing-playground.ts apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts
git commit -m "feat: add manuscript annotation client methods"
```

### Task 7: Bind Active Saved Manuscript Scene To The Editor

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useActiveManuscriptScene.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/index.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/ManuscriptTreePanel.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx`

- [ ] **Step 1: Write failing hook tests**

Cases:

- does not query when active node type is not `"scene"`
- loads saved `content_plain` and `content` into editor state when a scene becomes active
- tracks `isSceneBound`, `sceneId`, `sceneVersion`, and dirty state
- save calls `updateManuscriptScene` with the saved version
- annotation range actions are disabled when editor text differs from saved scene text
- switching to a different active manuscript scene while the editor is dirty preserves unsaved content or blocks/prompts navigation; it must not silently overwrite or rebind the editor to another scene

- [ ] **Step 2: Run failing hook tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx
```

Expected: FAIL because the hook does not exist.

- [ ] **Step 3: Implement saved-scene binding hook**

Hook contract:

```ts
export type ActiveManuscriptSceneBinding = {
  scene: ManuscriptSceneResponse | null
  sceneId: string | null
  sceneVersion: number | null
  isSceneBound: boolean
  isSceneLoading: boolean
  isSceneDirty: boolean
  canCreateRangeAnnotation: boolean
  saveScene: () => Promise<ManuscriptSceneResponse | null>
  reloadScene: () => void
}
```

The hook must:

- query `getManuscriptScene(activeNodeId)` only when `activeNodeType === "scene"`
- load saved scene content into `editorText` and `tipTapContent`
- record the last saved plain text and rich JSON signature
- prevent annotation range actions unless the editor matches the saved scene baseline
- expose a save action using `updateManuscriptScene(sceneId, { content_plain, content }, scene.version)`
- export through `hooks/index.ts` when `index.tsx` consumes the hook through the existing hook barrel

- [ ] **Step 4: Wire the hook in `index.tsx`**

Replace the existing active-node comment with the hook. Add a small save-status affordance to the existing editor toolbar/statusbar and keep session save behavior separate from scene save behavior. Wire dirty scene navigation at the active-node selection boundary in `ManuscriptTreePanel.tsx` or its owning handler so scene switches cannot discard unsaved editor text without an explicit user action.

- [ ] **Step 5: Pass scene binding tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/hooks/useActiveManuscriptScene.ts apps/packages/ui/src/components/Option/WritingPlayground/hooks/index.ts apps/packages/ui/src/components/Option/WritingPlayground/index.tsx apps/packages/ui/src/components/Option/WritingPlayground/ManuscriptTreePanel.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
git commit -m "feat: bind writing editor to saved manuscript scenes"
```

### Task 8: Add Annotation State Hook And Inspector Tab

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-types.ts`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-anchor-utils.ts`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingAnnotations.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/index.ts`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationList.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx`

- [ ] **Step 1: Write failing utility and hook tests**

Test:

- client prefix/suffix capture clamps to 240 chars
- client selection conversion maps UTF-16 DOM/ProseMirror positions to backend Unicode code-point offsets when text contains astral symbols
- selected range validation rejects empty or stale selections
- query key includes project id, target context, and filters
- `enabled === false` keeps annotation queries unpopulated
- mutations invalidate the exact annotation keys

- [ ] **Step 2: Write failing inspector tab tests**

Test:

- Annotations tab appears in the inspector tablist with keyboard navigation.
- Scene range comment form requires saved scene binding.
- Chapter/project notes remain available without range selection.
- Resolve/reopen/update actions call the hook.
- Anchor `needs_review` state is visible in the row.

- [ ] **Step 3: Run failing frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx
```

Expected: FAIL for new modules and Annotations tab.

- [ ] **Step 4: Implement types, utilities, hook, and inspector tab**

Important contracts:

- Default filters prioritize open annotations for the active scene.
- Chapter/project notes are rendered in the inspector, never in the margin rail.
- Scene range creation sends `scene_version`, offsets, selected text, prefix, suffix, and fingerprint only when `canCreateRangeAnnotation` is true.
- Disable AI review actions if provider/model is unavailable.
- Keep UI components small; do not grow annotation rendering inside `index.tsx`.
- Export `useWritingAnnotations` through `hooks/index.ts` when the caller imports it from the existing hook barrel.

- [ ] **Step 5: Pass frontend inspector tests**

Run the same Vitest command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-types.ts apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-anchor-utils.ts apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingAnnotations.ts apps/packages/ui/src/components/Option/WritingPlayground/hooks/index.ts apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationList.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx apps/packages/ui/src/components/Option/WritingPlayground/index.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx
git commit -m "feat: add writing annotations inspector"
```

### Task 9: Add TipTap Range Measurement And Desktop Margin Rail

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/writing-editor-adapter.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationMarginRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx`
- Modify: `apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts`

- [ ] **Step 1: Write failing adapter measurement tests**

Extend adapter tests to verify:

- TipTap adapter exposes `getRangeClientRect(selection)` or equivalent.
- Measurement returns null for invalid/stale ranges.
- Textarea adapter does not claim margin measurement support.

- [ ] **Step 2: Write failing margin rail tests**

Test pure layout behavior:

- sorts cards by anchor top, then `created_at`, then id
- collision avoidance pushes later cards down by a fixed gap
- active card expands and pushes following cards down
- rail hides when measurement is unavailable
- resolved comments are excluded by default
- focus callbacks sync card to editor selection
- browser smoke path renders two attached comments at desktop width with non-null measurements, no overlapping cards, and hidden rail when measurement is unavailable

- [ ] **Step 3: Run failing rail tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx
```

Expected: FAIL for new measurement and rail.

- [ ] **Step 4: Implement TipTap measurement**

Use ProseMirror position mapping already present in `writing-editor-adapter.ts`, then call editor view coordinate APIs for the mapped positions. Do not approximate textarea positions by line counts.

Adapter extension:

```ts
export type WritingEditorRangeMeasurement = {
  top: number
  bottom: number
  height: number
}

export type WritingEditorAdapter = {
  getSelection: () => WritingEditorSelection
  setSelection: (selection: WritingEditorSelection) => void
  getSelectedText: (currentValue: string) => string
  focus: () => void
  measureRange?: (selection: WritingEditorSelection) => WritingEditorRangeMeasurement | null
}
```

- [ ] **Step 5: Implement rail and card components**

Rules:

- Render only wide-enough layouts.
- Render only open scene range annotations with `attached` or `reattached` measurements, plus visible warning state for `needs_review`.
- Keep the rail in the same scroll context as the editor or recompute positions on scroll/resize.
- Use immediate placement or transform/opacity transitions only.
- Provide stable aria ids connecting highlights, cards, and inspector rows.

- [ ] **Step 6: Wire the rail in `index.tsx`**

Place the rail alongside the editor area, not inside a nested card. Hide it for preview-only and plain textarea measurement-unavailable states.

- [ ] **Step 7: Pass rail tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Run browser rail smoke**

Run the extension parity smoke from the extension package so Playwright uses the existing extension configuration:

```bash
cd apps/extension
bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts --reporter=line
```

Expected: PASS, including checks that desktop TipTap geometry produces non-overlapping margin cards and measurement-unavailable modes fall back to the inspector.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/writing-editor-adapter.ts apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationMarginRail.tsx apps/packages/ui/src/components/Option/WritingPlayground/index.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts
git commit -m "feat: add manuscript annotation margin rail"
```

### Task 10: Wire Selected-Text Review, Scene Review, And Suggested-Fix Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingAnnotations.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts` only if a small source field is needed for annotation-origin proposals.
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx`
- Modify: `apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts`

- [ ] **Step 1: Write failing action tests**

Test:

- selected-text AI review sends active provider/model, scene version, offsets, and selected text
- selected-text review is disabled when scene is dirty
- scene review enqueues a job and displays returned job id/status
- suggested-fix "Create revision" creates a revision proposal and does not mutate the editor
- suggested-fix action falls back to copy/manual behavior when anchor needs review

- [ ] **Step 2: Run failing action tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
```

Expected: FAIL for new actions.

- [ ] **Step 3: Implement selected-text and scene-review actions**

In `useWritingAnnotations`, expose:

```ts
reviewSelection: (input: ReviewSelectionInput) => Promise<ManuscriptAnnotationResponse>
reviewScene: (input: ReviewSceneInput) => Promise<ManuscriptAnnotationReviewJobResponse>
```

Use active Writing Playground provider/model. Do not fall back to unrelated chat defaults when provider/model is absent.

- [ ] **Step 4: Implement suggested-fix handoff**

In `index.tsx`, bridge annotation suggested fixes into the existing `useWritingRevisions` queue:

- resolve attached or reattached anchor
- create a replacement proposal with `beforeText` from the current saved editor text and `replacementText` from `suggested_fix`
- keep apply/reject/conflict behavior inside `WritingRevisionQueue`
- if anchor needs review, expose copy/manual action only

- [ ] **Step 5: Pass action tests**

Run the same Vitest command from Step 2.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingAnnotations.ts apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx apps/packages/ui/src/components/Option/WritingPlayground/index.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
git commit -m "feat: connect annotation review actions"
```

### Task 11: End-To-End Verification, Accessibility, And Cleanup

**Files:**
- Modify docs only if endpoint behavior or worker env flags need short user-facing documentation.
- Modify tests found flaky or incomplete during verification.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent backend regression tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_writing_error_mapping.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py -q
```

Expected: PASS.

- [ ] **Step 3: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts apps/packages/ui/src/services/__tests__/writing-playground.annotations.test.ts
```

Expected: PASS.

- [ ] **Step 4: Run adjacent frontend regression tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run browser layout checks**

Start the WebUI dev server from its package, then use Playwright from the extension package to check:

- desktop width shows editor plus margin rail when TipTap and attached scene comments are present
- medium width keeps cards compact and active card expanded
- narrow extension/options width hides the rail and uses inspector fallback
- keyboard focus moves from highlight to card/list action and back to editor selection

Command shape:

```bash
# terminal 1
cd apps/tldw-frontend
bun run dev

# terminal 2
cd apps/extension
bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts --reporter=line
```

Expected: PASS, or document any environment-specific skip with the exact missing dependency.

- [ ] **Step 6: Run static checks for changed files**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 7: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Writing tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py tldw_Server_API/app/services/startup_primary_jobs_pollers.py -f json -o /tmp/bandit_writing_manuscript_annotations.json
```

Expected: no new high or medium findings in touched code. Fix new findings before continuing.

- [ ] **Step 8: Self-review the implementation**

Check:

- no manuscript text, selected text, model output, annotation body, or suggested fix is logged
- no `as any` casts were added in new Writing Playground surfaces
- no broad `anchor_status` SQL filtering gives stale totals
- normal list/get never mutates anchor offsets
- expected-version headers are used for update/delete
- Jobs payloads do not include raw manuscript text
- annotation suggested fixes only create revision proposals and never mutate draft text directly

- [ ] **Step 9: Commit final cleanup**

```bash
git add <changed files>
git commit -m "test: verify manuscript annotations workflow"
```

Skip this commit if no files changed during verification.

## Acceptance Checklist

- Backend annotation records are durable, versioned, soft-deletable, and sync-logged.
- Range annotations validate against saved scene text and scene version before creation.
- List/get responses derive anchor status side-effect free.
- Broad `anchor_status` filters do not return stale project-wide totals.
- Manual scene range comments are blocked until the active editor is bound to a saved scene and has no unsaved scene changes.
- Chapter and project notes work without fake text ranges.
- The inspector tab is the full management and responsive fallback surface.
- The desktop rail shows only when reliable range measurement is available.
- Plain textarea mode uses inspector fallback unless a real coordinate implementation is proven by tests.
- Selected-text AI review requires explicit provider/model and creates exactly one validated annotation.
- Scene review uses Jobs and does not put manuscript text in the Jobs payload.
- Suggested fixes feed the existing revision proposal queue and do not create a second mutation path.
- Keyboard users can access annotation actions and return focus to the editor selection.
- WebUI and extension continue using the shared Writing Playground implementation.
- Browser/editor offsets round-trip through Unicode code-point offsets without breaking emoji-containing scenes.
- Annotation review request contracts use `provider` and `model` consistently across backend schemas and frontend services.
