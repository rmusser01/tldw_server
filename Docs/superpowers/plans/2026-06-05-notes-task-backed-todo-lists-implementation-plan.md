# Notes Task-Backed To-Do Lists Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class task-backed to-do list support for Notes, with markdown projection, `/notes` and Notes Dock interaction, and MCP Unified task tools.

**Architecture:** Add a small task domain beside Notes: parser/reconciler logic in `tldw_Server_API/app/core/Notes_Tasks`, persistence under `tldw_Server_API/app/core/DB_Management/chacha`, API schemas/endpoints under `api/v1`, MCP tools in the existing Notes MCP module, and task UI helpers in the WebUI. Markdown remains the editing/export surface; task tables hold durable state and audit/activity records.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL-compatible ChaChaNotes DB helpers, pytest, React, TypeScript, TanStack Query, Vitest, MCP Unified.

---

## References

- Spec: `Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md`
- Planning Backlog: `TASK-513`
- PRD Backlog: `TASK-512`
- Execution Backlog: create or identify implementation task(s) in Task 0 before any code edits.
- Existing note persistence: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Existing note API: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Router registration: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Existing notes MCP module: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py`
- Full notes page: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Notes editor state: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
- Notes editor pane: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Notes Dock: `apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx`
- Notes Dock store: `apps/packages/ui/src/store/notes-dock.tsx`

## Scope

This plan implements the PRD as four releasable slices:

1. Core task foundation: parser, schema, persistence, reconciler, REST API.
2. WebUI task interaction: `/notes` preview/split and Notes Dock.
3. MCP task tools for discovery, user-confirmed writes, approval-required writes, and denied autonomous writes.
4. Autonomous activity surfacing, autonomous-write enablement, and final verification.

Autonomous MCP task writes must remain disabled or approval-required until persistent activity delivery is implemented and tested in Task 9.

Do not implement Kanban, recurring tasks, reminders, calendar sync, hidden markdown IDs, or cross-user assignment.

`TASK-513` is a planning task for implementing the PRD captured in `TASK-512`; it does not replace the PRD task.

Execution agents must not edit code, tests, generated API guards, docs, or tracked UI assets until Task 0 has created or identified the implementation Backlog task(s) for those edits.

## File Structure

### Backend Core

- Create `tldw_Server_API/app/core/Notes_Tasks/__init__.py`
  - Package exports for task parser/reconciler/service.
- Create `tldw_Server_API/app/core/Notes_Tasks/models.py`
  - Dataclasses/enums for parsed checklist items, locators, task status, projection status, reconciliation result, and mutation result.
- Create `tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py`
  - Markdown checklist parser, metadata token parser, child-content detection, projection rewrite helpers.
- Create `tldw_Server_API/app/core/Notes_Tasks/reconciler.py`
  - Idempotent mapping between saved note markdown and task records.
- Create `tldw_Server_API/app/core/Notes_Tasks/service.py`
  - Public service methods used by REST and MCP: list/get/create/update/status/delete/reconcile/activity.

### Backend Persistence

- Create `tldw_Server_API/app/core/DB_Management/chacha/task_store.py`
  - Raw SQL task persistence only, kept inside the DB management boundary.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Bump `_CURRENT_SCHEMA_VERSION` from current `47` to the next version at implementation time.
  - Add migration SQL for `tasks`, `task_events`, `task_event_read_state`, `task_note_projections`, and `note_task_reconciliation_state`.
  - Instantiate `TaskStore` and delegate task methods like the existing `NoteStore`.
- Modify `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py` if PostgreSQL RLS coverage is needed for new task tables.

### Backend API

- Create `tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py`
  - Pydantic request/response models for task list/get/create/update/status/delete/reconcile/activity.
- Create `tldw_Server_API/app/api/v1/endpoints/notes_tasks.py`
  - Routes under `/api/v1/notes/tasks` and `/api/v1/notes/{note_id}/tasks`.
- Modify `tldw_Server_API/app/api/v1/router_groups/content.py`
  - Register `notes_tasks` after `notes_graph` and before generic `notes` so static `/tasks` routes are not shadowed by `/{note_id}`.
- Modify `tldw_Server_API/app/api/v1/endpoints/notes.py`
  - Trigger reconciliation after create/update/bulk-create/import paths that save note content.

### MCP Unified

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py`
  - Add task tool definitions, validators, execution dispatch, and service calls.
- Modify MCP tests under `tldw_Server_API/tests/MCP_unified/`
  - Add task tool schema/validation/policy behavior tests.

### Frontend

- Create `apps/packages/ui/src/services/notes-tasks.ts`
  - Typed `bgRequest` wrappers for task endpoints.
- Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Add new task endpoint paths used by WebUI.
- Create `apps/packages/ui/src/components/Notes/task-markdown.ts`
  - Client-side checklist parser and local-only dirty toggle helper.
- Create `apps/packages/ui/src/components/Notes/TaskChecklistPreview.tsx`
  - Shared renderer for task-backed checkboxes in `/notes` and dock.
- Modify `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
  - Render task-backed checkboxes in preview/split mode and activity notices.
- Modify `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
  - Load task state/activity, coordinate clean vs dirty checkbox toggles, preserve stale-save conflict behavior.
- Modify `apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx`
  - Add compact checklist interaction for the active note.
- Modify `apps/packages/ui/src/store/notes-dock.tsx`
  - Track pending task activity notices and local-only dirty checkbox changes if the component needs store-level state.
- Modify `apps/packages/ui/src/public/_locales/en/option.json`
  - Add task UI strings only for user-visible labels/errors.

## Task 0: Implementation Backlog Tracking

**Files:**
- Backlog task records only.

- [ ] **Step 1: Read Backlog workflow instructions**

Use the official Backlog.md MCP workflow if available. Read the workflow overview/resource first, then use the task creation/execution instructions as needed.

- [ ] **Step 2: Search for existing implementation tasks**

Search for open tasks covering Notes task-backed to-do list implementation. Do not create duplicates.

- [ ] **Step 3: Create or identify execution task structure before code edits**

Use one of these structures:

- preferred: one parent implementation task linked to `TASK-512`, `TASK-513`, and this plan, with child tasks for each releasable slice
- acceptable: one implementation task per releasable slice, each linked to `TASK-512`, `TASK-513`, and this plan

Minimum child/slice coverage:

- parser/reconciler/storage/API foundation
- MCP task tools and permissions
- WebUI task client/renderer plus `/notes` interaction
- Notes Dock interaction
- activity notices, autonomous-write enablement, and final verification

- [ ] **Step 4: Mark the active execution task In Progress**

Record:

- plan path: `Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md`
- PRD/spec path: `Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md`
- planning task: `TASK-513`
- PRD task: `TASK-512`
- planned verification commands for the active slice

- [ ] **Step 5: Keep Backlog records current during execution**

At the end of each implementation task:

- update status, touched files, verification output, blockers, and commit hash
- only mark Done after the slice tests pass or any skipped verification is explicitly documented
- commit Backlog task updates with the related implementation slice unless the user asks otherwise

## Task 1: Parser And Projection Utilities

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Tasks/__init__.py`
- Create: `tldw_Server_API/app/core/Notes_Tasks/models.py`
- Create: `tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py`
- Test: `tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py`

- [ ] **Step 1: Create the parser test file with failing checklist detection tests**

Add tests for standard GitHub task syntax, checked state, line range, locator version, text hash, and raw line preservation.

```python
def test_parse_basic_checklist_lines_with_locators() -> None:
    markdown = "Intro\n- [ ] Review source @due(2026-06-10)\n- [x] Summarize findings\n"

    result = parse_note_checklists(note_id="note-1", note_version=7, content=markdown)

    assert [item.checked for item in result.items] == [False, True]
    assert result.items[0].text == "Review source"
    assert result.items[0].raw_line == "- [ ] Review source @due(2026-06-10)"
    assert result.items[0].locator.note_version == 7
    assert result.items[0].metadata["due_date"] == "2026-06-10"
```

Also add a nested checklist participant test:

```python
def test_parse_nested_checklist_lines_as_tasks() -> None:
    markdown = "- [ ] Parent\n  - [ ] Nested child task\n    - supporting note\n"

    result = parse_note_checklists(note_id="note-1", note_version=2, content=markdown)

    assert [item.text for item in result.items] == ["Parent", "Nested child task"]
    assert result.items[0].has_child_content is True
    assert result.items[1].has_child_content is True
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py -v
```

Expected: FAIL with missing module/import errors.

- [ ] **Step 3: Add parser models**

Implement focused dataclasses/enums in `models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    OPEN = "open"
    DONE = "done"


class ProjectionStatus(StrEnum):
    LIVE = "live"
    UNLINKED = "unlinked"
    AMBIGUOUS = "ambiguous"
    DELETED = "deleted"


@dataclass(frozen=True)
class TaskLocator:
    note_id: str
    note_version: int
    line_number: int
    start_offset: int
    end_offset: int
    normalized_text_hash: str
    occurrence_index: int
    block_fingerprint: str


@dataclass(frozen=True)
class ParsedChecklistItem:
    note_id: str
    checked: bool
    text: str
    raw_line: str
    metadata: dict[str, Any]
    warnings: list[str]
    locator: TaskLocator
    has_child_content: bool = False
```

- [ ] **Step 4: Implement minimal parser**

In `markdown_parser.py`, support:

- unindented and indented `- [ ]`, `* [ ]`, `+ [ ]`
- checked markers `[x]` and `[X]`
- allowlisted tokens `@due(...)`, `@priority(...)`, `@estimate(...)`
- duplicate allowlisted token handling: last valid wins
- malformed allowlisted token warnings
- unknown token preservation in raw text
- strict `YYYY-MM-DD` due dates
- priority `high|medium|low`
- estimate `30m|2h|1d`

Keep implementation deterministic; do not use hidden IDs.

- [ ] **Step 5: Add parser edge-case tests**

Add tests for:

- duplicate tokens last valid wins
- malformed dates/priorities/estimates produce warnings
- duplicate checklist text gets distinct occurrence indexes
- nested checklist lines participate as tasks
- nested child content is detected
- unknown tokens remain in raw line
- parser is idempotent on repeated calls

- [ ] **Step 6: Run parser tests to verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit parser slice**

```bash
git add tldw_Server_API/app/core/Notes_Tasks tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py
git commit -m "feat: add notes task markdown parser"
```

## Task 2: Task Storage And Migration

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/task_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py`
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py`

- [ ] **Step 1: Write failing migration test**

Use the existing migration test pattern from `test_chacha_migration_v39.py`. If `_CURRENT_SCHEMA_VERSION` has moved above 47 before implementation, name the file and migration after the next schema version.

```python
def test_sqlite_migration_adds_task_tables(tmp_path) -> None:
    db_path = tmp_path / "tasks.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="bootstrap")
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        for table in (
            "task_event_read_state",
            "task_note_projections",
            "task_events",
            "note_task_reconciliation_state",
            "tasks",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")  # nosec B608 - test-only fixed table list
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (CharactersRAGDB._CURRENT_SCHEMA_VERSION - 1, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.commit()

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="migrate")
    migrated.close_connection()

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {
        "tasks",
        "task_events",
        "task_event_read_state",
        "task_note_projections",
        "note_task_reconciliation_state",
    } <= tables
```

- [ ] **Step 2: Write failing task store tests**

Cover:

- create task record
- update status with optimistic locking
- clear `completed_at` on reopen while recording event history
- soft-delete projected task transactionally
- allow record-only soft-delete for unlinked task
- reject ambiguous projection deletion
- record reconciliation state per note/version

- [ ] **Step 3: Run storage tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py \
  -v
```

Expected: FAIL with missing tables/store.

- [ ] **Step 4: Add migration SQL**

In `ChaChaNotes_DB.py`:

- bump `_CURRENT_SCHEMA_VERSION`
- add migration block from previous version to new version
- add table definitions compatible with SQLite and existing PostgreSQL helpers
- add indexes for user/client scope, note ID, status, projection status, activity read state, and reconciliation status

Planned logical schema:

```sql
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  note_id TEXT NOT NULL,
  text TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('open','done')),
  metadata_json TEXT NOT NULL DEFAULT '{}',
  projection_status TEXT NOT NULL DEFAULT 'live',
  deleted INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  client_id TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_note_projections (
  task_id TEXT PRIMARY KEY,
  note_id TEXT NOT NULL,
  note_version INTEGER NOT NULL,
  line_number INTEGER NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  normalized_text_hash TEXT NOT NULL,
  occurrence_index INTEGER NOT NULL,
  block_fingerprint TEXT NOT NULL,
  raw_line TEXT NOT NULL,
  has_child_content INTEGER NOT NULL DEFAULT 0,
  projection_status TEXT NOT NULL DEFAULT 'live',
  updated_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
  FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_events (
  id TEXT PRIMARY KEY,
  task_id TEXT,
  note_id TEXT,
  event_type TEXT NOT NULL,
  actor_type TEXT NOT NULL,
  actor_id TEXT,
  tool_name TEXT,
  policy_mode TEXT,
  approval_id TEXT,
  old_value_json TEXT,
  new_value_json TEXT,
  created_at TEXT NOT NULL,
  client_id TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL,
  FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS task_event_read_state (
  event_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  read_at TEXT,
  dismissed_at TEXT,
  PRIMARY KEY(event_id, user_id),
  FOREIGN KEY(event_id) REFERENCES task_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS note_task_reconciliation_state (
  note_id TEXT PRIMARY KEY,
  note_version INTEGER NOT NULL,
  status TEXT NOT NULL,
  reconciled_at TEXT NOT NULL,
  item_count INTEGER NOT NULL DEFAULT 0,
  warning_count INTEGER NOT NULL DEFAULT 0,
  cursor TEXT,
  FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE
);
```

- [ ] **Step 5: Implement `TaskStore`**

`TaskStore` should expose methods that the service can compose:

- `create_task(...)`
- `get_task(task_id, include_deleted=False)`
- `list_tasks(...)`
- `update_task_record(...)`
- `set_task_projection(...)`
- `mark_task_unlinked(...)`
- `soft_delete_task(...)`
- `record_task_event(...)`
- `list_task_activity(...)`
- `mark_task_activity_read(...)`
- `mark_task_activity_dismissed(...)`
- `get_task_activity_read_state(...)`
- `get_reconciliation_state(note_id)`
- `set_reconciliation_state(...)`
- `candidate_notes_for_task_discovery(...)`

Use transaction arguments like `NoteStore.add_note(..., conn=None)` where multi-step service mutations need note and task writes in one transaction.

- [ ] **Step 6: Delegate store methods through `CharactersRAGDB`**

Mirror the existing `NoteStore` pattern:

- instantiate `self.task_store = TaskStore(self)`
- add a delegate loop at the bottom of `ChaChaNotes_DB.py`
- keep raw SQL inside `task_store.py`

- [ ] **Step 7: Run storage tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit storage slice**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py
git commit -m "feat: add notes task storage"
```

## Task 3: Reconciler And Note Save Integration

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Tasks/reconciler.py`
- Create: `tldw_Server_API/app/core/Notes_Tasks/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Test: `tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py`
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py`

- [ ] **Step 1: Write failing reconciler tests**

Cover:

- unchanged content is idempotent
- same locator/hash preserves task ID
- unique reordered item preserves task ID
- duplicate text reorder becomes ambiguous or distinct
- missing line marks previous task `unlinked`
- manual line removal does not hard-delete task history
- dirty/stale save conflict means no reconciliation state update

- [ ] **Step 2: Write failing note-save integration test**

In `test_notes_tasks_reconciliation_api.py`, create a note with checklist content through the existing notes API and assert tasks are reconciled after save. Then update content with expected version and assert task status follows markdown.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py \
  -v
```

Expected: FAIL with missing service/reconciliation.

- [ ] **Step 4: Implement reconciler**

Rules:

- parser output is the source of truth after successful note save
- match by live projection locator/hash first
- match by unique hash plus occurrence/block context second
- never merge ambiguous duplicates
- mark old live tasks `unlinked` when source line disappears
- create new task records for unmatched parsed items
- write `note_task_reconciliation_state` after successful reconciliation only

Minimal service signature:

```python
class NotesTaskService:
    def reconcile_note(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        note_version: int,
        content: str,
        actor: TaskActor,
    ) -> ReconciliationResult:
        ...
```

- [ ] **Step 5: Integrate reconciliation after note writes**

In `notes.py`, after successful create/update/bulk-create/import content saves:

- fetch saved note ID/version/content
- call `NotesTaskService.reconcile_note(...)`
- do not reconcile on failed/conflicted note writes
- preserve existing response shape for notes; add task data only if future endpoint explicitly asks for it

- [ ] **Step 6: Run reconciliation and notes integration tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit reconciler slice**

```bash
git add \
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py \
  tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py
git commit -m "feat: reconcile note checklists into tasks"
```

## Task 4: REST API For Task Operations

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/notes_tasks.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- list tasks for note reconciles stale note
- broad list returns incomplete reconciliation metadata when work limit is reached
- get task includes note/projection details
- create task requires expected note version and inserts checklist line
- set status on clean note rewrites marker and records event
- update text requires expected task/note versions
- metadata update preserves unknown tokens
- delete projected task removes line transactionally
- delete with nested child content conflicts
- unlinked task record-only delete succeeds
- unlinked task status, text, and metadata updates conflict unless explicitly record-only and non-projection-changing
- ambiguous projected status, text, metadata, and delete mutations conflict instead of rewriting arbitrary note content
- recent activity returns unread agent events and supports dismissal/read state

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py -v
```

Expected: FAIL with missing schemas/routes.

- [ ] **Step 3: Add Pydantic schemas**

Include:

- `TaskResponse`
- `TaskProjectionResponse`
- `TaskMetadata`
- `TaskListResponse`
- `TaskListReconciliationStatus`
- `TaskCreateRequest`
- `TaskUpdateRequest`
- `TaskStatusUpdateRequest`
- `TaskDeleteRequest`
- `TaskReconcileResponse`
- `TaskActivityResponse`

Use tight bounds on text, IDs, status enum, metadata tokens, and batch sizes.

- [ ] **Step 4: Add endpoint module**

Route shape:

```text
GET    /api/v1/notes/tasks
POST   /api/v1/notes/tasks/status
GET    /api/v1/notes/tasks/activity
PATCH  /api/v1/notes/tasks/activity/{event_id}
GET    /api/v1/notes/tasks/{task_id}
PATCH  /api/v1/notes/tasks/{task_id}
DELETE /api/v1/notes/tasks/{task_id}
GET    /api/v1/notes/{note_id}/tasks
POST   /api/v1/notes/{note_id}/tasks
POST   /api/v1/notes/{note_id}/tasks/reconcile
```

Important: define literal/static routes such as `/tasks/status` and `/tasks/activity` before dynamic `/tasks/{task_id}` in `notes_tasks.py`, then register `notes_tasks.py` before generic `notes.py`.

- [ ] **Step 5: Register router**

In `content.py`, add `notes_tasks` between `notes_graph` and `notes`:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.notes_tasks",
    log_name="notes_tasks",
    prefix=f"{API_V1_PREFIX}/notes",
    tags=("notes",),
    route_key="notes",
),
```

- [ ] **Step 6: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Run route contract smoke**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit REST API slice**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py
git commit -m "feat: add notes task API"
```

## Task 5: MCP Unified Task Tools

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py`

- [x] **Step 1: Write failing MCP tool tests**

Cover:

- `get_tools()` exposes `notes.tasks.list`, `notes.tasks.get`, `notes.tasks.create`, `notes.tasks.update`, `notes.tasks.set_status`, `notes.tasks.delete`, `notes.tasks.reconcile_note`
- read tools are read-only
- write tools are management/write tools with auth required
- write tools require user confirmation or explicit approval-required policy when invoked by an agent context
- autonomous write attempts are denied until Task 9 enables them after persistent activity notices are verified
- validators reject invalid status, missing note/task IDs, oversized text, invalid metadata, invalid expected versions, and oversized batches
- list uses reconciliation-aware discovery response
- write responses include succeeded/failed/skipped for batch status changes
- status, text, metadata, and delete writes conflict for ambiguous live projections
- projection-changing writes conflict for unlinked tasks unless the request is explicitly record-only
- idempotency keys prevent duplicate creates/status updates when a tool call is retried
- governance preflight, persona scope, user scope, and note ownership checks run before every write
- denied autonomous writes do not mutate notes/tasks and still return an auditable policy decision

- [x] **Step 2: Run MCP tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py -v
```

Expected: FAIL with missing tools.

- [x] **Step 3: Add tool definitions**

Use the existing `create_tool_definition` style. Proposed schemas:

- `notes.tasks.list`: note_id, status, query, metadata filters, limit, offset, include_unlinked
- `notes.tasks.get`: task_id
- `notes.tasks.create`: note_id, expected_note_version, text, metadata, insertion, idempotency_key
- `notes.tasks.update`: task_id, expected_task_version, expected_note_version, text, metadata, idempotency_key
- `notes.tasks.set_status`: idempotency_key plus items array of task_id, status, expected_task_version, expected_note_version
- `notes.tasks.delete`: task_id, expected_task_version, expected_note_version, record_only_if_unlinked, idempotency_key
- `notes.tasks.reconcile_note`: note_id, expected_note_version optional, work_limit optional

- [x] **Step 4: Implement validators**

Use `validate_tool_arguments` with strict checks. Do not rely only on JSON schema.

- [x] **Step 5: Implement execution methods**

Each method should open the user ChaCha DB with `_open_db`, call `NotesTaskService`, close connections, and sanitize debug logs on close failures like existing MCP modules.

Before each write, use the existing MCP Unified permission/governance hooks instead of custom bypass logic:

- auth and user-scope validation
- persona/session policy validation where the tool invocation supplies persona context
- note ownership/scope validation before reading or mutating a note
- approval-required flow for agent writes without autonomous permission
- autonomous write denial until Task 9 flips the policy after activity notices are durable

Write event metadata must include actor type, actor ID where available, tool name, policy mode, approval ID if present, idempotency key, and note/task IDs.

- [x] **Step 6: Run MCP tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py -v
```

Expected: PASS.

- [x] **Step 7: Commit MCP slice**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
git commit -m "feat: expose notes task tools through MCP"
```

## Task 6: Frontend Task API Client And Shared Renderer

**Files:**
- Create: `apps/packages/ui/src/services/notes-tasks.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Create: `apps/packages/ui/src/components/Notes/task-markdown.ts`
- Create: `apps/packages/ui/src/components/Notes/TaskChecklistPreview.tsx`
- Test: `apps/packages/ui/src/services/__tests__/notes-tasks.test.ts`
- Test: `apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts`
- Test: `apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx`

- [x] **Step 1: Write failing frontend service tests**

Assert each wrapper calls the correct endpoint with expected method, query, and body.

- [x] **Step 2: Write failing local markdown helper tests**

Cover:

- parse checklist lines for local dirty toggles
- toggle local checkbox marker without changing task record
- preserve unknown metadata tokens
- detect nested child content

- [x] **Step 3: Write failing renderer tests**

Render a note with two task items and assert:

- accessible checkbox labels
- checked state reflects task status
- dirty mode calls local toggle callback only
- clean mode calls task status callback
- conflict/ambiguous/unlinked badges render non-blocking status

- [x] **Step 4: Run frontend tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/notes-tasks.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx
```

Expected: FAIL with missing files.

- [x] **Step 5: Implement `notes-tasks.ts`**

Use `bgRequest` and `AllowedPath`. Export typed helpers:

- `listNoteTasks`
- `listTasks`
- `getTask`
- `createNoteTask`
- `updateNoteTask`
- `setNoteTaskStatus`
- `deleteNoteTask`
- `reconcileNoteTasks`
- `listTaskActivity`
- `markTaskActivityRead`

- [x] **Step 6: Update `openapi-guard.ts`**

Add the task paths used by WebUI:

```typescript
| "/api/v1/notes/tasks"
| "/api/v1/notes/tasks/{task_id}"
| "/api/v1/notes/tasks/status"
| "/api/v1/notes/tasks/activity"
| "/api/v1/notes/tasks/activity/{event_id}"
| "/api/v1/notes/{note_id}/tasks"
| "/api/v1/notes/{note_id}/tasks/reconcile"
```

- [x] **Step 7: Implement `task-markdown.ts` and `TaskChecklistPreview.tsx`**

Keep renderer compact and reusable. Do not put full note editing logic inside the renderer.

- [x] **Step 8: Run frontend shared tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/notes-tasks.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx
```

Expected: PASS.

- [x] **Step 9: Commit frontend shared slice**

```bash
git add \
  apps/packages/ui/src/services/notes-tasks.ts \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/components/Notes/task-markdown.ts \
  apps/packages/ui/src/components/Notes/TaskChecklistPreview.tsx \
  apps/packages/ui/src/services/__tests__/notes-tasks.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx
git commit -m "feat: add notes task frontend client and renderer"
```

## Task 7: `/notes` Page Task Interaction

**Files:**
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: `apps/packages/ui/src/components/Notes/notes-manager-types.ts`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx`

- [x] **Step 1: Write failing `/notes` interaction tests**

Cover:

- preview/split mode renders task-backed checkboxes
- edit mode keeps raw markdown visible
- clean checkbox toggle calls backend status endpoint and refreshes content/task snapshot
- dirty checkbox toggle changes local markdown only
- dirty save after remote version change reports conflict and preserves local draft
- agent activity notice appears for unread events and can be dismissed

- [x] **Step 2: Run `/notes` tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx
```

Expected: FAIL with missing task integration.

- [x] **Step 3: Add task state to editor hook**

In `useNotesEditorState.tsx`:

- load tasks for `selectedId`
- load unread task activity for selected note
- expose clean/dirty checkbox toggle callbacks
- clean toggle calls `setNoteTaskStatus`
- dirty toggle rewrites local markdown with `task-markdown.ts`
- on dirty save, keep using `selectedVersion`; backend conflict must surface existing stale-version behavior
- after successful save, reload tasks and activity

- [x] **Step 4: Render task checklist in preview/split**

In `NotesEditorPane.tsx`:

- pass preview markdown and task state to `TaskChecklistPreview`
- keep raw textarea/WYSIWYG behavior unchanged
- render conflict/activity notices near existing save/monitoring notices

- [x] **Step 5: Add copy strings**

Add concise labels in `option.json` for:

- task checkbox accessible label
- task conflict notice
- agent activity notice
- incomplete reconciliation warning
- task continuity notice: "Portable markdown with best-effort task continuity"

- [x] **Step 6: Run `/notes` tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx
```

Expected: PASS.

- [x] **Step 7: Commit `/notes` UI slice**

```bash
git add \
  apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx \
  apps/packages/ui/src/components/Notes/NotesEditorPane.tsx \
  apps/packages/ui/src/components/Notes/notes-manager-types.ts \
  apps/packages/ui/src/public/_locales/en/option.json \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx
git commit -m "feat: add task-backed checklists to notes page"
```

## Task 8: Notes Dock Task Interaction

**Files:**
- Modify: `apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx`
- Modify: `apps/packages/ui/src/store/notes-dock.tsx`
- Test: `apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx`

- [ ] **Step 1: Write failing Notes Dock tests**

Cover:

- dock renders compact task list for active saved note
- clean toggle calls backend status endpoint and refreshes note snapshot
- dirty toggle updates local markdown only and preserves dirty state
- failed save leaves durable task state unchanged
- remote/autonomous change while dirty queues pending-change notice
- dock does not overwrite dirty content when refreshed

- [ ] **Step 2: Run dock tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx
```

Expected: FAIL with missing dock task UI.

- [ ] **Step 3: Add dock task loading and rendering**

In `NotesDockPanel.tsx`:

- load tasks when active note has an ID
- use `TaskChecklistPreview` in compact mode
- use backend toggle only when `activeNote.isDirty === false`
- use local markdown toggle and `updateNote` when dirty
- after successful save, refresh tasks and call existing cache invalidation path

- [ ] **Step 4: Add store fields only if component state is not enough**

If pending activity notices must survive panel remount, extend `notes-dock.tsx` with:

- `taskActivityByNoteId`
- `dismissTaskActivity(noteId, eventId)`
- `pendingTaskChangeByLocalId`

Keep store additions minimal.

- [ ] **Step 5: Run dock tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit dock slice**

```bash
git add \
  apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx \
  apps/packages/ui/src/store/notes-dock.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx
git commit -m "feat: add task-backed checklists to notes dock"
```

## Task 9: Activity Notices And End-To-End Verification

**Files:**
- Modify as needed from previous tasks only.
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx`
- Test: `apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx`
- Optional E2E: `tests/e2e/notes-task-backed-todos.spec.ts` if this repo already has a running WebUI E2E harness in the active branch.

- [ ] **Step 1: Write activity notice tests**

Cover:

- agent task event appears in `/notes`
- agent task event appears next time dock opens the affected note
- dismissal marks event read/dismissed
- dismissed event is not shown repeatedly
- unread events survive reload with server response
- read/dismiss state is per user and backed by `task_event_read_state`
- autonomous task write remains blocked until the activity delivery path is verified

- [ ] **Step 2: Run activity tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx
```

Expected: FAIL until activity UI is complete.

- [ ] **Step 3: Implement activity notice behavior**

Use the task activity endpoint. Keep UI concise:

- actor/tool label
- count of changed tasks
- affected note/list
- inspect action
- dismiss action

No global notification center in v1.

Use `task_event_read_state` for per-user read/dismiss state. Do not store read/dismiss flags directly on `task_events`, because the same event can be unread for one user and dismissed for another.

- [ ] **Step 4: Run focused frontend tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Enable autonomous MCP writes after activity delivery passes**

Only after Step 4 activity UI tests pass:

- enable the autonomous-write policy path for scoped, permitted MCP task mutations
- keep approval-required behavior for agent contexts without autonomous task permission
- add/extend MCP tests proving autonomous allowed policy succeeds and records actor/tool/policy metadata
- keep denied autonomous policy tests from Task 5 passing

- [ ] **Step 6: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Notes_Tasks \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py \
  tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py \
  -f json -o /tmp/bandit_notes_tasks.json
```

Expected: PASS or only documented pre-existing/non-applicable findings.

- [ ] **Step 8: Run OpenAPI guard if frontend paths changed**

Run from `apps/packages/ui`:

```bash
bun run verify:openapi
```

Expected: PASS. If it fails because the generated spec has not been refreshed, update the checked-in path guard or snapshot according to the verifier output.

- [ ] **Step 9: Run browser smoke when dev server is available**

Scenario:

1. Open chat/workspace route with Notes Dock visible.
2. Create/open a note with:

```markdown
## Follow-up
- [ ] Review source @due(2026-06-10) @priority(high)
- [ ] Summarize findings
```

3. Mark first item done in Notes Dock.
4. Open `/notes` and verify first item is done.
5. Reopen first item from `/notes`.
6. Return to dock and verify it refreshes after save/sync.
7. Simulate MCP/agent activity response and verify activity notice appears.
8. Simulate an allowed autonomous MCP task status update and verify it writes an event before the UI marks it read.

- [ ] **Step 10: Commit final UI/activity slice**

```bash
git add \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx \
  apps/packages/ui/src/components/Notes \
  apps/packages/ui/src/components/Common/NotesDock \
  apps/packages/ui/src/store/notes-dock.tsx \
  apps/packages/ui/src/public/_locales/en/option.json
git commit -m "feat: surface notes task activity"
```

## Final Verification

- [ ] **Step 1: Run all focused tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notes_Tasks \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_reconciliation_api.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py \
  -v

bunx vitest run \
  apps/packages/ui/src/services/__tests__/notes-tasks.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts \
  apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx \
  apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx \
  apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx
```

- [ ] **Step 2: Run broader regression tests if time allows**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW tldw_Server_API/tests/MCP_unified -v
bunx vitest run apps/packages/ui/src/components/Notes apps/packages/ui/src/components/Common/NotesDock
```

- [ ] **Step 3: Run Bandit**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Notes_Tasks \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py \
  tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py \
  -f json -o /tmp/bandit_notes_tasks.json
```

- [ ] **Step 4: Final commit if any verification-only fixes were made**

```bash
git status --short
git add <only task-related changed files>
git commit -m "test: verify notes task-backed todos"
```

## Known Risks To Watch During Implementation

- `notes.py` is already large. Keep task endpoints in `notes_tasks.py` and touch `notes.py` only for post-save reconciliation hooks.
- Broad reconciliation-aware discovery can become expensive. Respect page size, scope, candidate detection, and server-side work limits.
- No hidden markdown IDs means identity is best-effort after arbitrary edits. Do not promise perfect identity in UI copy.
- Dirty-note toggles must not call durable status APIs. They are local markdown edits until save succeeds.
- Projection-changing writes against unlinked/ambiguous tasks must conflict instead of appending or rewriting arbitrary note content.
- Default delete must preserve nested child content by conflicting.
- Autonomous activity notices are in-app audit/activity notices, not reminders or global notifications.
