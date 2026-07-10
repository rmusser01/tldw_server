# Source-Grounded Spaced Repetition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Flashcards-owned source-review schedule slice from `TASK-12932`: users create source-grounded review plans, see due occurrences, start/resume them, and manually complete or skip them.

**Architecture:** Store source-review plans and occurrences in `ChaChaNotes_DB` so the feature inherits Flashcards auth, ownership, sync/version behavior, and local-first persistence. Keep date math and launch-state construction in a small Flashcards core helper module, expose routes under `/api/v1/flashcards/source-review-plans`, and wire the WebUI through `services/flashcards.ts`, React Query hooks, a planner drawer, and a due panel inside the Flashcards Study tab. Start actions return launch metadata only; no quiz/card/cloze artifacts are generated in this task.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL-compatible ChaChaNotes DB patterns, pytest, React, TypeScript, TanStack Query, Ant Design, Vitest.

---

## File Structure

Create:

- `tldw_Server_API/app/core/Flashcards/source_review.py` - date math, schedule validation helpers, launch metadata builders, and source-bundle normalization.
- `tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py` - DB and helper coverage for schedule creation, due filtering, transitions, soft delete, sync rows, and month-end math.
- `tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py` - PostgreSQL schema/migration parity coverage using `pg_database_config`.
- `tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py` - route and schema integration coverage for create/list/due/start/complete/skip/delete.
- `apps/packages/ui/src/services/__tests__/flashcards-source-review.test.ts` - client typing/path tests.
- `apps/packages/ui/src/components/Flashcards/hooks/useSourceReviewQueries.ts` - React Query wrappers for source-review APIs.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx` - hook invalidation and mutation tests.
- `apps/packages/ui/src/components/Flashcards/components/SourceReviewPlanDrawer.tsx` - schedule creation drawer.
- `apps/packages/ui/src/components/Flashcards/components/SourceReviewDuePanel.tsx` - due occurrence panel and start/complete/skip actions.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx` - planner validation and create payload tests.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx` - empty, start, resume, complete, and skip UI tests.
- `apps/packages/ui/src/services/tldw/source-review-handoff.ts` - helpers that turn source-review launch state into reread text, in-memory/session-backed Flashcards generate intent, and tokenized Quiz generate intent.
- `apps/packages/ui/src/services/tldw/__tests__/source-review-handoff.test.ts` - route/payload derivation tests.

Modify:

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` - add schema version, tables, sync triggers, sequence table entries, JSON field handling, and CRUD/transition methods.
- `tldw_Server_API/app/api/v1/schemas/flashcards.py` - add source-review request/response DTOs and validators, reusing `StudyPackSourceSelection`.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py` - add Flashcards-owned source-review route handlers.
- `apps/packages/ui/src/services/flashcards.ts` - add source-review types and client functions.
- `apps/packages/ui/src/services/tldw/openapi-guard.ts` - add narrow new Flashcards paths to `ClientPath`.
- `apps/packages/ui/src/components/Flashcards/hooks/index.ts` - export source-review hooks.
- `apps/packages/ui/src/components/Flashcards/components/index.ts` - export new components.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` - render source-review due panel and planner entry point.
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx` - route source-review launch handoffs to Flashcards transfer tasks when needed.
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx` - accept source-review generate handoff fields for cloze/basic defaults if needed.
- `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx` - parse source-review quiz handoff and default to Generate tab.
- `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx` - preselect supported media/note sources from source-review handoff and show snapshot summary.
- `backlog/tasks/task-12932 - Add-source-grounded-spaced-repetition-review-schedules.md` - reference this plan and record verification.

Do not create a root-level source-review API, a scheduler engine, external notifications, or automatic artifact generation.

---

### Task 1: Backend Helper And Schema Tests

**Files:**
- Create: `tldw_Server_API/app/core/Flashcards/source_review.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py`

- [x] **Step 1: Write failing helper tests**

Add tests for day/month date math, duplicate computed `(due_at, activity_type)`, offset caps, launch metadata size, and source-bundle normalization.

```python
def test_source_review_month_offset_clamps_to_month_end():
    due_at = compute_source_review_due_at(
        starts_on=date(2026, 1, 31),
        timezone_name="America/Los_Angeles",
        offset_value=1,
        offset_unit="month",
    )
    assert due_at == datetime(2026, 2, 28, 8, 0, tzinfo=timezone.utc)
```

- [x] **Step 2: Run helper tests and confirm they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py -q
```

Expected: fails with missing `source_review` module or missing helper functions.

- [x] **Step 3: Implement minimal helper module**

Implement these public helpers:

```python
SourceReviewActivity = Literal["reread", "quiz", "flashcards", "cloze"]
SourceReviewOffsetUnit = Literal["day", "month"]

def compute_source_review_due_at(
    *, starts_on: date, timezone_name: str, offset_value: int, offset_unit: SourceReviewOffsetUnit
) -> datetime: ...

def normalize_source_review_bundle(source_items: Sequence[StudyPackSourceSelection | Mapping[str, Any]]) -> dict[str, Any]: ...

def build_source_review_launch_metadata(
    *, activity_type: SourceReviewActivity, plan_id: int, occurrence_id: int, created_at: str
) -> dict[str, Any]: ...
```

Use `zoneinfo.ZoneInfo`, `calendar.monthrange`, and stdlib `datetime`. Validate offset caps with `3650` for days and `120` for months.

- [x] **Step 4: Run helper tests and confirm they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py -q
```

Expected: helper-only tests pass; DB-specific tests can still be absent until Task 2.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Flashcards/source_review.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py
git commit -m "test: cover source review scheduling helpers"
```

---

### Task 2: ChaChaNotes Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py`

- [x] **Step 1: Add failing DB tests**

Cover:

- create plan stores wrapped `{"items": [...]}` source bundle and creates occurrences
- due query includes `pending` and `in_progress` due rows, excludes future/completed/skipped/deleted rows
- list and due ordering are stable
- start is idempotent and stores thin launch metadata
- full transition table
- delete soft-deletes plan plus occurrences in one transaction
- delete writes sync rows once; repeated delete does not mutate versions or sync rows
- PostgreSQL schema creation has matching tables, indexes, and sync triggers

```python
def test_source_review_delete_is_idempotent_and_deletes_occurrences_once(db):
    plan_id = db.create_source_review_plan(...)
    assert db.soft_delete_source_review_plan(plan_id) is True
    assert db.soft_delete_source_review_plan(plan_id) is False
    rows = db.execute_query(
        "SELECT entity, operation FROM sync_log WHERE entity IN ('source_review_plans','source_review_occurrences')"
    ).fetchall()
    assert [row["operation"] for row in rows].count("delete") == expected_delete_count
```

Use `test_source_review_plans_postgres.py` for real PostgreSQL schema coverage patterned after `test_note_folders_postgres.py`:

```python
def test_postgres_source_review_schema_has_tables_indexes_and_sync_triggers(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="source-review-pg-test", backend=backend)
    try:
        # Assert source_review_plans/source_review_occurrences exist, core indexes exist,
        # trigger names for create/update/delete sync rows are registered, and at least
        # one create/update/delete path writes the expected sync_log row in PostgreSQL.
        ...
    finally:
        db.close_connection()
```

- [x] **Step 2: Run DB tests and confirm they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py -q
```

Expected: fails on missing DB methods/tables.

- [x] **Step 3: Add schema migration**

In `ChaChaNotes_DB.py`:

- increment `_CURRENT_SCHEMA_VERSION`
- add `source_review_plans` and `source_review_occurrences`
- add indexes for plan, due/status, deleted, and stable ordering
- add sync triggers for create/update/delete on both tables
- add table IDs to `_POSTGRES_SEQUENCE_TABLES`
- keep SQLite and PostgreSQL branches equivalent where the file has dialect-specific sections
- add a dedicated Postgres ensure path, for example `_ensure_source_review_schema_postgres(conn)`, and call it from `_initialize_schema_postgres()` alongside other Flashcards/Study Pack ensure hooks

Use entity names exactly:

```text
source_review_plans
source_review_occurrences
```

- [x] **Step 4: Add DB methods**

Add methods near the Flashcards/Study Packs section:

```python
def create_source_review_plan(self, *, title: str, starts_on: str, timezone_name: str, source_bundle_json: dict[str, Any], schedule: list[dict[str, Any]]) -> int: ...
def list_source_review_plans(self, *, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]: ...
def list_due_source_review_occurrences(self, *, now_utc: str, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]: ...
def start_source_review_occurrence(self, occurrence_id: int) -> dict[str, Any]: ...
def complete_source_review_occurrence(self, occurrence_id: int, *, completion_source: str = "manual") -> dict[str, Any]: ...
def skip_source_review_occurrence(self, occurrence_id: int) -> dict[str, Any]: ...
def soft_delete_source_review_plan(self, plan_id: int) -> bool: ...
```

Return joined plan/occurrence rows with deserialized JSON fields where route serialization needs them.

- [x] **Step 5: Run DB tests and confirm they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py -q
```

Expected: all source-review DB tests pass.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py
git commit -m "feat: persist source review plans"
```

---

### Task 3: Flashcards API Routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Test: `tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py`

- [x] **Step 1: Write failing API/schema tests**

Cover validation and route behavior:

- title/source/schedule/timezone validation
- source item count, excerpt length, locator size
- offset caps and duplicate computed due/activity rows
- create/list/due/start/complete/skip/delete happy paths
- delete missing plan is 404; repeated delete returns `{ "deleted": false }`
- start/complete/skip against a deleted plan return 404
- start/complete/skip against a deleted occurrence return 404
- due `now` is UTC ISO
- start response includes composed `launch_state.source_bundle`; stored DB JSON remains thin

```python
def test_create_source_review_plan_rejects_duplicate_computed_due_activity(client_with_flashcards_db):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/source-review-plans",
        json={...},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 422
```

- [x] **Step 2: Run API tests and confirm they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py -q
```

Expected: fails on missing schemas/routes.

- [x] **Step 3: Add Pydantic DTOs**

In `schemas/flashcards.py`, add:

```python
SourceReviewActivity = Literal["reread", "quiz", "flashcards", "cloze"]
SourceReviewOffsetUnit = Literal["day", "month"]

class SourceReviewScheduleRow(BaseModel): ...
class SourceReviewPlanCreateRequest(BaseModel): ...
class SourceReviewOccurrenceResponse(BaseModel): ...
class SourceReviewLaunchStateResponse(BaseModel): ...
class SourceReviewPlanResponse(BaseModel): ...
class SourceReviewPlanListResponse(BaseModel): ...
class SourceReviewDueListResponse(BaseModel): ...
class SourceReviewOccurrenceActionResponse(BaseModel): ...
class SourceReviewPlanDeleteResponse(BaseModel): ...
```

Reuse `StudyPackSourceSelection` from `schemas.study_packs`. Accept `source_title` through that model and serialize canonical `label`.

- [x] **Step 4: Add route handlers under the existing Flashcards router**

In `endpoints/flashcards.py`, add handlers for:

```text
POST   /source-review-plans
GET    /source-review-plans
GET    /source-review-plans/due
POST   /source-review-plans/occurrences/{occurrence_id}/start
POST   /source-review-plans/occurrences/{occurrence_id}/complete
POST   /source-review-plans/occurrences/{occurrence_id}/skip
DELETE /source-review-plans/{plan_id}
```

Use `get_chacha_db_for_user`, `get_request_user`, and existing `map_db_error_to_http` patterns. Cap list/due `limit` at 100 and default to 50.

- [x] **Step 5: Run API and existing Flashcards endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -q
```

Expected: new source-review API tests pass; existing Flashcards route tests remain green.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py
git commit -m "feat: expose source review plan APIs"
```

---

### Task 4: Frontend Service, Hooks, And Handoff Helpers

**Files:**
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Create: `apps/packages/ui/src/services/tldw/source-review-handoff.ts`
- Create: `apps/packages/ui/src/services/__tests__/flashcards-source-review.test.ts`
- Create: `apps/packages/ui/src/services/tldw/__tests__/source-review-handoff.test.ts`
- Create: `apps/packages/ui/src/components/Flashcards/hooks/useSourceReviewQueries.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/index.ts`
- Create: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx`

- [x] **Step 1: Write failing service and handoff tests**

Assert client functions hit the exact paths, query params, methods, and payloads. Assert handoff helpers:

- derive `source_items` from `launch_state.source_bundle.items`
- build reread display content
- build bounded in-memory Flashcards generate handoff text from excerpts without putting the full payload in a URL
- build Quiz generate route with a short `source_review_token` and store the full source-review payload in `sessionStorage`
- reject missing/expired source-review handoff tokens without throwing

- [x] **Step 2: Run service tests and confirm they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/flashcards-source-review.test.ts src/services/tldw/__tests__/source-review-handoff.test.ts
```

Expected: fails on missing functions/modules.

- [x] **Step 3: Add service types and functions**

Add types mirroring backend DTOs to `services/flashcards.ts`:

```ts
export type SourceReviewActivity = "reread" | "quiz" | "flashcards" | "cloze"
export type SourceReviewPlanCreateRequest = { ... }
export type SourceReviewPlanResponse = { ... }
export type SourceReviewDueListResponse = { items: SourceReviewOccurrence[]; total: number; now: string }
export type SourceReviewOccurrenceActionResponse = SourceReviewOccurrence & { launch_state?: SourceReviewLaunchState | null }
```

Add functions:

```ts
createSourceReviewPlan
listSourceReviewPlans
listDueSourceReviewOccurrences
startSourceReviewOccurrence
completeSourceReviewOccurrence
skipSourceReviewOccurrence
deleteSourceReviewPlan
```

- [x] **Step 4: Update `ClientPath`**

Add only:

```ts
| "/api/v1/flashcards/source-review-plans"
| "/api/v1/flashcards/source-review-plans/due"
| "/api/v1/flashcards/source-review-plans/{plan_id}"
| "/api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/start"
| "/api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/complete"
| "/api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/skip"
```

- [x] **Step 5: Add React Query hooks**

Create hooks:

```ts
useSourceReviewPlansQuery
useDueSourceReviewOccurrencesQuery
useCreateSourceReviewPlanMutation
useStartSourceReviewOccurrenceMutation
useCompleteSourceReviewOccurrenceMutation
useSkipSourceReviewOccurrenceMutation
useDeleteSourceReviewPlanMutation
```

Invalidate query keys prefixed with `flashcards:source-review`.

- [x] **Step 6: Add bounded source-review handoff storage**

In `source-review-handoff.ts`, do not serialize source snapshots into query strings. Implement tokenized handoff helpers:

```ts
const SOURCE_REVIEW_HANDOFF_PREFIX = "tldw:source-review-handoff:"

export function saveSourceReviewHandoff(payload: SourceReviewHandoffPayload): string
export function loadSourceReviewHandoff(token: string): SourceReviewHandoffPayload | null
export function buildSourceReviewQuizRoute(payload: SourceReviewHandoffPayload): string
```

`buildSourceReviewQuizRoute` returns a short route such as:

```text
/quiz?tab=generate&source_review=1&source_review_token=<token>
```

It must not include excerpt text or the full source bundle in the URL. Store only in `sessionStorage`; if storage is unavailable, return a safe fallback route to `/quiz?tab=generate` and let the caller show a non-blocking error.

- [x] **Step 7: Run service and hook tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/flashcards-source-review.test.ts src/services/tldw/__tests__/source-review-handoff.test.ts src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx
```

Expected: all service/hook tests pass.

- [x] **Step 8: Commit**

```bash
git add apps/packages/ui/src/services/flashcards.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/services/tldw/source-review-handoff.ts apps/packages/ui/src/services/__tests__/flashcards-source-review.test.ts apps/packages/ui/src/services/tldw/__tests__/source-review-handoff.test.ts apps/packages/ui/src/components/Flashcards/hooks/useSourceReviewQueries.ts apps/packages/ui/src/components/Flashcards/hooks/index.ts apps/packages/ui/src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx
git commit -m "feat: add source review frontend client"
```

---

### Task 5: Flashcards UI Planner And Due Panel

**Files:**
- Create: `apps/packages/ui/src/components/Flashcards/components/SourceReviewPlanDrawer.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/SourceReviewDuePanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/index.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx`

- [x] **Step 1: Write failing component tests**

Planner tests:

- preset rows are Day 1/3/7/14/28/3 months/6 months
- invalid rows block create and show row errors
- exact duplicate rows show row errors
- valid create payload includes timezone and all schedule rows

Due panel tests:

- empty state
- pending start calls start mutation
- in-progress resume uses existing launch state
- reread launch shows source snapshot inline
- complete calls complete mutation
- skip calls skip mutation
- flashcards/cloze launch calls generate handoff without auto-generating artifacts

- [x] **Step 2: Run component tests and confirm they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx
```

Expected: fails on missing components.

- [x] **Step 3: Implement planner drawer**

Build a drawer similar in footprint to `StudyPackCreateDrawer`:

- title input
- native date input for `starts_on`
- source entry fields for type/id/label/excerpt/locator JSON
- schedule table/list with editable value/unit/activity
- browser timezone default
- row validation without silently dropping invalid rows
- create button disabled until title, source, and schedule are valid

- [x] **Step 4: Implement due panel**

Place the panel near the top of `ReviewTab`. Keep it compact so normal flashcard review remains primary. Use icons in action buttons where existing UI patterns do.

Behavior:

- fetch due items only when Review tab is active
- start/resume occurrence
- display reread source snapshot inline
- complete/skip mutation buttons
- for `flashcards` and `cloze`, route to Flashcards generate prefill via manager handoff
- for `quiz`, navigate to `/quiz` with only a short source-review token in the query string; the source snapshot remains in `sessionStorage`

- [x] **Step 5: Add manager and generate handoff support**

Wire `FlashcardsManager` so `SourceReviewDuePanel` can request:

```ts
onSourceReviewGenerate(intent)
onSourceReviewQuiz(intent)
```

Extend `GeneratePanel` only as needed to select cloze/default card type from the source-review generate handoff. Do not trigger generation automatically.

- [x] **Step 6: Run component tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx
```

Expected: new component tests pass and existing ReviewTab coverage remains green.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Flashcards/components/SourceReviewPlanDrawer.tsx apps/packages/ui/src/components/Flashcards/components/SourceReviewDuePanel.tsx apps/packages/ui/src/components/Flashcards/components/index.ts apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx
git commit -m "feat: add source review flashcards UI"
```

---

### Task 6: Quiz Handoff UI

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
- Create or modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx`

- [x] **Step 1: Write failing quiz handoff tests**

Cover:

- `/quiz?tab=generate&source_review=1&source_review_token=...` opens Generate tab
- media source item preselects media source ID
- note source item preselects note source ID
- message or unsupported source item appears as snapshot context instead of breaking form
- generate button is not auto-clicked
- source excerpt text is not present in `window.location.href`
- missing/expired token falls back to Generate tab with a recoverable message

- [x] **Step 2: Run tests and confirm they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx
```

Expected: fails on missing handoff parse/props.

- [x] **Step 3: Implement quiz source-review handoff**

Use `source-review-handoff.ts` to parse the short route token, load the full payload from `sessionStorage`, and pass an `initialSourceReviewIntent` prop into `GenerateTab`. Preselect media/note IDs using existing state setters and render a small source-review context summary for snapshot-only items. Never read excerpt text directly from query parameters.

- [x] **Step 4: Run quiz handoff tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx
```

Expected: tests pass.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Quiz/QuizPlayground.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx
git commit -m "feat: wire source review quiz handoff"
```

---

### Task 7: Final Verification And Task Record

**Files:**
- Modify: `backlog/tasks/task-12932 - Add-source-grounded-spaced-repetition-review-schedules.md`

- [x] **Step 1: Run backend focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py -q
```

Expected: all pass.

- [x] **Step 2: Run frontend focused tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/flashcards-source-review.test.ts src/services/tldw/__tests__/source-review-handoff.test.ts src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx
```

Expected: all pass.

- [x] **Step 3: Run OpenAPI/path guard**

Run:

```bash
cd apps/packages/ui && bun run verify:openapi
```

Expected: new Flashcards paths are accepted by the guard.

- [x] **Step 4: Run Bandit on touched backend paths**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Flashcards/source_review.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_task_12932.json
```

Expected: no new findings in touched code. If the command reports existing unrelated findings, document them and fix any finding introduced by this task.

- [x] **Step 5: Run diff checks**

Run:

```bash
git diff --check
rg -n 'TO''DO|TB''D|PLACE''HOLDER' tldw_Server_API/app/core/Flashcards/source_review.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py apps/packages/ui/src/services/flashcards.ts apps/packages/ui/src/components/Flashcards apps/packages/ui/src/components/Quiz
```

Expected: `git diff --check` exits 0; placeholder scan has no matches.

- [x] **Step 6: Update Backlog task**

Record:

- implementation summary
- verification commands and results
- Bandit result path
- modified files
- known skips or blockers, if any

- [x] **Step 7: Commit final task metadata**

```bash
git add Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md 'backlog/tasks/task-12932 - Add-source-grounded-spaced-repetition-review-schedules.md'
git commit -m "docs: finalize source review task record"
```

---

## Execution Notes

- Use `source .venv/bin/activate` before Python test, Bandit, or pytest commands.
- Keep public API routes under `/api/v1/flashcards`; do not add a root-level source-review router.
- Keep launch storage thin. Full source snapshots live in plan `source_bundle_json` and are composed into API responses.
- Do not call quiz/flashcard/cloze generation from the start endpoint or from the due panel automatically.
- Prefer existing `StudyPackSourceSelection`, Flashcards query hooks, and existing handoff helpers over new parallel abstractions.
- If schema migration work gets large, implement SQLite first with tests, then verify PostgreSQL compatibility paths in the same DB task before moving to API routes.
