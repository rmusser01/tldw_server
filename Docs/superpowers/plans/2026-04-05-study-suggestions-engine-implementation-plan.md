# Study Suggestions Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first `Study Suggestions` release for quizzes and flashcards: durable suggestion snapshots, Jobs-backed async generation and refresh, persisted flashcard review sessions, editable follow-up topics, and duplicate-safe follow-up generation.

**Architecture:** Add a shared `StudySuggestions` backend domain with quiz and flashcard adapters, a permission-safe snapshot store, and a thin shared API surface for status, read, refresh, and follow-up actions. Keep quiz attempts as the native anchor, add a new persisted `flashcard_review_sessions` anchor for flashcards, and integrate both into the existing quiz results and flashcard review surfaces with shared frontend hooks/components.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL via `ChaChaNotes_DB`, Jobs `JobManager` + `WorkerSDK`, Loguru, pytest, React, TanStack Query, Ant Design, Vitest, Bandit

---

## Scope Lock

Keep these decisions fixed during implementation:

- v1 covers exactly:
  - quiz attempts
  - flashcard review sessions
- quiz attempts remain the native quiz anchor
- flashcards gain a new persisted `flashcard_review_sessions` table plus `flashcard_reviews.review_session_id`
- suggestion generation and refresh are `Jobs`-backed
- a shared `/api/v1/study-suggestions` router is allowed for:
  - anchor status
  - snapshot read
  - refresh
  - duplicate-safe follow-up actions
- follow-up actions may use a thin orchestration layer that wraps existing quiz/flashcard generation logic; do not add a second independent generator stack
- no global history page in v1
- flashcard history access stays inside the existing review workspace as `Recent study sessions`
- no notes/chat/research/presentation adapters in this plan
- no global ontology project
- no live in-session suggestions
- flashcard inactivity handling is lazy:
  - stale active sessions are closed when the next relevant read or write occurs
  - do not add a background sweeper in v1

## Canonical V1 Contracts

The implementation should converge on these shapes before frontend integration:

```python
class SuggestionStatusResponse(BaseModel):
    anchor_type: Literal["quiz_attempt", "flashcard_review_session"]
    anchor_id: int
    status: Literal["none", "pending", "ready", "failed"]
    job_id: int | None = None
    snapshot_id: int | None = None
    error: str | None = None


class SuggestionTopicResponse(BaseModel):
    canonical_label: str
    display_label: str
    evidence_class: Literal["grounded", "derived", "exploratory"]
    weakness: bool
    adjacent: bool
    selected: bool = True
    source_available: bool = True
    safe_source_refs: list[dict[str, Any]] = Field(default_factory=list)
    alternate_labels: list[str] = Field(default_factory=list)


class SuggestionSnapshotResponse(BaseModel):
    id: int
    service: Literal["quiz", "flashcards"]
    activity_type: Literal["quiz_attempt", "flashcard_review_session"]
    anchor_type: Literal["quiz_attempt", "flashcard_review_session"]
    anchor_id: int
    status: Literal["active", "superseded"]
    refreshed_from_snapshot_id: int | None = None
    payload_json: dict[str, Any]
    user_selection_json: dict[str, Any] | None = None
    live_evidence: list[dict[str, Any]] = Field(default_factory=list)
```

```python
class SuggestionActionRequest(BaseModel):
    target_service: Literal["quiz", "flashcards"]
    action_kind: Literal[
        "follow_up_quiz",
        "follow_up_flashcards",
    ]
    selected_topics: list[str]
    force_regenerate: bool = False


class SuggestionActionResponse(BaseModel):
    disposition: Literal["opened_existing", "generated"]
    snapshot_id: int
    selection_fingerprint: str
    target_service: Literal["quiz", "flashcards"]
    target_type: str
    target_id: str
```

```python
class FlashcardReviewSessionSummary(BaseModel):
    id: int
    deck_id: int | None = None
    review_mode: Literal["due", "cram"]
    tag_filter: str | None = None
    scope_key: str
    status: Literal["active", "completed", "abandoned"]
    review_count: int
    lapse_count: int
    started_at: datetime
    completed_at: datetime | None = None
    last_activity_at: datetime
    summary_metrics_json: dict[str, Any]
    suggestion_status: SuggestionStatusResponse | None = None
```

Route targets for v1:

- `GET /api/v1/study-suggestions/anchors/{anchor_type}/{anchor_id}/status`
- `GET /api/v1/study-suggestions/snapshots/{snapshot_id}`
- `POST /api/v1/study-suggestions/snapshots/{snapshot_id}/refresh`
- `POST /api/v1/study-suggestions/snapshots/{snapshot_id}/actions`
- `GET /api/v1/flashcards/review-sessions`
- `POST /api/v1/flashcards/review-sessions/end`

## File Structure

- `tldw_Server_API/app/api/v1/schemas/study_suggestions.py`
  Purpose: define shared status, snapshot, refresh, live-evidence, and follow-up action API models.
- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
  Purpose: extend flashcard review and review-session response models with `review_session_id` and recent-session payloads.
- `tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`
  Purpose: expose the shared status, snapshot read, refresh, and duplicate-safe action endpoints.
- `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
  Purpose: enqueue suggestion generation when attempts are submitted and preserve existing quiz behavior.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
  Purpose: assign review submissions to persisted review sessions, end sessions, list recent sessions, and enqueue flashcard suggestion generation.
- `tldw_Server_API/app/core/StudySuggestions/__init__.py`
  Purpose: export the new shared study-suggestions domain helpers.
- `tldw_Server_API/app/core/StudySuggestions/types.py`
  Purpose: hold the normalized context, topic candidate, snapshot payload, and action request dataclasses used across adapters and services.
- `tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py`
  Purpose: convert durable quiz attempts into normalized suggestion contexts.
- `tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py`
  Purpose: convert persisted flashcard review sessions plus linked reviews into normalized suggestion contexts, including grounding eligibility.
- `tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py`
  Purpose: resolve evidence, normalize labels, rank weakness-first, and preserve evidence classes.
- `tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`
  Purpose: persist snapshots, read frozen payloads, resolve live evidence safely, manage refresh lineage, and store anchor status.
- `tldw_Server_API/app/core/StudySuggestions/actions.py`
  Purpose: normalize user-selected topics, compute selection fingerprints, detect duplicates, dispatch existing generation flows, and record output links.
- `tldw_Server_API/app/core/StudySuggestions/jobs.py`
  Purpose: build Jobs payloads/results, normalize queue names, and centralize job metadata for snapshot generation and refresh.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: add schema/migration support and CRUD helpers for suggestion snapshots, generation links, flashcard review sessions, and `flashcard_reviews.review_session_id`.
- `tldw_Server_API/app/services/study_suggestions_jobs_worker.py`
  Purpose: run Jobs-backed snapshot generation and refresh tasks.
- `tldw_Server_API/app/main.py`
  Purpose: register the new router and start the study-suggestions Jobs worker under an env guard.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py`
  Purpose: lock request/response validation and shared API contracts.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`
  Purpose: verify snapshot and output-link persistence, sync/version fields, and refresh lineage.
- `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`
  Purpose: verify session lifecycle, stale-session cleanup, same-scope dedupe, and review row linkage.
- `tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py`
  Purpose: verify evidence classes, normalization, weakness-first ranking, adjacency filtering, and exploratory-topic rules.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py`
  Purpose: verify quiz and flashcard context extraction plus `is_source_grounded_session` behavior.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`
  Purpose: verify anchor-status, snapshot read, refresh, recent-session, and follow-up action routes.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py`
  Purpose: verify Jobs execution, result persistence, failure handling, and no-mutation refresh behavior.
- `apps/packages/ui/src/services/studySuggestions.ts`
  Purpose: add shared client types and API calls for anchor status, snapshot read, refresh, and follow-up actions.
- `apps/packages/ui/src/services/flashcards.ts`
  Purpose: extend the flashcards client with `review_session_id`, recent-session, and end-session APIs.
- `apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts`
  Purpose: own anchor polling, snapshot loading, refresh, and follow-up action mutations.
- `apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx`
  Purpose: verify pending-to-ready polling, failure handling, and action result mapping.
- `apps/packages/ui/src/components/StudySuggestions/TopicBuilder.tsx`
  Purpose: render the editable topic list with add/remove/rename/reset behavior and evidence-class affordances.
- `apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx`
  Purpose: render loading, failure, summary, topics, refresh, and follow-up actions for both quiz and flashcard surfaces.
- `apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx`
  Purpose: verify panel states, evidence badges, duplicate handling UI, and retry behavior.
- `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`
  Purpose: mount the study-suggestions panel for the selected attempt without regressing existing remediation flows.
- `apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx`
  Purpose: verify post-attempt loading, ready, failed, and action states in the results surface.
- `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
  Purpose: render the `Recent study sessions` list inside the review workspace.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx`
  Purpose: verify session list rendering, reopen behavior, and snapshot-status display.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  Purpose: track the active persisted session id, call `End Session`, auto-finish on queue exhaustion, and render recent sessions plus the shared study-suggestions panel.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx`
  Purpose: verify manual end, queue-complete auto-summary, recent-session reopen, and grounded/exploratory copy.
- `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
  Purpose: add review-session-aware mutations and query invalidation for the review workspace.

## Stages

### Stage 1: Contracts And Persistence

**Goal:** Land the new storage model and API contracts without changing behavior in the UI yet.

**Success Criteria:** Shared study-suggestion schemas validate, snapshot and output-link tables persist correctly, flashcard review sessions are durable, and `flashcard_reviews.review_session_id` can be written safely without breaking legacy rows.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py -v`

**Status:** Not Started

### Stage 2: Context Derivation And Snapshot Engine

**Goal:** Turn quiz attempts and flashcard review sessions into normalized contexts and persisted snapshots.

**Success Criteria:** Quiz and flashcard adapters emit normalized contexts, topic normalization/ranking is deterministic, grounding eligibility is explicit, and snapshot reads return frozen payloads plus best-effort live evidence.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py -v`

**Status:** Not Started

### Stage 3: Async Orchestration And Trigger Wiring

**Goal:** Add Jobs-backed generation/refresh and hook the engine to real quiz and flashcard lifecycle events.

**Success Criteria:** Quiz submission and flashcard session completion enqueue suggestions, anchor-status polling works, refresh creates a new snapshot, and follow-up actions are duplicate-safe.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py -v`

**Status:** Not Started

### Stage 4: Shared Frontend Plumbing And Quiz Surface

**Goal:** Build the reusable frontend client/hooks/components and mount them in quiz results.

**Success Criteria:** The quiz results surface shows pending, ready, failed, refresh, and follow-up states using the shared panel without regressing current remediation UI.

**Tests:** `bunx vitest run apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx`

**Status:** Not Started

### Stage 5: Flashcards Surface, History, And Verification

**Goal:** Finish the flashcards review integration, recent-session reopen flow, and verify the touched scope.

**Success Criteria:** Review mode supports manual end and queue-complete summaries, recent sessions reopen correctly, targeted backend/frontend tests pass, and Bandit reports no new findings in touched backend code.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -v && bunx vitest run apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx && python -m bandit -r tldw_Server_API/app/core/StudySuggestions tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/endpoints/study_suggestions.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/services/study_suggestions_jobs_worker.py -f json -o /tmp/bandit_study_suggestions.json`

**Status:** Not Started

## Task 1: Add Shared Study-Suggestion Schemas And Persistent Storage

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/study_suggestions.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`

- [ ] **Step 1: Write the failing schema and storage tests**

Add tests that prove:

1. `SuggestionStatusResponse.status` only accepts `none|pending|ready|failed`
2. `suggestion_snapshots.status` only accepts `active|superseded`
3. `payload_json` stores only safe labels/refs and not large quotes by default
4. `user_selection_json` is optional and survives refresh lineage
5. `suggestion_generation_links` persists one row per concrete snapshot/action result
6. sync/version fields are present on all new durable tables

Use assertions like:

```python
payload = SuggestionStatusResponse(
    anchor_type="quiz_attempt",
    anchor_id=101,
    status="pending",
    job_id=22,
)
assert payload.status == "pending"
assert payload.snapshot_id is None
```

```python
snapshot_id = db.create_suggestion_snapshot(
    service="quiz",
    activity_type="quiz_attempt",
    anchor_type="quiz_attempt",
    anchor_id=101,
    suggestion_type="study_suggestions",
    payload_json={"summary": {"score": 7}, "topics": [{"display_label": "Renal basics"}]},
)
row = db.get_suggestion_snapshot(snapshot_id)
assert row["service"] == "quiz"
assert row["status"] == "active"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  -v
```

Expected: missing schema modules, missing DB helpers, or migration failures.

- [ ] **Step 3: Implement the minimal schemas and DB helpers**

Add:

```python
class SuggestionRefreshRequest(BaseModel):
    reason: str | None = None


class SuggestionActionResponse(BaseModel):
    disposition: Literal["opened_existing", "generated"]
    snapshot_id: int
    selection_fingerprint: str
    target_service: Literal["quiz", "flashcards"]
    target_type: str
    target_id: str
```

Add DB helpers:

- `create_suggestion_snapshot(...)`
- `get_suggestion_snapshot(...)`
- `list_suggestion_snapshots_for_anchor(...)`
- `create_suggestion_generation_link(...)`
- `find_suggestion_generation_link(...)`

Keep the payload permission-safe:

- store safe refs, labels, flags, counts
- do not persist rich excerpts or long quotes by default

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same pytest command from Step 2.

Expected: PASS for schema validation and persistence tests.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/study_suggestions.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py
git commit -m "feat: add study suggestion storage contracts"
```

## Task 2: Add Flashcard Review Session Persistence And Review Linkage

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`

- [ ] **Step 1: Write the failing flashcard session tests**

Add tests that prove:

1. the first review submission in a scope creates an active session
2. stale active sessions older than 30 minutes are marked `abandoned`
3. same-scope duplicates collapse to the newest active session
4. every new review row gets `review_session_id`
5. manual end marks the session `completed`
6. legacy rows with null `review_session_id` remain readable

Use assertions like:

```python
session = db.get_or_create_flashcard_review_session(
    deck_id=12,
    review_mode="due",
    tag_filter=None,
    scope_key="due:deck:12",
)
assert session["status"] == "active"
```

```python
updated = db.review_flashcard(card_uuid, rating=1, answer_time_ms=900, review_session_id=session["id"])
review_row = db.get_latest_flashcard_review(card_uuid)
assert review_row["review_session_id"] == session["id"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py -v
```

Expected: FAIL on missing session table, missing helper methods, or response-schema mismatch.

- [ ] **Step 3: Implement the minimal session lifecycle**

Add DB helpers and endpoint wiring for:

- `get_or_create_flashcard_review_session(...)`
- `mark_flashcard_review_session_completed(...)`
- `list_flashcard_review_sessions(...)`
- `abandon_stale_flashcard_review_sessions(...)`

Extend the review response shape so the frontend can retain the active session id:

```python
class FlashcardReviewResponse(BaseModel):
    uuid: UUID
    ef: float
    interval_days: int
    repetitions: int
    lapses: int
    due_at: str | None = None
    last_reviewed_at: str | None = None
    last_modified: str | None = None
    version: int
    scheduler_type: DeckSchedulerType
    queue_state: Literal["new", "learning", "review", "relearning", "suspended"]
    step_index: int | None = None
    suspended_reason: Literal["manual", "leech"] | None = None
    next_intervals: FlashcardReviewIntervalPreviews
    review_session_id: int | None = None
```

Inside `review_flashcard(...)`:

```python
session = db.get_or_create_flashcard_review_session(...)
updated = db.review_flashcard(
    payload.card_uuid,
    payload.rating,
    payload.answer_time_ms,
    review_session_id=session["id"],
)
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same pytest command from Step 2.

Expected: PASS for lifecycle, dedupe, and write-through linkage behavior.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py
git commit -m "feat: persist flashcard review sessions"
```

## Task 3: Implement Topic Resolution, Normalization, And Service Adapters

**Files:**
- Create: `tldw_Server_API/app/core/StudySuggestions/__init__.py`
- Create: `tldw_Server_API/app/core/StudySuggestions/types.py`
- Create: `tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py`
- Create: `tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py`
- Create: `tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py`

- [ ] **Step 1: Write the failing topic-pipeline and adapter tests**

Add tests that prove:

1. source metadata outranks tags, and tags outrank derived labels
2. obvious near-duplicates normalize to one canonical label
3. weakness-first ranking preserves `weakness` before `adjacent`
4. `exploratory` topics never claim source-aware adjacency
5. quiz attempts emit stable `SuggestionContext`
6. flashcard sessions expose `is_source_grounded_session` correctly

Use concrete assertions like:

```python
candidates = resolve_topic_candidates(
    source_labels=["Kidney function"],
    tag_labels=["renal basics"],
    derived_labels=["how kidneys work"],
)
assert candidates[0].evidence_class == "grounded"
```

```python
normalized = normalize_topic_labels([" Renal Basics ", "renal-basics", "Kidney basics"])
assert normalized[0].canonical_label == "renal basics"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py \
  -v
```

Expected: missing modules and failing normalization/ranking assertions.

- [ ] **Step 3: Implement the minimal shared pipeline**

Create the normalized context types:

```python
@dataclass(slots=True)
class SuggestionContext:
    service: str
    activity_type: str
    anchor_type: str
    anchor_id: int
    workspace_id: str | None
    summary_metrics: dict[str, Any]
    performance_signals: dict[str, Any]
    source_bundle: list[dict[str, Any]]
```

Implement:

- `build_quiz_suggestion_context(...)`
- `build_flashcard_suggestion_context(...)`
- `resolve_topic_candidates(...)`
- `normalize_topic_labels(...)`
- `rank_suggestion_topics(...)`
- `is_source_grounded_session(...)`

Keep flashcard grounding rules explicit:

- provenance-backed study-pack or citation data => grounded
- coarse tags only => weakly grounded/derived
- manual or mixed-no-lineage decks => exploratory-only adjacency suppressed

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same pytest command from Step 2.

Expected: PASS for ranking, normalization, and adapter behavior.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/__init__.py \
  tldw_Server_API/app/core/StudySuggestions/types.py \
  tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py \
  tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py \
  tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py
git commit -m "feat: add study suggestion context pipeline"
```

## Task 4: Add Snapshot Service, Shared API Routes, And Jobs Worker

**Files:**
- Create: `tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`
- Create: `tldw_Server_API/app/core/StudySuggestions/jobs.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`
- Create: `tldw_Server_API/app/services/study_suggestions_jobs_worker.py`
- Modify: `tldw_Server_API/app/main.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py`

- [ ] **Step 1: Write the failing endpoint and Jobs tests**

Add tests that prove:

1. anchor status returns `none|pending|ready|failed` by `anchor_type/anchor_id`
2. snapshot read returns frozen payload plus best-effort `live_evidence`
3. refresh creates a new snapshot with `refreshed_from_snapshot_id`
4. failed Jobs surface `status="failed"` without mutating prior snapshots
5. permission failures in live evidence degrade to `source_available=False`

Use assertions like:

```python
response = client.get("/api/v1/study-suggestions/anchors/quiz_attempt/101/status")
assert response.status_code == 200
assert response.json()["status"] in {"none", "pending", "ready", "failed"}
```

```python
refresh = client.post(f"/api/v1/study-suggestions/snapshots/{snapshot_id}/refresh", json={})
assert refresh.status_code == 202
assert refresh.json()["job"]["status"] == "queued"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py \
  -v
```

Expected: FAIL on missing router, missing worker, or missing snapshot service.

- [ ] **Step 3: Implement the minimal snapshot service and Jobs path**

Add job payloads like:

```python
def build_study_suggestions_job_payload(*, job_type: str, anchor_type: str | None = None, anchor_id: int | None = None, snapshot_id: int | None = None) -> dict[str, Any]:
    return {
        "job_type": job_type,
        "anchor_type": anchor_type,
        "anchor_id": anchor_id,
        "snapshot_id": snapshot_id,
    }
```

Expose routes:

```python
@router.get("/anchors/{anchor_type}/{anchor_id}/status", response_model=SuggestionStatusResponse)
def get_suggestion_status(...): ...


@router.get("/snapshots/{snapshot_id}", response_model=SuggestionSnapshotResponse)
def get_suggestion_snapshot(...): ...
```

Wire worker startup in `main.py` using the same pattern as `study_pack_jobs_worker.py` and guard it with `STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED`.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same pytest command from Step 2.

Expected: PASS for status, read, refresh, and Jobs handling.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/snapshot_service.py \
  tldw_Server_API/app/core/StudySuggestions/jobs.py \
  tldw_Server_API/app/api/v1/endpoints/study_suggestions.py \
  tldw_Server_API/app/services/study_suggestions_jobs_worker.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py
git commit -m "feat: add study suggestion jobs and routes"
```

## Task 5: Wire Lifecycle Triggers And Duplicate-Safe Follow-Up Actions

**Files:**
- Create: `tldw_Server_API/app/core/StudySuggestions/actions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`
- Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py`

- [ ] **Step 1: Write the failing trigger and action tests**

Add tests that prove:

1. `submit_attempt(...)` enqueues study suggestions for the completed attempt
2. `POST /flashcards/review-sessions/end` completes the session and enqueues suggestions
3. `POST /study-suggestions/snapshots/{id}/actions` returns `opened_existing` when the same fingerprint already exists
4. `force_regenerate=True` bypasses duplicate open behavior
5. selection fingerprints include snapshot id, target service/type, normalized topics, action kind, and generator version

Use assertions like:

```python
fingerprint = build_selection_fingerprint(
    snapshot_id=10,
    target_service="quiz",
    target_type="follow_up_quiz",
    selected_topics=["renal basics", "electrolyte handling"],
    generator_version="v1",
)
assert "10" in fingerprint or fingerprint
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py \
  -v
```

Expected: FAIL on missing enqueue hooks, missing end-session route, or duplicate-handling gaps.

- [ ] **Step 3: Implement the minimal trigger and action wiring**

In `quizzes.py`, enqueue after a successful submit:

```python
attempt = db.submit_attempt(...)
enqueue_study_suggestions_job(
    anchor_type="quiz_attempt",
    anchor_id=attempt["id"],
)
return attempt
```

In `flashcards.py`, add:

```python
@router.post("/review-sessions/end")
def end_review_session(payload: FlashcardReviewSessionEndRequest, ...):
    session = db.mark_flashcard_review_session_completed(payload.review_session_id)
    enqueue_study_suggestions_job(anchor_type="flashcard_review_session", anchor_id=session["id"])
    return session
```

In `actions.py`, normalize topics before fingerprinting:

```python
fingerprint_payload = {
    "snapshot_id": snapshot_id,
    "target_service": payload.target_service,
    "target_type": payload.action_kind,
    "topics": sorted(normalize_topic_label(topic) for topic in payload.selected_topics),
    "generator_version": STUDY_SUGGESTIONS_GENERATOR_VERSION,
}
```

Then:

- find prior generation link by fingerprint
- if found and `force_regenerate` is false, return `opened_existing`
- else dispatch the existing quiz or flashcard generation flow and record the new link

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same pytest command from Step 2.

Expected: PASS for lifecycle triggers and duplicate-safe follow-up actions.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/actions.py \
  tldw_Server_API/app/api/v1/endpoints/quizzes.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/api/v1/endpoints/study_suggestions.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py
git commit -m "feat: wire study suggestion lifecycle triggers"
```

## Task 6: Add Shared Frontend Services, Hooks, And Panel Components

**Files:**
- Create: `apps/packages/ui/src/services/studySuggestions.ts`
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Create: `apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts`
- Create: `apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx`
- Create: `apps/packages/ui/src/components/StudySuggestions/TopicBuilder.tsx`
- Create: `apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx`
- Create: `apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`

- [ ] **Step 1: Write the failing shared frontend tests**

Add tests that prove:

1. the hook polls by anchor until `ready` or `failed`
2. topic builder supports add/remove/rename/reset
3. manual topics render as exploratory
4. duplicate follow-up responses show `Open existing`
5. refresh keeps the old snapshot visible until the new one resolves

Use assertions like:

```tsx
expect(screen.getByText("Renal basics")).toBeInTheDocument()
await user.click(screen.getByRole("button", { name: /add topic/i }))
expect(screen.getByDisplayValue("")).toBeInTheDocument()
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
```

Expected: FAIL on missing services, missing hooks, and missing panel components.

- [ ] **Step 3: Implement the minimal shared frontend layer**

Add shared client types:

```ts
export type SuggestionAnchorType = "quiz_attempt" | "flashcard_review_session"
export type SuggestionStatus = "none" | "pending" | "ready" | "failed"
```

Expose service calls:

- `getSuggestionStatus(anchorType, anchorId)`
- `getSuggestionSnapshot(snapshotId)`
- `refreshSuggestionSnapshot(snapshotId)`
- `runSuggestionAction(snapshotId, payload)`
- `endFlashcardReviewSession(reviewSessionId)`
- `listRecentFlashcardReviewSessions(params)`

Keep topic-builder state local to the panel and submit normalized labels back to the server.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same Vitest command from Step 2.

Expected: PASS for polling, topic editing, and panel states.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/services/studySuggestions.ts \
  apps/packages/ui/src/services/flashcards.ts \
  apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts \
  apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  apps/packages/ui/src/components/StudySuggestions/TopicBuilder.tsx \
  apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx \
  apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx \
  apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts
git commit -m "feat: add shared study suggestions frontend plumbing"
```

## Task 7: Mount Study Suggestions In Quiz Results

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx`

- [ ] **Step 1: Write the failing quiz-results UI test**

Add tests that prove:

1. selecting an attempt shows the study-suggestions loading state
2. a ready snapshot renders summary, topic builder, and both actions
3. a failed status renders retry without hiding existing remediation controls
4. follow-up action results route to the right destination or open existing artifacts

Use assertions like:

```tsx
expect(screen.getByText(/generate follow-up quiz/i)).toBeInTheDocument()
expect(screen.getByText(/generate flashcards/i)).toBeInTheDocument()
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx
```

Expected: FAIL because `ResultsTab` does not yet mount the shared study-suggestions panel.

- [ ] **Step 3: Implement the minimal quiz integration**

In `ResultsTab.tsx`:

- derive the anchor from `selectedAttemptId`
- mount `StudySuggestionsPanel` beside the existing attempt summary/remediation UI
- use:
  - primary action => `follow_up_quiz`
  - secondary action => `follow_up_flashcards`
- keep the existing remediation drawer/modal behavior untouched

- [ ] **Step 4: Run the targeted test to verify it passes**

Run the same Vitest command from Step 2.

Expected: PASS for ready, pending, failed, and action states.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx
git commit -m "feat: add study suggestions to quiz results"
```

## Task 8: Mount Study Suggestions And Recent Sessions In Flashcard Review

**Files:**
- Create: `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Test: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx`

- [ ] **Step 1: Write the failing flashcard review UI tests**

Add tests that prove:

1. the review surface stores the active `review_session_id` returned by review submissions
2. clicking `End Session` completes the session and reveals the panel
3. queue exhaustion auto-calls the end-session path once
4. `Recent study sessions` lists completed sessions and reopens linked snapshots
5. exploratory-only sessions use weaker copy and suppress source-aware adjacency claims

Use assertions like:

```tsx
await user.click(screen.getByRole("button", { name: /end session/i }))
expect(await screen.findByText(/generate focused flashcards/i)).toBeInTheDocument()
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx
```

Expected: FAIL because review sessions are not yet surfaced in the UI.

- [ ] **Step 3: Implement the minimal flashcards integration**

In `ReviewTab.tsx`:

- store the latest `review_session_id` from review responses
- show an `End Session` button only when there is an active persisted session
- detect the transition from `activeCard != null` to `activeCard == null` and call the end-session mutation once
- mount `StudySuggestionsPanel` for the completed session using:
  - primary action => `follow_up_flashcards`
  - secondary action => `follow_up_quiz`
- render `RecentStudySessions` below the active review area

Keep existing review analytics and undo behavior intact.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run the same Vitest command from Step 2.

Expected: PASS for manual end, queue-complete summary, and history reopen behavior.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx
git commit -m "feat: add study suggestions to flashcard review"
```

## Task 9: Verify The Integrated Slice And Clean Up Regressions

**Files:**
- Modify: any touched file only if verification reveals real regressions

- [ ] **Step 1: Run the targeted backend test suite**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/StudySuggestions -v
```

Expected: PASS for the full backend study-suggestions suite.

- [ ] **Step 2: Run the targeted frontend test suite**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.study-suggestions.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx
```

Expected: PASS for the full UI slice.

- [ ] **Step 3: Run Bandit on the touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/StudySuggestions \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/endpoints/study_suggestions.py \
  tldw_Server_API/app/api/v1/endpoints/quizzes.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/services/study_suggestions_jobs_worker.py \
  -f json -o /tmp/bandit_study_suggestions.json
```

Expected: zero new findings in the touched scope.

- [ ] **Step 4: Run one manual smoke pass if the local app can be started**

Run, if the environment is already configured:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --reload
```

Smoke paths:

1. finish a quiz attempt and verify suggestions move from pending to ready
2. finish or end a flashcard session and verify recent-session reopen works
3. use the same selected topics twice and verify `Open existing` appears

If manual smoke is not possible in the session, record that explicitly instead of claiming it ran.

- [ ] **Step 5: Commit the final verification fixes**

```bash
git add <touched-files>
git commit -m "fix: finalize study suggestions integration"
```
