# Study Packs Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 of study packs: source-backed flashcard generation with strict provenance, Jobs-backed creation, citation-backed remediation, and Flashcards workspace entry points.

**Architecture:** Add a new `StudyPacks` backend domain on top of the existing flashcards system rather than inventing a second scheduler. Persist study-pack membership and citations in `ChaChaNotes_DB`, resolve only the approved V1 source classes into a canonical source bundle, run strict generation through Jobs, and expose remediation through the existing flashcard review/study-assistant surfaces.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL via `ChaChaNotes_DB`, Media DB package-native APIs, Jobs `JobManager` + `WorkerSDK`, Loguru, pytest, React, TanStack Query, Ant Design, Vitest, Bandit

---

## Scope Lock

Keep these decisions fixed during implementation:

- Phase 1 only
- supported sources are exactly:
  - notes
  - ingested media
  - selected chat messages
- chat support is selected-message only, not whole-conversation generation
- destination deck mode is `create new deck` only
- deck names must use collision-safe suffixing because deck names are globally unique
- new decks inherit the current flashcards default scheduler type and settings unless explicitly overridden
- generation is Jobs-backed
- do not reuse `run_flashcard_generate_adapter(...)` as the persistence path for study packs
- no calendar work
- no `append to existing deck`
- no automatic remediation-card creation
- Phase 1 should include backend regeneration through the same Jobs path even if the first UI remains create-first
- do not mutate old packs in place during regeneration; use `status = active|superseded` plus `superseded_by_pack_id`

## Canonical Phase 1 Contracts

The implementation should converge on these request and response shapes before frontend work starts:

```python
class StudyPackSourceSelection(BaseModel):
    source_type: Literal["note", "media", "message"]
    source_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    excerpt_text: str | None = None


class StudyPackCreateJobRequest(BaseModel):
    title: str
    workspace_id: str | None = None
    deck_mode: Literal["new"] = "new"
    deck_title: str | None = None
    scheduler_type: Literal["sm2_plus", "fsrs"] | None = None
    scheduler_settings: DeckSchedulerSettingsEnvelope | None = None
    source_items: list[StudyPackSourceSelection]


class FlashcardCitationResponse(BaseModel):
    id: int
    ordinal: int
    source_type: Literal["note", "media", "message"]
    source_id: str
    label: str | None = None
    quote: str
    chunk_id: str | None = None
    timestamp_seconds: float | None = None
    source_url: str | None = None
    locator_json: dict[str, Any] = Field(default_factory=dict)


class FlashcardDeepDiveTarget(BaseModel):
    route: str | None = None
    route_kind: Literal["exact_locator", "workspace_route", "citation_only"] = "citation_only"
    available: bool = True
    fallback_reason: str | None = None
```

Do not add Phase 2 calendar types or append-mode enums to these contracts.

## File Structure

- `tldw_Server_API/app/api/v1/schemas/study_packs.py`
  Purpose: define study-pack API requests, job-status responses, citation responses, and deep-dive response models.
- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
  Purpose: extend flashcard assistant response types with citation and study-pack metadata without breaking legacy cards.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
  Purpose: expose study-pack create/regenerate job routes, status/detail routes, and assistant response extensions under the existing flashcards router.
- `tldw_Server_API/app/core/StudyPacks/__init__.py`
  Purpose: export the new domain helpers.
- `tldw_Server_API/app/core/StudyPacks/types.py`
  Purpose: define the internal normalized source-bundle and citation dataclasses used by resolver, generator, and provenance store.
- `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
  Purpose: resolve notes, media, and selected chat messages into durable evidence snippets plus locator metadata.
- `tldw_Server_API/app/core/StudyPacks/provenance.py`
  Purpose: read/write flashcard citations, choose the deterministic primary citation, and resolve deep-dive targets.
- `tldw_Server_API/app/core/StudyPacks/generation_service.py`
  Purpose: run strict study-pack generation, validation, optional repair, deck naming, and transactional persistence orchestration.
- `tldw_Server_API/app/core/StudyPacks/jobs.py`
  Purpose: centralize study-pack job payload normalization and job-result mapping so endpoints and worker stay consistent.
- `tldw_Server_API/app/core/Flashcards/study_assistant.py`
  Purpose: extend the existing flashcard assistant context with citations, primary citation summary, study-pack summary, and deep-dive target.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: add schema/migration support and CRUD helpers for `study_packs`, `study_pack_cards`, and `flashcard_citations`, including sync/version semantics.
- `tldw_Server_API/app/services/study_pack_jobs_worker.py`
  Purpose: execute Jobs-backed study-pack generation and cleanup rules.
- `tldw_Server_API/app/main.py`
  Purpose: start the study-pack Jobs worker under an env-guarded startup flag.
- `tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py`
  Purpose: lock new table creation, CRUD helpers, sync/version defaults, regeneration status handling, and soft-delete semantics.
- `tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py`
  Purpose: verify request/response model validation and backward-compatible flashcard assistant payload shapes.
- `tldw_Server_API/tests/StudyPacks/test_source_resolver.py`
  Purpose: verify note/media/message resolution, capability gating, excerpt handling, and locator normalization.
- `tldw_Server_API/tests/StudyPacks/test_provenance.py`
  Purpose: verify primary-citation selection, deep-dive fallback ordering, and citation persistence rules.
- `tldw_Server_API/tests/StudyPacks/test_generation_service.py`
  Purpose: verify strict validation, single repair pass, collision-safe deck naming, and empty-deck cleanup rules.
- `tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py`
  Purpose: verify create/status/detail routes, owner scoping, and job-to-pack result mapping.
- `tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py`
  Purpose: verify worker execution, cancellation handling, and no-partial-data failure behavior.
- `apps/packages/ui/src/services/flashcards.ts`
  Purpose: add study-pack client types and API calls.
- `apps/packages/ui/src/services/tldw/study-pack-handoff.ts`
  Purpose: build and parse Flashcards workspace handoff routes for note/media/message study-pack entry points.
- `apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts`
  Purpose: verify handoff route parsing and serialization.
- `apps/packages/ui/src/components/Flashcards/hooks/useStudyPackQueries.ts`
  Purpose: own create-job, poll-status, and fetch-pack React Query hooks.
- `apps/packages/ui/src/components/Flashcards/hooks/index.ts`
  Purpose: export the new study-pack hooks.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx`
  Purpose: verify optimistic state, polling lifecycle, and terminal job handling.
- `apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx`
  Purpose: provide the Flashcards workspace launcher for Phase 1 study-pack creation.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx`
  Purpose: verify form validation, source prefill, job submission, and success-state navigation.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx`
  Purpose: render citation-backed remediation affordances without breaking legacy assistant flows.
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
  Purpose: host the study-pack launcher in the Flashcards workspace and bridge existing generate-from-text flows with the new source-backed flow.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  Purpose: pass citation-backed assistant context into the review UI and expose deep-dive/open-source actions.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx`
  Purpose: verify quote/deep-dive rendering and legacy-card fallback.
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
  Purpose: parse study-pack handoff intent and open the correct tab.
- `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`
  Purpose: hand off the current media item into the Flashcards study-pack launcher.
- `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
  Purpose: hand off the current note into the Flashcards study-pack launcher.
- `apps/packages/ui/src/routes/sidepanel-chat.tsx`
  Purpose: hand off selected chat messages into the Flashcards study-pack launcher.

## Stages

### Stage 1: Contracts And Persistence

**Goal:** Add the study-pack data model, API schemas, and DB helpers with repository-standard sync/version behavior.

**Success Criteria:** The backend can persist `study_packs`, `study_pack_cards`, and `flashcard_citations`; assistant response models can carry citations and deep-dive metadata; regeneration state can be represented without mutating the original pack in place; storage tests pass for SQLite and PostgreSQL-backed paths already covered by current DB abstractions.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py -v`

**Status:** Complete

### Stage 2: Source Resolution And Provenance

**Goal:** Resolve the approved Phase 1 sources into a canonical bundle and expose deterministic provenance/deep-dive behavior.

**Success Criteria:** Notes, media, and selected messages resolve into bounded evidence text plus locators, and flashcard assistant context can read citations plus deep-dive targets for generated cards.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudyPacks/test_source_resolver.py tldw_Server_API/tests/StudyPacks/test_provenance.py -v`

**Status:** Complete

### Stage 3: Strict Generation And Jobs

**Goal:** Add strict study-pack generation, Jobs-backed execution, and API status surfaces.

**Success Criteria:** The server can enqueue create and regenerate study-pack jobs, validate or repair generated output, persist only validated packs, supersede old packs during regeneration, and return pack/job status without partial visible data.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudyPacks/test_generation_service.py tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py -v`

**Status:** Complete

### Stage 4: Flashcards Workspace Launcher And Entry Points

**Goal:** Add the primary Flashcards workspace launcher plus note/media/chat handoff entry points.

**Success Criteria:** Users can launch study-pack generation from the Flashcards workspace or from supported source surfaces, submit a job, poll it, and land in the created deck.

**Tests:** `bunx vitest run apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx`

**Status:** Complete

### Stage 5: Review Remediation And Verification

**Goal:** Expose supporting quote and deep-dive remediation inside review mode, then verify the touched scope.

**Success Criteria:** Review mode shows provenance-backed remediation for study-pack cards, legacy cards still work, targeted tests pass, and Bandit reports zero new findings in the touched backend scope.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudyPacks/test_provenance.py tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py -v && bunx vitest run apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx && python -m bandit -r tldw_Server_API/app/core/StudyPacks tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/core/Flashcards/study_assistant.py tldw_Server_API/app/services/study_pack_jobs_worker.py -f json -o /tmp/bandit_study_packs_phase1.json`

**Status:** Complete

## Task 1: Add Study-Pack API Schemas And Persistent Storage

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/study_packs.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py`

- [ ] **Step 1: Write the failing schema and storage tests**

Add tests that prove:

1. the create-job request only accepts `deck_mode="new"`
2. `study_packs.status` only allows `active` or `superseded`
3. `study_pack_cards` membership survives manual deck changes because it does not derive from `deck_id`
4. `flashcard_citations` rows keep `ordinal`, `client_id`, `version`, and `deleted`
5. `StudyAssistantContextResponse` can carry `citations`, `primary_citation`, `deep_dive_target`, and `study_pack`
6. `study_packs.workspace_id` persists alongside the destination deck for workspace-scoped visibility rules

Use concrete assertions like:

```python
payload = StudyPackCreateJobRequest(
    title="Operating Systems",
    source_items=[StudyPackSourceSelection(source_type="note", source_id="note-123")],
)
assert payload.deck_mode == "new"
assert payload.source_items[0].source_type == "note"
```

```python
pack_id = db.create_study_pack(
    title="Networking",
    workspace_id="ws-1",
    deck_id=deck_id,
    source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
    generation_options_json={"deck_mode": "new"},
)
db.add_study_pack_cards(pack_id, ["fc-1", "fc-2"])
rows = db.list_study_pack_cards(pack_id)
assert [row["flashcard_uuid"] for row in rows] == ["fc-1", "fc-2"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py \
  -v
```

Expected: FAIL because the schema file, response extensions, and DB helpers do not exist yet.

- [ ] **Step 3: Implement the minimal schema and DB layer**

Make the smallest changes that establish the storage contract:

- add `StudyPackCreateJobRequest`, `StudyPackJobStatusResponse`, `StudyPackSummaryResponse`, `FlashcardCitationResponse`, and `FlashcardDeepDiveTarget`
- extend the flashcard assistant response models with optional provenance fields
- add SQLite and PostgreSQL schema ensure helpers for:
  - `study_packs`
  - `study_pack_cards`
  - `flashcard_citations`
- add CRUD helpers for pack creation, pack membership, citation insert/list, pack soft-delete, and supersede updates
- add sync-log trigger coverage matching the existing flashcards/decks patterns

Keep the new SQL boring and explicit:

```python
CREATE TABLE IF NOT EXISTS study_packs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
  source_bundle_json TEXT NOT NULL,
  generation_options_json TEXT,
  status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
  superseded_by_pack_id INTEGER REFERENCES study_packs(id) ON DELETE SET NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1
)
```

- [ ] **Step 4: Re-run the schema and storage tests**

Run the same pytest command from Step 2.

Expected: PASS with the new schema models, DB tables, and CRUD helpers in place.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/study_packs.py \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py
git commit -m "feat: add study pack storage contracts"
```

## Task 2: Build The Canonical Source Resolver

**Files:**
- Create: `tldw_Server_API/app/core/StudyPacks/__init__.py`
- Create: `tldw_Server_API/app/core/StudyPacks/types.py`
- Create: `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_source_resolver.py`

- [ ] **Step 1: Write the failing source-resolver tests**

Add tests that prove:

1. a note source resolves through `db.get_note_by_id(...)`
2. a media source resolves through package-native media DB helpers and includes either chunk locators or transcript/timestamp fallback
3. a message source requires both stable message identity and conversation identity
4. unsupported source types or missing locators fail fast
5. excerpt text is allowed only when a durable parent source still exists

Use a concrete expected bundle shape:

```python
bundle = resolve_study_sources(
    note_db=db,
    media_db=media_db,
    selections=[StudySourceSelection(source_type="message", source_id="msg-1", conversation_id="conv-1")],
)
assert bundle.items[0].source_type == "message"
assert bundle.items[0].locator["conversation_id"] == "conv-1"
assert bundle.items[0].evidence_text
```

- [ ] **Step 2: Run the resolver tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/StudyPacks/test_source_resolver.py -v
```

Expected: FAIL because the `StudyPacks` package and resolver do not exist yet.

- [ ] **Step 3: Implement the minimal resolver and helper fetches**

Implement:

- internal normalized dataclasses for `StudySourceSelection`, `StudySourceBundle`, `StudySourceBundleItem`, and `StudyCitationDraft`
- capability-gated resolver functions for:
  - notes
  - media
  - selected messages
- tiny `ChaChaNotes_DB` fetch helpers only if missing, for example `get_message_by_id(...)`
- media resolution using existing package-native helpers such as `get_media_by_id(...)`, `get_latest_transcription(...)`, and `get_unvectorized_chunks_in_range(...)`

Keep the resolver output explicit:

```python
StudySourceBundleItem(
    source_type="media",
    source_id=str(media_id),
    label=media_row.get("title") or f"Media {media_id}",
    evidence_text=snippet_text,
    locator={
        "media_id": media_id,
        "chunk_id": chunk_id,
        "timestamp_seconds": timestamp_seconds,
    },
)
```

- [ ] **Step 4: Re-run the resolver tests**

Run the same pytest command from Step 2.

Expected: PASS with deterministic resolution and capability checks.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudyPacks/__init__.py \
  tldw_Server_API/app/core/StudyPacks/types.py \
  tldw_Server_API/app/core/StudyPacks/source_resolver.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py
git commit -m "feat: add study pack source resolver"
```

## Task 3: Add Citation Persistence And Deep-Dive Resolution

**Files:**
- Create: `tldw_Server_API/app/core/StudyPacks/provenance.py`
- Modify: `tldw_Server_API/app/core/Flashcards/study_assistant.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_provenance.py`

- [ ] **Step 1: Write the failing provenance tests**

Add tests that prove:

1. `ordinal = 0` is chosen deterministically as the primary citation
2. deep-dive resolution prefers exact locator, then workspace route, then citation-only fallback
3. legacy `source_ref_type/source_ref_id` mirrors only the primary citation
4. assistant context returns empty citation lists for legacy cards without failing

Use assertions like:

```python
target = resolve_flashcard_deep_dive_target(
    citations=[
        {"ordinal": 0, "source_type": "media", "source_id": "42", "locator_json": {"chunk_id": "c7"}},
    ]
)
assert target.route_kind == "exact_locator"
assert "/media/" in (target.route or "")
```

- [ ] **Step 2: Run the provenance tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/StudyPacks/test_provenance.py -v
```

Expected: FAIL because the provenance store and deep-dive resolver do not exist yet.

- [ ] **Step 3: Implement the provenance store and assistant-context extension**

Implement:

- citation read/write helpers in `provenance.py`
- deterministic primary-citation selection
- deep-dive target resolution
- assistant-context enrichment in `build_flashcard_assistant_context(...)`

Keep the assistant payload additive so legacy callers survive:

```python
return {
    **existing_context,
    "study_pack": study_pack_summary,
    "citations": citation_rows,
    "primary_citation": primary_citation,
    "deep_dive_target": deep_dive_target,
}
```

- [ ] **Step 4: Re-run the provenance tests**

Run the same pytest command from Step 2.

Expected: PASS with deterministic primary-citation and graceful legacy fallback behavior.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudyPacks/provenance.py \
  tldw_Server_API/app/core/Flashcards/study_assistant.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/tests/StudyPacks/test_provenance.py
git commit -m "feat: add study pack provenance and deep dive support"
```

## Task 4: Implement Strict Study-Pack Generation

**Files:**
- Create: `tldw_Server_API/app/core/StudyPacks/generation_service.py`
- Modify: `tldw_Server_API/app/core/StudyPacks/types.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_generation_service.py`

- [ ] **Step 1: Write the failing generation-service tests**

Add tests that prove:

1. uncited cards are rejected
2. citations outside the allowed source bundle are rejected
3. one malformed-but-salvageable response gets exactly one repair pass
4. deck names use collision-safe suffixing
5. the service does not leave an empty deck behind on persistence failure
6. regeneration marks the prior pack `superseded` and records `superseded_by_pack_id` only after the replacement pack is committed

Use a concrete strict-output expectation:

```python
result = await service.generate_validated_cards(bundle, request)
assert result.cards[0].front
assert result.cards[0].back
assert result.cards[0].citations[0].source_id in {"note-1", "42", "msg-1"}
```

- [ ] **Step 2: Run the generation-service tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/StudyPacks/test_generation_service.py -v
```

Expected: FAIL because the strict generation service does not exist yet.

- [ ] **Step 3: Implement the strict generation and persistence orchestration**

Implement:

- a tight study-pack prompt contract that requests cards plus citations
- JSON extraction and validation logic separate from `run_flashcard_generate_adapter(...)`
- one repair pass for malformed but recoverable outputs
- deterministic primary-citation mirroring to legacy summary fields
- deck naming helper that reuses current flashcards scheduler defaults
- transactional persistence that writes:
  - deck
  - flashcards
  - study pack row
  - pack membership rows
  - citation rows

Keep the persistence entry point explicit:

```python
async def create_study_pack_from_request(
    *,
    note_db: CharactersRAGDB,
    media_db: MediaDatabase,
    request: StudyPackCreateJobRequest,
    regenerate_from_pack_id: int | None = None,
    provider: str | None,
    model: str | None,
) -> StudyPackCreationResult:
    ...
```

- [ ] **Step 4: Re-run the generation-service tests**

Run the same pytest command from Step 2.

Expected: PASS with strict validation, deck naming, and cleanup rules in place.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudyPacks/generation_service.py \
  tldw_Server_API/app/core/StudyPacks/types.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudyPacks/test_generation_service.py
git commit -m "feat: add strict study pack generation service"
```

## Task 5: Wire Study-Pack Jobs And API Endpoints

**Files:**
- Create: `tldw_Server_API/app/core/StudyPacks/jobs.py`
- Create: `tldw_Server_API/app/services/study_pack_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/main.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py`
- Test: `tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py`

- [ ] **Step 1: Write the failing Jobs and API tests**

Add tests that prove:

1. `POST /api/v1/flashcards/study-packs/jobs` enqueues a user-scoped job
2. `GET /api/v1/flashcards/study-packs/jobs/{job_id}` returns mapped job status plus pack result when complete
3. `POST /api/v1/flashcards/study-packs/{pack_id}/regenerate` enqueues a replacement job using the stored source bundle
4. failed jobs expose diagnostics but do not expose partial packs
5. the worker finalizes cancelled jobs cleanly
6. the worker obeys the empty-deck cleanup rule on late persistence failure and only marks the old pack superseded after the replacement pack commits

Use assertions like:

```python
response = client.post(
    "/api/v1/flashcards/study-packs/jobs",
    json={"title": "Biology", "source_items": [{"source_type": "note", "source_id": "note-1"}]},
)
assert response.status_code == 202
assert response.json()["job"]["status"] in {"queued", "running"}
```

- [ ] **Step 2: Run the Jobs and API tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py \
  -v
```

Expected: FAIL because the job helpers, worker, and endpoints do not exist yet.

- [ ] **Step 3: Implement the job helpers, worker, and routes**

Implement:

- job payload helpers in `jobs.py`
- worker execution in `study_pack_jobs_worker.py`
- startup wiring in `main.py` behind `STUDY_PACK_JOBS_WORKER_ENABLED`
- route handlers in `flashcards.py` for:
  - create job
  - regenerate job
  - get job status
  - get persisted pack detail

Keep the job creation boring and aligned with other user-visible jobs:

```python
job = jm.create_job(
    domain="study_packs",
    queue="default",
    job_type="study_pack_generate",
    payload=payload,
    owner_user_id=str(current_user.user_id),
    priority=5,
    max_retries=2,
)
```

- [ ] **Step 4: Re-run the Jobs and API tests**

Run the same pytest command from Step 2.

Expected: PASS with enqueue, polling, worker execution, and no-partial-data guarantees.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudyPacks/jobs.py \
  tldw_Server_API/app/services/study_pack_jobs_worker.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py
git commit -m "feat: add study pack jobs and api routes"
```

## Task 6: Add Frontend Client Types, Handoff Parsing, And Query Hooks

**Files:**
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Create: `apps/packages/ui/src/services/tldw/study-pack-handoff.ts`
- Create: `apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts`
- Create: `apps/packages/ui/src/components/Flashcards/hooks/useStudyPackQueries.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/index.ts`
- Test: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx`

- [ ] **Step 1: Write the failing frontend service and hook tests**

Add tests that prove:

1. study-pack handoff routes round-trip source ids and titles
2. `useStudyPackCreateMutation` posts the create-job request
3. `useStudyPackJobQuery` polls until terminal status
4. hooks stop polling after `completed|failed|cancelled`

Use assertions like:

```ts
const route = buildStudyPackRoute({
  title: "Networks",
  sourceItems: [{ sourceType: "media", sourceId: "42", sourceTitle: "Lecture 5" }]
})
expect(route).toContain("study_pack=1")
expect(parseStudyPackIntentFromLocation({ search: route.split("?")[1] ?? "" })?.sourceItems[0]?.sourceId).toBe("42")
```

- [ ] **Step 2: Run the frontend service and hook tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx \
```

Expected: FAIL because the service types, handoff helper, and hooks do not exist yet.

- [ ] **Step 3: Implement the minimal client and query layer**

Implement:

- study-pack request and response types in `flashcards.ts`
- API calls for create job, get job, and get pack
- a dedicated handoff helper mirroring the existing flashcard-generate route style
- React Query hooks for create and status polling

Keep the polling hook explicit:

```ts
export function useStudyPackJobQuery(jobId: number | null) {
  return useQuery({
    queryKey: ["flashcards:study-packs:job", jobId],
    queryFn: () => getStudyPackJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) =>
      isTerminalStudyPackJobStatus(query.state.data?.job?.status) ? false : 1500
  })
}
```

- [ ] **Step 4: Re-run the frontend service and hook tests**

Run the same Vitest command from Step 2.

Expected: PASS with route parsing and polling behavior in place.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/services/flashcards.ts \
  apps/packages/ui/src/services/tldw/study-pack-handoff.ts \
  apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/useStudyPackQueries.ts \
  apps/packages/ui/src/components/Flashcards/hooks/index.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx
git commit -m "feat: add study pack frontend client hooks"
```

## Task 7: Build The Flashcards Workspace Launcher And Source Entry Points

**Files:**
- Create: `apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-chat.tsx`

- [ ] **Step 1: Write the failing launcher and handoff UI tests**

Add tests that prove:

1. the drawer opens from the Flashcards workspace
2. a handoff intent from media, note, or selected message pre-fills the source list
3. submit is disabled until title and at least one source are present
4. a successful job result navigates to the created deck or review view

Use concrete expectations like:

```ts
render(<StudyPackCreateDrawer open initialIntent={{
  title: "Lecture 5",
  sourceItems: [{ sourceType: "media", sourceId: "42", sourceTitle: "Lecture 5" }]
}} />)
expect(screen.getByDisplayValue("Lecture 5")).toBeInTheDocument()
expect(screen.getByText("Lecture 5")).toBeInTheDocument()
```

- [ ] **Step 2: Run the launcher UI tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx
```

Expected: FAIL because the drawer and handoff-aware launcher do not exist yet.

- [ ] **Step 3: Implement the drawer, workspace host, and entry-point links**

Implement:

- a focused `StudyPackCreateDrawer`
- a study-pack launcher area inside `ImportExportTab`
- handoff parsing in `FlashcardsManager`
- source-entry buttons in:
  - `ViewMediaPage.tsx`
  - `NotesManagerPage.tsx`
  - `sidepanel-chat.tsx`

Keep entry points limited to supported Phase 1 sources:

```ts
navigate(buildStudyPackRoute({
  title: mediaTitle,
  sourceItems: [{ sourceType: "media", sourceId: String(mediaId), sourceTitle: mediaTitle }]
}))
```

- [ ] **Step 4: Re-run the launcher UI tests**

Run the same Vitest command from Step 2.

Expected: PASS with a working Flashcards workspace launcher and prefilled entry points.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx \
  apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx \
  apps/packages/ui/src/components/Review/ViewMediaPage.tsx \
  apps/packages/ui/src/components/Notes/NotesManagerPage.tsx \
  apps/packages/ui/src/routes/sidepanel-chat.tsx
git commit -m "feat: add study pack launcher and entry points"
```

## Task 8: Expose Citation-Backed Remediation In Review Mode

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Test: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx`

- [ ] **Step 1: Write the failing review-remediation tests**

Add tests that prove:

1. a study-pack card shows the primary quote
2. a deep-dive button/link appears when the backend supplies a route
3. legacy cards without citations still render the assistant panel without errors
4. review controls continue working while remediation UI is visible

Use assertions like:

```ts
expect(screen.getByText(/supporting quote/i)).toBeInTheDocument()
expect(screen.getByRole("link", { name: /deep dive to source/i })).toHaveAttribute("href", expect.stringContaining("/media/"))
expect(screen.getByRole("button", { name: /good/i })).toBeEnabled()
```

- [ ] **Step 2: Run the review-remediation tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx
```

Expected: FAIL because the review UI does not render citation-backed remediation yet.

- [ ] **Step 3: Implement the minimal review-mode remediation UI**

Implement:

- assistant-panel rendering for `citations`, `primary_citation`, and `deep_dive_target`
- `ReviewTab` plumbing to pass the richer assistant payload through unchanged
- graceful fallback when citations or routes are missing

Keep the UI additive:

```tsx
{primaryCitation ? (
  <Alert
    type="info"
    message="Supporting quote"
    description={primaryCitation.quote}
  />
) : null}
```

- [ ] **Step 4: Re-run the review-remediation tests**

Run the same Vitest command from Step 2.

Expected: PASS with provenance-backed remediation and legacy-card compatibility.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx
git commit -m "feat: add study pack remediation in review mode"
```

## Task 9: Run Final Verification And Security Checks

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-02-study-packs-phase1-implementation-plan.md`

- [ ] **Step 1: Run the full targeted backend and frontend test slate**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py \
  tldw_Server_API/tests/StudyPacks/test_provenance.py \
  tldw_Server_API/tests/StudyPacks/test_generation_service.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py \
  -v
bunx vitest run \
  apps/packages/ui/src/services/__tests__/study-pack-handoff.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx
```

Expected: PASS across the targeted backend and frontend scope.

- [ ] **Step 2: Run Bandit on the touched backend files**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/StudyPacks \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/core/Flashcards/study_assistant.py \
  tldw_Server_API/app/services/study_pack_jobs_worker.py \
  -f json -o /tmp/bandit_study_packs_phase1.json
```

Expected: JSON report written to `/tmp/bandit_study_packs_phase1.json` with zero new findings in the touched scope.

- [ ] **Step 3: Update the plan status markers**

Mark each stage in this document as `Complete` only after the commands above pass. Do not mark partial completion early.

- [ ] **Step 4: Commit the final verification pass**

```bash
git add Docs/superpowers/plans/2026-04-02-study-packs-phase1-implementation-plan.md
git commit -m "docs: finalize study packs phase1 implementation plan status"
```
