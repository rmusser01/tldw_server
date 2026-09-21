# OSCE Scenario Practice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-backed OSCE station generation, authoring, private self-assessed practice, and deterministic results to the existing Quiz workspace without introducing automated scoring or exposing marking-guide content before reveal.

**Architecture:** Extend quizzes with a first-class `activity_type`, while storing OSCE stations and practice attempts in dedicated per-user tables. Keep strict station contracts, protected nested IDs, lifecycle transitions, response projection, and generation verification in focused backend modules; expose them through an included quiz subrouter. Add focused shared WebUI services and OSCE components under `apps/packages/ui/src`, leaving the Next.js app and browser extension as consumers of the same Quiz workspace.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, SQLite, PostgreSQL, Pytest, Hypothesis where useful, React 18, TypeScript, Ant Design 6, TanStack Query 5, Vitest, Testing Library, Playwright, Ruff, Bandit.

**Spec:** `Docs/superpowers/specs/2026-09-10-osce-scenario-practice-design.md`

## Global Constraints

- OSCE is a quiz `activity_type` (`questions` or `osce`), never a `QuestionType`.
- Allocate the next free ChaChaNotes database schema version after rebasing; it is expected to be `67`, but do not reuse a version that has landed meanwhile.
- Keep the `osce_scenario` generation profile `planned` until Task 10 activates the compatible backend and shared WebUI together.
- Use `osce.station.v1`; reject unknown fields, unsupported schema versions, scoring fields, pass thresholds, generated feedback, and chain-of-thought fields.
- Enforce these bounds: title 1-200; instructions/task 1-4,000; patient context 1-10,000; duration 60-7,200 seconds; checklist 1-50; rubric domains 1-12 with 2-6 ordered levels; expected key points 1-50; notes at most 10,000.
- Nested checklist, rubric-domain, rubric-level, and key-point UUIDs are server-owned. Creates discard supplied/model IDs; updates preserve recognized IDs, assign new IDs, and reject unknown, duplicate, or moved IDs.
- Generated stations require resolvable citations on patient context, every checklist rationale, and every expected key point; manual stations may be uncited.
- Candidate notes may be persisted but must never enter LLM or verification calls, logs, analytics, browser marking-guide storage, or exports.
- Candidate-phase API models structurally omit marking guides, rationales, citation quotes, and expected answers. Do not return redacted placeholders.
- Practice lifecycle is monotonic: `in_progress` -> `self_assessment` -> `completed`; completed attempts are immutable.
- Checklist and rubric summaries remain separate. Do not calculate aggregate scores, percentages, thresholds, or pass/fail outcomes.
- Timer behavior is advisory, server-time based, includes time away, freezes on first reveal, and never auto-submits.
- Station snapshots are immutable attempt truth; existing attempts remain readable after live station edits or soft deletion.
- New user-visible work uses existing quiz authorization and per-user database isolation; inaccessible records return `404`.
- No new OSCE MCP tools, browser-extension-only UI, feature-flag framework, metrics, assignments, remediation, flashcard conversion, Study Assistant integration, or OSCE CSV export.
- Use the canonical shared UI under `apps/packages/ui/src`; keep large Generate/Create/Manage/Take/Results tabs limited to routing and composition changes.
- Run SQLite and PostgreSQL migration tests, focused backend/frontend tests, neighboring quiz regressions, OpenAPI verification, TypeScript checks, Playwright, Ruff, `compileall`, `git diff --check`, and Bandit before activation.
- Treat an unavailable required SQLite/PostgreSQL/E2E environment as blocked, not passing.

---

## File Structure

### Backend

- `tldw_Server_API/app/api/v1/schemas/osce.py`: strict station, citation, practice-attempt, summary, and transition contracts.
- `tldw_Server_API/app/services/osce_practice.py`: protected nested-ID reconciliation, verification invalidation, candidate/revealed projections, lifecycle validation, and deterministic summaries.
- `tldw_Server_API/app/services/osce_generator.py`: OSCE prompt construction, provider-output normalization, source citation resolution, verification units, and test-mode fixtures.
- `tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py`: station authoring and OSCE practice REST routes.
- Existing quiz schemas, endpoint, generator, and ChaChaNotes DB files receive only integration changes required by the new activity.

### Shared WebUI

- `apps/packages/ui/src/services/osce.ts`: OSCE API types and HTTP methods.
- `apps/packages/ui/src/components/Quiz/hooks/useOsceQueries.ts`: query keys and mutations with serialized autosave support.
- `apps/packages/ui/src/components/Quiz/osce/OsceStationEditor.tsx`: shared Create/Manage inline authoring surface.
- `apps/packages/ui/src/components/Quiz/osce/OscePracticePanel.tsx`: candidate, reveal, assessment, timer, and resume flow.
- `apps/packages/ui/src/components/Quiz/osce/OsceResultsPanel.tsx`: independently paginated completed-attempt summaries and detail.
- `apps/packages/ui/src/components/Quiz/osce/osceDraftStore.ts`: 24-hour writable-field-only browser fallback; no guide fields.
- Existing Quiz tabs compose these modules and keep question behavior intact.

---

### Task 1: Strict OSCE Contracts And Quiz Activity Shapes

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/osce.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_schema_contract.py`
- Test: `tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py`

**Interfaces:**
- Consumes: existing `QuizCreate`, `QuizUpdate`, `QuizGenerateRequest`, `QuizGenerateResponse`, and strict Pydantic configuration patterns.
- Produces: `QuizActivityType`, `OsceVerificationState`, `OsceCitation`, `OscePatientContext`, create/update/stored checklist/rubric/key-point models, `OsceStationCreateContent`, `OsceStationUpdateContent`, `OsceStationStoredContent`, `OsceStationSummary`, `OsceStationCreateRequest`, `OsceStationAuthoringResponse`, `OsceAttemptState`, `OsceAttemptCreate`, `OsceAttemptPatch`, `OsceAttemptTransition`, `OsceCandidateAttemptResponse`, `OsceRevealedAttemptResponse`, and `OsceAttemptSummary`.

- [ ] **Step 1: Add failing strict-contract tests**

Create table-driven tests that prove valid manual content parses and that each bound, unknown field, numeric scoring field, duplicate UUID, duplicate rubric-level label, invalid source locator, invalid duration, and unsupported schema version fails. Add generation-request tests proving OSCE accepts `num_stations` 1-10 and rejects explicitly supplied `num_questions`, `question_types`, or `question_plan`.

```python
def test_station_create_rejects_scoring_fields(valid_station_payload):
    payload = {**valid_station_payload, "passing_score": 70}
    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


@pytest.mark.parametrize("field", ["num_questions", "question_types", "question_plan"])
def test_osce_generation_rejects_question_only_fields(field, valid_generation_payload):
    payload = {**valid_generation_payload, "generation_profile": "osce_scenario", field: 2}
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(payload)
```

- [ ] **Step 2: Run the schema tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_schema_contract.py tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py -q`

Expected: FAIL because OSCE models and generation fields are absent.

- [ ] **Step 3: Implement strict OSCE schema families**

Use `ConfigDict(extra="forbid", strict=True)`. Separate create/update input models from stored models so protected IDs cannot be written on create. Use discriminated citation validation for `media`, `document`, `url`, and `note` source types, including source-type-specific locator checks.

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QuizActivityType(str, Enum):
    QUESTIONS = "questions"
    OSCE = "osce"


class OsceAttemptState(str, Enum):
    IN_PROGRESS = "in_progress"
    SELF_ASSESSMENT = "self_assessment"
    COMPLETED = "completed"


class OsceStationCreateContent(StrictModel):
    schema_version: Literal["osce.station.v1"] = "osce.station.v1"
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    candidate_instructions: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    candidate_task: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    patient_context: OscePatientContext
    recommended_duration_seconds: int = Field(ge=60, le=7200)
    checklist_items: list[OsceChecklistItemCreate] = Field(min_length=1, max_length=50)
    rubric_domains: list[OsceRubricDomainCreate] = Field(min_length=1, max_length=12)
    expected_key_points: list[OsceKeyPointCreate] = Field(min_length=1, max_length=50)
```

Define `OsceCitation` with required `source_type` and bounded `source_id`, then source-type locator fields checked by an `after` validator. Define `OsceChecklistItemCreate` without an ID, `OsceChecklistItemUpdate` with `id: UUID | None`, and `OsceChecklistItemStored` with `id: UUID`; apply the same three-shape pattern to rubric domains, rubric levels, and key points. `OsceStationUpdateContent` has optional top-level fields so omission means unchanged, while each supplied nested collection is validated as a complete replacement. Define tri-state checklist values and separate candidate/revealed response models. The candidate response contains only candidate context, quote-free source labels, notes, timestamps, version, state, and `server_time`.

```python
class OsceChecklistItemCreate(StrictModel):
    label: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    rationale: Annotated[str, StringConstraints(max_length=2000)] | None = None
    citations: list[OsceCitation] = Field(default_factory=list)


class OsceChecklistItemUpdate(OsceChecklistItemCreate):
    id: UUID | None = None


class OsceChecklistItemStored(OsceChecklistItemCreate):
    id: UUID
```

- [ ] **Step 4: Extend existing quiz request/response models compatibly**

Add `activity_type=questions`, nullable `generation_profile`, and `total_stations=0` to quiz responses. Reject question-only settings for OSCE creates/updates. Change generation-profile parsing to accept the complete enum, then validate catalog availability in Task 6 so `planned` profiles still return a stable public error until activation. Add `num_stations`, `output_kind`, and `osce_stations` while preserving question defaults and responses.

```python
class QuizGenerateResponse(BaseModel):
    output_kind: Literal["questions", "osce_stations"] = "questions"
    quiz: QuizResponse
    questions: list[QuestionResponse] = Field(default_factory=list)
    osce_stations: list[OsceStationAuthoringResponse] = Field(default_factory=list)
    claim_verification: ClaimVerificationSummary | None = None
```

- [ ] **Step 5: Run focused tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_schema_contract.py tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/schemas/osce.py tldw_Server_API/app/api/v1/schemas/quizzes.py tldw_Server_API/tests/Quizzes/test_osce_schema_contract.py tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py
git commit -m "feat(quizzes): define strict OSCE contracts"
```

### Task 2: Dual-Backend Migration And Quiz Activity Guards

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py`
- Test: `tldw_Server_API/tests/DB_Management/test_osce_postgres_migration_v67.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_quizzes_basic.py`

**Interfaces:**
- Consumes: `QuizActivityType` values and current migration registry/version discovered at execution time.
- Produces: migrated quiz columns, `osce_stations`, `osce_practice_attempts`, required indexes/constraints, and activity-aware existing quiz/question/attempt DB methods.

- [ ] **Step 1: Confirm and reserve the next migration number**

Run: `rg -n "_CURRENT_SCHEMA_VERSION|_POSTGRES_SCHEMA_VERSION|_migrate_from_v[0-9]+_to_v[0-9]+" tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py | tail -20`

Expected: both backends are still version 66 and `67` is free. If not, substitute the actual next version consistently in filenames, functions, registries, tests, and schema constants before editing.

- [ ] **Step 2: Write failing fresh-install and upgrade tests**

Assert legacy quiz rows become `questions`, both count columns obey defaults, fresh and upgraded schemas are equivalent, foreign keys/cascades exist, and required indexes are present on both engines.

```python
def test_v67_upgrade_preserves_legacy_quiz_as_questions(legacy_v66_db):
    db = CharactersRAGDB(legacy_v66_db, client_id="migration-test")
    quiz = db.get_quiz(1)
    assert quiz["activity_type"] == "questions"
    assert quiz["generation_profile"] is None
    assert quiz["total_stations"] == 0
```

For PostgreSQL, use the repository fixture and skip only when it explicitly reports PostgreSQL unavailable.

- [ ] **Step 3: Run migration tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py tldw_Server_API/tests/DB_Management/test_osce_postgres_migration_v67.py -q`

Expected: FAIL because the migration and OSCE tables do not exist.

- [ ] **Step 4: Add additive migration and indexes on both backends**

Add the three quiz columns and dedicated tables. Store validated content/snapshots as JSON (`TEXT` in SQLite and existing JSON-compatible PostgreSQL convention), keep provenance separate from editable content, and constrain states and non-negative counts.

```sql
CREATE TABLE osce_stations (
  id INTEGER PRIMARY KEY,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  schema_version TEXT NOT NULL CHECK (schema_version = 'osce.station.v1'),
  content_json TEXT NOT NULL,
  order_index INTEGER NOT NULL CHECK (order_index >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  origin TEXT NOT NULL CHECK (origin IN ('generated', 'manual')),
  provenance_json TEXT,
  source_bundle_json TEXT,
  verification_state TEXT NOT NULL,
  verification_timestamp TEXT,
  verification_summary TEXT,
  deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

Create equivalent PostgreSQL DDL and these indexes: active stations by `(quiz_id, deleted, order_index)`, attempts by `(station_id, state, last_modified_at DESC)`, attempts by `(quiz_id, state, last_modified_at DESC)`, and unique `(station_id, client_attempt_id)`.

The attempt table must contain the immutable snapshot and lifecycle fields explicitly rather than joining the live station for reads:

```sql
CREATE TABLE osce_practice_attempts (
  id INTEGER PRIMARY KEY,
  station_id INTEGER NOT NULL REFERENCES osce_stations(id) ON DELETE CASCADE,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  client_attempt_id TEXT NOT NULL,
  station_snapshot_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('in_progress', 'self_assessment', 'completed')),
  candidate_notes TEXT NOT NULL DEFAULT '' CHECK (length(candidate_notes) <= 10000),
  checklist_selections_json TEXT NOT NULL DEFAULT '{}',
  rubric_selections_json TEXT NOT NULL DEFAULT '{}',
  started_at TEXT NOT NULL,
  self_assessment_started_at TEXT,
  completed_at TEXT,
  frozen_elapsed_seconds INTEGER CHECK (frozen_elapsed_seconds IS NULL OR frozen_elapsed_seconds >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  last_modified_at TEXT NOT NULL,
  UNIQUE (station_id, client_attempt_id)
);
```

- [ ] **Step 5: Add activity checks to existing quiz paths**

Update quiz row serialization and create/list/get/update/delete. Reject activity changes once either count is nonzero; reject OSCE `passing_score`/`time_limit_seconds`; force irrelevant count to zero. Make Size sorting select `total_questions` for question quizzes and `total_stations` for OSCE quizzes. Make existing question CRUD and normal quiz-attempt start/submit paths raise the existing not-found/domain error for OSCE quizzes. Soft quiz deletion must also hide stations; hard authorized purge relies on foreign-key cascade.

```python
def _require_quiz_activity(self, quiz_id: int, activity_type: str) -> dict[str, Any]:
    quiz = self.get_quiz(quiz_id)
    if quiz is None or quiz["activity_type"] != activity_type:
        raise QuizNotFoundError(quiz_id)
    return quiz
```

- [ ] **Step 6: Run migration/activity regressions and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py tldw_Server_API/tests/DB_Management/test_osce_postgres_migration_v67.py tldw_Server_API/tests/ChaChaNotesDB/test_quizzes_basic.py -q`

Expected: PASS on SQLite and PostgreSQL, or an explicit environment skip for unavailable PostgreSQL.

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py tldw_Server_API/tests/DB_Management/test_osce_postgres_migration_v67.py tldw_Server_API/tests/ChaChaNotesDB/test_quizzes_basic.py
git commit -m "feat(quizzes): persist OSCE activity data"
```

### Task 3: Station Identity, Verification State, And Persistence

**Files:**
- Create: `tldw_Server_API/app/services/osce_practice.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_station_service.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_osce_stations.py`

**Interfaces:**
- Consumes: Task 1 station contracts and Task 2 OSCE tables/activity guard.
- Produces: `materialize_station_content`, `reconcile_station_update`, `evidence_fingerprint`, `project_station_summary`, and DB methods `create_osce_station`, `get_osce_station`, `list_osce_stations`, `update_osce_station`, `delete_osce_station`, `create_quiz_with_osce_stations_atomic`.

- [ ] **Step 1: Write failing protected-ID and verification tests**

Cover server UUID assignment, preservation on update, unknown/duplicate/moved rubric-level rejection, full-collection replacement, evidence edit invalidation, and presentation-only preservation.

```python
def test_evidence_edit_invalidates_source_verification(stored_station):
    updated = stored_station.model_copy(
        update={"patient_context": {**stored_station.patient_context.model_dump(), "text": "Changed fact"}}
    )
    result = reconcile_station_update(stored_station, updated, "source_verified")
    assert result.verification_state == "modified_after_verification"


def test_moving_existing_level_to_another_domain_is_rejected(stored_station):
    payload = move_first_level_to_second_domain(stored_station)
    with pytest.raises(OsceStationIdentityError, match="rubric level"):
        reconcile_station_update(stored_station, payload, "manually_authored")
```

- [ ] **Step 2: Run station service tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_station_service.py -q`

Expected: FAIL because the service does not exist.

- [ ] **Step 3: Implement station-domain helpers**

Materialize create content by rebuilding nested objects with `uuid4()` regardless of supplied/model IDs. On update, index every prior nested UUID with its kind and parent domain, preserve recognized IDs, assign new IDs where absent, and fail on unknown, duplicates, or changed parent. Compute verification invalidation only from normalized patient-context text/citations, checklist rationale/citations, and key-point text/citations.

```python
def evidence_fingerprint(content: OsceStationStoredContent) -> str:
    evidence = {
        "patient_context": content.patient_context.model_dump(mode="json"),
        "checklist": [
            {"id": str(item.id), "rationale": item.rationale, "citations": item.citations}
            for item in content.checklist_items
        ],
        "key_points": [point.model_dump(mode="json") for point in content.expected_key_points],
    }
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Write failing station DB tests**

Test ordered pagination, activity mismatch, optimistic `expected_version`, cross-quiz lookup as not found, soft delete/recount, create rollback, generated multi-station atomicity, and hard cascade.

```python
def test_create_station_updates_total_stations_in_same_transaction(osce_quiz, db):
    station = db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")
    assert station["version"] == 1
    assert db.get_quiz(osce_quiz["id"])["total_stations"] == 1
```

- [ ] **Step 5: Implement station DB methods with single transactions**

Use the existing DB transaction/context patterns and JSON serializer. `create_quiz_with_osce_stations_atomic(...)` must insert quiz, every station/provenance row, and final count inside one transaction; it accepts already validated stored station payloads and has no provider calls inside the transaction.

```python
def create_quiz_with_osce_stations_atomic(
    self,
    quiz_data: dict[str, Any],
    stations: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Persist one OSCE quiz and all validated stations atomically."""
```

- [ ] **Step 6: Run station suites and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_station_service.py tldw_Server_API/tests/ChaChaNotesDB/test_osce_stations.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/services/osce_practice.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Quizzes/test_osce_station_service.py tldw_Server_API/tests/ChaChaNotesDB/test_osce_stations.py
git commit -m "feat(quizzes): add OSCE station persistence"
```

### Task 4: Retry-Safe Practice Attempts And Lifecycle

**Files:**
- Modify: `tldw_Server_API/app/services/osce_practice.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_attempt_service.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_osce_attempts.py`

**Interfaces:**
- Consumes: Task 1 attempt contracts and Task 3 stored station content.
- Produces: `project_candidate_attempt`, `project_revealed_attempt`, `summarize_osce_attempt`, `validate_assessment_selections`; DB methods `start_osce_attempt`, `get_osce_attempt`, `list_osce_attempts`, `patch_osce_attempt`, `transition_osce_attempt`.

- [ ] **Step 1: Write failing lifecycle and privacy tests**

Cover retry-safe create, immutable snapshot after edit/delete, candidate structural omission, notes-only candidate updates, assessment-only revealed updates, stale version conflict, repeated transition success, no backward transitions, complete selection requirements, completed immutability, and deterministic separate summaries.

```python
def test_candidate_projection_structurally_omits_marking_guide(attempt_row):
    payload = project_candidate_attempt(attempt_row).model_dump(mode="json")
    forbidden = {"checklist_items", "rubric_domains", "expected_key_points", "rationale", "quote"}
    assert forbidden.isdisjoint(payload.keys())
    assert "private notes" not in json.dumps(payload["station"])


def test_repeated_reveal_is_idempotent_even_with_stale_version(db, attempt):
    first = db.transition_osce_attempt(attempt["id"], "self_assessment", expected_version=1)
    repeated = db.transition_osce_attempt(attempt["id"], "self_assessment", expected_version=1)
    assert repeated["id"] == first["id"]
    assert repeated["state"] == "self_assessment"
```

- [ ] **Step 2: Run attempt tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_attempt_service.py tldw_Server_API/tests/ChaChaNotesDB/test_osce_attempts.py -q`

Expected: FAIL because attempt methods/projections are absent.

- [ ] **Step 3: Implement immutable snapshot and response projection**

Snapshot content, provenance, source bundle, and title at creation. Candidate projection must construct an allowlisted object rather than serialize-then-delete. Convert citations to source type/ID/label only. Revealed projection may include the guide and quotes. Never include notes in summary projection.

```python
def project_candidate_attempt(row: Mapping[str, Any]) -> OsceCandidateAttemptResponse:
    snapshot = OsceStationStoredContent.model_validate_json(row["station_snapshot_json"])
    return OsceCandidateAttemptResponse(
        id=row["id"],
        quiz_id=row["quiz_id"],
        station_id=row["station_id"],
        client_attempt_id=row["client_attempt_id"],
        state=OsceAttemptState.IN_PROGRESS,
        version=row["version"],
        station=build_candidate_station(snapshot),
        notes=row["candidate_notes"],
        started_at=row["started_at"],
        server_time=utc_now(),
    )
```

- [ ] **Step 4: Implement atomic DB attempt operations**

Insert by `(station_id, client_attempt_id)` and return the existing owned attempt on uniqueness conflict. Derive elapsed seconds from server timestamps, freeze it only on first reveal, increment version on successful mutation, apply state/version predicates in SQL, and distinguish already-achieved idempotent transitions from genuine stale conflicts.

```python
def transition_osce_attempt(
    self,
    attempt_id: int,
    target_state: str,
    *,
    expected_version: int,
) -> dict[str, Any]:
    """Advance an attempt monotonically or return its already-achieved state."""
```

`list_osce_attempts` accepts `quiz_id`, `station_id`, and `states: Sequence[str]`, sorts by `last_modified_at DESC, id DESC`, and returns compact summaries from snapshots even when the station is deleted.

- [ ] **Step 5: Add concurrency/property coverage**

Use two DB handles or threads to assert one retry-key row is created and only one update wins an expected version. Add a property test generating valid checklist/rubric maps to prove completion accepts exactly complete, domain-valid selections.

```python
@given(checklist_states=st.lists(st.sampled_from(["met", "not_met"]), min_size=1, max_size=20))
def test_complete_summary_never_derives_score(checklist_states):
    summary = summarize_fixture_attempt(checklist_states)
    assert "score" not in summary.model_dump()
    assert "passed" not in summary.model_dump()
```

- [ ] **Step 6: Run attempt suites and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_attempt_service.py tldw_Server_API/tests/ChaChaNotesDB/test_osce_attempts.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/services/osce_practice.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Quizzes/test_osce_attempt_service.py tldw_Server_API/tests/ChaChaNotesDB/test_osce_attempts.py
git commit -m "feat(quizzes): add OSCE practice lifecycle"
```

### Task 5: Station Authoring And Practice REST APIs

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_endpoints.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_endpoint_privacy.py`

**Interfaces:**
- Consumes: Task 1 schemas, Task 3 station DB methods, Task 4 attempt lifecycle/projection methods, and existing quiz auth/DB dependencies.
- Produces: all station and practice routes specified in the design, included beneath the existing `/api/v1/quizzes` router.

- [ ] **Step 1: Write failing endpoint contract tests**

Test exact paths/status codes, pagination, parent mismatch/cross-user `404`, station activity mismatch, create/edit/delete, attempt filters with repeatable `state`, optimistic `409`, malformed/incomplete `422`, retry create, idempotent transitions, and deleted-station snapshot reads.

```python
def test_candidate_get_omits_marking_guide_fields(client, active_attempt_headers):
    response = client.get("/api/v1/quizzes/osce-attempts/1", headers=active_attempt_headers)
    assert response.status_code == 200
    body = response.json()
    serialized = json.dumps(body)
    for secret in ["expected_key_points", "rubric_domains", "rationale", '"quote"']:
        assert secret not in serialized
```

- [ ] **Step 2: Run API tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_endpoints.py tldw_Server_API/tests/Quizzes/test_osce_endpoint_privacy.py -q`

Expected: FAIL with route-not-found responses.

- [ ] **Step 3: Implement station subrouter**

Create an `APIRouter`, reuse existing auth/database dependency helpers, map domain not-found to `404`, stale/transition/activity conflict to `409`, and contract/selection errors to `422`. Return compact station summaries for lists and authoring detail only from station detail routes.

```python
router = APIRouter(tags=["quizzes"])


@router.post("/{quiz_id}/osce-stations", response_model=OsceStationAuthoringResponse, status_code=201)
async def create_station(
    quiz_id: int,
    request: OsceStationCreateRequest,
    db: ChaChaNotesDatabase = Depends(get_notes_database),
) -> OsceStationAuthoringResponse:
    stored = materialize_station_content(request.content)
    return map_station(db.create_osce_station(quiz_id, stored, origin="manual"))
```

- [ ] **Step 4: Implement practice routes with phase-specific responses**

Use a response union discriminated by `state`, but return only candidate model fields for `in_progress`. The list endpoint returns summaries only. Flush version checks through DB methods; do not log request bodies or notes.

```python
@router.get(
    "/osce-attempts/{attempt_id}",
    response_model=OsceCandidateAttemptResponse | OsceRevealedAttemptResponse,
)
async def get_attempt(attempt_id: int, db=Depends(get_notes_database)):
    row = db.get_osce_attempt(attempt_id)
    return project_candidate_attempt(row) if row["state"] == "in_progress" else project_revealed_attempt(row)
```

Include the subrouter once in `quizzes.py` after the main router is defined so existing router-group imports continue to work.

- [ ] **Step 5: Verify OpenAPI phase schemas and route ownership**

Assert candidate and revealed schemas are distinct in `app.openapi()`, all eleven routes are present, and no duplicate operation IDs are emitted.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_endpoints.py tldw_Server_API/tests/Quizzes/test_osce_endpoint_privacy.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the API slice**

```bash
git add tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/tests/Quizzes/test_osce_endpoints.py tldw_Server_API/tests/Quizzes/test_osce_endpoint_privacy.py
git commit -m "feat(api): expose OSCE authoring and practice"
```

### Task 6: Source-Backed OSCE Generation And Verification

**Files:**
- Create: `tldw_Server_API/app/services/osce_generator.py`
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_generator.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_generation_endpoint.py`
- Test: `tldw_Server_API/tests/Quizzes/test_quiz_generation_profiles.py`

**Interfaces:**
- Consumes: existing `resolve_quiz_sources`, `_call_quiz_generation_llm`, `_build_quiz_source_documents`, `verify_generated_artifact_against_sources`, Task 1 contracts, Task 3 atomic persistence, and Task 5 API mapping.
- Produces: `GeneratedOsceBundle`, `generate_osce_stations_from_sources`, `normalize_generated_station`, `build_osce_verification_units`, `ensure_generation_profile_available`, and OSCE dispatch through `generate_quiz_from_sources`.

- [ ] **Step 1: Write failing normalization and privacy tests**

Test test-mode deterministic output, model-ID stripping, fictional/deidentified prompt instructions, exact station count, strict citations, inaccessible/source-inconsistent citation rejection, bounded public provider errors, and absence of candidate notes from prompt/verification inputs.

```python
def test_generated_station_requires_citation_for_each_evidence_unit(valid_generated_station):
    valid_generated_station["expected_key_points"][0]["citations"] = []
    with pytest.raises(OsceCitationError, match="expected key point"):
        normalize_generated_station(valid_generated_station, resolved_sources())


def test_prompt_forbids_real_patient_data(resolved_sources):
    prompt = build_osce_generation_prompt(resolved_sources, num_stations=1, difficulty="medium")
    assert "fictional or deidentified" in prompt.lower()
    assert "candidate notes" not in prompt.lower()
```

- [ ] **Step 2: Run generation tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_generator.py tldw_Server_API/tests/Quizzes/test_osce_generation_endpoint.py -q`

Expected: FAIL because OSCE generation is not implemented.

- [ ] **Step 3: Implement focused OSCE generator**

Use the existing provider boundary and JSON extraction utilities. Normalize provider objects through create models after removing every nested `id`; resolve citations only against the canonical source bundle. Build verification units for patient context, checklist rationales, and expected key points, then call the existing verifier before converting to stored content.

```python
@dataclass(frozen=True)
class GeneratedOsceBundle:
    stations: tuple[OsceStationStoredContent, ...]
    verification_result: ArtifactVerificationResult
    provenance: dict[str, Any]


async def generate_osce_stations_from_sources(
    *,
    evidence: Sequence[dict[str, Any]],
    normalized_sources: Sequence[dict[str, str]],
    num_stations: int,
    difficulty: str,
    focus_topics: Sequence[str],
    model: str | None,
    api_provider: str | None,
    verification_provider: str | None,
    verification_model: str | None,
) -> GeneratedOsceBundle:
    """Generate, normalize, cite, and verify all stations before persistence."""
```

Stable public failures are `osce_provider_failure`, `osce_malformed_output`, `osce_unsupported_contract`, `osce_citation_failure`, and `osce_verification_failure`; log only bounded identifiers and exception classes, never provider payload/source content.

- [ ] **Step 4: Dispatch generation and enforce planned catalog state**

Add `output_kind` and `default_num_stations` to catalog entries. Until Task 10, `ensure_generation_profile_available("osce_scenario")` returns the same stable unavailable response as other planned profiles. Tests may monkeypatch the catalog entry to `available` to exercise the full endpoint while rollout remains hidden.

```python
def ensure_generation_profile_available(profile: QuizGenerationProfile) -> None:
    definition = _QUIZ_GENERATION_PROFILES[profile.value]
    if definition["status"] != "available":
        raise QuizGenerationRequestError("generation_profile_unavailable")
```

When enabled in tests, reject explicitly supplied question-only fields, generate/verify all stations first, then call `create_quiz_with_osce_stations_atomic`. Return `questions=[]`; preserve the question path as `output_kind="questions"`, `osce_stations=[]`.

- [ ] **Step 5: Inject failures at each boundary and prove atomicity**

Parameterize normalization, citation, verification, and second-station DB insertion failures. After each failure assert no quiz shell, station, or provenance row exists. Add neighboring question-generation regression tests.

```python
@pytest.mark.parametrize("failure_stage", ["normalize", "citation", "verify", "persist_second"])
def test_osce_generation_failure_leaves_no_partial_rows(failure_stage, db, generator):
    with inject_osce_failure(failure_stage):
        with pytest.raises(OsceGenerationError):
            generator()
    assert db.list_quizzes(activity_type="osce")["items"] == []
```

- [ ] **Step 6: Run generation suites and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_generator.py tldw_Server_API/tests/Quizzes/test_osce_generation_endpoint.py tldw_Server_API/tests/Quizzes/test_quiz_generation_profiles.py tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py -q`

Expected: PASS with the production OSCE catalog entry still `planned`.

```bash
git add tldw_Server_API/app/services/osce_generator.py tldw_Server_API/app/services/quiz_generator.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/app/api/v1/schemas/quizzes.py tldw_Server_API/tests/Quizzes/test_osce_generator.py tldw_Server_API/tests/Quizzes/test_osce_generation_endpoint.py tldw_Server_API/tests/Quizzes/test_quiz_generation_profiles.py
git commit -m "feat(quizzes): generate source-backed OSCE stations"
```

### Task 7: Export V2, Atomic Import, And MCP Compatibility

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/quizzes_module.py`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_import_v2.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_quizzes_module_sanitization.py`

**Interfaces:**
- Consumes: Task 1 authoring contracts, Task 3 protected-ID materialization and atomic station persistence, existing v1 import behavior, and existing MCP quiz tool methods.
- Produces: `tldw.quiz.export.v2` import contracts for mixed activity files, `import_osce_quiz_entry_atomic`, and activity guards/omissions for existing MCP tools.

- [ ] **Step 1: Write failing portability and MCP tests**

Cover v2 question and OSCE entries in one file, v1 unchanged, new protected IDs, untrusted provenance downgraded to manual, no attempts/notes, per-entry partial success, no shell on invalid OSCE entry, and MCP question/attempt/generation tools rejecting or omitting OSCE.

```python
def test_v2_import_rekeys_osce_and_downgrades_verification(client, exported_osce_entry):
    response = client.post(
        "/api/v1/quizzes/import/json",
        json={"export_format": "tldw.quiz.export.v2", "quizzes": [exported_osce_entry]},
    )
    assert response.status_code == 200
    item = response.json()["items"][0]
    station = client.get(
        f"/api/v1/quizzes/{item['quiz_id']}/osce-stations/{item['station_ids'][0]}"
    ).json()
    assert item["imported_stations"] == 1
    assert station["verification_state"] == "manually_authored"
    assert station["content"]["checklist_items"][0]["id"] != exported_osce_entry["stations"][0]["content"]["checklist_items"][0]["id"]
```

- [ ] **Step 2: Run import/MCP tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_import_v2.py tldw_Server_API/tests/MCP_unified/test_quizzes_module_sanitization.py -q`

Expected: FAIL because v2 and activity guards are absent.

- [ ] **Step 3: Implement discriminated v2 import contracts**

Keep v1 models unchanged and default every v1 entry to `questions`. Define a v2 entry union discriminated by `activity_type`; OSCE export station input may carry informational provenance and IDs, but import must strip IDs, ignore claimed verification, validate all stations, and materialize server-owned IDs before opening the transaction.

```python
class QuizExportV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    export_format: Literal["tldw.quiz.export.v2"]
    exported_at: datetime
    quizzes: list[QuestionQuizExportV2 | OsceQuizExportV2]
```

Extend import result models with `imported_stations`, `failed_stations`, and created `station_ids`; retain the existing question counters so v1 clients remain compatible.

- [ ] **Step 4: Implement per-entry atomic OSCE import**

Validate one entry entirely, construct `activity_type="osce"`, `generation_profile=None`, `origin="manual"`, `verification_state="manually_authored"`, and call the Task 3 atomic method. Preserve existing batch behavior: successful sibling entries stay imported and each invalid entry receives its own bounded error.

```python
def import_osce_quiz_entry_atomic(self, entry: OsceQuizExportV2) -> dict[str, Any]:
    stations = [
        station_for_import(station.content, origin="manual", verification_state="manually_authored")
        for station in entry.stations
    ]
    return self.create_quiz_with_osce_stations_atomic(imported_quiz_data(entry), stations)
```

- [ ] **Step 5: Guard existing MCP question semantics**

For MCP list responses, either omit OSCE quizzes or expose activity metadata without treating `total_questions=0` as a usable question quiz. Existing MCP get/question/attempt/generate operations must return a stable unsupported-activity error when their target quiz is OSCE. Do not register new tools.

```python
def _require_question_quiz(self, quiz: Mapping[str, Any]) -> None:
    if quiz.get("activity_type", "questions") != "questions":
        raise ValueError("This MCP quiz operation supports question quizzes only")
```

- [ ] **Step 6: Run portability/MCP regressions and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_import_v2.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py tldw_Server_API/tests/MCP_unified/test_quizzes_module_sanitization.py -q`

Expected: PASS.

```bash
git add tldw_Server_API/app/api/v1/schemas/quizzes.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/MCP_unified/modules/implementations/quizzes_module.py tldw_Server_API/tests/Quizzes/test_osce_import_v2.py tldw_Server_API/tests/MCP_unified/test_quizzes_module_sanitization.py
git commit -m "feat(quizzes): add portable OSCE import contract"
```

### Task 8: Shared WebUI Data Layer, Generate, Create, And Manage

**Files:**
- Create: `apps/packages/ui/src/services/osce.ts`
- Create: `apps/packages/ui/src/components/Quiz/hooks/useOsceQueries.ts`
- Create: `apps/packages/ui/src/components/Quiz/osce/OsceStationEditor.tsx`
- Create: `apps/packages/ui/src/components/Quiz/osce/oscePortability.ts`
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/CreateTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx`
- Test: `apps/packages/ui/src/services/__tests__/osce.test.ts`
- Test: `apps/packages/ui/src/components/Quiz/osce/__tests__/OsceStationEditor.test.tsx`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.osce.test.tsx`

**Interfaces:**
- Consumes: Tasks 5-7 REST shapes, existing `tldw` request client, `QuizMarkdown`, `SourceCitations`, query conventions, and tab navigation callbacks.
- Produces: typed OSCE API client/query hooks, reusable station editor, v2 client-side export builder, and Generate/Create/Manage composition.

- [ ] **Step 1: Write failing API-client and control tests**

Test exact URL/method/query encoding, repeated state parameters, expected-version bodies, OSCE generation controls, `Quiz | OSCE` activity selection, question-control suppression, station count 1-10, generated navigation to Manage, incompatible action suppression, and v2 export excluding attempts/notes.

```ts
it("encodes repeated attempt states", async () => {
  await listOsceAttempts({ states: ["in_progress", "self_assessment"] })
  expect(mockGet).toHaveBeenCalledWith(
    "/api/v1/quizzes/osce-attempts?state=in_progress&state=self_assessment"
  )
})

it("uses v2 only when an OSCE entry is present", () => {
  expect(buildQuizExport([questionEntry]).export_format).toBe("tldw.quiz.export.v1")
  expect(buildQuizExport([questionEntry, osceEntry]).export_format).toBe("tldw.quiz.export.v2")
})
```

- [ ] **Step 2: Run focused UI tests and verify red**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/osce.test.ts ../packages/ui/src/components/Quiz/osce/__tests__/OsceStationEditor.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.osce.test.tsx`

Expected: FAIL because OSCE UI modules are absent.

- [ ] **Step 3: Implement typed service and query hooks**

Define wire types matching OpenAPI, with candidate/revealed attempts as a discriminated union. Centralize keys and invalidate quiz/station/attempt lists after mutations. Keep autosave serialization in Task 9; authoring remains explicit-save only.

```ts
export const osceKeys = {
  all: ["quizzes", "osce"] as const,
  stations: (quizId: number) => [...osceKeys.all, "stations", quizId] as const,
  attempt: (attemptId: number) => [...osceKeys.all, "attempt", attemptId] as const,
  attempts: (filters: OsceAttemptFilters) => [...osceKeys.all, "attempts", filters] as const
}

export type OsceAttempt = OsceCandidateAttempt | OsceRevealedAttempt
```

- [ ] **Step 4: Build the shared station editor**

Render full-width candidate fields, duration, checklist, rubric domains/ordered levels, key points, and citations. Use existing Ant Design controls and Lucide/installed icon buttons with tooltips and accessible labels for add/remove/reorder. Enforce client bounds without relying on them for server security. Show the real-patient warning without compliance claims.

```tsx
<Tooltip title="Move checklist item up">
  <Button
    type="text"
    icon={<ArrowUp aria-hidden />}
    aria-label="Move checklist item up"
    onClick={() => moveChecklistItem(index, index - 1)}
  />
</Tooltip>
```

Track dirty state from the last acknowledged server payload. On `409`, preserve the local draft, fetch the latest server version, and present `Reload server version` and `Keep local draft`; a subsequent save uses the newly loaded version and requires confirmation.

- [ ] **Step 5: Compose Generate/Create/Manage**

Keep `osce_scenario` hidden while the fallback/catalog says `planned`. When tests supply `available`, show Stations 1-10 and hide question mix/count/plan, passing score, and flashcard companion. Create uses a segmented activity control and the shared editor. Manage lists compact station summaries, verification labels, explicit Save, dirty guards, and hides assignments/remediation/flashcard/Study Assistant actions for OSCE. Client-side v2 export includes editable station content, citations, and informational safe provenance, excludes attempts and candidate notes, and the importer accepts both v1 and v2 markers.

```tsx
{quiz.activity_type === "osce" ? (
  <OsceStationEditor quizId={quiz.id} station={selectedStation} onSaved={refreshStations} />
) : (
  <QuestionEditor quiz={quiz} />
)}
```

- [ ] **Step 6: Run focused UI tests, typecheck, and commit**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/osce.test.ts ../packages/ui/src/components/Quiz/osce/__tests__/OsceStationEditor.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.osce.test.tsx`

Run: `cd apps/tldw-frontend && bun run typecheck`

Expected: PASS.

```bash
git add apps/packages/ui/src/services/osce.ts apps/packages/ui/src/services/quizzes.ts apps/packages/ui/src/components/Quiz/hooks/useOsceQueries.ts apps/packages/ui/src/components/Quiz/osce/OsceStationEditor.tsx apps/packages/ui/src/components/Quiz/osce/oscePortability.ts apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx apps/packages/ui/src/components/Quiz/tabs/CreateTab.tsx apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx apps/packages/ui/src/services/__tests__/osce.test.ts apps/packages/ui/src/components/Quiz/osce/__tests__/OsceStationEditor.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.osce.test.tsx
git commit -m "feat(webui): add OSCE authoring workspace"
```

### Task 9: Shared Practice, Offline Drafts, Timer, And Results

**Files:**
- Create: `apps/packages/ui/src/components/Quiz/osce/osceDraftStore.ts`
- Create: `apps/packages/ui/src/components/Quiz/osce/OscePracticePanel.tsx`
- Create: `apps/packages/ui/src/components/Quiz/osce/OsceResultsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/hooks/useOsceQueries.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`
- Test: `apps/packages/ui/src/components/Quiz/osce/__tests__/osceDraftStore.test.ts`
- Test: `apps/packages/ui/src/components/Quiz/osce/__tests__/OscePracticePanel.test.tsx`
- Test: `apps/packages/ui/src/components/Quiz/osce/__tests__/OsceResultsPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.osce.test.tsx`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.osce.test.tsx`

**Interfaces:**
- Consumes: Task 8 service/query types and existing Quiz tab composition.
- Produces: serialized writable-field autosave, 24-hour local fallback, candidate/self-assessment UI, server-offset timer, deterministic results, and independent active/completed pagination streams.

- [ ] **Step 1: Write failing storage/privacy and fake-clock tests**

Test a strict allowlist of `notes`, checklist selections, rubric selections, attempt/version/timestamp metadata; reject guide-like keys recursively; expire after 24 hours; clear after acknowledged server save/completion. With fake clocks, cover server offset, reload, local clock jump, background suspension, reveal freeze, and resume.

```ts
it("never stores marking-guide fields", () => {
  saveOsceDraft({
    attemptId: 7,
    version: 2,
    notes: "private",
    checklistSelections: {},
    rubricSelections: {},
    expected_key_points: [{ text: "must not persist" }]
  } as never)
  expect(localStorage.getItem(osceDraftKey(7))).not.toContain("expected_key_points")
})

it("freezes elapsed time at reveal timestamp", () => {
  expect(computeElapsedSeconds({ startedAt, selfAssessmentStartedAt, serverNow })).toBe(480)
})
```

- [ ] **Step 2: Run practice/results tests and verify red**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Quiz/osce/__tests__/osceDraftStore.test.ts ../packages/ui/src/components/Quiz/osce/__tests__/OscePracticePanel.test.tsx ../packages/ui/src/components/Quiz/osce/__tests__/OsceResultsPanel.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.osce.test.tsx`

Expected: FAIL because practice/result modules are absent.

- [ ] **Step 3: Implement serialized autosave and local fallback**

Use one promise chain per attempt so patches never overtake each other. Persist only writable fields and expected version, debounce notes, write the same allowlisted draft locally before network mutation, reconcile version on success, and retain the draft on disconnect/conflict. `flush()` resolves only after the last server acknowledgement.

```ts
export interface OsceSaveQueue {
  enqueue(patch: OsceWritablePatch): Promise<OsceAttempt>
  flush(): Promise<void>
  hasPending(): boolean
}

export const OSCE_DRAFT_TTL_MS = 24 * 60 * 60 * 1000
```

Reveal/complete must call `flush()`, stop on save error, require an active connection, and then issue the transition with the acknowledged version.

- [ ] **Step 4: Build candidate and self-assessment phases**

Candidate phase renders only task/instructions/context/duration/source labels/private notes. It derives elapsed time from `server_time - started_at`, updates visually without changing state, and never auto-submits. Reveal requires confirmation. Self-assessment retains candidate context, renders `Met`/`Not met`, rubric radio groups, expected points and citations, and disables Complete until all selections exist.

```tsx
const canComplete =
  checklistItems.every((item) => selections.checklist[item.id] !== undefined) &&
  rubricDomains.every((domain) => selections.rubric[domain.id] !== undefined)
```

Use `QuizMarkdown` and `SourceCitations` for sanitization/rendering. Do not cache revealed response bodies in browser storage.

- [ ] **Step 5: Build active-attempt resume and Results**

Take fetches active states, selects the most recently modified attempt deterministically, and keeps other active attempts selectable. Multi-station quizzes render a picker. Deleted-station attempts remain resumable from snapshots. Results owns completed filters, pagination, summaries, detail, and station/domain labels. It offers no CSV export and no score/pass/fail UI.

```tsx
{quiz.activity_type === "osce" ? (
  <OsceResultsPanel quizId={quiz.id} />
) : (
  <QuestionQuizResults quizId={quiz.id} />
)}
```

- [ ] **Step 6: Verify accessibility, responsive behavior, and commit**

Add keyboard/focus assertions for reveal confirmation, rubric groups, conflict messages, and station picker; render at narrow and desktop container widths and assert no clipped command labels.

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Quiz/osce/__tests__ ../packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.osce.test.tsx`

Run: `cd apps/tldw-frontend && bun run typecheck`

Expected: PASS.

```bash
git add apps/packages/ui/src/components/Quiz/osce/osceDraftStore.ts apps/packages/ui/src/components/Quiz/osce/OscePracticePanel.tsx apps/packages/ui/src/components/Quiz/osce/OsceResultsPanel.tsx apps/packages/ui/src/components/Quiz/hooks/useOsceQueries.ts apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx apps/packages/ui/src/components/Quiz/osce/__tests__ apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.osce.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.osce.test.tsx
git commit -m "feat(webui): add OSCE practice and results"
```

### Task 10: End-To-End Qualification, Documentation, And Catalog Activation

**Files:**
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `Docs/Product/Advanced_Quiz_Customization_PRD.md`
- Create: `Docs/API/Quizzes.md`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/QuizPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts`
- Test: `tldw_Server_API/tests/Quizzes/test_osce_release_contract.py`

**Interfaces:**
- Consumes: every prior task and the catalog/fallback planned-state guard.
- Produces: one compatible release revision with OSCE catalog status `available`, shared frontend fallback enabled, documented recovery rules, and release-gate evidence.

- [ ] **Step 1: Add failing release-contract and E2E tests before activation**

Assert catalog output exposes `output_kind="osce_stations"`, `default_num_stations=1`, legacy `default_num_questions=1`, and `status="available"`. Extend the quiz page object with semantic operations for choosing OSCE, generating, editing, starting, revealing, assessing, completing, and opening results.

```python
def test_osce_profile_is_available_only_with_complete_contract(client):
    profile = next(
        item for item in client.get("/api/v1/quizzes/generation-profiles").json()["profiles"]
        if item["id"] == "osce_scenario"
    )
    assert profile == {
        **profile,
        "status": "available",
        "output_kind": "osce_stations",
        "default_num_stations": 1,
        "default_num_questions": 1,
    }
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_release_contract.py -q`

Expected: FAIL because the profile remains `planned`.

- [ ] **Step 2: Add complete Playwright workflows**

Use mocked deterministic backend responses for UI qualification and one real local backend path where the existing test harness supports it. Cover: generate -> Manage -> edit -> practice -> reveal -> complete -> Results; offline draft/reconnect/conflict; live station edit/delete with stable attempt snapshot; pagination; and a near-limit station fixture.

```ts
test("generates, reviews, practices, self-assesses, and completes an OSCE", async ({ page }) => {
  const quiz = new QuizPage(page)
  await quiz.open()
  await quiz.generateOsce({ stations: 2 })
  await quiz.editFirstStationTitle("Anticoagulant safety review")
  await quiz.startOscePractice()
  await quiz.beginSelfAssessment()
  await quiz.completeAllChecklistAndRubricSelections()
  await quiz.completeOsce()
  await quiz.expectOsceResultWithoutScore()
})
```

- [ ] **Step 3: Update OpenAPI guard and documentation while profile stays planned**

Add every OSCE path to `ClientPath`, run strict OpenAPI verification, and document station schemas, phase-dependent response privacy, retry/version semantics, v2 portability, no-score behavior, fictional/deidentified data guidance, and recovery constraints. Record OSCE implementation status in the advanced quiz PRD; leave metrics assigned to TASK-12102.3.7 and broader docs work to TASK-12102.3.6.

Run: `cd apps/packages/ui && TLDW_VERIFY_OPENAPI_STRICT=1 bun run verify:openapi`

Expected: PASS against generated backend OpenAPI.

- [ ] **Step 4: Run backend qualification before activation**

Run focused and neighboring suites:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes \
  tldw_Server_API/tests/ChaChaNotesDB/test_quizzes_basic.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_osce_stations.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_osce_attempts.py \
  tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py \
  tldw_Server_API/tests/DB_Management/test_osce_postgres_migration_v67.py \
  tldw_Server_API/tests/MCP_unified/test_quizzes_module_sanitization.py -q
```

Run: `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/api/v1/schemas/osce.py tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py tldw_Server_API/app/services/osce_practice.py tldw_Server_API/app/services/osce_generator.py`

Run: `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/api/v1/schemas/osce.py tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py tldw_Server_API/app/services/osce_practice.py tldw_Server_API/app/services/osce_generator.py`

Expected: all pass; PostgreSQL must run unless the fixture explicitly reports it unavailable, in which case resolve the environment before activation.

- [ ] **Step 5: Run shared UI and browser-extension qualification before activation**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/osce.test.ts ../packages/ui/src/components/Quiz/osce/__tests__ ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.osce.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.osce.test.tsx`

Run: `cd apps/tldw-frontend && bun run test:extension -- --run`

Run: `cd apps/tldw-frontend && bun run typecheck`

Run: `cd apps/tldw-frontend && bun run compile`

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/quiz.spec.ts --reporter=line`

Expected: all pass with no browser-storage marking-guide leakage, responsive overlap, or question-quiz regression.

- [ ] **Step 6: Run security/privacy checks before activation**

Run Bandit on touched backend scope and inspect findings rather than treating command success alone as sufficient.

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/schemas/osce.py \
  tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py \
  tldw_Server_API/app/services/osce_practice.py \
  tldw_Server_API/app/services/osce_generator.py \
  -f json -o /tmp/bandit_task_12102_3_5.json
```

Run: `git diff --check`

Expected: no new Bandit findings in changed code and no whitespace errors.

- [ ] **Step 7: Activate server catalog and bundled frontend fallback together**

Only after Steps 3-6 pass, change the server `osce_scenario` profile to `available` and add the same available profile to the frontend fallback. Keep `default_num_questions=1` solely for old catalog parsers and use `default_num_stations=1` in the UI/API.

```python
"osce_scenario": {
    "status": "available",
    "output_kind": "osce_stations",
    "default_num_stations": 1,
    "default_num_questions": 1,
}
```

- [ ] **Step 8: Re-run release contracts after activation**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_osce_release_contract.py tldw_Server_API/tests/Quizzes/test_osce_generation_endpoint.py -q`

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.osce.test.tsx ../packages/ui/src/services/__tests__/osce.test.ts`

Run: `cd apps/packages/ui && TLDW_VERIFY_OPENAPI_STRICT=1 bun run verify:openapi`

Expected: PASS with both catalog sources presenting OSCE as available.

- [ ] **Step 9: Commit qualification and activation**

```bash
git add tldw_Server_API/app/services/quiz_generator.py apps/packages/ui/src/services/quizzes.ts apps/packages/ui/src/services/tldw/openapi-guard.ts Docs/Product/Advanced_Quiz_Customization_PRD.md Docs/API/Quizzes.md apps/tldw-frontend/e2e/utils/page-objects/QuizPage.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts tldw_Server_API/tests/Quizzes/test_osce_release_contract.py
git commit -m "feat(quizzes): activate qualified OSCE practice"
```

Document in the Backlog task which required gates ran, their exact results, the migration version used, the Bandit report path, and the recovery rule: hide generation by returning the profile to `planned` only on an OSCE-aware deployment; never deploy a pre-OSCE server after OSCE rows exist.

---

## Final Review Checklist

- [ ] Every design-spec requirement maps to a task above; no OSCE marking-guide field is present in candidate schemas, summaries, logs, local drafts, analytics, or exports.
- [ ] Existing clients that omit `activity_type` still create and operate question quizzes unchanged.
- [ ] SQLite and PostgreSQL migrations, constraints, indexes, counts, and cascades match.
- [ ] Generation and v2 import failure injection proves atomic rollback.
- [ ] Nested IDs, optimistic versions, retry keys, repeated transitions, soft deletion, and immutable snapshots have concurrency coverage.
- [ ] UI conflict handling never silently overwrites and transitions never proceed before pending saves are acknowledged.
- [ ] No score, percentage, pass threshold, pass/fail language, OSCE CSV, or incompatible action is present.
- [ ] Server catalog and bundled frontend fallback activate in the same final commit only after release gates pass.
- [ ] Backlog TASK-12102.3.5 contains plan/spec links, touched files, verification evidence, PR link, and final summary before closure.
