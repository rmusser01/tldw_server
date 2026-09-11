# OSCE Scenario Practice Design

Task: TASK-12102.3.5

## Goal

Add source-backed Objective Structured Clinical Examination (OSCE) station generation, authoring, and self-assessed practice to the existing Quiz workspace. OSCE remains study practice: the product does not calculate a percentage, decide pass/fail, or claim professional assessment.

This design extends the existing Quiz domain while keeping OSCE stations and practice attempts separate from normal questions and quiz attempts. Advisory LLM feedback, automated grading, metrics, assignments, remediation, and flashcard conversion are outside this task.

## Current State

The Quiz API and WebUI support generated and manually authored question quizzes. Generation profiles are returned by `GET /api/v1/quizzes/generation-profiles`; `osce_scenario` exists there with `status: "planned"` and is excluded from generation request validation. Generated quizzes use `POST /api/v1/quizzes/generate`, while normal attempts use quiz-level attempt routes and percentage-based grading.

The per-user notes/chat database stores quizzes and attempts on SQLite or PostgreSQL. At the time this design was approved, both backends use schema version 66. The implementation must allocate the next free version after rebasing; it is expected to be version 67 but must not assume that number if another migration lands first.

The Next.js WebUI already provides Generate, Create, Manage, Take, and Results quiz views. The browser extension hands captured material into that shared workspace and does not maintain an independent quiz-taking interface.

## Product Decisions

- OSCE is a first-class quiz activity, not a special `QuestionType`.
- The MVP includes source-backed generation, a basic station editor, advisory timing, private candidate notes, checklist self-assessment, and rubric-level selection.
- Candidate notes may be persisted, but are never sent to an LLM, included in analytics, logged as content, or included in quiz exports.
- Checklist and rubric results remain separate. There is no aggregate score, percentage, pass threshold, or pass/fail label.
- Marking-guide content is unavailable through practice APIs until the user explicitly begins self-assessment.
- Once a quiz contains questions or stations, its activity type cannot change.
- Generated patient details must be fictional or deidentified. The UI warns users not to enter real patient information but makes no HIPAA or regulatory-compliance claim.

## Domain Model

### Quiz Activity

Add the following persisted quiz fields:

- `activity_type`: `questions` or `osce`; existing rows migrate to `questions`.
- `generation_profile`: nullable profile identifier; existing rows remain null unless already recoverable from stored metadata.
- `total_stations`: non-negative integer defaulting to `0`; existing rows migrate to `0`.

Quiz responses add:

- `activity_type`
- `generation_profile`
- `total_stations`

For OSCE quizzes, `total_questions` is always `0`. For question quizzes, `total_stations` is always `0`. List sorting by Size uses the count selected by `activity_type`.

`QuizCreate` accepts `activity_type`. `QuizUpdate` may change it only while the quiz contains no questions or stations and must use the existing optimistic version check. An OSCE create or update request rejects question-only settings, including `passing_score` and `time_limit_seconds`. Question routes reject OSCE quizzes, and station routes reject question quizzes. Existing clients that omit `activity_type` continue to create question quizzes. Manually created quizzes have a null `generation_profile`; generated quizzes persist the selected profile.

### OSCE Station

Add an `osce_stations` table owned by a quiz. A row contains:

- Integer station ID and quiz ID
- `schema_version`, initially `osce.station.v1`
- Validated station content as structured JSON
- Stable server-owned UUIDs for nested checklist items, rubric domains, rubric levels, and expected key points
- `order_index`
- Monotonic edit `version`
- Provenance metadata and source bundle
- Verification state
- Soft-delete state and timestamps

The station contract is versioned independently of the database schema. Requests cannot set station IDs, edit versions, provenance, or verification state. Create and generation requests omit nested IDs; any model-supplied IDs are discarded before validation. Update requests include the server-issued IDs of existing nested objects and omit IDs for new objects. The server preserves recognized IDs, assigns IDs to new objects, and rejects unknown IDs, duplicate IDs, or moving a rubric level to a different domain.

Station content has this logical shape:

```json
{
  "schema_version": "osce.station.v1",
  "title": "Discuss safe anticoagulant use",
  "candidate_instructions": "You are speaking with a simulated patient.",
  "candidate_task": "Explain key safety advice and respond to concerns.",
  "patient_context": {
    "text": "A fictional adult has recently started warfarin.",
    "citations": []
  },
  "recommended_duration_seconds": 480,
  "checklist_items": [
    {
      "id": "server-owned-uuid",
      "label": "Explains the purpose of treatment",
      "rationale": "The source identifies understanding of indication as a safety requirement.",
      "citations": []
    }
  ],
  "rubric_domains": [
    {
      "id": "server-owned-uuid",
      "label": "Communication",
      "levels": [
        {"id": "server-owned-uuid", "label": "Needs development", "description": "Explanation is incomplete or unclear."},
        {"id": "server-owned-uuid", "label": "Effective", "description": "Explanation is clear and checks understanding."}
      ]
    }
  ],
  "expected_key_points": [
    {
      "id": "server-owned-uuid",
      "text": "Discusses monitoring and clinically important warning signs.",
      "citations": []
    }
  ]
}
```

Contract bounds are:

- Title: 1-200 characters
- Candidate instructions and task: 1-4,000 characters each
- Patient-context text: 1-10,000 characters
- Recommended duration: 60-7,200 seconds
- Checklist: 1-50 items; label 1-1,000 characters; optional rationale up to 2,000 characters
- Rubric: 1-12 domains; 2-6 levels per domain, ordered lowest to highest
- Rubric domain and level labels: 1-200 characters; level descriptions: 1-2,000 characters
- Expected key points: 1-50 items; text 1-2,000 characters

All contract models use strict validation and reject unknown fields. They also reject malformed or duplicate UUIDs, empty required collections, duplicate rubric-level labels within a domain, unsupported schema versions, numeric scoring fields, pass thresholds, generated feedback, chain-of-thought fields, and invalid citation or timing data.

### Citations And Provenance

OSCE citations use a strict, OSCE-specific model rather than weakening compatibility of the existing question citation model. Every citation identifies a source type and source ID. Optional labels are limited to 200 characters, quotes to 1,000 characters, source IDs to 512 characters, and URLs to 2,048 characters. Media citations validate non-negative timestamps and valid media/chunk locators. Empty citation objects and locators inconsistent with their source type are rejected.

Citations are embedded directly on patient context, checklist rationales, and expected key points. There is no separate evidence-reference graph, so dangling citation references cannot occur. Manually authored stations may omit citations. Generated stations require at least one resolvable citation on patient context, every checklist rationale, and every expected key point; generated checklist rationales are therefore required even though they are optional for manual stations.

Provenance is server-managed and stored outside editable content:

- `origin`: `generated` or `manual`
- Generation provider and model identifiers, excluding secrets
- Canonical source bundle
- Verification state: `source_verified`, `modified_after_verification`, or `manually_authored`
- Verification timestamp and bounded verification summary where applicable

Claims verification covers factual patient context, expected key points, and checklist rationales. Candidate instructions, task wording, rubric labels, and generic performance descriptions are not treated as factual claims. An edit to evidence-bearing text or citations changes `source_verified` to `modified_after_verification`; presentation-only edits preserve the current state.

### OSCE Practice Attempt

Add an `osce_practice_attempts` table containing:

- Integer attempt ID, station ID, and quiz ID
- Caller-provided `client_attempt_id` UUID
- Immutable station snapshot, including provenance and marking-guide content
- Lifecycle state: `in_progress`, `self_assessment`, or `completed`
- Candidate notes, limited to 10,000 characters
- Checklist selections keyed by checklist-item UUID
- Rubric selections keyed by domain UUID and referencing a level UUID in that domain
- `started_at`, `self_assessment_started_at`, `completed_at`, and frozen elapsed seconds
- Monotonic update `version`

The unique key `(station_id, client_attempt_id)` makes attempt creation retry-safe for the current user database. Repeating creation returns the existing attempt rather than creating another.

Checklist state is tri-state: unanswered, `met`, or `not_met`. A rubric domain remains unanswered until one level is selected. Completion requires every checklist item and rubric domain to have a valid selection. Deterministic results report checklist `met` count and total plus the selected label for each rubric domain; they never derive an aggregate rubric score.

The station snapshot is the source of truth for every attempt. Later station edits or deletion do not change an existing attempt. Completed attempts are immutable. Hard deletion during an authorized quiz purge cascades to stations and attempts; ordinary station deletion is soft deletion, and any station with attempts may only be soft-deleted.

Indexes cover active station ordering by quiz, attempts by station and lifecycle state, attempts by quiz and lifecycle state, and the retry-safe attempt key. `total_stations` is updated in the same transaction as station creation or deletion and is also recoverable by recount during migration or repair.

## API Design

All routes remain under `/api/v1/quizzes` and existing authorization behavior. Inaccessible resources return `404` rather than revealing ownership.

### Generation Profiles

Profile definitions add:

- `output_kind`: `questions` or `osce_stations`
- `default_num_stations`: nullable integer

Existing question-profile fields remain compatible. The OSCE profile remains `planned` until persistence, API, generation, and WebUI support ship together. Its `output_kind` is `osce_stations`, `default_num_stations` is `1`, and its legacy `default_num_questions` catalog value remains `1` for older catalog parsers; OSCE generation does not consume that field.

### Quiz Generation

Continue using `POST /api/v1/quizzes/generate`. For `generation_profile: "osce_scenario"`:

- `num_stations` defaults to 1 and accepts 1-10.
- Explicit `num_questions`, `question_types`, and `question_plan` fields are rejected.
- Existing source, difficulty, focus-topic, provider, verification-provider, and workspace fields remain available.
- The service validates every station and verifies all evidence-bearing claims before opening one persistence transaction.
- The quiz and all stations persist atomically. Any generation, normalization, citation, verification, or database failure leaves no quiz shell or partial station set.

Generation responses add `output_kind` and `osce_stations`:

```json
{
  "output_kind": "osce_stations",
  "quiz": {},
  "questions": [],
  "osce_stations": [],
  "claim_verification": {}
}
```

Question generation continues to return `output_kind: "questions"`, populated `questions`, and an empty station list. Provider failures use stable public errors and never expose raw provider payloads or source content.

### Station Authoring

- `POST /api/v1/quizzes/{quiz_id}/osce-stations` creates a manually authored station.
- `GET /api/v1/quizzes/{quiz_id}/osce-stations` returns paginated station summaries.
- `GET /api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` returns authoring detail, including the marking guide.
- `PATCH /api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` updates editable fields and requires `expected_version`.
- `DELETE /api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` soft-deletes the station.

On PATCH, omitted top-level fields remain unchanged and supplied nested collections replace that collection atomically. A stale `expected_version` returns `409`. Malformed content returns `422`. Lists omit deleted stations by default and return compact summaries rather than full marking guides.

### Practice

- `POST /api/v1/quizzes/osce-stations/{station_id}/attempts` creates or returns a retry-safe attempt.
- `GET /api/v1/quizzes/osce-attempts` returns paginated attempt summaries and accepts optional `quiz_id`, `station_id`, and repeatable lifecycle `state` filters.
- `GET /api/v1/quizzes/osce-attempts/{attempt_id}` returns the attempt appropriate to its lifecycle phase.
- `PATCH /api/v1/quizzes/osce-attempts/{attempt_id}` updates allowed notes or assessment selections and requires `expected_version`.
- `POST /api/v1/quizzes/osce-attempts/{attempt_id}/begin-self-assessment` reveals the snapshot marking guide and freezes elapsed time.
- `POST /api/v1/quizzes/osce-attempts/{attempt_id}/complete` validates all selections and completes the attempt.

Practice uses distinct candidate-phase and revealed response models. During `in_progress`, responses omit checklist items, rubric domains, expected key points, checklist rationales, citation quotes, and all other marking-guide content rather than returning redacted placeholders. The candidate-facing station context, source labels without quotes, attempt timestamps, server time, and notes remain available. Checklist and rubric updates are accepted only in `self_assessment`; notes may be updated in `in_progress` or `self_assessment`.

Attempt summaries never include marking-guide content or candidate notes. They include snapshot title, quiz and station IDs, lifecycle state, version, timestamps, last-modified time, elapsed seconds when available, and deterministic completed-summary counts. Lists sort by most recently modified first. The Take view requests `in_progress` and `self_assessment` states and resumes the most recently modified matching attempt; other unfinished attempts remain selectable. Results requests completed summaries through its own pagination stream.

Transition requests include `expected_version`. If the attempt has not reached the requested state, a stale version returns `409`; if it is already in that state or a later state, the repeated request returns the current representation successfully. Transitions never move state backward. Invalid forward transitions return `409`; incomplete completion or invalid selections return `422`.

The timer is advisory. Elapsed time is derived from server timestamps, includes time away, and freezes at the first transition to self-assessment. It never submits, locks, or changes the attempt automatically.

Starting a new attempt from a deleted station returns `404`. Listing or loading an attempt that began before station deletion remains available from its immutable snapshot and does not require the live station to remain active.

### Import And Export

Introduce `tldw.quiz.export.v2` with an explicit activity type. V2 supports both question and OSCE entries so one bulk file may contain either activity. OSCE entries include quiz metadata, station authoring content, citations, and provenance safe for informational export. They do not include attempts or candidate notes. Import treats all provenance and verification fields as untrusted: it validates an entire OSCE quiz, assigns new protected IDs, records imported stations as manually authored, and persists that quiz entry atomically. Invalid station content leaves no imported quiz shell, while existing per-entry partial success for other quizzes in the same batch remains unchanged. Export/import therefore round-trips editable authoring content, not trusted verification status or internal IDs.

Manage continues to assemble downloadable JSON through the existing client-side export path; no new server export endpoint is added. Question-only exports may remain `tldw.quiz.export.v1`. Any export containing an OSCE entry uses v2. The existing v1 question-quiz import/export behavior remains unchanged, and a v1 entry always imports as `activity_type: "questions"`.

## WebUI Design

OSCE uses the existing Quiz workspace and does not add a top-level navigation tab.

### Generate

Selecting the OSCE profile changes the count label to Stations, defaults it to 1, and enforces the 1-10 range. Question mix, question count, question plan, passing score, and flashcard companion controls are hidden. Source selection, difficulty, focus topics, generation provider, and claims-verification provider remain available.

After generation, the user is routed to Manage to inspect and edit stations before practice. The browser extension continues to use its existing handoff into this shared generation flow.

### Create And Manage

Create uses a `Quiz | OSCE` segmented activity control. The activity type becomes immutable once the quiz contains questions or stations.

A shared `OsceStationEditor` supports both Create and Manage. It is a full-width inline editor, not a large modal. It edits candidate content, duration, checklist items, rubric domains and ordered levels, expected key points, and citations. Familiar arrow icon buttons reorder items and include accessible labels and tooltips. Authoring uses an explicit Save action; dirty navigation guards protect unsaved work.

Manage lists paginated station summaries with duration, checklist count, rubric-domain count, and one of these labels:

- Source verified
- Modified after verification
- Manually authored

Updates use optimistic versioning. On conflict, saving stops and the user chooses Reload server version or Keep local draft; the UI never silently overwrites the server. Keep local draft preserves the unsaved editor state while loading the latest server version for comparison, and a later overwrite requires a new explicit Save confirmation against that version.

### Take

OSCE quizzes show Practice station instead of normal quiz-taking actions. Multi-station quizzes open a station picker. If the selected station has an active attempt, Resume is primary and Start new is secondary.

A dedicated `OscePracticePanel` owns the flow instead of adding more state to the normal question-taking component. Candidate phase shows the task, candidate instructions, patient context, recommended duration, advisory elapsed time, source labels, and private notes. It does not show the marking guide. A separate `OsceResultsPanel` owns OSCE filters, summaries, and result detail rather than adding OSCE state to the normal results implementation.

Begin self-assessment requires confirmation, flushes pending notes, and stops if the save fails. The self-assessment phase keeps candidate context visible, reveals expected points and citations, presents each checklist item as `Met` or `Not met`, and renders rubric levels as labeled radio groups. Complete remains disabled until every required selection is made.

Assignments, remediation, flashcard conversion, and Study Assistant actions are hidden for OSCE activities. Revealing or completing an assessment requires a server connection.

### Results

Results uses independent `Quiz attempts | OSCE practice` segments rather than combining separately paginated feeds. OSCE results show:

- A clear self-marked study-practice label
- Checklist `met` count and total
- Selected rubric level for each domain
- Elapsed time
- Candidate notes
- Expected points and source citations

Results do not show percentage, pass/fail, remediation recommendations, or claims of clinical or professional competence.

The existing results CSV export remains available only in the Quiz attempts segment. The OSCE practice segment does not expose attempt or candidate-note export in this task.

### Saving, Offline Behavior, And Timing

OSCE practice serializes server saves so later responses cannot overwrite newer local edits. It shows concise saving, saved, offline, and conflict status. A local draft fallback may contain candidate notes and unsent selections, expires after 24 hours, and clears after server acknowledgement or completion. Draft keys are scoped by user and attempt. Station authoring remains explicit-save rather than adding a second autosave system.

The local fallback stores only writable draft fields and never stores marking-guide content at any phase. Candidate notes are not sent to logging, analytics, generation, verification, or any LLM feature. The UI warns users not to enter real patient information.

Timer display uses server-provided time or elapsed values as its baseline and a monotonic browser clock between responses. Reload, client clock changes, and background-tab suspension cannot reduce server-derived elapsed time.

Existing `QuizMarkdown` and `SourceCitations` rendering are reused. Layouts must remain usable on supported desktop and mobile widths with keyboard navigation and visible focus states.

## Validation And Safety

- Generated station requests ask for fictional or deidentified patient context.
- Input and output models use `extra="forbid"` at OSCE trust boundaries.
- Generated nested IDs are discarded and replaced by server-owned UUIDs.
- Source verification receives only evidence-bearing station claims and configured source content.
- Candidate notes never enter generation or verification requests.
- Semantic spoiler detection is not guaranteed. Prompts and the editor tell authors to keep expected answers out of candidate-facing text.
- Generated stations are not persisted when source verification fails.
- API logs contain identifiers and state transitions, not candidate-note or source-content bodies.
- All reads and writes use existing per-user database and authorization boundaries.

## Error Handling

- `404`: quiz, station, or attempt is absent, deleted where applicable, mismatched with its parent route, or inaccessible.
- `409`: stale `expected_version`, invalid lifecycle transition, or an activity type that can no longer change.
- `422`: malformed station content, incompatible generation fields, incomplete self-assessment, or invalid nested selections.
- Stable generation errors distinguish provider failure, malformed output, unsupported station contract, citation failure, and verification failure without returning raw source or provider content.
- Database transactions roll back every write in failed generation and import operations.

## Testing Strategy

Implementation follows test-driven development in reviewable stages.

### Backend

- Strict station and attempt validation: bounds, protected IDs, rubric ordering, citations, timing, unknown fields, and forbidden scoring/advisory fields.
- Fresh-install and upgrade migrations on SQLite and PostgreSQL, including schema parity, indexes, defaults, and constraints.
- CRUD, pagination, activity enforcement, optimistic conflicts, soft deletion, immutable snapshots, and cross-user `404` behavior.
- Attempt-list filters and summaries support active-attempt discovery and independently paginated Results without exposing notes or marking guides.
- Lifecycle and idempotency behavior: duplicate attempt creation, repeated transitions, stale updates, concurrent requests, and no state regression.
- Candidate-phase response serialization proves marking-guide fields are absent.
- Candidate notes are absent from mocked LLM calls, verification calls, logs, analytics payloads, and exports.
- Failure injection proves generation and v2 import leave no partial quiz, station, or provenance records.
- Evidence-bearing edits invalidate verification while presentation-only edits preserve it.
- Generated stations reject missing, inaccessible, or source-inconsistent citations while manual stations may remain uncited.
- Export v2 round-trips OSCE authoring data atomically, excludes attempts and notes, and leaves v1 question behavior unchanged.
- Existing question generation, attempts, profiles, imports, exports, and grading retain regression coverage.
- Existing MCP quiz tools reject or omit OSCE activities rather than treating them as empty question quizzes; new OSCE MCP tools are not introduced.

### Frontend

- Profile controls, incompatible-field removal, immutable activity type, and suppression of incompatible actions.
- Editor validation, accessible ordering, dirty guards, verification labels, serialized saves, and conflict recovery.
- Candidate, reveal, self-assessment, completion, resume, and segmented-results workflows.
- Local-draft expiry and cleanup, disconnect/reconnect behavior, and transition save flushing.
- Browser storage never contains marking-guide content, and candidate-phase error states never reveal it.
- Fake-clock tests cover server offset, reload, local clock changes, background-tab suspension, and resume.
- Responsive layouts, keyboard and focus behavior, sanitization, and existing citation rendering.

### End To End

- Generate, review, practice, self-assess, complete, and inspect results.
- Disconnect, edit locally, reconnect, and resolve a server-version conflict.
- Edit or delete a station after starting an attempt and confirm the attempt snapshot remains stable.
- Exercise stations near supported size limits and paginated station and result lists.

Stage-level checks use focused Pytest and Vitest suites plus compile and type checks. Final release gates require real SQLite and PostgreSQL runs, neighboring quiz regressions, Playwright workflows, OpenAPI contract checks, Ruff, TypeScript checks, `compileall`, `git diff --check`, and Bandit on touched backend code. A required environment that is unavailable is blocked, not passing.

## Rollout And Recovery

1. Rebase on current `origin/dev` and allocate the next free database migration version.
2. Ship additive persistence and API support while the OSCE catalog profile remains `planned`.
3. Ship generation and WebUI support while the server catalog and bundled frontend fallback still hide OSCE.
4. Run release gates against the same server and frontend revision.
5. Change the server catalog profile to `available` only in the compatible release.
6. Leave generation metrics and operational dashboards to TASK-12102.3.7.

Returning the profile to `planned` in an OSCE-aware deployment is the operational rollback. It hides generation and new-practice entry points without deleting data; resume and results access for existing attempts remains available. The additive database schema remains in place and no destructive schema rollback is attempted. Once OSCE rows exist, the server must not roll back to a pre-OSCE version that would interpret an OSCE quiz as an empty question quiz; rollback builds must retain activity-type guards and snapshot read support. A separate runtime feature-flag subsystem is not introduced for this task.

## Implementation Boundaries

Implementation should extend the existing quiz schemas, endpoints, generation service, database abstraction, and the canonical shared Quiz UI under `apps/packages/ui/src`. The Next.js app and browser extension continue to consume that shared package. New focused modules are appropriate for station contracts, OSCE query hooks, `OsceStationEditor`, `OscePracticePanel`, and `OsceResultsPanel`; the existing multi-thousand-line Manage, Take, and Results tabs should only receive routing and composition changes. Unrelated quiz refactors are not part of this work.

The implementation plan should stage the work as:

1. Schema contracts and dual-backend persistence
2. Station and attempt APIs
3. Source-backed generation and verification
4. WebUI authoring, practice, and results
5. Import/export, end-to-end verification, documentation, and catalog availability

## Out Of Scope

- Automated or LLM-assisted scoring, coaching, or free-text evaluation
- Aggregate rubric scores, percentages, pass thresholds, and pass/fail decisions
- Authoritative clinical assessment or certification claims
- Candidate-answer audio/video capture, simulated-patient chat, or examiner collaboration
- Assignments, remediation, flashcard conversion, and Study Assistant integration
- Attempt or candidate-note export
- OSCE analytics and observability, owned by TASK-12102.3.7
- A separate browser-extension OSCE interface
- New MCP tools for authoring or practicing OSCE stations
- A new feature-flag framework or destructive migration rollback
