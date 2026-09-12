# Quiz API

The Quiz API supports ordinary scored question quizzes and OSCE scenario
practice. OSCE practice is self-marked study guidance. It does not calculate a
score, percentage, passing threshold, or pass/fail result.

## Generation Profiles

`GET /api/v1/quizzes/generation-profiles` is the source of truth for profile
availability and defaults. Clients must inspect `status` and `output_kind`
before offering a profile. All profiles below are currently available.

| Profile | Output | Default size | Main constraint |
| --- | --- | --- | --- |
| `standard_recall` | Questions | 10 questions | General recall and application |
| `mixed_assessment` | Questions | 10 questions | Mixed recall, interpretation, and application |
| `best_of_five` | Questions | 5 questions | Exactly five options and one best answer |
| `emq` | Questions | 5 stems | At least two stems per shared option bank |
| `assertion_reasoning` | Questions | 5 questions | Canonical A-E assertion/reason scale |
| `osce_scenario` | OSCE stations | 1 station | Self-assessed practice with no numeric score |

Question-profile requests accept `sources`, `generation_profile`,
`num_questions`, `difficulty`, optional `focus_topics`, and either
`question_types` or a `question_plan`. `question_plan` is supported only by
`standard_recall` and `mixed_assessment`. OSCE requests use `num_stations` and
reject question-count and question-shape controls.

Every generated question is normalized to the same base shape. Optional fields
are `null` or omitted when they do not apply:

```json
{
  "output_kind": "questions",
  "questions": [{
    "question_type": "multiple_choice",
    "question_text": "Question text",
    "group_id": null,
    "group_prompt": null,
    "options": ["A", "B", "C", "D"],
    "correct_answer": 1,
    "explanation": "Concise source-backed rationale.",
    "hint": null,
    "hint_penalty_points": 0,
    "source_citations": [{
      "source_type": "note",
      "source_id": "note-advanced-quiz",
      "quote": "Bounded supporting excerpt."
    }],
    "tags": ["topic"],
    "points": 1
  }]
}
```

Citations must reference a selected source. Generation fails when a question
has no citation or names an unselected source. Explanations are concise
rationales, not hidden chain-of-thought. The complete machine-validated example
and malformed-output matrix is in
`tldw_Server_API/tests/Quizzes/fixtures/advanced_quiz_generation_profiles.json`.

<a id="profile-standard_recall"></a>
### `standard_recall`

Use this profile for a conventional quiz or a precise per-type plan:

```json
{
  "sources": [{"source_type": "note", "source_id": "note-advanced-quiz"}],
  "generation_profile": "standard_recall",
  "num_questions": 3,
  "question_plan": [
    {"question_type": "multiple_choice", "count": 2, "option_count": 4},
    {"question_type": "true_false", "count": 1}
  ],
  "difficulty": "mixed"
}
```

The response uses ordinary question records with no reserved subtype or group
metadata. The Generate tab exposes the complete question-plan editor; Take Quiz
uses the normal controls for each base question type. A plan is exact: missing,
extra, or malformed generated questions reject the generation rather than
silently changing the requested mix.

<a id="profile-mixed_assessment"></a>
### `mixed_assessment`

```json
{
  "sources": [{"source_type": "note", "source_id": "note-advanced-quiz"}],
  "generation_profile": "mixed_assessment",
  "num_questions": 2,
  "question_types": ["true_false", "fill_blank"],
  "difficulty": "mixed",
  "focus_topics": ["mechanisms", "application"]
}
```

The output uses the same ordinary question schema as `standard_recall`, while
the prompt asks for a broader cognitive mix. The WebUI and extension expose the
same per-type controls. Difficulty is a generation instruction, not a calibrated
psychometric guarantee, and source quality limits the quality of application
questions.

<a id="profile-best_of_five"></a>
### `best_of_five`

```json
{
  "sources": [{"source_type": "note", "source_id": "note-advanced-quiz"}],
  "generation_profile": "best_of_five",
  "num_questions": 5,
  "question_types": ["multiple_choice"],
  "difficulty": "mixed"
}
```

Each output is a `multiple_choice` question with exactly five distinct options, a
zero-based `correct_answer`, a source-backed explanation, and one canonical
`best_of_five` tag in addition to optional topic tags. Generation rejects an
invalid option count or answer, or a missing, blank, or non-string explanation,
instead of guessing. Take Quiz renders the five
options as an ordinary single-answer question. The profile constrains shape but
cannot guarantee that model-generated distractors are equally plausible.

<a id="profile-emq"></a>
### `emq`

```json
{
  "sources": [{"source_type": "note", "source_id": "note-advanced-quiz"}],
  "generation_profile": "emq",
  "num_questions": 4,
  "question_types": ["multiple_choice"],
  "difficulty": "mixed"
}
```

Every stem repeats the same nonempty `group_id`, `group_prompt`, and option bank
for its group. A group contains at least two stems, each with its own
zero-based answer, explanation, and citations. The UI presents the shared bank
once with its related stems. Because groups are atomic, asking for one stem can
produce two and result limiting never splits a group.

```json
{
  "question_type": "multiple_choice",
  "question_text": "A patient has episodic reversible wheeze.",
  "group_id": "respiratory-diagnosis",
  "group_prompt": "Choose the single most likely diagnosis for each stem.",
  "options": ["Asthma", "Pneumonia", "Pulmonary embolism"],
  "correct_answer": 0,
  "explanation": "Variable reversible airflow obstruction supports asthma.",
  "source_citations": [{
    "source_type": "note",
    "source_id": "note-advanced-quiz",
    "quote": "Asthma causes episodic reversible airflow obstruction."
  }]
}
```

<a id="profile-assertion_reasoning"></a>
### `assertion_reasoning`

```json
{
  "sources": [{"source_type": "note", "source_id": "note-advanced-quiz"}],
  "generation_profile": "assertion_reasoning",
  "num_questions": 5,
  "question_types": ["multiple_choice"],
  "difficulty": "mixed"
}
```

Generated input contains separate `assertion` and `reason` fields. Persisted
questions combine them into explicitly labeled `question_text`, own the
canonical five-option A-E scale, and carry exactly one
`assertion_reasoning` tag. Answers may arrive as a zero-based index, A-E letter,
or exact canonical label and are stored as a zero-based index. The UI shows the
scale once and keeps option order fixed. Explanations must state the concise
evidential relationship; hidden reasoning fields are discarded.

<a id="profile-osce_scenario"></a>
### `osce_scenario`

Generate OSCE content with `POST /api/v1/quizzes/generate`:

```json
{
  "sources": [{"source_type": "note", "source_id": "note-123"}],
  "generation_profile": "osce_scenario",
  "num_stations": 2,
  "difficulty": "mixed"
}
```

The response has `output_kind: "osce_stations"`, an OSCE quiz shell, an empty
`questions` array, and authoring-detail entries in `osce_stations`. Generation
validates and persists the quiz and every station atomically. The Generate tab
switches to station-count controls, and practice uses the dedicated scenario,
notes, reveal, and self-assessment interface. Marking guides remain hidden until
self-assessment; results contain no score, percentage, or pass/fail outcome.
`default_num_questions` remains `1` only for compatibility with older catalog
parsers; use `default_num_stations` for the station count. Likewise,
`default_question_types` remains `["fill_blank"]` for catalog compatibility and
is ignored for OSCE station generation.

## Station Authoring

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/quizzes/{quiz_id}/osce-stations` | Create a manual station |
| `GET` | `/api/v1/quizzes/{quiz_id}/osce-stations` | List compact active station summaries |
| `GET` | `/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` | Read complete authoring detail |
| `PATCH` | `/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` | Update content with optimistic locking |
| `DELETE` | `/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}` | Soft-delete a station |

Station content uses schema version `osce.station.v1` and contains candidate
instructions, a candidate task, patient context, a recommended duration,
checklist items, rubric domains and levels, and expected key points. Nested
checklist, rubric, level, and key-point IDs are stable UUIDs.

Updates require `expected_version`. A stale version returns `409`; clients must
reload the current station and explicitly choose whether to discard or reapply
their local draft. Deletes accept `expected_version` as a query parameter.
Create and delete calls can be retried only after reconciling the station list
when the transport result is ambiguous.

Station lists use offset pagination with `limit` from 1 to 200 and include both
canonical `pagination` metadata and top-level compatibility aliases.

## Practice Attempts

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/quizzes/osce-stations/{station_id}/attempts` | Start or recover an attempt by `client_attempt_id` |
| `GET` | `/api/v1/quizzes/osce-attempts` | List note-free attempt summaries |
| `GET` | `/api/v1/quizzes/osce-attempts/{attempt_id}` | Read phase-dependent attempt detail |
| `PATCH` | `/api/v1/quizzes/osce-attempts/{attempt_id}` | Save notes or revealed self-assessment selections |
| `POST` | `/api/v1/quizzes/osce-attempts/{attempt_id}/begin-self-assessment` | Reveal the marking guide |
| `POST` | `/api/v1/quizzes/osce-attempts/{attempt_id}/complete` | Complete a fully self-assessed attempt |

Starting an attempt requires a UUID `client_attempt_id`. Repeating the same UUID
is retry-safe and returns the original attempt. Each attempt stores an immutable
station snapshot, so later station edits or deletion do not alter active or
completed practice.

While `state` is `in_progress`, the response includes only candidate-safe
station fields. It excludes checklist rationales, rubric content, expected key
points, provenance, verification details, and source quotes. After
`begin-self-assessment`, the immutable marking guide is returned and the elapsed
time is frozen. Completion requires a selection for every checklist item and
rubric domain.

Attempt patches and transitions require `expected_version`. A `409` means the
server changed since the client loaded the attempt. Local drafts should be kept
under a server/auth/organization/user-scoped key and must never contain an
unrevealed marking guide. Reconnect logic must replay the draft against its last
acknowledged version rather than silently overwriting newer server state.

## Results And Portability

Completed OSCE summaries expose checklist counts, rubric labels, and elapsed
time, but no numeric grade. OSCE results are not included in question-quiz CSV
exports.

JSON export/import uses `tldw.quiz.export.v2`. An OSCE entry has
`activity_type: "osce"`, quiz metadata, and a `stations` array. Imports reject
mixed or malformed station content and roll back an individual quiz atomically.
The v1 format remains for question-only quizzes.

## Safety And Recovery

Use fictional or deidentified scenarios. Candidate notes must not contain real
patient information. This feature is study practice, not clinical decision
support, professional assessment, or certification.

If generation must be disabled after release, return `osce_scenario` to
`status: "planned"` in both the server catalog and bundled frontend fallback,
but deploy that change only on an OSCE-aware server revision. Never deploy a
pre-OSCE server after migration v67 has created OSCE rows; older code does not
understand those records or their lifecycle guarantees.
