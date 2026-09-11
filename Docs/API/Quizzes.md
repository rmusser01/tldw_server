# Quiz API

The Quiz API supports ordinary scored question quizzes and OSCE scenario
practice. OSCE practice is self-marked study guidance. It does not calculate a
score, percentage, passing threshold, or pass/fail result.

## Generation Profiles

`GET /api/v1/quizzes/generation-profiles` returns the available and planned
generation profiles. Clients must inspect `status` before offering a profile.
The `osce_scenario` profile is available. Use `default_num_stations` for its
station count; `default_num_questions` remains `1` only for compatibility with
older catalog parsers.

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
validates and persists the quiz and every station atomically.

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
