# Source-Grounded Spaced Repetition Design

Task: TASK-12932

## Goal

Add first-pass source-grounded spaced repetition so users can schedule repeated review of source-backed material from Flashcards.

This design creates source review plans with per-occurrence study activities and due dates. It reuses existing Flashcards, quiz, study-pack source item shapes, and review UI patterns where practical. It does not add a new scheduler engine, external notifications, visual recognition questions, extended matching questions, or Notes/Research highlight wiring in this task.

## Current State

Flashcards already has deck-level SM2/FSRS scheduling, due cards, review sessions, source reference fields, study packs, and source-bundle JSON. Quizzes and flashcard generation already support source-grounded generation flows. The generic reminders API can schedule one-time or recurring reminders, but it is not a study review plan and would scatter per-occurrence progress state outside Flashcards.

The next feature needs something different from deck card scheduling: a user selects source-grounded material, defines a custom review timeline, and chooses what kind of review happens on each due date.

## Scope

Included:

- Flashcards-owned source review plan routes under `/api/v1/flashcards/source-review-plans`.
- Neutral DB tables named `source_review_plans` and `source_review_occurrences`.
- Planner drawer in Flashcards.
- Due source-review panel in Flashcards.
- Preset editable schedule rows: Day 1, Day 3, Day 7, Day 14, Day 28, 3 months, 6 months.
- Per-occurrence activities: `reread`, `quiz`, `flashcards`, `cloze`.
- Study Pack-style source items with optional excerpt and locator snapshots.
- Server-computed due dates from a stored anchor date.
- Idempotent start/resume metadata.
- Manual completion for all occurrence types.

Out of scope:

- Root-level or standalone source-review API.
- Notes/Research highlight action wiring.
- Schedule editing after creation.
- External reminders, notifications, or background delivery.
- New scheduler engine.
- Mandatory quiz/flashcard/cloze artifact generation on start.
- Required auto-completion hooks.
- Visual recognition and extended matching activities.

## API Contract

Keep the public API under Flashcards:

- `POST /api/v1/flashcards/source-review-plans`
- `GET /api/v1/flashcards/source-review-plans`
- `GET /api/v1/flashcards/source-review-plans/due`
- `POST /api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/start`
- `POST /api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/complete`
- `POST /api/v1/flashcards/source-review-plans/occurrences/{occurrence_id}/skip`
- `DELETE /api/v1/flashcards/source-review-plans/{plan_id}`

Implementation can live in a small Flashcards subrouter module if that keeps `endpoints/flashcards.py` from growing further. The public route ownership remains Flashcards either way.

Response shapes:

- `SourceReviewPlanResponse`: plan fields plus occurrence summaries.
- `SourceReviewPlanListResponse`: `{ "items": [...], "total": N }`.
- `SourceReviewDueListResponse`: `{ "items": [...], "total": N, "now": "..." }`, where `now` is the backend's current UTC ISO timestamp.
- `SourceReviewOccurrenceActionResponse`: occurrence fields plus optional `launch_state`.
- `SourceReviewPlanDeleteResponse`: `{ "deleted": true | false }`.

`SourceReviewOccurrenceActionResponse.launch_state` is assembled by the API from occurrence metadata plus the plan's stored source snapshot. The stored `launch_state_json` column contains only thin route/action metadata and timestamps; it does not duplicate `source_bundle_json`. It never contains generated quiz/card/cloze content.

List and due query behavior:

- Plan list defaults to `limit = 50`, caps `limit` at 100, supports `offset`, and orders by `created_at DESC, id DESC`.
- Due list defaults to `limit = 50`, caps `limit` at 100, supports `offset`, and orders by `due_at ASC, id ASC`.
- Both list responses include `total` for the filtered result set before `limit`/`offset`.

Create request:

```json
{
  "title": "Cardiac physiology review",
  "starts_on": "2026-07-09",
  "timezone": "America/Los_Angeles",
  "source_items": [
    {
      "source_type": "media",
      "source_id": "42",
      "label": "Lecture 3",
      "excerpt_text": "Frank-Starling mechanism...",
      "locator": { "page": 12, "chunk_id": "chunk_4" }
    }
  ],
  "schedule": [
    { "offset_value": 1, "offset_unit": "day", "activity_type": "reread" },
    { "offset_value": 3, "offset_unit": "day", "activity_type": "cloze" },
    { "offset_value": 7, "offset_unit": "day", "activity_type": "quiz" },
    { "offset_value": 14, "offset_unit": "day", "activity_type": "flashcards" },
    { "offset_value": 28, "offset_unit": "day", "activity_type": "quiz" },
    { "offset_value": 3, "offset_unit": "month", "activity_type": "flashcards" },
    { "offset_value": 6, "offset_unit": "month", "activity_type": "quiz" }
  ]
}
```

Field rules:

- `title`: non-empty string, max 200 characters.
- `starts_on`: ISO date. This is the authoritative anchor for schedule offsets.
- `timezone`: required IANA timezone string. The WebUI should default it from the browser timezone. The backend validates it with `zoneinfo`.
- `source_items`: at least one item using the existing `StudyPackSourceSelection` shape, max 10 items.
- `source_items[].excerpt_text`: optional, max 20,000 characters per item.
- `source_items[].locator`: optional JSON object, max 8 KiB serialized per item.
- `schedule`: at least one row, max 24 rows.
- `offset_value`: strict positive integer, max `3650` for `day` rows and max `120` for `month` rows.
- `offset_unit`: `day` or `month`.
- `activity_type`: `reread`, `quiz`, `flashcards`, or `cloze`.

Validation rules:

- Use server-side date math only. The client never recomputes due dates after creation.
- Compute each occurrence date in the plan timezone, at local `00:00:00`, then convert that instant to UTC for `due_at`.
- Due queries compare stored UTC `due_at` with current UTC time.
- Month offsets use a stdlib helper that clamps to month end, for example January 31 plus 1 month becomes February 28 or 29.
- Reject duplicate computed `(due_at, activity_type)` rows after date math. The same due date may have multiple activities, but not the same activity twice.

Example: `starts_on = 2026-07-09`, `timezone = America/Los_Angeles`, and `{ "offset_value": 1, "offset_unit": "day" }` produces a local due date of `2026-07-10 00:00:00` and a stored UTC `due_at` for that instant.

Source snapshot behavior:

- The first slice stores client-supplied `source_items` as wrapped plan `source_bundle_json`: `{ "items": [...] }`.
- Import and reuse `StudyPackSourceSelection` directly where practical. Accept its `source_title` alias on input, but serialize the canonical `label` field in stored snapshots and responses.
- Create validates the Study Pack source selection shape but does not resolve or refresh source content through a separate resolver.
- Reread/start payloads use the stored snapshot. If `excerpt_text` is missing, the UI shows the source label/type/id and locator instead of trying to fetch source text.

## Data Model

Add two tables to `ChaChaNotes_DB` using the DB's existing migration, timestamp, soft-delete, `client_id`, `version`, and sync-log patterns used by Study Pack storage.

`source_review_plans`:

- `id`
- `title`
- `starts_on`
- `timezone`
- `source_bundle_json`
- `created_at`
- `last_modified`
- `deleted`
- `client_id`
- `version`

`source_review_occurrences`:

- `id`
- `plan_id`
- `offset_value`
- `offset_unit`
- `activity_type`
- `due_at`
- `status`: `pending`, `in_progress`, `completed`, `skipped`
- `launch_state_json`
- `started_at`
- `completed_at`
- `completion_source`
- `created_at`
- `last_modified`
- `deleted`
- `client_id`
- `version`

Indexes:

- occurrence plan ID
- occurrence due/status
- plan deleted
- list/due composite indexes that support `deleted = 0`, `status`, `due_at`, and stable `id` ordering

Soft-delete plans and occurrences; never hard-delete either table in this feature. Plan delete runs in one transaction and sets `source_review_plans.deleted = 1` plus `source_review_occurrences.deleted = 1` for that plan's undeleted occurrences.

Plan completion is derived from occurrences. The first slice does not need a plan lifecycle status column; a plan is active while `deleted = 0`.

`DELETE /source-review-plans/{plan_id}` is idempotent for existing plans: the first successful call returns `{ "deleted": true }`; repeated calls for an already-deleted plan return `{ "deleted": false }` and do not mutate versions or sync rows. A missing plan ID returns 404.

Sync/version requirements:

- SQLite and any PostgreSQL schema/migration path for `ChaChaNotes_DB` must stay equivalent for these two tables.
- Create sets `version = 1`, current timestamps, and the request `client_id`.
- Start/complete/skip/delete increment `version`, update `last_modified`, preserve/update `client_id` using the DB's existing convention, and write matching `sync_log` rows.
- Sync entities are `source_review_plans` and `source_review_occurrences`; operations are `create`, `update`, and `delete`.
- Plan delete writes a delete sync row for the plan and for each occurrence it newly soft-deletes. Repeated idempotent delete writes no additional sync rows.

## Start And Completion

Starting an occurrence is idempotent:

- `pending` occurrence: set `status = in_progress`, store `started_at`, and write thin `launch_state_json` metadata in one DB transaction.
- `in_progress` occurrence with launch state: return the stored launch state.
- `completed` occurrence: return the completed occurrence without creating new launch state.
- `skipped` occurrence: return a conflict.
- deleted plan or occurrence: return 404 for start/complete/skip.

Launch state is intentionally thin in storage. Stored `launch_state_json` fields:

- `activity_type`
- `plan_id`
- `occurrence_id`
- `target_route`
- `target_surface`
- `action`
- `source_payload_field`
- `completion_required`: always `true` in this slice
- `created_at`

The API response `launch_state` includes those stored fields plus `source_bundle` assembled from the plan's `source_bundle_json`. When `source_payload_field = "source_items"`, the UI derives that payload from `launch_state.source_bundle.items`; the response does not duplicate the items under a second key. The 16 KiB cap applies only to stored `launch_state_json`, not to the assembled response's source snapshot.

Do not store generated quiz/card/cloze content in `launch_state_json`. Cap serialized launch state to 16 KiB; if an activity later needs larger generated content, store it through that feature's existing artifact/storage path and keep only a reference here.

Activity launch behavior:

- `reread`: `target_route = "/flashcards"`, `target_surface = "source_review_due_panel"`, `action = "show_reread_snapshot"`, `source_payload_field = "source_bundle"`.
- `quiz`: `target_route = "/quiz"`, `target_surface = "quiz_generation"`, `action = "prefill_generation_sources"`, `source_payload_field = "source_items"`.
- `flashcards`: `target_route = "/flashcards"`, `target_surface = "flashcard_generation"`, `action = "prefill_generation_sources"`, `source_payload_field = "source_items"`.
- `cloze`: `target_route = "/flashcards"`, `target_surface = "cloze_flashcard_generation"`, `action = "prefill_generation_sources"`, `source_payload_field = "source_items"`.

For `quiz`, `flashcards`, and `cloze`, start does not call generation endpoints. It only returns enough state for the UI to prefill the existing generation flow and require the user to complete the occurrence manually.

Completion:

- Manual complete is required for all activity types.
- `complete` is idempotent: repeated calls for a completed occurrence return the completed occurrence.
- `skip` after complete conflicts.
- `complete` after skip conflicts.
- Required auto-complete is out of scope. If implementation finds a tiny existing hook, such as quiz attempt completion or flashcard review session end, it may call the same completion helper and set `completion_source`, but the MVP must not depend on it.

State transitions:

| Current status | `start` | `complete` | `skip` |
| --- | --- | --- | --- |
| `pending` | Set `in_progress` and create launch state | Conflict; start first | Set `skipped` |
| `in_progress` | Return existing launch state | Set `completed` | Set `skipped` |
| `completed` | Return completed occurrence | Return completed occurrence | Conflict |
| `skipped` | Conflict | Conflict | Return skipped occurrence |

## WebUI

Add Flashcards-only UI:

- `SourceReviewPlanDrawer` for creating plans.
- `SourceReviewDuePanel` inside the Flashcards Review/Study area.
- Client methods in the existing Flashcards service module.

Planner drawer:

- Title input.
- Native date input or existing lightweight date control for `starts_on`; no new date-picker dependency.
- Source item entry using Study Pack-style fields: source type, source ID, optional title/label, optional excerpt, optional locator.
- Preset schedule rows seeded from Day 1, 3, 7, 14, 28, 3 months, 6 months.
- Editable offset value, offset unit, and activity per row.
- Add/remove rows.
- Disable create until title, at least one source, and at least one schedule row are present.
- All submitted schedule rows must be valid; the UI must not silently drop invalid rows.
- Show row-level errors for invalid offset/activity and exact duplicate `(offset_value, offset_unit, activity_type)` rows.
- The backend remains authoritative for computed duplicate `(due_at, activity_type)` validation; the UI should surface that API error rather than recomputing due dates.

Due panel:

- List due pending/in-progress occurrences where `due_at <= now`.
- Empty state when nothing is due.
- Start action for pending occurrences.
- Resume action for in-progress occurrences.
- Manual complete action for started occurrences.
- Skip action.
- Display the plan title, due date, activity label, and source label/excerpt summary.
- `reread` start shows the source snapshot in the due panel.
- `quiz`, `flashcards`, and `cloze` start use `launch_state.target_route`, `target_surface`, and `source_payload_field` to prefill the existing generation UI; they do not auto-generate artifacts.

Do not add schedule editing after creation in this task.

## Permissions And Compatibility

Use the same auth/dependency behavior as existing Flashcards endpoints. Do not add a new permission family.

Keep all existing Flashcards, quiz, study-pack, extension, sidepanel, and Research Workspace behavior backward compatible. No hidden default changes outside the new source review plan UI.

Avoid broad generated OpenAPI/client churn unless the repository's existing workflow requires it for touched routes. If this checkout has generated API typings or route allowlists for new paths, update only the narrow Flashcards source-review entries.

## Tests

Backend DB tests:

- Create a plan with preset rows and source snapshot.
- Due query returns pending and in-progress occurrences with `due_at <= now`.
- Due query hides future, skipped, completed, and soft-deleted plan occurrences.
- Start is idempotent and writes status plus launch state atomically.
- Stored launch state remains below the 16 KiB cap and does not duplicate large source snapshots.
- Complete retry is idempotent.
- Skip/complete conflict rules are enforced.
- Full state transition table coverage: complete on pending, repeated skip, start on completed, start on skipped, complete after skip, and skip after complete.
- Soft-deleted plans and occurrences are hidden.
- Missing/deleted plan or occurrence actions return 404.
- Plan delete soft-deletes plan and undeleted occurrences in one transaction.
- Plan delete writes delete sync rows once and repeated idempotent delete writes none.
- Month-end math clamps correctly for January 31 plus 1 month.

Backend API tests:

- Reject empty title.
- Reject overlong title.
- Reject missing source items.
- Reject too many source items.
- Reject overlong source excerpt or locator payload.
- Reject missing schedule rows.
- Reject too many schedule rows.
- Reject missing or invalid timezone.
- Reject invalid activity.
- Reject non-positive or non-integer offset.
- Reject offset values over the day/month caps.
- Reject duplicate computed due date plus activity.
- Delete returns `{ "deleted": true }` on first soft delete and `{ "deleted": false }` on repeat.
- Delete missing plan returns 404.
- List and due ordering are stable and enforce the `limit` cap.
- Create/list/due/start/complete/skip happy paths.
- Start returns stored launch state when repeated.
- Start action response assembles `source_bundle` from the plan snapshot while stored `launch_state_json` remains thin.
- Complete after skip and skip after complete conflict.

Frontend tests:

- Planner validation covers invalid rows, duplicate rows, and valid create payload.
- Due panel covers empty state plus start/resume/manual complete/skip states.

Security validation:

- Run Bandit on touched backend Python.

## Implementation Notes

- Keep the first implementation boring: DB helpers, Flashcards route handlers, service client, and two small UI components.
- Store source excerpts and locators as snapshots so reread remains useful if the upstream source changes.
- Prefer stdlib `datetime`, `zoneinfo`, and `calendar` for date math.
- Manual completion is the product guarantee; auto-complete is opportunistic only.
- Do not overload flashcard review sessions or generic reminders to represent plans.
