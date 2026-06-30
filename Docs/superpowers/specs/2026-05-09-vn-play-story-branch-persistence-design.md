# VN Play Story Branch Persistence Design

Status: Draft
Date: 2026-05-09
Owner: Core/WebUI maintainers
Scope: Persist selected Story/CYOA choices as VN Play branch metadata
Tracking: https://github.com/rmusser01/tldw_server/issues/1434
Backlog: TASK-179

## Summary

Persist Story/CYOA choice selection as a first-class branch transition before the
model call starts. A selected visible choice creates one `vn_play_branches` row,
appends one `choice_selected` input event with the branch ID, clears stale visible
choices through existing scene replay, persists the accepted-choice scene state
before model work, and carries the active branch into model context, failure
recovery, checkpoints, and branch listing.

This is a backend-first runtime change. The API server remains authoritative for
choice validation, branch creation, event ordering, idempotency, retry behavior,
and derived scene state. Existing and custom frontends continue to submit choices
through `POST /api/v1/vn-play/sessions/{session_id}/turn`.

## Goals

- Validate Story choices against the current server-authoritative
  `visible_choices`.
- Reject unsupported `choice_id` submits before model work starts.
- Create exactly one branch row per accepted selected choice.
- Append `choice_selected` as the accepted input event for Story choices.
- Persist `active_branch_node_id` in derived scene state before model work starts,
  including model-failure paths and process-crash recovery.
- Preserve turn idempotency and per-session turn concurrency guarantees.
- Make `retry-last-turn` reuse the existing failed choice branch instead of
  creating a duplicate branch.
- Document behavior for API clients and add focused regression coverage.

## Non-Goals

- No authored story graph editor.
- No realtime image generation.
- No frontend-only branch persistence.
- No branch delete, merge, rename, or graph visualization workflow.
- No multi-choice batch submission.
- No new public export/import format for branch history.

## Existing Project Context

The current VN Play runtime already has most of the storage needed:

- `vn_play_branches` stores branch metadata in the user's `ChaChaNotes.db`.
- `vn_play_events` already includes `choice_selected`.
- `derive_scene_state()` already clears `visible_choices` and applies
  `active_branch_node_id` when replaying `choice_selected`.
- `vn_play_scene_state` already stores `active_branch_node_id`.
- `GET /sessions/{session_id}/branches` already lists branch rows.
- `choice_presented` events already carry model-generated visible choices.

The missing behavior is in the turn service. Today the accepted-turn helper always
appends `user_turn`, and Story choices are not converted into branch rows or
`choice_selected` events before the model call.

## Chosen Approach

Use a backend branch-on-choice flow inside `VNPlayService.submit_turn()`:

1. Resolve idempotent duplicates before any validation with side effects.
2. Load the session and current persisted scene state.
3. Check `client_scene_version`, session active-turn state, and mode/input
   compatibility.
4. For Story `choice_id` submits, validate the choice against the current
   `visible_choices`.
5. Create the turn request and acquire the session turn lock.
6. In one accepted-turn transaction, create one branch row, append
   `turn_started`, append `choice_selected`, update the turn request, and persist
   the derived scene state with `last_event_id` set to the `choice_selected`
   event.
7. Mark the turn request `model_calling` with `input_event_id` pointing at
   `choice_selected`.
8. Build model context from replayed state, including the new branch.
9. Complete or fail the turn using the existing second transaction boundary.

Freeform text and custom actions continue to append `user_turn`. Story selected
choices append `choice_selected` instead of a duplicate `user_turn`; that event is
the accepted user input for retry and idempotency purposes.

## API Semantics

### Accepted Inputs

The turn endpoint continues to accept exactly one of `input_text`, `choice_id`, or
`custom_action`.

V1 compatibility rules:

- Freeform sessions accept `input_text` and non-branch custom actions.
- Freeform sessions reject `choice_id` with `choice_not_allowed`.
- Story sessions accept `choice_id` when it matches the current
  `visible_choices`.
- Story sessions reject unknown choices with `invalid_choice_id`.
- Story sessions accept `custom_action` as a non-branching `user_turn` in V1.
  Custom actions do not create branch rows in this sprint.
- Story sessions reject `input_text` with `choice_not_allowed` unless a future
  API setting explicitly enables freeform Story turns.

Choice validation errors happen before turn request creation, event append,
branch creation, scene-state mutation, or model work.

### Stable Error Codes

Add stable VN Play error constants:

- `choice_not_allowed`
- `invalid_choice_id`

These should be returned as normal VN turn errors. They are not retryable
conflicts like stale scene versions or active turns.

## Branch Row Contract

For each accepted Story `choice_id` submit, create one branch row after the turn
lock is acquired and before appending the input event.

Branch fields:

- `session_id`: current session.
- `owner_user_id`: authenticated owner.
- `parent_event_id`: the current choice-presented event when it can be identified
  from the active replay window. The lookup must be bounded by the persisted scene
  state's `last_event_id` and must not cross the latest `session_restored` event.
  If restored snapshot choices do not map to a `choice_presented` event in that
  bounded window, use the scene state's `last_event_id`.
- `branch_label`: selected choice text, normalized to a bounded length such as
  160 characters.
- `branch_path_json`: a list of compact path-step objects. Keep the top-level
  value list-shaped so it remains compatible with the current
  `VNPlayBranchResponse.branch_path: list[Any]` API contract:

```json
[
  {
    "schema_version": 1,
    "type": "choice",
    "choice_id": "open-the-door",
    "choice_text": "Open the door",
    "choice_presented_event_id": 10,
    "scene_version": 1
  }
]
```

- `status`: `active`.

Do not store a top-level object in `branch_path_json` unless the API schema and
all branch consumers are changed in the same implementation. V1 should avoid that
schema churn and append a path-step object to the existing list.

The branch row and `choice_selected` event must be committed atomically after the
turn lock is acquired. A process crash or DB error must not leave a branch row
without its matching accepted input event.

The implementation should add a repository helper such as
`record_story_choice_selection(...)` instead of composing the existing
`create_branch()`, `append_event()`, and `update_turn_request()` helpers. Those
helpers each open their own transactions today, so composing them in the service
would not satisfy the atomicity requirement.

## Event Contract

For Story choice turns, append `choice_selected` as the accepted input event.

Payload:

```json
{
  "schema_version": 1,
  "turn_request_id": 5,
  "choice_id": "open-the-door",
  "choice": {
    "id": "open-the-door",
    "text": "Open the door"
  },
  "branch_node_id": 12,
  "scene_version": 1
}
```

Event metadata:

- `event_type`: `choice_selected`
- `source`: `user`
- `branch_node_id`: created branch ID

`derive_scene_state()` already clears `visible_choices` and applies
`active_branch_node_id` when it replays this event. The implementation should
preserve that behavior and add tests around it rather than creating a second
state-mutating path.

The accepted-choice repository helper should persist replay-derived state
immediately, with:

- `last_event_id`: the `choice_selected` event ID.
- `scene_version`: the submitted `client_scene_version`.
- `active_branch_node_id`: the created branch ID.
- `visible_choices`: empty.

## Idempotency And Concurrency

Idempotency remains scoped to `(owner_user_id, session_id, idempotency_key)`.

Rules:

- Duplicate completed requests with the same normalized payload hash replay the
  stored response.
- Duplicate in-flight or failed requests with the same hash return the stored
  turn status and do not append events, create branches, or call the model.
- Same key with a different payload hash returns `409 idempotency_key_conflict`.
- Branch creation only happens on the first accepted request after the turn lock
  is acquired.
- If lock acquisition fails after turn request creation, mark the request
  `abandoned` as today and do not create a branch.
- Duplicate requests that arrive after accepted choice persistence but before
  model completion return the in-flight turn status and observe the already
  persisted scene state. They do not revalidate the now-cleared choices.

The normalized payload hash must include `choice_id` but not the derived
`choice` text or branch ID. Replayed submissions should not fail because a choice
label was later changed by an implementation detail.

## Model Context

After appending `choice_selected`, rebuild or derive scene state before building
the model context. The Story adapter should receive:

- `input_payload.choice_id`.
- selected choice text from the validated visible choice payload.
- `active_branch_node_id`.
- recent events including `choice_selected`.
- previous branch path context from existing events or branch rows.

The model call must happen only after the database transaction has created the
branch, appended `choice_selected`, updated the turn request, persisted the
accepted-choice scene state, and committed.

## Failure And Retry Behavior

If the model call fails or parsing fails after the branch row and
`choice_selected` event are committed:

- Keep the branch row.
- Keep the `choice_selected` event.
- Append `turn_failed` or `model_turn_parse_failed`.
- Persist derived scene state with `active_branch_node_id` set to the branch and
  `visible_choices` cleared.
- Clear `active_turn_request_id`.
- Leave `scene_version` at the accepted base version unless a later successful
  turn advances it.

`POST /sessions/{session_id}/retry-last-turn` is failure retry only in this
sprint. It retries the latest accepted input only when the associated turn request
ended in `model_failed` or `parse_failed`, or when a recovery pass marks an
expired in-flight request `abandoned` with a non-null `input_event_id`. Completed
Story turns are not rerolled through this endpoint; a future explicit reroll
endpoint can define that behavior separately.

Retry must not resubmit a selected choice through normal visible-choice
validation because the original `choice_selected` already cleared
`visible_choices`.

Retry design:

- Find the latest failed turn request, then use its `input_event_id` as the source
  of truth.
- The source input event may be `choice_selected` for Story choices or `user_turn`
  for Freeform and non-branch custom actions.
- Create a new turn request with a new idempotency key and a retry metadata field
  in event payloads such as `retry_source_event_id` and
  `retry_of_turn_request_id`.
- Set the new retry turn request's `input_event_id` to the original accepted input
  event ID. Do not append a second input event for Story choice retries.
- Reuse the existing `choice_selected.branch_node_id` for Story choices.
- Do not append another `choice_selected`.
- Do not create another branch row.
- Rebuild model context from the original input event and current replayed state.

If the latest accepted input already completed, return a stable non-retryable
error such as `retry_last_turn_not_failed`. This keeps the branch model clear:
retry means recovery from a failed generation, not a new branch reroll.

## Checkpoints And Restore

Checkpoints already snapshot `vn_play_scene_state`. Because
`choice_selected` updates `active_branch_node_id` through replay, checkpoint
creation after an accepted Story choice should include that branch ID.

Restore behavior:

- Restoring a checkpoint restores `active_branch_node_id` from the snapshot.
- Later events are not deleted.
- Branch rows remain listable.
- A restored scene with old `visible_choices` uses the snapshot choices as the
  current valid choice set until a new turn changes them.

## API Documentation Updates

Update `Docs/API-related/VN_PLAY_API.md` with:

- Story choice validation behavior.
- `choice_selected` event shape.
- Branch row creation semantics.
- New error codes: `choice_not_allowed`, `invalid_choice_id`.
- New retry error code: `retry_last_turn_not_failed`.
- Retry behavior for failed Story choices.
- Note that custom frontends can rely on the backend branch list and current
  scene `active_branch_node_id`.

## Test Plan

Backend tests:

- Valid Story choice creates one branch row and one `choice_selected` event in a
  repository helper that commits them atomically.
- `choice_selected` sets persisted `active_branch_node_id` and clears stale
  `visible_choices` before the adapter/model call starts.
- Branch paths remain top-level lists and `GET /branches` validates against
  `VNPlayBranchResponse`.
- Duplicate idempotency replay does not create a second branch or input event.
- Duplicate in-flight replay after accepted choice persistence observes the
  existing branch state and does not revalidate cleared choices.
- Unknown Story `choice_id` returns `invalid_choice_id` before model work.
- Freeform `choice_id` returns `choice_not_allowed` before model work.
- Story `custom_action` follows the non-branch `user_turn` behavior and does not
  create a branch row.
- Model failure after a valid choice keeps the branch row, keeps
  `choice_selected`, persists branch state, clears active turn, and records a
  failure event.
- `retry-last-turn` after a failed Story choice reuses the original branch and
  does not append a duplicate `choice_selected`.
- `retry-last-turn` after a completed Story choice returns
  `retry_last_turn_not_failed`.
- Checkpoint creation and restore preserve `active_branch_node_id`.
- Restored or repeated choice IDs resolve `parent_event_id` only inside the active
  replay window and do not attach to stale pre-restore choices.
- Branch listing returns the created branch metadata for the owner and excludes
  other users' rows.

API/docs tests:

- Turn responses expose the updated scene state after successful Story choices.
- Existing Freeform turn tests still pass without branch rows.
- OpenAPI/schema tests include the new stable error codes if the constants are
  surfaced in schema documentation.

## Rollout

Implement behind normal VN Play runtime behavior, not a feature flag. The change
only affects Story sessions that submit `choice_id`.

Compatibility notes:

- Existing sessions without branch rows remain readable.
- Existing `choice_presented` events remain valid.
- Existing frontends that submit `choice_id` get stronger backend behavior
  without changing request shape.
- Custom frontends can ignore branch listing and still rely on
  `current_scene.visible_choices` and `active_branch_node_id`.

## Risks And Mitigations

- Duplicate branches from browser retries: handled by existing idempotency key
  storage, creating branches only after first lock acquisition, and returning
  in-flight turn state without revalidating cleared choices.
- Invalid choices after stale clients: handled by `client_scene_version` and
  server-side `visible_choices` validation.
- Retry after failed choice losing context: handled by using the failed turn
  request's `input_event_id` as the source of truth and reusing the original
  `choice_selected` event and branch ID.
- Branch parent ambiguity: mitigated by matching `choice_presented` only inside
  the active replay window bounded by scene `last_event_id` and latest restore,
  then falling back to the current scene state's `last_event_id`.
- Schema drift in `branch_path_json`: mitigate by keeping the top-level value a
  list of versioned path-step objects until a deliberate API migration changes
  `VNPlayBranchResponse`.
