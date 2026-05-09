# VN Play Branch Navigation API Design

Date: 2026-05-09

Issue: https://github.com/rmusser01/tldw_server/issues/1463

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1391

## Summary

Expose a backend-owned VN Play branch navigation contract for Story/CYOA sessions. The API should let custom frontends render branch menus, active paths, branch event ranges, and safe restore controls without reverse-engineering raw events or reading VN asset-pack internals.

This sprint should add a derived branch navigation read model plus guarded branch restore. The read model is computed from existing branch rows, ordered events, and persisted scene state. Restore uses the existing event replay model: the server validates the request, appends an auditable restore event, derives scene state through replay, persists the resulting state, and returns a fresh scene/navigation payload.

## Current State

Merged VN Play work already provides:

- sessions with per-session `scene_version` and `active_turn_request_id`
- ordered events in `vn_play_events`
- scene state in `vn_play_scene_state`
- checkpoints in `vn_play_checkpoints`
- branches in `vn_play_branches`
- idempotent turn requests in `vn_play_turn_requests`
- Story choices persisted as durable branch rows plus `choice_selected` events
- current branch state via `scene_state.active_branch_node_id`
- a flat `GET /api/v1/vn-play/sessions/{session_id}/branches` endpoint

The gap is that clients still need to infer branch navigation behavior from flat branch rows plus raw events. That couples frontend behavior to database/event internals and makes branch switching risky.

## Goals

- Provide a server-shaped branch navigation payload for one VN Play session.
- Identify the active branch and active branch path.
- Provide stable labels and choice metadata for user-facing branch lists.
- Provide event cursor/range metadata so clients can page branch-related history without fetching and interpreting every event.
- Support safe branch restore/resume with stale-scene and active-turn guards.
- Make restore idempotency real for branch restore and harden the existing checkpoint restore path, which already accepts `idempotency_key`.
- Keep existing Story turn idempotency, retry-last-turn, checkpoint restore, and Freeform behavior compatible.

## Non-Goals

- Full story graph authoring.
- Branch merge/rebase.
- Visual timeline UI implementation.
- Realtime image generation.
- Client-side branch reconstruction as the source of truth.
- Destructive deletion of branch history.

## Design Decisions

### 1. Add A Derived Navigation Endpoint

Add:

`GET /api/v1/vn-play/sessions/{session_id}/branch-navigation`

The endpoint returns a stable read model derived from existing state:

```json
{
  "session_id": 1,
  "mode": "story",
  "scene_version": 6,
  "last_event_id": 41,
  "active_branch_node_id": 12,
  "active_path": [
    {
      "branch_id": 8,
      "branch_label": "Open the archive door",
      "choice_id": "open-door",
      "choice_text": "Open the archive door",
      "depth": 1
    },
    {
      "branch_id": 12,
      "branch_label": "Step inside",
      "choice_id": "step-inside",
      "choice_text": "Step inside",
      "depth": 2
    }
  ],
  "branches": [
    {
      "branch_id": 12,
      "parent_branch_id": 8,
      "parent_event_id": 32,
      "choice_selected_event_id": 33,
      "branch_label": "Step inside",
      "choice_id": "step-inside",
      "choice_text": "Step inside",
      "branch_path": [
        {
          "type": "choice",
          "choice_id": "open-door",
          "choice_text": "Open the archive door",
          "choice_presented_event_id": 20,
          "scene_version": 2
        },
        {
          "type": "choice",
          "choice_id": "step-inside",
          "choice_text": "Step inside",
          "choice_presented_event_id": 32,
          "scene_version": 5
        }
      ],
      "depth": 2,
      "status": "active",
      "is_active": true,
      "is_on_active_path": true,
      "event_range": {
        "start_event_id": 33,
        "start_sequence_number": 18,
        "latest_event_id": 41,
        "latest_sequence_number": 26
      },
      "subtree_event_range": {
        "start_event_id": 33,
        "start_sequence_number": 18,
        "latest_event_id": 41,
        "latest_sequence_number": 26
      },
      "restore": {
        "supported": true,
        "default_target": "branch_latest",
        "targets": ["branch_latest", "choice_point"]
      },
      "warnings": []
    }
  ],
  "warnings": []
}
```

The flat `GET /branches` endpoint remains for compatibility. New clients should prefer `branch-navigation`.

### 2. Derive, Do Not Persist, The Navigation Read Model

Do not add a branch navigation table in V1.

The service should build the payload from:

- `repo.list_branches(session_id)`
- `repo.list_events(session_id)`
- `repo.get_scene_state(session_id)`
- current session metadata

This keeps the read model reversible and avoids synchronization drift between branch rows, events, and scene state.

The implementation may add small repository helper queries for efficiency, but the source of truth remains branches/events/scene state.

### 3. Infer Parent Branches From Branch Paths

`vn_play_branches` has `parent_event_id` and `branch_path_json`, but not `parent_branch_id`. V1 should derive `parent_branch_id` by matching each branch path prefix to another branch row:

- A depth-1 branch has no parent branch.
- A depth-N branch's parent is the branch whose normalized `branch_path` equals the first `N - 1` path steps.
- If no parent can be resolved, return `parent_branch_id: null` and add a warning code such as `parent_branch_unresolved`.

The implementation plan should include normalization rules for path comparison:

- compare `choice_id`
- compare `choice_presented_event_id` when present
- compare `scene_version`
- do not compare user-facing `choice_text` alone

### 4. Expose Branch Event Ranges

Each branch node should expose two range concepts:

- `event_range`: events owned by this branch node while it is the active leaf.
- `subtree_event_range`: events owned by this branch and any descendant branch nodes.

This distinction is required because an ancestor branch can be on the active path while a child branch owns the current turn. Clients that render a flat branch list generally want `event_range`; clients that render collapsible history groups may want `subtree_event_range`.

For `event_range`:

- `start_event_id` / `start_sequence_number`: the `choice_selected` event for that branch.
- `latest_event_id` / `latest_sequence_number`: the newest event in the branch's latest direct active-leaf interval.

For `subtree_event_range`, `latest_event_id` is the newest event associated with the branch or any branch whose parent chain includes it.

For new events after this sprint, `_complete_turn()` and visual directive event creation should attach the current active `branch_node_id` to model, visual directive, choice-presented, scene-state, and turn-completed events whenever a Story branch is active.

For existing sessions where only `choice_selected` has `branch_node_id`, the read model should fall back to replay-derived active branch intervals:

- iterate events in sequence
- maintain replayed `active_branch_node_id`
- assign events to that active branch until the next branch choice or restore changes it
- record a warning if the range is ambiguous

`branch_latest` restore uses the branch's latest direct `event_range`, not the subtree range. Restoring a parent branch should resume the parent branch immediately before or at its latest direct active-leaf state, not jump into a descendant choice.

### 5. Add Optional Branch-Aware Event Filtering

Extend the existing event endpoint in a backward-compatible way:

`GET /api/v1/vn-play/sessions/{session_id}/events?branch_id=12&after_sequence=18&limit=100&include_descendants=false`

Rules:

- omitted query parameters preserve current behavior
- `limit` is bounded, default `100`, maximum `250`
- `branch_id` must belong to the session owner
- `include_descendants=false` returns direct branch-owned events
- `include_descendants=true` returns branch-owned events plus descendant branch events
- branch filtering uses explicit `branch_node_id` when present and bounded replay-derived active branch intervals as fallback
- invalid branch ids return `404 not_found`

Fallback replay must be bounded. Define a server constant such as `VN_PLAY_BRANCH_NAV_MAX_REPLAY_EVENTS = 5000`. If a session exceeds that cap and explicit `branch_node_id` tags are insufficient, the endpoint should return explicit-tag matches plus a warning such as `branch_interval_replay_limit_exceeded`; it must not silently perform an unbounded full replay on every request.

This gives custom frontends a paginated history path without requiring them to fetch every session event.

### 6. Add Guarded Branch Restore

Add:

`POST /api/v1/vn-play/sessions/{session_id}/branches/{branch_id}/restore`

Request:

```json
{
  "client_scene_version": 6,
  "idempotency_key": "session-1-restore-branch-12",
  "target": "branch_latest"
}
```

Targets:

- `branch_latest`: resume the latest server-resolved state on that branch.
- `choice_point`: rewind to the choice-presentation state that produced the branch, so the user can choose that branch again or pick a sibling branch.

Default target: `branch_latest`.

Response:

```json
{
  "status": "completed",
  "replayed": false,
  "restore_event_id": 42,
  "branch_id": 12,
  "target": "branch_latest",
  "target_event_id": 41,
  "scene_version": 7,
  "session": {},
  "current_scene": {},
  "branch_navigation": {}
}
```

The response should include the same session/current scene shape clients already use, plus the updated branch navigation payload.

### 7. Restore Must Be Serialized And Idempotent

Branch restore mutates scene state, so it must use the same safety posture as turns:

- require `client_scene_version == session.scene_version`
- reject when `session.active_turn_request_id` is not null
- reject when another non-expired `active_session_action_id` is set
- reject missing or foreign branch ids
- reject non-Story branch restore attempts with `branch_restore_not_allowed`
- reject ambiguous restore targets with stable error codes
- use an idempotency key scoped to owner + session across all restore action types

The current checkpoint restore request already accepts `idempotency_key`, but the service does not yet persist restore request keys. This sprint should add a small action-request table rather than duplicating JSON-event scans.

Add a nullable `active_session_action_id` column to `vn_play_sessions`. Restore actions set it while they are pending. Turn acquisition must also check it, so turns and restores share one session mutation gate:

- `try_acquire_turn_lock` must require `active_session_action_id IS NULL`.
- restore action acquisition must require `active_turn_request_id IS NULL`.
- both paths must update the session row with a scene-version compare-and-swap.
- both paths clear their active marker in the same transaction that commits completion, failure, or abandonment.

Recommended table:

```sql
CREATE TABLE IF NOT EXISTS vn_play_session_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    response_payload_json TEXT,
    error_json TEXT,
    lease_owner TEXT,
    locked_until DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, idempotency_key)
);
```

Use it for both:

- `branch_restore`
- `checkpoint_restore`

The idempotency key is session-global for restore actions. `request_payload_hash` must include `action_type`, target id, target mode, and `client_scene_version`. Reusing the same key for a different action type or payload returns `idempotency_key_conflict`; reusing it for the same payload replays the stored response.

This hardens the existing checkpoint restore API without changing its public request shape.

### 8. Action Reservation Is Transactional

Restore actions must be reserved before appending restore events. The implementation should provide one repository helper that:

1. Looks up or creates the `vn_play_session_actions` row by idempotency key.
2. Verifies the request payload hash for duplicate keys.
3. Rejects if another non-expired pending restore action exists for the same session.
4. Verifies `session.scene_version == client_scene_version`.
5. Verifies `session.active_turn_request_id IS NULL`.
6. Verifies `session.active_session_action_id` is either null or the same replayed action.
7. Sets `active_session_action_id` with a scene-version compare-and-swap, so a concurrent turn or restore cannot commit against the same visible state.
8. Appends the restore event, persists scene state, updates session `scene_version`, clears `active_session_action_id`, and stores the response payload before committing.

If a process dies after reserving an action but before completion, a later retry may take over only after `locked_until` expires. A retry with the same idempotency key may mark the action abandoned only when no restore event was committed. If the restore event exists, the retry must reconstruct and store the completed response rather than appending another event.

### 9. Restore Uses `session_restored` Events

Do not add a new event type for branch restore unless implementation proves it is necessary. Use the existing `session_restored` event with a richer payload:

```json
{
  "restore_kind": "branch",
  "branch_id": 12,
  "target": "branch_latest",
  "target_event_id": 41,
  "scene_state_snapshot": {},
  "previous_scene_version": 6,
  "scene_version": 7,
  "idempotency_key": "session-1-restore-branch-12"
}
```

For checkpoint restore, use:

```json
{
  "restore_kind": "checkpoint",
  "checkpoint_id": 3,
  "scene_state_snapshot": {},
  "previous_scene_version": 6,
  "scene_version": 7,
  "idempotency_key": "session-1-restore-checkpoint-3"
}
```

The important semantic change is that restore is a new scene mutation and should advance the session `scene_version` by one. The snapshot may contain older scene data, but the restore event payload's `scene_version` becomes the new current scene version.

### 10. Restore Snapshot Derivation

For `branch_latest`, derive a scene snapshot by replaying events up to the branch's direct `event_range.latest_event_id`.

For `choice_point`, derive a scene snapshot by replaying events through the branch's `parent_event_id`, which should be the `choice_presented` event that made the chosen branch visible. If `parent_event_id` is missing or no longer points to a choice-presented state, reject with `branch_restore_target_unavailable`.

`choice_point` snapshot semantics:

- visible choices come from the parent `choice_presented` event
- `active_branch_node_id` is the parent branch active at that choice point, or `null` for a root choice
- the selected branch is not active after restore
- sibling branches remain durable history but are not selected
- the next submitted `choice_id` creates or reuses normal Story branch persistence through the existing turn path

Then append `session_restored` with the snapshot and new scene version, replay the full event stream, persist scene state, and update session `scene_version`.

This keeps checkpoint restore and branch restore on the same replay path.

### 11. Error Codes

Add stable error details:

- `branch_navigation_unavailable`
- `branch_not_found`
- `branch_restore_not_allowed`
- `branch_restore_target_unavailable`
- `branch_restore_ambiguous`
- `restore_action_in_progress`
- `branch_interval_replay_limit_exceeded`
- `idempotency_key_conflict`
- `stale_scene_version`
- `turn_in_progress`

HTTP mapping:

- `404`: missing session or branch
- `409`: stale scene version, turn in progress, idempotency conflict
- `400`: restore not allowed, unsupported target, ambiguous target

### 12. Existing Behavior Compatibility

Story choice turns:

- keep current `choice_id` validation
- keep one durable branch row per idempotent accepted choice
- keep retry-last-turn failure-only and branch-preserving

Freeform turns:

- unchanged
- branch navigation returns an empty branch list with `mode: "freeform"` and no restore capabilities

Checkpoint restore:

- public endpoint remains
- request body remains compatible
- idempotency key becomes enforced
- restore scene version should advance rather than silently reverting the visible version

Existing `GET /branches`:

- unchanged response shape
- no branch navigation-only fields added to that flat schema

## Warning Payload Shape

Branch navigation warnings must use a stable, non-exception schema:

```json
{
  "code": "parent_branch_unresolved",
  "severity": "warning",
  "message": "Parent branch could not be resolved from branch_path.",
  "branch_id": 12,
  "event_id": 33,
  "recoverable": true
}
```

Fields:

- `code`: stable machine-readable code.
- `severity`: `info`, `warning`, or `high_risk`.
- `message`: optional frontend-safe text.
- `branch_id`: optional affected branch.
- `event_id`: optional affected event.
- `recoverable`: whether retrying after more events/tags are available could resolve it.

Do not expose raw exception strings or stack details in warning payloads.

## API Surface

New schemas:

- `VNPlayBranchNavigationResponse`
- `VNPlayBranchNavigationNode`
- `VNPlayBranchPathStep`
- `VNPlayBranchEventRange`
- `VNPlayBranchRestoreRequest`
- `VNPlayBranchRestoreResponse`
- `VNPlayBranchRestoreTarget`

New endpoint:

- `GET /sessions/{session_id}/branch-navigation`
- `POST /sessions/{session_id}/branches/{branch_id}/restore`

Extended endpoint:

- `GET /sessions/{session_id}/events`
  - add optional `branch_id`
  - add optional `after_sequence`
  - add bounded `limit`
  - add optional `include_descendants`

## Implementation Notes For The Future Plan

Likely file map:

- `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
  - pure navigation builder and range derivation helpers
- `tldw_Server_API/app/core/VN_Play/service.py`
  - service orchestration and branch restore flow
- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
  - session action table and helper methods
  - `active_session_action_id` schema migration/update support
  - optional branch-aware event query helpers
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
  - new request/response schemas
- `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - new endpoints and query parameters
- `tldw_Server_API/app/core/VN_Play/state.py`
  - no major change expected; replay already honors `session_restored`
- `Docs/API-related/VN_PLAY_API.md`
  - document branch navigation and restore contracts

## Test Plan

Repository tests:

- creates `vn_play_session_actions` idempotency rows
- duplicate action key with same payload replays stored response
- duplicate action key with different payload raises conflict
- branch-aware event query filters by explicit branch id
- branch-aware event query falls back to replay-derived active branch intervals
- branch-aware event query respects max replay cap and emits a stable warning rather than scanning unbounded history
- session-global restore idempotency conflicts when the same key is reused for a different action type

Pure branch navigation tests:

- derives active branch path from branch rows and current scene state
- resolves parent branch ids from normalized branch path prefixes
- emits warnings for unresolved parents or ambiguous event ranges
- computes event ranges for explicit `branch_node_id` rows
- computes event ranges for older rows using replay fallback
- distinguishes direct branch-owned ranges from subtree ranges
- validates warning payload shape

Service tests:

- branch navigation response contains active path and branch restore capabilities
- Freeform sessions return empty navigation and no restore capability
- branch restore rejects stale scene versions
- branch restore rejects active in-flight turns
- branch restore rejects when another restore action is active
- turn acquisition rejects when a restore action is active
- branch restore rejects foreign/missing branches
- branch restore appends `session_restored`, advances scene version, and persists scene state
- `choice_point` restore returns visible choices from the parent choice-presented event and restores the parent active branch rather than the selected branch
- branch restore idempotency replay does not append duplicate restore events
- branch restore idempotency conflict rejects different payloads
- checkpoint restore idempotency is enforced without changing request shape
- Story choice retry and completed-turn idempotency remain unchanged

API tests:

- `GET /branch-navigation` response shape for Story session with two branch choices
- `GET /events?branch_id=...&limit=...&include_descendants=...` returns bounded filtered events
- `POST /branches/{branch_id}/restore` returns session/current scene/navigation payload
- expected HTTP mappings for `404`, `409`, and `400`

Verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py -f json -o /tmp/bandit_vn_play_branch_navigation.json
git diff --check
```

## Risks And Mitigations

- Risk: restore idempotency is currently accepted but not enforced for checkpoint restore.
  - Mitigation: add `vn_play_session_actions` and use it for checkpoint and branch restore.

- Risk: branch event ranges may be ambiguous for existing sessions that only tagged `choice_selected`.
  - Mitigation: use replay-derived active branch intervals and return warning metadata when confidence is low.

- Risk: branch-aware event filtering could become an unbounded replay scan.
  - Mitigation: define a replay cap and return explicit-tag results plus warning metadata when fallback replay exceeds it.

- Risk: returning tree-shaped data could overfit one frontend.
  - Mitigation: return normalized nodes plus active path and event ranges, not a presentation-specific tree layout.

- Risk: branch restore could race with a model turn.
  - Mitigation: add a shared session mutation gate with `active_turn_request_id`, `active_session_action_id`, and exact `client_scene_version`.

- Risk: restore could make scene versions move backward.
  - Mitigation: restore appends a new event and advances scene version by one.

## Open Questions

- Should branch status support archived/hidden in this sprint? Recommendation: no new status behavior; keep `status` passthrough only.
