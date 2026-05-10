# VN Play API

VN Play provides durable visual-novel runtime sessions backed by approved VN asset packs. V1 supports Freeform and Story/CYOA modes, ordered event history, server-authoritative scene state, turn idempotency, checkpoints, branch navigation, and guarded branch restore.

Base path: `/api/v1/vn-play`

## Authentication And Ownership

All endpoints require the normal tldw_server API authentication:

- Single-user mode: `X-API-KEY: <key>`
- Multi-user mode: `Authorization: Bearer <jwt>`

Session metadata, events, turn requests, scene state, checkpoints, and branches are stored in the authenticated user's `ChaChaNotes.db`. VN Play references an existing VN asset pack by `vn_asset_pack_id`; it does not generate image files in realtime.

Linked chat is read-only in V1. `linked_chat_id` may be stored as session context, but VN Play turns do not write messages back to the linked chat and do not continuously ingest new chat messages after the runtime turn request starts.

## Endpoint Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/setup-options` | Return backend-computed character and asset pack setup selectors. |
| `POST` | `/sessions` | Create a Freeform or Story session. |
| `GET` | `/sessions` | List the current user's non-deleted sessions. |
| `GET` | `/sessions/{session_id}` | Get one session with current scene state. |
| `PATCH` | `/sessions/{session_id}` | Update session title, status, linked chat, settings, or soft-delete flag. |
| `DELETE` | `/sessions/{session_id}` | Soft-delete one session. |
| `POST` | `/sessions/{session_id}/turn` | Submit one user turn. |
| `POST` | `/sessions/{session_id}/retry-last-turn` | Retry the latest failed accepted turn with a new idempotency key. |
| `GET` | `/sessions/{session_id}/events` | List ordered event history, optionally filtered by branch. |
| `POST` | `/sessions/{session_id}/checkpoint` | Create a named checkpoint at the current scene state. |
| `GET` | `/sessions/{session_id}/checkpoints` | List checkpoints. |
| `POST` | `/sessions/{session_id}/restore` | Restore a checkpoint. |
| `GET` | `/sessions/{session_id}/branches` | List branch metadata. |
| `GET` | `/sessions/{session_id}/branch-navigation` | Get the backend-derived branch navigation read model. |
| `POST` | `/sessions/{session_id}/branches/{branch_id}/restore` | Restore a Story session to a branch target. |

## Setup Options

`GET /setup-options` returns a bounded, selector-safe setup contract for WebUI and custom frontend session creation. The backend owns character option shaping, VN asset pack readiness fanout, compatibility classification, content-rating warnings, trust provenance, warning severity, and empty-state hints.

Example:

```bash
curl "http://127.0.0.1:8000/api/v1/vn-play/setup-options?mode=story&selected_character_id=42&content_rating=general" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

Supported query parameters:

- `mode`: optional `freeform` or `story`.
- `character_query` / `pack_query`: optional selector search text.
- `character_limit` / `pack_limit`: bounded page sizes, default `25`, maximum `100`.
- `character_offset` / `pack_offset`: zero-based selector offsets.
- `selected_character_id`: optional active character; included as `selected_character` even when outside the current character page.
- `content_rating`: intended session content rating, default `general`.

The response includes:

- `characters`: selector-safe character rows with `id`, `name`, `description_preview`, `tags`, `favorite`, `deleted`, and `has_image`; full prompts and image bytes are not embedded.
- `asset_packs`: bounded pack rows with readiness, compatibility, trust, warning summary, and `recommended` metadata.
- `pagination`: separate character and pack pagination metadata. Pack readiness is computed only for returned pack rows.
- `empty_states`: scoped hints such as `no_characters`, `no_asset_packs`, `no_ready_packs`, `no_compatible_packs`, or `selected_character_not_found`.
- `defaults`: optional unambiguous defaults for frontend convenience.

High-risk warning summaries require frontend acknowledgement before submit, but V1 keeps `POST /sessions` compatible and does not enforce setup acknowledgement server-side. Clients that accept high-risk warnings should persist acknowledgement metadata in `settings.setup_acknowledgements`.

## Create A Session

Freeform session:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "freeform",
    "title": "Orbital Library",
    "primary_character_id": 42,
    "vn_asset_pack_id": 7,
    "content_rating": "general",
    "linked_chat_id": null,
    "linked_chat_mode": "read_only_context",
    "settings": {}
  }'
```

Story/CYOA session:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "story",
    "title": "Door Under The Archive",
    "primary_character_id": 42,
    "additional_character_ids": [43],
    "vn_asset_pack_id": 7,
    "source_world_book_ids": [9],
    "content_rating": "general",
    "seed": "story-seed-1"
  }'
```

Minimal response:

```json
{
  "id": 1,
  "owner_user_id": 1,
  "mode": "story",
  "title": "Door Under The Archive",
  "status": "active",
  "primary_character_id": 42,
  "additional_character_ids": [43],
  "linked_chat_id": null,
  "vn_asset_pack_id": 7,
  "source_world_book_ids": [9],
  "content_rating": "general",
  "trust_level": "local",
  "linked_chat_mode": "read_only_context",
  "scene_version": 0,
  "active_turn_request_id": null,
  "current_scene": {
    "session_id": 1,
    "owner_user_id": 1,
    "scene_version": 0,
    "visible_choices": [],
    "warnings": []
  },
  "scene_state": {
    "session_id": 1,
    "owner_user_id": 1,
    "scene_version": 0,
    "visible_choices": [],
    "warnings": []
  },
  "deleted": false
}
```

## Turn Requests

`POST /sessions/{session_id}/turn` accepts exactly one of:

- `input_text` for Freeform text.
- `choice_id` for a Story/CYOA choice.
- `custom_action` for structured client actions.

Every turn request must include:

- `client_scene_version`: the scene version the client rendered before submitting.
- `idempotency_key`: a unique key scoped to the session and user.

The runtime stores turn request keys before model work starts. If the same key and same request payload are submitted again, the stored turn status or completed response is replayed. If the same key is reused with a different payload, the API returns `409 idempotency_key_conflict`.

Model `visual_directives` are resolved by the backend against the session's approved VN asset-pack manifest. Resolved assets update scene state and unresolved directives remain warning-only: the text/model turn can still complete, while the runtime records a rejection event with a stable reason such as `asset_not_found` or `manifest_unavailable`.

Story mode is backend-authoritative. A `choice_id` is accepted only when it matches the current persisted `scene_state.visible_choices` for the submitted `client_scene_version`. Accepted Story choices atomically create branch metadata, append `turn_started` and `choice_selected`, set the scene state's `active_branch_node_id`, clear `visible_choices`, and only then call the turn adapter/model. This lets custom frontends submit simple choice IDs without owning branch persistence. Story `custom_action` is non-branching and is stored as a normal `user_turn`.

Freeform turn:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/turn" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Look around the archive.",
    "client_scene_version": 0,
    "idempotency_key": "session-1-turn-1"
  }'
```

Story choice:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/turn" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "choice_id": "open-the-door",
    "client_scene_version": 1,
    "idempotency_key": "session-1-choice-2"
  }'
```

Minimal response:

```json
{
  "turn_request_id": 5,
  "status": "completed",
  "scene_version": 2,
  "replayed": false,
  "session": {
    "id": 1,
    "mode": "story",
    "title": "Door Under The Archive",
    "scene_version": 2
  },
  "current_scene": {
    "session_id": 1,
    "scene_version": 2,
    "location_key": "archive",
    "active_branch_node_id": 12,
    "visible_choices": [
      { "id": "step-inside", "text": "Step inside" }
    ],
    "warnings": []
  },
  "events": [
    {
      "id": 10,
      "session_id": 1,
      "sequence_number": 4,
      "event_type": "model_turn",
      "event_payload": {
        "dialogue": [
          { "speaker": "Mira", "text": "The door opens onto the archive." }
        ],
        "scene_version": 2
      },
      "source": "model"
    }
  ],
  "warnings": []
}
```

## Concurrency And Conflict Responses

The session scene version is server-authoritative. Clients must submit the version they rendered.

- `409 stale_scene_version`: the session moved forward after the client rendered. Reload the session and events, then let the user resubmit.
- `409 turn_in_progress`: another accepted turn is still active for this session. Poll or reload the session until `active_turn_request_id` clears.
- `409 idempotency_key_conflict`: the idempotency key was already used with a different payload. Generate a new key only if this is a genuinely new user action.

Model or parse failures are recorded as turn/event state and returned as `502 model_failed` or `502 parse_failed` when the accepted turn cannot complete.

Validation errors returned as HTTP 400 include:

- `choice_not_allowed`: Freeform received `choice_id`, or Story received freeform `input_text`.
- `invalid_choice_id`: Story `choice_id` is not in the persisted current `visible_choices`.
- `retry_last_turn_not_failed`: no latest failed accepted turn is available to retry.

## Events And Scene State

`GET /sessions/{session_id}/events` returns append-only ordered events. Scene state is derived from these events and persisted for fast reads. Omitting branch filter query parameters preserves the legacy unbounded event list behavior.

Optional branch filter query parameters:

- `branch_id`: filter events to one Story branch.
- `after_sequence`: return only events after a sequence number.
- `limit`: bounded page size, maximum `250`. For branch-filtered requests the default is `100`; for unfiltered requests omission means unbounded.
- `include_descendants`: include descendant branch events when `true`.

Example:

```bash
curl "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/events?branch_id=12&include_descendants=true&limit=100" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

The response body remains the existing event list for compatibility. If branch filtering needs to report replay-cap or ambiguity warnings, the endpoint sets `X-VN-Play-Warnings` to a compact JSON list using the stable warning shape documented below.

Important event types include:

- `session_started`
- `turn_started`
- `user_turn`
- `model_turn`
- `choice_presented`
- `choice_selected`
- `scene_state_changed`
- `turn_completed`
- `turn_failed`
- `model_turn_parse_failed`
- `session_checkpoint_created`
- `session_restored`

The current scene state includes background/depth item ids, active sprite item payloads, location, mood, time of day, weather, visible choices, transcript cursor, scene version, and warnings.

Story choice selection appends `choice_selected` instead of `user_turn`. Its payload includes:

```json
{
  "schema_version": 1,
  "turn_request_id": 5,
  "choice_id": "open-the-door",
  "choice": { "id": "open-the-door", "text": "Open the door" },
  "branch_node_id": 12,
  "scene_version": 1
}
```

When a session has resolved visual state, scene responses also include render-ready asset payloads:

- `background`: approved background asset payload for `current_background_item_id`, including `item_id`, `slot_key`, `content_url`, labels, dimensions, and storage metadata where available.
- `depth`: approved depth companion payload for `current_depth_item_id`, when present.
- `active_sprites`: approved sprite asset payloads for the current character staging.

Visual directive event behavior:

- `visual_directive_requested`: appended for each model/runtime visual directive considered by the backend.
- `visual_directive_applied`: appended when the directive resolves to an approved manifest item; replay uses this to restore background, depth, and sprite state.
- `visual_directive_rejected`: appended when the directive cannot be resolved safely. The event payload includes `code=visual_directive_rejected`, `reason`, the original directive context, and `scene_version`.

Custom frontends should prefer `scene_state.background`, `scene_state.depth`, and `scene_state.active_sprites` from VN Play responses instead of calling VN asset-pack internals directly for runtime rendering.

## Branch Navigation

`GET /sessions/{session_id}/branch-navigation` returns a backend-derived read model for Story/CYOA navigation. Custom frontends should use this endpoint instead of reconstructing branch menus from raw events.

Example:

```bash
curl "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/branch-navigation" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

Abbreviated response:

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
        "target_names": ["branch_latest", "choice_point"],
        "targets": {
          "branch_latest": { "event_id": 41, "sequence_number": 26 },
          "choice_point": { "event_id": 32 }
        }
      },
      "warnings": []
    }
  ],
  "warnings": []
}
```

Warning payloads are frontend-safe and never contain stack traces:

```json
{
  "code": "parent_branch_unresolved",
  "severity": "warning",
  "message": "Parent branch could not be resolved from branch path prefix.",
  "branch_id": 12,
  "recoverable": true
}
```

## Branches And Retry

`GET /sessions/{session_id}/branches` remains a compatibility endpoint for durable Story branch metadata. New clients should prefer `GET /branch-navigation`. `branch_path` is always a list so clients can render path breadcrumbs without special-casing single-choice branches:

```json
[
  {
    "id": 12,
    "session_id": 1,
    "parent_event_id": 9,
    "branch_label": "Open the door",
    "branch_path": [
      {
        "schema_version": 1,
        "type": "choice",
        "choice_id": "open-the-door",
        "choice_text": "Open the door",
        "choice_presented_event_id": 9,
        "scene_version": 1
      }
    ],
    "status": "active"
  }
]
```

`POST /sessions/{session_id}/retry-last-turn` is failure-only. It retries the newest accepted turn whose turn request is still `model_failed`, `parse_failed`, or recoverable `abandoned`, and it uses that failed request's stored `input_event_id` as the source of truth. For failed Story choices, retry does not append another `choice_selected` event and does not create another branch; it reuses the original branch and calls the model again from the replayed scene state.

A retry after a completed turn returns `400 retry_last_turn_not_failed`. A retry request still requires the current `client_scene_version` and a fresh `idempotency_key`.

## Checkpoints And Restore

Create a checkpoint:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/checkpoint" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "label": "Before opening the door" }'
```

Restore a checkpoint:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/restore" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_id": 3,
    "idempotency_key": "restore-3"
  }'
```

Checkpoint restore appends a `session_restored` event, advances the visible `scene_version`, and returns the updated session. The `idempotency_key` is enforced through the session action table: a duplicate request with the same key and payload replays the stored response, while reusing the key for a different restore payload returns `409 idempotency_key_conflict`.

## Branch Restore

`POST /sessions/{session_id}/branches/{branch_id}/restore` restores a Story session to a backend-resolved branch target. It is guarded by the same session mutation gate as turns and checkpoint restore.

Targets:

- `branch_latest`: resume from the latest direct event range for the branch.
- `choice_point`: rewind to the choice-presented state that produced the branch, so the user can choose that branch again or choose a sibling.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/branches/12/restore" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "client_scene_version": 6,
    "idempotency_key": "session-1-restore-branch-12",
    "target": "choice_point"
  }'
```

Abbreviated response:

```json
{
  "status": "completed",
  "replayed": false,
  "restore_event_id": 42,
  "branch_id": 12,
  "target": "choice_point",
  "target_event_id": 32,
  "scene_version": 7,
  "session": {},
  "current_scene": {},
  "branch_navigation": {}
}
```

Restore conflict and validation responses:

- `404 branch_not_found`: the branch does not belong to the session/user.
- `409 stale_scene_version`: the client restored from an old scene version.
- `409 turn_in_progress`: a turn is currently mutating the session.
- `409 restore_action_in_progress`: another restore action is active for the session.
- `409 idempotency_key_conflict`: the key was reused for a different restore payload.
- `400 branch_restore_not_allowed`: branch restore was requested for a non-Story session.
- `400 branch_restore_target_unavailable`: the requested restore target cannot be resolved.
- `400 branch_restore_ambiguous`: the target is ambiguous and the server refused to guess.

## Character Safety Metadata

VN Play uses explicit character safety metadata rules for content-rating gates:

- Adult metadata is allowed for all ratings.
- Minor metadata is blocked for `mature`, `adult`, `explicit`, and `nsfw` ratings.
- Conflicting metadata is blocked.
- Unknown metadata is allowed for `general` sessions with a `character_safety_unknown` warning.
- Unknown metadata is blocked for mature ratings unless `settings.allow_unknown_character_safety` is `true`.
- Imported or mixed-trust unknown metadata requires `settings.allow_untrusted_character_safety` when `trust_level` is `untrusted_import` or `mixed`.

Recognized metadata fields include `safety_metadata.age_status`, `safety_metadata.status`, `age_status`, `safety_status`, `is_minor`, `minor`, `is_adult`, `adult`, `age_years`, and integer `age`.

## Frontend Workspace

The Next.js workspace is available at `/vn-play`. It can create Freeform and Story sessions, list sessions, submit Freeform turns, submit Story choices, render returned dialogue/events, and show current scene metadata.
