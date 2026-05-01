# VN Play API

VN Play provides durable visual-novel runtime sessions backed by approved VN asset packs. V1 supports Freeform and Story/CYOA modes, ordered event history, server-authoritative scene state, turn idempotency, checkpoints, and branch metadata.

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
| `POST` | `/sessions` | Create a Freeform or Story session. |
| `GET` | `/sessions` | List the current user's non-deleted sessions. |
| `GET` | `/sessions/{session_id}` | Get one session with current scene state. |
| `PATCH` | `/sessions/{session_id}` | Update session title, status, linked chat, settings, or soft-delete flag. |
| `DELETE` | `/sessions/{session_id}` | Soft-delete one session. |
| `POST` | `/sessions/{session_id}/turn` | Submit one user turn. |
| `POST` | `/sessions/{session_id}/retry-last-turn` | Retry the latest user turn input with a new idempotency key. |
| `GET` | `/sessions/{session_id}/events` | List ordered event history. |
| `POST` | `/sessions/{session_id}/checkpoint` | Create a named checkpoint at the current scene state. |
| `GET` | `/sessions/{session_id}/checkpoints` | List checkpoints. |
| `POST` | `/sessions/{session_id}/restore` | Restore a checkpoint. |
| `GET` | `/sessions/{session_id}/branches` | List branch metadata. |

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

## Events And Scene State

`GET /sessions/{session_id}/events` returns append-only ordered events. Scene state is derived from these events and persisted for fast reads. Important event types include:

- `session_started`
- `turn_started`
- `user_turn`
- `model_turn`
- `choice_presented`
- `scene_state_changed`
- `turn_completed`
- `turn_failed`
- `model_turn_parse_failed`
- `session_checkpoint_created`
- `session_restored`

The current scene state includes background/depth item ids, active sprite item payloads, location, mood, time of day, weather, visible choices, transcript cursor, scene version, and warnings.

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

Restore appends a `session_restored` event and returns the updated session. V1 restore uses the checkpoint id as the durable action; clients should still send a fresh idempotency key for request tracing and forward compatibility.

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
