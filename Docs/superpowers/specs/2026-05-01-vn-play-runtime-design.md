# VN Play Runtime Design

Status: Draft
Date: 2026-05-01
Owner: Core/WebUI maintainers
Scope: Shared visual-novel runtime for freeform character chat and structured CYOA play

## Summary

Add a dedicated VN Play runtime that consumes approved VN asset pack manifests and
turns character/persona interactions into visual-novel style sessions. The runtime
supports two separate play paths:

- Freeform VN chat: open-ended character chat rendered with VN backgrounds,
  sprites, narration, and scene transitions.
- Story/CYOA: model-generated persisted branches, choices, scene beats, and
  checkpoint/restore controls.

Both paths share the same session model, event log, asset resolver, safety gates,
and `/vn-play` workspace. V1 uses offline-approved assets only. It does not perform
realtime image generation during play.

## Goals

- Provide a new `/vn-play` WebUI workspace with Freeform and Story mode tabs.
- Store VN play as separate durable sessions, optionally linked to existing chat
  sessions.
- Keep the backend authoritative for turn execution, visual directives, branching,
  and current scene state.
- Use event sourcing as the source of truth for replay, debugging, branching, and
  future export/import.
- Consume only approved items from `vn_asset_manifest.v1`.
- Support deterministic visual selection with optional validated model-directed
  scene changes.
- Persist model-generated CYOA branches and choices in Story mode.
- Add explicit runtime gates for pack readiness, content rating, character metadata,
  provider capability, and structured output support.

## Non-Goals

- No realtime image generation in V1.
- No authored story graph editor in V1.
- No dependency on a future multi-character asset generation workflow.
- No automatic migration of existing chat sessions into VN sessions.
- No replacement of the existing `/chat` route.
- No public sharing or export/import format for VN play sessions in V1.
- No guarantee that all LLM providers can satisfy the structured VN turn contract.

## Existing Project Context

The runtime should build on these existing pieces:

- Character cards, character chat, personas, world books, and chat history live in
  per-user `ChaChaNotes.db`.
- VN asset packs already store metadata in per-user `ChaChaNotes.db` and expose
  approved-only `vn_asset_manifest.v1` responses.
- VN asset item content is served through authenticated pack/item content URLs.
- Jobs are available for user-visible background work, but VN turns are interactive
  request/response operations in V1.
- The Next.js WebUI already has separate routes for chat, characters, and VN asset
  authoring.

VN Play should stay separate from VN asset authoring. Asset packs create and review
assets. VN Play consumes runtime-ready manifests.

## Architecture

New backend package:

- `tldw_Server_API/app/core/VN_Play/`
  - session service
  - event log and replay helpers
  - scene-state derivation
  - asset resolver
  - freeform turn adapter
  - story/CYOA turn adapter
  - structured model-output parser
  - runtime gate evaluator

New DB ownership:

- Extend per-user `ChaChaNotes.db` with VN play tables.
- Add DB helpers in a new `VNPlay_DB.py` module or a clearly separated section of
  the existing character/chat DB layer if local patterns make that simpler.
- Do not store image bytes in VN play tables. Scene state references manifest item
  IDs and content URLs.

New API endpoint module:

- `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Router prefix: `/api/v1/vn-play`

New frontend workspace:

- `apps/tldw-frontend/pages/vn-play.tsx`
- `apps/tldw-frontend/components/vn-play/`
- `apps/tldw-frontend/lib/api/vnPlay.ts`
- `apps/tldw-frontend/types/vn-play.ts`

## Data Model

### `vn_play_sessions`

Stores durable session metadata.

Important fields:

- `id`
- `owner_user_id`
- `mode`: `freeform` or `story`
- `title`
- `status`: `active`, `paused`, `completed`, `archived`, `failed`
- `primary_character_id`
- `additional_character_ids_json`
- `linked_chat_id`
- `vn_asset_pack_id`
- `asset_manifest_version`
- `source_world_book_ids_json`
- `content_rating`
- `trust_level`: `local`, `trusted_restore`, `untrusted_import`, or `mixed`
- `linked_chat_mode`: V1 default `read_only_context`
- `seed`
- `settings_json`
- `scene_version`
- `active_turn_request_id`
- `created_at`
- `updated_at`
- `deleted`

V1 can use one primary rendered character by default. Additional character IDs are
reserved for text/persona context and future multi-character visual packs.

### `vn_play_events`

Append-only source of truth for session history.

Important fields:

- `id`
- `session_id`
- `owner_user_id`
- `sequence_number`
- `event_type`
- `event_payload_json`
- `source`: `user`, `model`, `runtime`, `system`
- `model_provider`
- `model_name`
- `branch_node_id`
- `created_at`

Required uniqueness:

- `(session_id, sequence_number)`

Core event types:

- `session_started`
- `turn_started`
- `turn_completed`
- `turn_failed`
- `user_turn`
- `model_turn`
- `choice_presented`
- `choice_selected`
- `scene_state_changed`
- `visual_directive_requested`
- `visual_directive_applied`
- `visual_directive_rejected`
- `model_turn_parse_failed`
- `safety_gate_triggered`
- `runtime_gate_failed`
- `session_settings_changed`
- `session_checkpoint_created`
- `session_restored`

The event payload must be JSON-schema-versioned so future event fields can be
added without breaking replay.

### `vn_play_turn_requests`

Durable idempotency and in-flight turn tracking. This table is required because
turns can be slow, clients can retry, and model calls happen outside database
transactions.

Important fields:

- `id`
- `session_id`
- `owner_user_id`
- `idempotency_key`
- `request_payload_hash`
- `base_scene_version`
- `status`: `pending`, `model_calling`, `model_failed`, `parse_failed`,
  `completed`, `abandoned`, `cancelled`
- `input_event_id`
- `turn_started_event_id`
- `turn_completed_event_id`
- `response_payload_json`
- `error_json`
- `lease_owner`
- `locked_until`
- `created_at`
- `updated_at`

Required uniqueness:

- `(owner_user_id, session_id, idempotency_key)`

Idempotency behavior:

- A duplicate request with the same key and identical `request_payload_hash`
  returns the stored response when status is `completed`.
- A duplicate request with the same key and different payload hash is rejected with
  `409 idempotency_key_conflict`.
- A duplicate request for an in-flight turn returns the current turn request status
  and does not append events or call the model again.
- A retry after `model_failed` or `parse_failed` must use `retry-last-turn` or an
  explicit retry flag so the runtime can preserve the original input event and
  append a new attempt event instead of duplicating the user turn.

### `vn_play_scene_state`

Derived cache for fast current-session reads.

Important fields:

- `session_id`
- `last_event_id`
- `current_background_item_id`
- `current_depth_item_id`
- `active_sprite_items_json`
- `location_key`
- `mood`
- `time_of_day`
- `weather`
- `active_branch_node_id`
- `visible_choices_json`
- `transcript_cursor`
- `scene_version`
- `warnings_json`
- `updated_at`

This table is rebuildable from `vn_play_events`. It is not the authoritative
record.

### `vn_play_branches`

Story/CYOA branch graph nodes.

Important fields:

- `id`
- `session_id`
- `parent_branch_node_id`
- `choice_event_id`
- `scene_title`
- `scene_summary`
- `branch_path_json`
- `depth`
- `status`: `active`, `available`, `abandoned`, `checkpointed`
- `created_at`

Model-generated choices are persisted as events and linked to branch nodes when
selected. V1 does not need a full authoring graph editor, but the schema should not
block authored graph import later.

### `vn_play_checkpoints`

Named restore points.

Important fields:

- `id`
- `session_id`
- `name`
- `event_id`
- `branch_node_id`
- `scene_state_snapshot_json`
- `created_at`

Restore appends a `session_restored` event and rebuilds current scene state from
the target checkpoint or branch node. It does not delete later events.

## Turn Flow

### Freeform VN Chat

1. Client submits user text to `POST /vn-play/sessions/{session_id}/turn`.
2. Backend opens a short transaction to validate session ownership, mode, status,
   pack readiness, character links, content gates, provider capability, token
   budget, idempotency key, scene version, and per-session turn availability.
3. Backend creates or reuses a `vn_play_turn_requests` row. New accepted turns append
   `turn_started` and `user_turn` events before the model call.
4. Backend releases the transaction before calling the model.
5. Freeform adapter builds a chat request using the selected character/persona,
   world books, recent VN events, linked chat context if configured, and VN
   structured-output instructions.
6. Model returns dialogue/narration plus optional scene directives.
7. Backend parses model output into a normalized turn result.
8. Asset resolver validates visual directives against the approved manifest.
9. Backend opens a second short transaction to append model, directive,
   scene-state, warning, and `turn_completed` events, update derived scene state,
   store the response payload, and clear the active turn.
10. Response returns the stored turn response, appended events, and current scene.

### Story/CYOA

Story turns use the same turn request, idempotency, concurrency, and transaction
envelope as Freeform turns.

1. Client submits either a selected choice ID or freeform custom action, depending
   on session settings.
2. Backend validates that the choice is visible for the active branch node, or that
   custom actions are allowed.
3. Backend appends `choice_selected` or `user_turn`.
4. Story adapter builds a model request with branch path, scene summary, current
   state, character/persona context, and constraints for next choices.
5. Model returns scene narration, dialogue, visual directives, and 2-5 next
   choices.
6. Backend persists the branch node and choice events.
7. Asset resolver validates and applies scene changes.
8. Response returns the updated scene state, appended events, and visible choices.

Invalid model directives produce structured warning/error events. They should not
crash a session or expose unapproved assets.

## Turn Idempotency, Concurrency, And Failure Boundaries

The turn endpoint must be safe under retries, slow model calls, browser refreshes,
and double-submits.

Idempotency:

- `idempotency_key` is required for turn requests.
- Keys are scoped by `(owner_user_id, session_id, idempotency_key)`.
- The backend hashes the normalized request body into `request_payload_hash`.
- Duplicate completed requests with the same hash return the stored
  `response_payload_json`.
- Duplicate in-flight requests return the current turn request status without
  appending events or making another model call.
- Same key with a different hash returns `409 idempotency_key_conflict`.

Concurrency:

- V1 allows one active turn per session.
- `client_scene_version` must equal the current `vn_play_scene_state.scene_version`.
- Stale scene versions return `409 stale_scene_version` with the current scene
  version and current scene state summary.
- If another turn is active, the endpoint returns `409 turn_in_progress` with the
  active turn request ID and status.
- The active-turn lock is stored on `vn_play_sessions.active_turn_request_id` and
  mirrored by `vn_play_turn_requests.status`.

Transaction boundaries:

- The pre-model transaction creates the turn request, records `base_scene_version`,
  appends `turn_started`, appends the input event, marks the request `model_calling`,
  sets `locked_until`, and sets the active turn lock. It does not advance the
  visible scene version before the model result or failure state is committed.
- The model call and parsing happen outside the database transaction.
- The post-model transaction appends output/state events, stores the exact response
  payload, marks the turn `completed`, increments `scene_version`, and clears the
  active turn lock.
- Model timeout or provider failure marks the turn `model_failed`, appends
  `turn_failed`, stores `error_json`, updates scene warnings/version as needed, and
  clears the active turn lock.
- Parse failure marks the turn `parse_failed`, appends `model_turn_parse_failed` and
  `turn_failed`, stores `error_json`, updates scene warnings/version as needed, and
  clears the active turn lock.
- Process crash recovery treats expired `locked_until` rows as `abandoned` only
  after confirming the session still points at that active turn. A retry must use
  `retry-last-turn` or an explicit retry flag so the original input event remains
  linked to the new attempt.

`retry-last-turn` creates a new turn request linked to the previous failed or
abandoned request. It does not append a second user input event unless the caller
changes the input.

## Structured Model Contract

V1 should prefer a structured JSON block or provider-native structured output when
available. The normalized model turn shape should include:

- `narration`
- `dialogue`: list of speaker/text pairs
- `scene_directives`: optional background, sprite, mood, time, weather, and camera
  hints
- `choices`: Story mode only, 2-5 options by default
- `summary`: short durable scene summary
- `safety_notes`: optional provider/model refusal or constraint notes

If a provider cannot support native structured output, the adapter may use a
delimited JSON instruction and strict parser. Parse failures append
`runtime_gate_failed` or `model_turn_parse_failed` events and return a recoverable
error.

## Linked Chat Semantics

V1 treats `linked_chat_id` as read-only context.

- VN turns do not write user or model messages back to the linked chat session.
- VN Play may read a bounded snapshot of linked chat history when building model
  context.
- The snapshot boundary is captured per turn, for example latest linked chat message
  ID or timestamp, inside the turn request/event payload.
- VN events remain authoritative for VN playback. Linked chat history is auxiliary
  context only.
- If the linked chat changes while a VN session is active, later VN turns may include
  newer chat context only if the session setting allows live context refresh.
- Future write-back or two-way sync must be an explicit mode, not an implicit side
  effect of linking a chat.

## Asset Selection

Use a hybrid resolver.

Deterministic defaults:

- Current location label chooses background.
- Speaker/character state chooses sprite.
- Emotion labels map to expression slots.
- Time, weather, and mood labels refine background variants when present.
- Preferred approved items win over non-preferred approved items.

Model-directed changes:

- Model may request `asset_type`, `slot_key`, or labels such as
  `emotion=happy`, `location=library`, `time=evening`.
- Runtime accepts only matches from the approved manifest.
- Ambiguous matches resolve by preferred item, then stable seeded selection.
- Missing or unapproved matches append `visual_directive_rejected` and keep the
  prior valid scene.

Randomization:

- Seed per session or branch so replay is stable.
- Optional shuffle setting can vary newly entered scenes, but applied item IDs must
  be recorded in events so replay remains exact.

Depth/background effects are frontend presentation hints from manifest metadata.
They do not affect runtime readiness unless a pack explicitly marks depth as
required.

## Runtime Gates

The runtime should fail clearly before model turn execution when configuration is
unsupported.

Required gates:

- Session and linked records belong to the current user.
- VN asset pack is runtime-ready.
- Runtime manifest contains approved assets required for the selected mode.
- Pack/session content rating is compatible with runtime settings.
- Character metadata passes configured age/status checks.
- Provider/model supports the selected mode and structured output path.
- Story mode respects configured max choices, max branch depth, and token budget.
- Visual directives resolve only to approved manifest assets.
- Imported pack trust metadata is visible. Untrusted or warning-bearing packs can
  require explicit opt-in before play.

Gate failures should return structured API errors. When a session exists, they
should also append `safety_gate_triggered` or `runtime_gate_failed` events where
that helps audit/replay.

Character safety metadata:

- Character safety status is derived at session creation and refreshed before each
  turn when linked character records changed.
- Known disallowed or conflicting metadata fails closed for all modes.
- Unknown, absent, or unparseable metadata is represented explicitly as
  `character_safety_status="unknown"`, never silently treated as passing.
- `general` sessions may proceed with unknown metadata and a visible runtime warning
  unless deployment policy requires fail-closed behavior.
- `suggestive`, `mature`, violent, or custom adult-oriented content ratings require
  either affirmative compatible character metadata or an explicit local override
  stored in `settings_json`.
- Imported or untrusted character payloads require explicit opt-in before use when
  safety metadata is missing, conflicting, or came from an untrusted bundle.
- Freeform and Story modes use the same safety metadata rules so mode switching
  cannot bypass gates.

## API Surface

Session endpoints:

- `POST /api/v1/vn-play/sessions`
- `GET /api/v1/vn-play/sessions`
- `GET /api/v1/vn-play/sessions/{session_id}`
- `PATCH /api/v1/vn-play/sessions/{session_id}`
- `DELETE /api/v1/vn-play/sessions/{session_id}`

Turn endpoints:

- `POST /api/v1/vn-play/sessions/{session_id}/turn`
- `POST /api/v1/vn-play/sessions/{session_id}/retry-last-turn`

Event and replay endpoints:

- `GET /api/v1/vn-play/sessions/{session_id}/events`
- `POST /api/v1/vn-play/sessions/{session_id}/rebuild-state`

Branch/checkpoint endpoints:

- `POST /api/v1/vn-play/sessions/{session_id}/checkpoint`
- `GET /api/v1/vn-play/sessions/{session_id}/checkpoints`
- `POST /api/v1/vn-play/sessions/{session_id}/restore`
- `GET /api/v1/vn-play/sessions/{session_id}/branches`

V1 can keep `rebuild-state` admin/debug-only if it is not needed by the public UI.

### Create Session Request

Important fields:

- `mode`
- `title`
- `primary_character_id`
- `additional_character_ids`
- `vn_asset_pack_id`
- `linked_chat_id`
- `linked_chat_mode`: V1 default `read_only_context`
- `world_book_ids`
- `content_rating`
- `seed`
- `settings`

### Turn Request

Important fields:

- `input_text`
- `choice_id`
- `custom_action`
- `client_scene_version`: required
- `idempotency_key`: required

Only one of `input_text`, `choice_id`, or `custom_action` should be accepted per
turn.

## Frontend Design

Add `/vn-play` as a separate workspace.

Layout:

- Left rail: session list, mode filter, new Freeform button, new Story button.
- Center stage: background layer, optional depth/parallax layer, character sprite
  layer, dialogue/narration box, choice area.
- Right inspector: active pack, character/persona, scene state, event warnings,
  branch path, and checkpoint controls.
- Compact toolbar: continue, auto-advance, checkpoint, restore, and settings.

Freeform mode:

- Open text input remains primary.
- Character replies render as VN dialogue.
- Scene state and visuals update from validated runtime events.

Story mode:

- Choice buttons are primary.
- Optional custom action input is a per-session setting.
- Branch/checkpoint state is visible but secondary to the scene.

Visual effects should stay modest in V1: fade transitions, sprite swaps, background
changes, and optional depth/parallax when manifest metadata exists.

## Testing Strategy

Backend tests:

- Session CRUD ownership and mode validation.
- Create-session gates for pack readiness and character links.
- Character safety metadata gates for known, unknown, conflicting, imported, and
  override cases.
- Event append ordering and sequence uniqueness.
- Replay-derived state from event logs.
- Turn idempotency replay, in-flight duplicate behavior, and payload-hash conflict.
- Per-session turn concurrency and stale `client_scene_version` rejection.
- Model timeout, provider failure, parse failure, abandoned turn recovery, and
  `retry-last-turn` behavior.
- Linked chat read-only context snapshots and no write-back side effects.
- Freeform turn with mocked model output and visual directives.
- Story turn creates branch node and visible choices.
- Asset resolver accepts approved manifest matches and rejects unapproved or missing
  matches.
- Runtime gate failures return structured errors and append events when applicable.
- Checkpoint and restore append events and derive expected scene state.
- Idempotent turn requests do not duplicate events.

Frontend tests:

- `/vn-play` renders session list and mode tabs.
- Creating Freeform and Story sessions calls expected APIs.
- Mock turn response updates background, sprite, dialogue, and choices.
- Rejected visual directive warning appears in inspector.
- Story choice click posts the selected choice.
- Checkpoint/restore controls call expected APIs and update scene state.

E2E smoke:

- Create a mocked Freeform session and complete one turn.
- Create a mocked Story session, select one choice, and verify branch state updates.

## Rollout

- Guard `/vn-play` behind a route/config flag initially.
- Keep VN asset pack creation/review independent from play sessions.
- Start with mocked LLM tests and one provider-agnostic structured-output adapter.
- Do not require realtime generation or authored graph tooling for V1.
- Keep the first implementation focused on one primary rendered character.
- Document provider limitations and structured-output requirements.

## Risks And Mitigations

- Model output can be malformed. Mitigate with strict parsing, structured output
  where available, recoverable parse-failure events, and retry controls.
- Slow turns can be retried or double-submitted. Mitigate with required
  idempotency keys, per-session active-turn locks, stored responses, and stale scene
  version rejection.
- Visual directives can request unavailable assets. Mitigate with approved-manifest
  validation and deterministic fallback to prior scene state.
- Event logs can grow large. Mitigate with pagination, compact scene summaries, and
  checkpoint snapshots.
- Story branches can expand without bound. Mitigate with max depth, max choices,
  archive controls, and explicit session settings.
- Existing chat and VN play state can drift when linked. Mitigate by treating VN
  events as authoritative for VN playback and linked chat as optional context, not
  source of truth.
- Content/provider configuration can be ambiguous. Mitigate with explicit gates,
  visible trust/rating metadata, and audit-friendly failure events.

## Future Work

- Authored story graph import/export and editor.
- Multi-character visual stage support once asset packs support multi-primary
  generation cleanly.
- Session export/import.
- Shareable playback-only bundles.
- Optional realtime generation hooks after offline pack play is stable.
- More advanced transitions, pose composition, lip sync, voice/TTS integration, and
  timeline playback.
- Runtime analytics for branch coverage and frequently rejected visual directives.
