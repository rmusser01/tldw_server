# VN Scripted Model Generation Runtime Design

Date: 2026-05-10

Issue: <https://github.com/rmusser01/tldw_server/issues/1535>

Parent tracker: <https://github.com/rmusser01/tldw_server/issues/1391>

## Summary

Add backend-owned model expansion and regeneration for `scripted_story` VN Play
sessions. Published scripts can contain explicit generation opcodes that call
configured LLM providers through pinned generation-profile snapshots, persist
immutable generation revisions, and replay deterministically through session
events, checkpoints, save slots, and idempotency records.

The work splits into two PR-sized paths:

1. Backend runtime/API: data model, interpreter flow, model calls, moderation,
   revisions, generation history API, and docs/tests.
2. WebUI generation-history inspector: a session-scoped route that consumes the
   stable backend API for revision browsing, debug reveal, regeneration,
   cancellation, and activation.

## Goals

- Make scripted `generate` opcodes capable of real model-backed output without
  moving prompt construction into clients.
- Preserve deterministic replay by persisting every model attempt and revision.
- Keep V1 authored-control-flow safe while allowing generated narrative,
  dialogue, scene-update hints, and generated choices.
- Support hosted and local providers through existing provider configuration,
  rate limits, policy checks, and LLM usage accounting.
- Give API clients a stable generation-history surface with owner/admin-only
  debug detail access.
- Keep implementation reviewable by landing backend runtime/API before WebUI.

## Non-Goals

- Realtime image generation.
- Streaming token updates, SSE, or WebSocket subscriptions.
- Full text DSL authoring.
- Manual editing of generated revisions.
- Runtime script graph creation by the model.
- Session export/import implementation.
- Multiplayer/co-op generation.

The data model should still leave enough structure for future session
export/import to include generation events and revisions.

## Existing Baseline

PR #1516 added:

- canonical VN platform namespace under `/api/v1/vn/vn-*`;
- `scripted_story` sessions with pinned script/profile snapshots;
- `script/advance`, `script/choices/{choice_id}`, `script/regenerate`,
  `script/state`, and `script/debug-state`;
- literal scripted generation support where `generate` opcodes must provide
  authored `narrative_text` or `regeneration_text`;
- `model_invoked=false` for literal generation results;
- `script_generation_unavailable` when a generation opcode lacks literal text;
- save slots/checkpoints and per-session idempotent action locking.

This sprint replaces the literal-only limitation for explicit generation opcodes
while keeping literal behavior valid.

## Design Decisions

### Generation Execution Mode

`POST /script/advance` runs generation automatically by default when the
interpreter reaches a generation opcode. A generation opcode may set
`requires_user_confirm=true`; in that case the interpreter pauses and returns a
normal script state response with:

- `waiting_generation_confirmation`;
- stable `generation_request_id`;
- safe generation profile summary;
- estimated cost/risk metadata where available.

Confirmation is not an error state.

### Generation Profiles

Published scripts can include multiple pre-approved generation-profile snapshots.
Each published script version owns an immutable
`generation_profile_snapshots` map:

```json
{
  "default": 44,
  "choice_writer": 45,
  "scene_director": 46
}
```

Map keys are authored, stable identifiers matching
`^[a-z0-9_.-]{1,64}$`. The `default` key is required when any generation
opcode omits `profile_key`. Snapshot IDs point to profile snapshot rows/content
captured at publish time, not mutable live profile records.

Each generation opcode references a profile with `profile_key`. Publish-time
validation rejects:

- unknown profile keys;
- profile snapshots that do not allow the opcode's `output_schema`;
- hosted/public snapshots without required moderation policy;
- automatic generation batch caps below `1`;
- dynamic choices when the profile does not allow `choice_set`.

Generation opcodes cannot pass arbitrary provider/model/API routing parameters
at runtime. Profile snapshots own provider, model, hosted/local classification,
moderation requirements, max batch count, and supported output schemas.

Runtime resolution always uses the script-version snapshot map. Generation
requests and revisions store both `generation_profile_key` and
`generation_profile_snapshot_id`; replay and regeneration use those stored
values rather than re-reading a mutable script or profile.

If the referenced provider/model is unavailable at runtime, the runtime persists
a failed generation attempt with stable `provider_unavailable` detail and leaves
the session at the generation point. There is no implicit fallback profile.

### Confirmation And Cancellation

Generation confirmation uses an explicit endpoint with `idempotency_key` and
`client_scene_version`.

Cancellation is also explicit. A pending generation request can be canceled while
the session is still waiting on that exact request. Cancel behavior:

- records `script_generation_canceled`;
- clears `waiting_generation_confirmation`;
- if opcode has `on_cancel`, jumps there and continues/pauses through normal
  script flow;
- if no `on_cancel`, leaves the session at stable `generation_canceled`.

`on_cancel` is optional. Authors who need narrative fallback can provide it;
simple scripts are not forced to define cancellation branches.

### Revision History

Every successful, failed, blocked, or canceled attempt is represented in a
generation request/revision history. Successful generated/regenerated output is
immutable. Exactly one successful revision may be active for each generation
point.

The owner of a generation point is a `vn_play_generations` row. It represents
one authored generation opcode occurrence in one session, not one attempt. Its
`active_revision_id` pointer is the source of truth for what public state
renders. Revision rows do not own independent active flags.

Generated revisions are read-only runtime artifacts in V1. Manual editing is a
future authoring feature, not part of this sprint.

Save slots/checkpoints pin the session's full generation activation map at that
moment. Restoring a checkpoint updates `vn_play_generations.active_revision_id`
in one transaction to match the checkpoint snapshot exactly. Generation rows not
present in the restored snapshot keep their history but have
`active_revision_id=null` until the script reaches that generation point again.
The full revision history remains attached to the session.

Public script state exposes only active generated output. Full revision history,
raw failed output, raw prompts, parser diagnostics, and moderation diagnostics
are owner/debug-only.

### Failure Semantics

Model failure:

- persist generation request/revision status `model_failed`;
- store stable public error code such as `provider_unavailable`,
  `model_timeout`, or `model_error`;
- keep session at the generation point;
- retry creates a new `generation_request_id`.

Parser/schema failure:

- persist raw model output in debug-only metadata;
- mark the revision `parse_failed`;
- expose stable public `parse_failed`;
- retry/regenerate creates a new revision/request.

Moderation block:

- for hosted/public profiles, run configured moderation/safety checks before a
  revision becomes active;
- blocked output is persisted as debug-only metadata;
- revision status is `moderation_blocked`;
- public state exposes stable `moderation_blocked`;
- blocked revisions are never activatable.

Moderation service failure:

- for hosted/public profiles, moderation timeout, unavailable service, or policy
  adapter error is fail-closed;
- persist the request/revision as `moderation_failed`;
- store public error code `moderation_unavailable`;
- do not activate the revision;
- keep the session at the generation point;
- retry/regenerate creates a new request.

Local/self-hosted profiles may opt out of output moderation through policy.
Opt-out is recorded on the request/revision as `moderation_skipped_by_policy` so
history can distinguish local policy from a missing moderation check.

### Checkpoints

The backend creates a checkpoint before executing any confirmed or automatic
model generation request. The backend also creates a checkpoint before activating
an older successful revision.

This makes nondeterministic generation recoverable without requiring clients to
remember to create save slots first.

Checkpoint payloads store generation state by generation point:

```json
{
  "active_generation_revisions": {
    "archive:3:map-clue": {
      "generation_id": 12,
      "active_revision_id": 31
    }
  }
}
```

Restore does not delete generation rows or revision rows. It changes active
revision pointers to the checkpoint's map, clears active pointers for every
generation point absent from the map, records
`script_generation_checkpoint_restored`, and increments scene version. This is
true even when a later script path could reach the same generation point again;
that future execution must create or reactivate output through normal runtime
flow.

### Batch Generation

`script/advance` may execute multiple automatic model generation opcodes in one
request, but only up to the selected profile's cap. Default cap is `1`.

Confirmation-gated generation always pauses. If the automatic generation cap is
reached, the response includes `waiting_reason=generation_batch_limit` and the
next pending generation request.

Each model call still has separate request/revision records, usage accounting,
failure state, and replay behavior.

### Output Schemas

Generation opcodes choose one of a small set of strict output schemas:

- `narrative_dialogue`: narrative lines and/or dialogue only.
- `choice_set`: generated choices plus optional short lead-in narration.
- `scene_update`: narrative/dialogue plus visual directive hints.

All model outputs are parsed as JSON objects with `schema` equal to the selected
opcode schema. Unknown fields fail validation at every object level, not only
the top level. Implement schemas with `additionalProperties: false` or Pydantic
`extra="forbid"` for:

- root output objects;
- narrative line objects;
- dialogue line objects;
- choice objects;
- visual directive objects.

The only intentional free-form object is `choice.metadata`, which remains
bounded and is never interpreted as control flow.

Shared limits:

- string fields are UTF-8 text, max 2,000 characters unless otherwise noted;
- arrays use server-enforced maximums from the profile snapshot;
- `character_id`, when present, must refer to a character attached to the
  session;
- `speaker`, when present without `character_id`, is plain display text with max
  length 128;
- `metadata` is a JSON object capped at 4 KB after serialization and may contain
  arbitrary JSON-compatible values;
- `visual_directives.labels`, when present, is a manifest-label filter object
  capped at 4 KB after serialization;
- generated choice IDs must match `^[a-zA-Z0-9_-]{1,80}$` and be unique within
  the revision;
- no schema may set arbitrary script variables, script labels, or next targets.

`narrative_dialogue` shape:

```json
{
  "schema": "narrative_dialogue",
  "narrative": [
    {"text": "The archive door clicks open."}
  ],
  "dialogue": [
    {
      "speaker": "Mira",
      "character_id": "character_mira",
      "text": "Someone was here before us."
    }
  ]
}
```

Validation requires at least one `narrative` or `dialogue` item. Default caps:
12 narrative lines and 24 dialogue lines.

`choice_set` shape:

```json
{
  "schema": "choice_set",
  "lead_in": "Mira watches your reaction.",
  "choices": [
    {
      "id": "ask-about-the-map",
      "text": "Ask about the map",
      "metadata": {"tone": "curious"}
    }
  ]
}
```

Validation requires 1-8 choices by default. The runtime injects
`source`, `generation_id`, and `revision_id`; the model may not provide them.

`scene_update` shape:

```json
{
  "schema": "scene_update",
  "narrative": [
    {"text": "Dust hangs in the lantern light."}
  ],
  "dialogue": [],
  "visual_directives": [
    {
      "asset_type": "background",
      "slot_key": "archive_night",
      "labels": {
        "location": "archive",
        "time": "night"
      }
    },
    {
      "asset_type": "sprite",
      "slot_key": "mira_concerned",
      "labels": {
        "character": "mira",
        "emotion": "concerned",
        "position": "left"
      }
    }
  ]
}
```

Default cap is 12 visual directives. `visual_directives` use the existing VN
Play resolver input shape: declared `asset_type`, optional `slot_key`, and
optional `labels`. They do not contain command fields such as `directive_type`,
`set_location`, `background_slot_key`, `expression_key`, or `position` outside
the `labels` filter. `asset_type` is one of the resolver-supported asset
families such as `background`, `sprite`, `depth_companion`, or `cg`.

Visual directives are suggestions until the existing VN visual directive
resolver validates them against the approved manifest. Resolved assets update
scene state through existing `visual_directive_applied` events; unresolved
directives are warning-only. `scene_update` does not generate choices in V1;
opcodes that need choices use `choice_set`.

No arbitrary variable mutation is allowed in V1. The runtime may set system
variables such as `last_generated_choice.id`, `last_generated_choice.text`, and
`last_generated_choice.metadata`.

### Generated Choices

Generated choices are allowed only when output validates against `choice_set`.
Generated choices are public and distinguishable:

```json
{
  "id": "ask-about-the-map",
  "text": "Ask about the map",
  "source": "generated",
  "generation_id": 12,
  "revision_id": 31
}
```

Generated choices do not define arbitrary next labels. A generation opcode that
allows choices must define `on_generated_choice`. When the user selects a
generated choice:

- runtime records durable branch event metadata with generation/revision IDs;
- selected choice metadata is exposed through system variables;
- interpreter jumps to `on_generated_choice`.

This keeps control flow authored while allowing dynamic choice content.

### Scene Updates

`scene_update` output may propose visual directives. The backend validates the
schema, then routes directives directly through the existing VN visual directive
resolver without translating a separate command language.

Only approved/owned assets may be applied. Rejected directives are warning-only
with stable reason codes. Public state exposes applied visuals plus safe
warnings, not raw prompt internals.

### Revision Activation

Users can activate an older successful revision through an idempotent backend
command. Activation:

- is allowed only for `succeeded` revisions;
- creates a checkpoint first;
- records `script_generation_revision_activated`;
- increments scene version;
- changes the active revision for that generation point only;
- does not move the script cursor;
- does not delete later revisions;
- does not rewrite branches or replay later events.

Activation is blocked if downstream state depends on generated choices from the
currently active revision. In V1 this means: if a generated choice from the old
active revision has already been selected into branch history, activation fails
with a stable conflict such as `generated_choice_dependency_exists`.

Activation is also blocked when the generation point has later visual or script
events that depend on the old revision's `scene_update` output and cannot be
resolved as a read-time overlay. The implementation must choose the conservative
path: if dependency analysis cannot prove the swap is independent, return
`revision_activation_blocked`.

Public/current scene rendering uses a read-time active-revision overlay. Events
remain immutable audit history, but generated output shown to clients is loaded
from `vn_play_generations.active_revision_id`. Activation updates that pointer
and appends `script_generation_revision_activated`; it does not rewrite the
original generation event payload. Scene-version increments make clients refetch
the newly active overlay.

## Data Model

Metadata lives in the per-user ChaChaNotes database with VN Play sessions. This
matches existing VN session/event/checkpoint ownership and backup behavior.

### `vn_play_generations`

One row per generated content point in a session. This is the owner for
`generation_id`.

Suggested fields:

- `id`
- `session_id`
- `owner_user_id`
- `script_id`
- `script_version_id`
- `generation_point_key`
- `opcode_id`
- `opcode_label`
- `opcode_index`
- `output_schema`
- `generation_profile_key`
- `generation_profile_snapshot_id`
- `active_revision_id`
- `latest_request_id`
- `status`: `not_started`, `pending_confirmation`, `in_progress`, `completed`,
  `canceled`, `model_failed`, `parse_failed`, `moderation_blocked`,
  `moderation_failed`, `abandoned`
- `created_at`
- `updated_at`

Constraints:

- unique `(owner_user_id, session_id, generation_point_key)`;
- `active_revision_id` must point to a `succeeded` revision for the same
  generation;
- `generation_profile_key` and `generation_profile_snapshot_id` are copied from
  the published script-version snapshot map and are immutable for the generation
  point after creation.

### `vn_play_generation_requests`

One row per generation attempt/confirmation action.

Suggested fields:

- `id`
- `generation_id`
- `session_id`
- `owner_user_id`
- `script_id`
- `script_version_id`
- `generation_point_key`
- `generation_profile_key`
- `generation_profile_snapshot_id`
- `request_kind`: `automatic`, `confirmation`, `regenerate`
- `status`: `pending_confirmation`, `in_progress`, `completed`, `canceled`,
  `model_failed`, `parse_failed`, `moderation_blocked`, `moderation_failed`,
  `abandoned`
- `create_action_id`
- `execute_action_id`
- `cancel_action_id`
- `client_scene_version`
- `opcode_snapshot_json`
- `prompt_fingerprint`
- `checkpoint_id_before`
- `provider_call_started_at`
- `provider_call_completed_at`
- `lease_expires_at`
- `public_error_code`
- `created_at`
- `updated_at`

Unique constraints:

- `(owner_user_id, session_id, generation_id, id)` as the stable request
  identity surface.

### `vn_play_generation_actions`

One row per idempotent generation-related API action. This table owns payload
hash comparison and completed response replay for create, confirm/execute,
cancel, regenerate, and activate actions.

Suggested fields:

- `id`
- `session_id`
- `owner_user_id`
- `generation_id`
- `generation_request_id`
- `generation_revision_id`
- `action_kind`: `create_pending`, `execute`, `cancel`, `regenerate`,
  `activate`
- `idempotency_key`
- `payload_hash`
- `status`: `pending`, `in_progress`, `completed`, `failed`, `canceled`,
  `abandoned`
- `completed_action_response_json`
- `public_error_code`
- `created_at`
- `updated_at`

Unique constraints:

- `(owner_user_id, session_id, idempotency_key)`.
- `(owner_user_id, session_id, action_kind, generation_request_id,
  idempotency_key)` for request-scoped lookup convenience.

Idempotency behavior:

- If a duplicate idempotency key arrives with the same payload hash and the
  action is complete, replay `completed_action_response_json`.
- If the same idempotency key is reused with a different payload hash, return
  `idempotency_key_conflict`.
- If it arrives while the request is `pending_confirmation`, return the same
  pending confirmation state.
- If it arrives while the request is `in_progress`, return 409
  `generation_request_in_progress` with the current request ID; do not start a
  second provider call.
- If a worker restarts and finds stale `in_progress` with
  `provider_call_started_at` set and no persisted provider result, mark the
  request `abandoned` with public error `generation_attempt_abandoned`. Retry
  requires a new idempotency key/request.
- If `provider_call_started_at` is not set, the request may be safely reclaimed
  by the same idempotency key because no provider call was made.

HTTP action replay remains scoped to the existing session action idempotency
surface from PR #1516. The generation action row stores the payload hash and
response needed to recover the model attempt itself. For confirmation-gated
generation, `script/advance` creates the pending request and a
`create_pending` action; `confirm` creates or reuses an `execute` action when it
starts the provider call; `cancel` creates or reuses a `cancel` action when it
cancels the pending request. The request row stores the current action IDs for
quick joins, but the action rows are the idempotency source of truth.

### `vn_play_generation_revisions`

One immutable row per generated output attempt.

Suggested fields:

- `id`
- `generation_id`
- `generation_request_id`
- `session_id`
- `owner_user_id`
- `generation_point_key`
- `generation_profile_key`
- `generation_profile_snapshot_id`
- `revision_number`
- `status`: `succeeded`, `model_failed`, `parse_failed`,
  `moderation_blocked`, `moderation_failed`, `canceled`, `abandoned`
- `output_schema`
- `public_output_json`
- `public_error_code`
- `raw_output_debug_json`
- `parser_diagnostics_json`
- `moderation_diagnostics_json`
- `model_metadata_json`
- `usage_metadata_json`
- `source`: `model`, `literal`, `regenerated`
- `created_at`

Revision rows do not have an `active` flag in V1. Active status is derived by
joining `vn_play_generations.active_revision_id` to the revision row. This avoids
conflicts between checkpoint restore, activation, and immutable revision history.

### Script Position Additions

Script position can reference active generation state by ID rather than embedding
large output blobs:

```json
{
  "label": "archive",
  "index": 4,
  "variables": {},
  "waiting_generation_request_id": 91,
  "waiting_reason": "generation_confirmation_required",
  "active_generation_revisions": {
    "archive:3:map-clue": {
      "generation_id": 12,
      "active_revision_id": 31
    }
  },
  "last_generation_id": 12
}
```

## API Contract

Canonical base path:

`/api/v1/vn/vn-play`

Legacy aliases may exist for compatibility, but the VN platform contract should
document the canonical path.

### Confirm Pending Generation

`POST /sessions/{session_id}/script/generation-requests/{generation_request_id}/confirm`

Request:

```json
{
  "client_scene_version": 4,
  "idempotency_key": "session-1-generation-91-confirm"
}
```

Response: normal script action response with updated `current_scene`,
`script_state`, events, warnings, and generated output if completed. If the
profile batch cap is reached after this generation, response may include the
next waiting generation request.

### Cancel Pending Generation

`POST /sessions/{session_id}/script/generation-requests/{generation_request_id}/cancel`

Request:

```json
{
  "client_scene_version": 4,
  "idempotency_key": "session-1-generation-91-cancel"
}
```

Response: normal script action response. If no `on_cancel` exists, script state
contains `waiting_reason=generation_canceled`.

### Regenerate

`POST /sessions/{session_id}/script/generations/{generation_id}/regenerate`

Request:

```json
{
  "client_scene_version": 7,
  "idempotency_key": "session-1-generation-12-regenerate-2"
}
```

Response includes a new revision. A successful regenerated revision becomes
active unless request includes a future explicit `activate=false` extension.

### Activate Revision

`POST /sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/activate`

Request:

```json
{
  "client_scene_version": 9,
  "idempotency_key": "session-1-generation-12-activate-31"
}
```

Only `succeeded` revisions may be activated. The backend checkpoints first and
blocks activation when downstream generated-choice dependencies exist.

### Generation History

`GET /sessions/{session_id}/script/generations`

Uses the repository's existing offset pagination conventions:
`limit`, `offset`, top-level legacy aliases, and a nested `pagination` object
using `OffsetPaginationMeta` with `mode="offset"`. Results are ordered by
`created_at desc, id desc`.

Suggested filters:

- `generation_id`
- `generation_point_key`
- `status`
- `active`
- `source`
- `created_after`
- `created_before`
- `limit`
- `offset`

Response item shape is owner-safe:

```json
{
  "id": 31,
  "generation_id": 12,
  "generation_point_key": "archive:3:map-clue",
  "revision_number": 2,
  "status": "succeeded",
  "active": true,
  "output_schema": "choice_set",
  "public_output": {
    "lead_in": "The map trembles in Mira's hand.",
    "choices": [
      {
        "id": "ask-about-the-map",
        "text": "Ask about the map",
        "source": "generated",
        "generation_id": 12,
        "revision_id": 31
      }
    ]
  },
  "profile": {
    "profile_key": "choice_writer",
    "snapshot_id": 44,
    "provider_class": "hosted",
    "moderation_required": true,
    "estimated_cost_class": "low"
  },
  "created_at": "2026-05-10T19:00:00Z"
}
```

Response envelope:

```json
{
  "items": [],
  "total": 42,
  "limit": 25,
  "offset": 0,
  "has_more": true,
  "next_offset": 25,
  "pagination": {
    "mode": "offset",
    "total": 42,
    "limit": 25,
    "offset": 0,
    "has_more": true,
    "next_offset": 25
  }
}
```

### Revision Debug Detail

`GET /sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/debug`

Owner/admin-only. Authorization requires the authenticated principal to own the
session, or to be an existing AuthNZ admin in multi-user mode. The route must
verify that `revision_id` belongs to `generation_id` and that both belong to the
session before returning data.

This is separate from the normal history endpoint so raw model output, prompts,
parser diagnostics, and moderation details are never returned accidentally in
list views.

Debug detail includes:

- raw model output;
- parser diagnostics;
- moderation diagnostics;
- prompt fingerprint and redaction status;
- generation profile/model metadata;
- usage/accounting metadata;
- request/revision lineage.

Moderation-blocked raw output is redacted by default even on this debug route.
To reveal it, clients must call the same endpoint with explicit reveal
parameters, for example
`?include_blocked_raw=true&confirm=REVEAL_MODERATION_BLOCKED`. The WebUI should
gate that call behind a second confirmation.

API access still requires normal authentication. Successful, denied, and
moderation-blocked reveal reads should emit `vn.script_generation.debug_read`
through the existing unified audit path where that service is configured; in
single-user deployments without audit storage, the read proceeds and logs a
structured warning instead of blocking local use.

## Public Script State

Public state may include:

- active generated output;
- generated choices with `source="generated"`;
- active generation IDs/revision IDs needed for rendering and branch metadata;
- waiting generation confirmation state;
- safe warnings/error codes.

Public state must not include:

- raw prompts;
- raw failed model output;
- parser diagnostics;
- moderation diagnostics;
- hidden profile secrets or provider configuration.

## Capabilities And Setup Metadata

`GET /api/v1/vn/vn-capabilities` should add feature flags:

- `scripted_model_generation`
- `scripted_generation_revision_history`
- `scripted_generation_batching`
- `scripted_generation_moderation`
- `scripted_generation_dynamic_choices`

Setup options and script version metadata should expose concrete selected
profile details:

- profile key and immutable snapshot ID;
- provider class: `hosted`, `local`, or `self_hosted`;
- max automatic generation batch count;
- moderation requirement;
- estimated cost class;
- supported output schemas;
- dynamic choice support;
- scene update support;
- whether confirmation is required by profile or opcode.

## Usage Accounting

Model-backed generation uses the existing LLM usage/accounting path with VN
metadata attached:

- `vn_session_id`
- `script_id`
- `script_version_id`
- `generation_id`
- `generation_request_id`
- `generation_revision_id`
- `generation_profile_key`
- `generation_profile_snapshot_id`
- `generation_point_key`

The API should not introduce a VN-only usage ledger unless existing accounting
cannot carry this metadata.

## Model Call Transaction Boundaries

Generation is intentionally split around the nondeterministic provider call:

1. In a short database transaction, validate session/user/scene version,
   acquire the per-session action lock, create or reuse `vn_play_generations`,
   create the checkpoint, create `vn_play_generation_requests`, create the
   matching `vn_play_generation_actions` row, and set request/action status to
   `in_progress`.
2. Commit before calling the model provider. The request row must contain
   `provider_call_started_at` immediately before the call begins.
3. Call the provider outside the database transaction.
4. In a second transaction, persist the revision, parser/moderation result,
   usage metadata, events, active revision pointer, scene version, and the
   generation action's `completed_action_response_json`.

Recovery rules:

- A duplicate request with the same idempotency key never starts another
  provider call once `provider_call_started_at` is set.
- If the first transaction committed but the provider call never started, the
  same idempotency key may reclaim the request.
- If the provider returned and the revision was persisted but the HTTP response
  was lost, duplicate submit replays the completed generation action response.
- If the process crashed during the provider call and no result was persisted,
  stale lease recovery marks the request `abandoned`; retry uses a new request.
- The per-session action lock serializes generation, confirmation, cancel,
  regenerate, activate, and normal script action endpoints. Stale
  `client_scene_version` returns 409 before any provider call.

## Runtime Flow

### Automatic Generation

1. Client calls `POST /script/advance`.
2. Runtime validates scene version and action idempotency.
3. Interpreter reaches a non-confirmation generation opcode.
4. Runtime resolves `profile_key` through the published script-version snapshot
   map.
5. Runtime creates or reuses the `vn_play_generations` row.
6. Runtime creates a checkpoint.
7. Runtime creates generation request/action rows and marks them `in_progress`.
8. Runtime calls the selected provider through the pinned profile snapshot.
9. Runtime validates output schema.
10. Runtime runs moderation if required.
11. Runtime writes immutable revision and updates
    `vn_play_generations.active_revision_id` if successful.
12. Runtime appends generation events, updates script position/scene state, and
    continues until stop point or batch cap.

### Confirmation-Gated Generation

1. Interpreter reaches `requires_user_confirm=true`.
2. Runtime creates or reuses a pending generation request.
3. Response returns normal script state with `waiting_generation_confirmation`.
4. Client calls confirm or cancel endpoint.
5. Confirm follows automatic generation steps from checkpoint onward.
6. Cancel records cancellation and follows `on_cancel` or stable canceled state.

### Regeneration

1. Client calls regenerate with generation ID and idempotency key.
2. Runtime validates session, scene version, and activation constraints.
3. Runtime creates checkpoint.
4. Runtime creates a new generation request/action tied to the existing
   generation row.
5. Runtime reuses the generation row's stored `generation_profile_key` and
   `generation_profile_snapshot_id`.
6. Runtime creates a new immutable revision.
7. Successful revision becomes active by updating
   `vn_play_generations.active_revision_id`.
8. Public state updates for that generation point only.

### Revision Activation

1. Client calls activate with generation/revision IDs.
2. Runtime validates revision is `succeeded`.
3. Runtime checks downstream generated-choice dependency guard.
4. Runtime creates checkpoint.
5. Runtime updates `vn_play_generations.active_revision_id` in a transaction.
6. Runtime records activation event and increments scene version.

## WebUI Inspector Path

WebUI work lands after the backend PR.

Route:

`/vn-play/sessions/:sessionId/generations`

The page should show:

- generation points and revisions;
- active revision status;
- public generated output;
- parser/model/moderation status;
- profile/provider metadata and cost class where available;
- regenerate, cancel, and activate controls;
- links back to session, script, and profile context;
- raw debug reveal per revision;
- second confirmation before revealing moderation-blocked raw output.

No manual editing in V1.

The WebUI must consume the dedicated generation-history APIs. It should not
derive history from debug-state or raw events.

## Implementation Split

### PR 1: Backend Runtime/API

Scope:

- database tables/migrations/repository methods;
- published script-version generation profile snapshot map and validator rules;
- generation request/action/revision services;
- interpreter integration and generation batching;
- confirmation/cancel/regenerate/activate endpoints;
- history endpoints with offset pagination and debug detail endpoint;
- provider invocation through pinned generation-profile snapshots;
- moderation and parser failure persistence;
- generated choices and generated-choice branch metadata;
- capabilities/setup metadata additions;
- API docs and backend tests.

Verification:

- focused repository/service tests;
- `VN_Play` scripted generation API tests;
- `VN_Scripts` validator/publish tests for new opcode fields;
- `VN_Platform` capabilities tests;
- OpenAPI contract tests;
- compileall;
- Bandit on touched backend scope;
- `git diff --check`.

### PR 2: WebUI Generation-History Inspector

Scope:

- API client helpers/types;
- `/vn-play/sessions/:sessionId/generations` route;
- generation history table/list;
- revision detail drawer/panel;
- controls for confirm/cancel/regenerate/activate;
- raw debug reveal and moderation-blocked second confirmation;
- links from existing VN Play session state/inspector;
- frontend tests.

Verification:

- focused Vitest for API client and page behavior;
- interaction tests for debug reveal and moderation-blocked confirmation;
- route smoke/e2e if the existing VN Play smoke harness supports it;
- TypeScript/lint scope checks where practical;
- `git diff --check`.

## Error Codes

New or formalized stable error codes:

- `generation_confirmation_required`
- `generation_request_not_found`
- `generation_request_not_pending`
- `generation_request_in_progress`
- `generation_attempt_abandoned`
- `generation_canceled`
- `generation_batch_limit`
- `generation_profile_unavailable`
- `provider_unavailable`
- `model_timeout`
- `model_failed`
- `parse_failed`
- `moderation_blocked`
- `moderation_failed`
- `moderation_unavailable`
- `revision_not_found`
- `revision_not_succeeded`
- `revision_activation_blocked`
- `generated_choice_dependency_exists`
- `debug_payload_forbidden`

Existing errors such as `stale_scene_version`, `turn_in_progress`,
`idempotency_key_conflict`, and `scripted_story_required` remain unchanged.

## Security And Safety

- Debug output is never returned by default list/state endpoints.
- Hosted/public profiles must run configured moderation before activation; block,
  timeout, or unavailable moderation is fail-closed.
- Moderation-blocked output is never activatable.
- Generated visual directives must resolve through approved owned assets.
- Raw prompt/provider secrets are not stored in public output.
- Sensitive debug reads should use the existing unified audit path where
  available, with explicit `vn.script_generation.debug_read` event names.
- Idempotency keys must be scoped to session/user and compared by payload hash.

## Implementation Notes To Resolve In PR 1

1. Which existing LLM call abstraction should scripted generation use for the
   first backend PR: the same adapter used by Story mode or a narrower service
   wrapper around the provider manager?
2. Confirm the exact existing moderation service call shape to use for the
   fail-closed hosted/public policy path.
3. Confirm whether single-user debug-read audit warnings should be emitted only
   to structured logs or also to a lightweight local audit table.

These are implementation-plan details, not design blockers. Pagination,
profile-snapshot identity, moderation failure behavior, debug endpoint shape,
and generation/revision ownership are defined above.
