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
- Give API clients a stable generation-history surface with owner-only debug
  detail access.
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
Each generation opcode references one profile snapshot by stable key or ID.

Generation opcodes cannot pass arbitrary provider/model/API routing parameters
at runtime. Profile snapshots own provider, model, hosted/local classification,
moderation requirements, max batch count, and supported output schemas.

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

Generated revisions are read-only runtime artifacts in V1. Manual editing is a
future authoring feature, not part of this sprint.

Save slots/checkpoints pin active revision IDs at that moment. The full revision
history remains attached to the session.

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

Local/self-hosted profiles may opt out of output moderation through policy.

### Checkpoints

The backend creates a checkpoint before executing any confirmed or automatic
model generation request. The backend also creates a checkpoint before activating
an older successful revision.

This makes nondeterministic generation recoverable without requiring clients to
remember to create save slots first.

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
schema, then routes directives through the existing VN visual directive resolver.

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

## Data Model

Metadata lives in the per-user ChaChaNotes database with VN Play sessions. This
matches existing VN session/event/checkpoint ownership and backup behavior.

### `vn_play_generation_requests`

One row per generation attempt/confirmation action.

Suggested fields:

- `id`
- `session_id`
- `owner_user_id`
- `script_id`
- `script_version_id`
- `generation_point_key`
- `generation_profile_snapshot_id`
- `request_kind`: `automatic`, `confirmation`, `regenerate`
- `status`: `pending_confirmation`, `in_progress`, `completed`, `canceled`,
  `model_failed`, `parse_failed`, `moderation_blocked`
- `idempotency_key`
- `request_payload_hash`
- `client_scene_version`
- `opcode_snapshot_json`
- `prompt_fingerprint`
- `checkpoint_id_before`
- `created_at`
- `updated_at`

Unique constraints:

- `(owner_user_id, session_id, idempotency_key)` for mutating generation actions.
- `(owner_user_id, session_id, generation_point_key, id)` as the stable request
  identity surface.

### `vn_play_generation_revisions`

One immutable row per generated output attempt.

Suggested fields:

- `id`
- `generation_request_id`
- `session_id`
- `owner_user_id`
- `generation_point_key`
- `revision_number`
- `status`: `succeeded`, `model_failed`, `parse_failed`,
  `moderation_blocked`, `canceled`
- `active`
- `output_schema`
- `public_output_json`
- `raw_output_debug_json`
- `parser_diagnostics_json`
- `moderation_diagnostics_json`
- `model_metadata_json`
- `usage_metadata_json`
- `source`: `model`, `literal`, `regenerated`
- `created_at`

At most one `active=true` successful revision per
`(owner_user_id, session_id, generation_point_key)`.

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
    "archive:3:map-clue": 31
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

Uses the repository's standard pagination envelope and metadata conventions.

Suggested filters:

- `generation_id`
- `generation_point_key`
- `status`
- `active`
- `source`
- `created_after`
- `created_before`
- `limit`
- `cursor` or standard project pagination equivalent

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
    "narrative": ["The map trembles in Mira's hand."],
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
    "profile_id": "story_default",
    "snapshot_id": 44,
    "provider_class": "hosted",
    "moderation_required": true,
    "estimated_cost_class": "low"
  },
  "created_at": "2026-05-10T19:00:00Z"
}
```

### Revision Debug Detail

`GET /sessions/{session_id}/script/generations/{revision_id}/debug`

Owner/admin-only. This is separate from the normal history endpoint so raw model
output, prompts, parser diagnostics, and moderation details are never returned
accidentally in list views.

Debug detail includes:

- raw model output;
- parser diagnostics;
- moderation diagnostics;
- prompt fingerprint and redaction status;
- generation profile/model metadata;
- usage/accounting metadata;
- request/revision lineage.

Moderation-blocked raw output should require an explicit second confirmation in
the WebUI before calling/revealing this endpoint. API access still requires
normal authentication and should be audit-loggable if an existing sensitive-read
audit path is available.

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
- `generation_request_id`
- `generation_revision_id`
- `generation_profile_snapshot_id`
- `generation_point_key`

The API should not introduce a VN-only usage ledger unless existing accounting
cannot carry this metadata.

## Runtime Flow

### Automatic Generation

1. Client calls `POST /script/advance`.
2. Runtime validates scene version and action idempotency.
3. Interpreter reaches a non-confirmation generation opcode.
4. Runtime creates a checkpoint.
5. Runtime creates generation request and marks it `in_progress`.
6. Runtime calls the selected provider through the pinned profile snapshot.
7. Runtime validates output schema.
8. Runtime runs moderation if required.
9. Runtime writes immutable revision and marks it active if successful.
10. Runtime appends generation events, updates script position/scene state, and
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
4. Runtime creates a new generation request/revision.
5. Successful revision becomes active.
6. Public state updates for that generation point only.

### Revision Activation

1. Client calls activate with generation/revision IDs.
2. Runtime validates revision is `succeeded`.
3. Runtime checks downstream generated-choice dependency guard.
4. Runtime creates checkpoint.
5. Runtime flips active revision in a transaction.
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
- generation request/revision services;
- interpreter integration and generation batching;
- confirmation/cancel/regenerate/activate endpoints;
- history/debug endpoints with standard pagination;
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
- `generation_canceled`
- `generation_batch_limit`
- `generation_profile_unavailable`
- `provider_unavailable`
- `model_timeout`
- `model_failed`
- `parse_failed`
- `moderation_blocked`
- `revision_not_found`
- `revision_not_succeeded`
- `revision_activation_blocked`
- `generated_choice_dependency_exists`
- `debug_payload_forbidden`

Existing errors such as `stale_scene_version`, `turn_in_progress`,
`idempotency_key_conflict`, and `scripted_story_required` remain unchanged.

## Security And Safety

- Debug output is never returned by default list/state endpoints.
- Hosted/public profiles must run configured moderation before activation.
- Moderation-blocked output is never activatable.
- Generated visual directives must resolve through approved owned assets.
- Raw prompt/provider secrets are not stored in public output.
- Sensitive debug reads should use the existing audit path where available.
- Idempotency keys must be scoped to session/user and compared by payload hash.

## Open Questions For Implementation

1. Which existing LLM call abstraction should scripted generation use for the
   first backend PR: the same adapter used by Story mode or a narrower service
   wrapper around the provider manager?
2. Which existing standard pagination schema should `GET /script/generations`
   reuse exactly?
3. Is there an existing moderation service contract that can be called
   synchronously for hosted/public profile output, or should PR 1 define a
   fail-closed adapter seam with tests?
4. Which audit path should debug revision reads use in single-user mode, where
   audit logging may be optional?

These are implementation-plan questions, not design blockers.
