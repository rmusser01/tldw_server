# VN Platform API Design

Status: Draft for review
Date: 2026-05-10
Owner: Core/WebUI maintainers
GitHub: #1391, #1486
Scope: Full backend-owned API contract for VN assets, authored scripts, play runtime, policy, and VN audio

## Summary

Define a full, resource-first VN platform API for tldw_server. The API serves
custom frontends and the bundled WebUI from a backend-owned contract rather than
requiring clients to reconstruct VN rules from internal tables, asset-pack
internals, or private `custom_action` conventions.

V1 covers:

- offline image asset packs and portability;
- authored VN scripts with immutable published versions;
- runtime sessions for `freeform`, model-generated `story`, and
  `scripted_story`;
- server-authoritative turns, events, branch navigation, checkpoints, and
  per-session save slots;
- configurable VN policy and generation profiles;
- VN-scoped TTS jobs for voice pre-generation and runtime on-demand voice;
- cross-namespace capabilities discovery.

Realtime image generation, multiplayer, session/script portability, public
marketplaces, subscriptions, collaborative editing, full media timelines, and
rich built-in gameplay systems are vNext.

## Goals

- Make the API server stand alone for VN/CYOA clients.
- Keep the backend authoritative for setup, policy, asset readiness, script
  validation, runtime replay, and visual/audio content access.
- Provide one canonical route family under `/api/v1/vn/vn-*`.
- Keep V1 implementable while reserving explicit vNext extension points.
- Preserve deterministic replay: script versions are immutable, asset manifest
  snapshots are pinned, model outputs are persisted, and random results are
  recorded.
- Use existing tldw_server auth, owner scoping, Jobs, generated-file storage,
  and VN asset/runtime foundations.

## Non-Goals

- No realtime image generation during play in V1.
- No session or script export/import in V1.
- No public or cross-user marketplace in V1.
- No multiplayer/co-op sessions in V1.
- No SSE, WebSocket, webhook, or subscription API in V1.
- No first-class text DSL authoring source in V1.
- No CRDT or collaborative editor semantics in V1.
- No multiple primary VN asset packs per published script version in V1.

## Namespace, Versioning, And Compatibility

Canonical namespace:

```text
/api/v1/vn/vn-capabilities
/api/v1/vn/vn-assets
/api/v1/vn/vn-scripts
/api/v1/vn/vn-play
/api/v1/vn/vn-policy
/api/v1/vn/vn-audio
```

The global API version remains `/api/v1`. Payloads carry schema versions where
payload shape must remain independently evolvable, for example
`vn_capabilities.v1`, `vn_script_program.v1`, or `vn_asset_manifest.v1`.

The currently shipped `/api/v1/vn-assets` and `/api/v1/vn-play` paths are
superseded in the target design by the canonical `/api/v1/vn/vn-*` route family.
The design intentionally documents a breaking route migration only. It does not
include deprecated aliases in the V1 target API.

## Authentication And Ownership

V1 reuses existing tldw_server auth:

- single-user mode: `X-API-KEY`;
- multi-user mode: JWT bearer auth.

All non-admin VN resources are owner-scoped to the authenticated user. VN asset
metadata, script metadata, play sessions, runtime events, scene state, branches,
checkpoints, save slots, and VN TTS metadata live in the user's normal data
boundary. Image/audio bytes live in generated-file storage and are served through
authenticated VN content endpoints.

Admin-only VN policy and generation-profile CRUD uses existing admin/RBAC checks.
Normal users can list usable policy and generation profiles but cannot mutate
global profiles.

## Response And Error Conventions

Successful responses use direct resource shapes. Endpoints add `warnings`,
`pagination`, `job`, or `meta` fields only where relevant.

Errors use FastAPI-compatible error transport, but VN endpoint `detail` payloads
must be stable objects:

```json
{
  "detail": {
    "code": "idempotency_key_conflict",
    "message": "The idempotency key was already used with a different payload.",
    "details": {
      "resource_type": "vn_play_session",
      "resource_id": 42
    },
    "retryable": false
  }
}
```

Important stable codes include:

- `not_found`
- `permission_denied`
- `validation_failed`
- `policy_blocked`
- `idempotency_key_required`
- `idempotency_key_conflict`
- `stale_scene_version`
- `turn_in_progress`
- `action_request_abandoned`
- `restore_action_in_progress`
- `draft_revision_conflict`
- `script_publish_validation_failed`
- `script_runtime_error`
- `job_not_cancellable`
- `cleanup_blocked`
- `content_unavailable`

Warning payloads should be frontend-safe and never include stack traces or
private prompt content.

## Cross-Cutting Contracts

### Idempotency

Mutating VN commands that create work, publish, or advance state require an
`idempotency_key` in the JSON body or multipart form fields.

Required examples:

- asset generation, export/import, cleanup execution;
- image item upload and import-preview archive upload;
- script publish;
- Story start;
- runtime turns and script commands;
- checkpoint/branch/save-slot restore;
- save-slot creation;
- VN TTS job creation.

The backend normalizes the request body, hashes the payload, and scopes keys by
owner plus resource/action. A duplicate key with the same payload replays the
stored response, current job status, or completed action result. The same key with
a different payload returns `409 idempotency_key_conflict`.

Multipart endpoints carry `idempotency_key` as a form field. Their payload hash
includes canonical form fields, file name where relevant, declared content type,
file size, and a streaming SHA-256 of uploaded bytes. Replaying the same item
upload key returns the existing draft item instead of creating a duplicate.
Replaying the same import-preview key returns the existing preview/job status.
Changing metadata or bytes under the same key returns
`409 idempotency_key_conflict`.

### Runtime Action Requests

Interactive runtime commands are not Jobs, but they still need durable request
state because clients retry slow turns and HTTP connections can close while a
model or script command is running.

Before executing a runtime command, the backend creates a per-session action
request row keyed by `(owner_user_id, session_id, idempotency_key)` with:

- request kind: `story_start`, `turn`, `script_advance`, `script_choice`,
  `script_regenerate`, `checkpoint_restore`, `branch_restore`,
  `save_slot_restore`;
- normalized payload hash;
- `client_scene_version` and starting `scene_version`;
- status: `pending`, `running`, `model_failed`, `parse_failed`,
  `runtime_failed`, `completed`, or `abandoned`;
- lease/heartbeat fields for active execution;
- stored response or stable error payload once terminal.

Only one non-terminal action request can hold the per-session runtime lease.
Concurrent commands return `409 turn_in_progress` or the more specific restore
conflict code with the active request ID and retry guidance. Stale
`client_scene_version` values return `409 stale_scene_version` before model or
script execution starts.

If the process crashes or the lease expires before a terminal result is written,
reads reconcile the row to `abandoned`. A duplicate submission with the same key
may atomically reacquire the lease and resume only when the operation is
explicitly marked safe to resume before side effects. Otherwise it returns
`409 action_request_abandoned` with instructions to refresh the session and submit
a new key if the user still wants the action. A duplicate completed key always
replays the stored response. A duplicate failed key replays the stored failure
unless the endpoint is explicitly documented as retryable.

### Policy And Generation Profile Snapshots

Published script versions and runtime sessions must not depend on mutable admin
profile rows for deterministic replay.

At publish time, the backend resolves named policy and generation profiles into
effective immutable snapshots containing provider/model routing, bounds,
structured-output requirements, content-rating constraints, persistence/quota
rules, and audit settings. The published script stores snapshot IDs and the
profile names/versions they came from.

At session creation, the backend records the effective runtime policy and
generation profile snapshots actually used by the session, including permitted
session overrides. Later admin edits, disables, or deletes of profile definitions
do not mutate existing published script versions or sessions. New sessions and
new published script versions resolve the latest permitted profile versions.

### Pagination

List endpoints use offset pagination:

```text
limit
offset
total
has_more
```

Append-only runtime event endpoints may also support `after_sequence` for replay
efficiency. Offset pagination remains the V1 default for normal lists.

### Jobs

Long-running VN operations return a VN domain status plus a generic `job_id`.
Generic Jobs remains the admin/ops source of truth.

V1 job-backed operations:

- VN asset image generation;
- VN asset portability/import/export;
- VN asset cleanup execution;
- VN TTS generation.

Runtime model calls are synchronous HTTP request/response operations with timeout
and persisted failure state. Streaming and subscriptions are vNext.

### Content And Preview

VN images and audio are served through authenticated API content endpoints only.
The canonical API does not expose raw storage paths or presigned direct storage
URLs.

Preview endpoints may serve reduced-size images, low-bitrate audio, or metadata
optimized for picker and save-slot UI. Content endpoints serve the canonical
stored asset bytes subject to ownership, policy, and generated-file checks.

## Resource Overview

| Resource | Prefix | Purpose |
| --- | --- | --- |
| Capabilities | `/api/v1/vn/vn-capabilities` | Cross-namespace feature and limit discovery. |
| Assets | `/api/v1/vn/vn-assets` | Offline image packs, generation, upload, review, manifest, portability. |
| Scripts | `/api/v1/vn/vn-scripts` | Authored VN scripts, drafts, validation, immutable published versions. |
| Play | `/api/v1/vn/vn-play` | Freeform, Story, and Scripted Story runtime sessions. |
| Policy | `/api/v1/vn/vn-policy` | Policy preflight, policy profiles, generation profiles. |
| Audio | `/api/v1/vn/vn-audio` | Reserved VN-scoped TTS jobs and output content; deferred until the VN audio router is implemented. |

## VN Capabilities API

`GET /api/v1/vn/vn-capabilities`

Returns:

- `schema_version`
- `generated_at`
- canonical namespace paths
- enabled modules
- feature flags
- route migration note
- limits
- supported content ratings
- visible policy profiles
- visible generation profiles
- supported image and TTS output media types
- docs/OpenAPI links where available

Example:

```json
{
  "schema_version": "vn_capabilities.v1",
  "generated_at": "2026-05-10T00:00:00Z",
  "base_path": "/api/v1/vn",
  "resources": {
    "assets": "/api/v1/vn/vn-assets",
    "scripts": "/api/v1/vn/vn-scripts",
    "play": "/api/v1/vn/vn-play",
    "policy": "/api/v1/vn/vn-policy",
    "audio": "/api/v1/vn/vn-audio"
  },
  "enabled_modules": {
    "assets": true,
    "scripts": true,
    "play": true,
    "policy": true,
    "audio": false
  },
  "features": {
    "asset_generation": true,
    "asset_portability": true,
    "scripted_story": true,
    "story_start": true,
    "tts_jobs": false,
    "realtime_image_generation": false,
    "subscriptions": false
  },
  "limits": {
    "max_pack_items": 300,
    "max_slot_variants": 6,
    "max_choices_per_scene": 8,
    "runtime_model_timeout_seconds": 120
  },
  "route_migration": {
    "canonical": "/api/v1/vn/vn-*",
    "supersedes": ["/api/v1/vn-assets", "/api/v1/vn-play"]
  }
}
```

## VN Assets API

Canonical prefix: `/api/v1/vn/vn-assets`.

V1 asset packs remain image-focused. They support one primary character, offline
generation, uploads, review, approved-only runtime manifests, cleanup, and
portability. Audio assets are not added to VN asset packs in V1.

### Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/starter-matrices` | List built-in pack matrices. |
| `POST` | `/packs` | Create a pack. |
| `GET` | `/packs` | List owned packs. |
| `GET` | `/packs/{pack_id}` | Read one pack. |
| `PATCH` | `/packs/{pack_id}` | Update pack metadata. |
| `DELETE` | `/packs/{pack_id}` | Soft-delete pack metadata only. |
| `POST` | `/packs/{pack_id}/cleanup` | Dry-run or confirmed generated-file cleanup. |
| `POST` | `/packs/{pack_id}/matrix/apply` | Expand a starter matrix into slots. |
| `GET` | `/packs/{pack_id}/slots` | List slots. |
| `POST` | `/packs/{pack_id}/slots` | Create a custom slot. |
| `PATCH` | `/packs/{pack_id}/slots/{slot_id}` | Update a slot. |
| `DELETE` | `/packs/{pack_id}/slots/{slot_id}` | Delete a slot if safe. |
| `GET` | `/packs/{pack_id}/items` | List generated/imported/uploaded candidates. |
| `POST` | `/packs/{pack_id}/items/upload` | Upload an image candidate. |
| `PATCH` | `/packs/{pack_id}/items/{item_id}/review` | Set review state and preferred flag. |
| `POST` | `/packs/{pack_id}/items/bulk-review` | Apply review state to many items. |
| `POST` | `/packs/{pack_id}/items/{item_id}/preferred` | Mark one item preferred. |
| `GET` | `/packs/{pack_id}/items/{item_id}/preview` | Serve authenticated preview bytes. |
| `GET` | `/packs/{pack_id}/items/{item_id}/content` | Serve authenticated original bytes. |
| `GET` | `/packs/{pack_id}/manifest` | Return approved-only runtime manifest. |
| `GET` | `/packs/{pack_id}/readiness` | Check runtime readiness. |
| `POST` | `/packs/{pack_id}/prompt-preview` | Preview prompt assembly without generation. |
| `POST` | `/packs/{pack_id}/generate` | Enqueue parent generation batch. |
| `GET` | `/packs/{pack_id}/generation` | Read latest generation status. |
| `POST` | `/packs/{pack_id}/generation/cancel` | Cancel active generation. |
| `POST` | `/packs/{pack_id}/slots/{slot_id}/retry` | Retry one slot. |
| `POST` | `/packs/{pack_id}/items/{item_id}/regenerate` | Regenerate one item variant. |
| `POST` | `/packs/{pack_id}/export` | Create export job. |
| `GET` | `/portability/exports/{job_id}` | Read export status. |
| `GET` | `/portability/exports/{job_id}/download` | Download completed export. |
| `POST` | `/portability/exports/{job_id}/cancel` | Cancel export. |
| `POST` | `/import/previews` | Upload archive and create preview. |
| `GET` | `/import/previews/{preview_id}` | Read import preview. |
| `POST` | `/import/previews/{preview_id}/cancel` | Cancel preview job. |
| `DELETE` | `/import/previews/{preview_id}` | Delete preview record/upload. |
| `POST` | `/import/commit` | Commit reviewed import plan. |
| `GET` | `/portability/imports/{job_id}` | Read import status. |
| `POST` | `/portability/imports/{job_id}/cancel` | Cancel import. |

### Core Rules

- Generated and uploaded variants start as `draft`.
- Uploaded images and import archives must pass through the existing upload,
  media validation, storage registration, and generated-file tracking pipeline;
  VN endpoints must not write trusted bytes directly to runtime asset paths.
- Runtime manifests include approved items only.
- Draft, rejected, and hidden items are workbench-only.
- Cleanup never happens through `DELETE /packs/{pack_id}`.
- Cleanup dry-runs must report generated-file blockers from published manifest
  snapshots, active or historical sessions, checkpoints, save slots, branch
  restore targets, and persisted VN TTS outputs before execution.
- Confirmed cleanup can physically delete only unreferenced generated files.
  Referenced files are skipped with `cleanup_blocked` details unless a future
  admin-only retention override is explicitly designed.
- Generation enqueues one parent fanout job and gradually creates child variant
  jobs.
- Local image backends remain globally concurrency-gated by backend configuration.

### Generate Example

```json
{
  "slot_ids": [10, 11],
  "variant_count": 2,
  "options": { "priority": "normal" },
  "idempotency_key": "pack-7-generate-2026-05-10"
}
```

Response:

```json
{
  "batch_id": 7,
  "job_id": "job_123",
  "job_batch_id": "vn_assets:user:1:pack:7:batch:7",
  "status": "queued",
  "planned_count": 4,
  "enqueued_count": 0,
  "completed_count": 0,
  "failed_count": 0,
  "warnings": []
}
```

## VN Scripts API

Canonical prefix: `/api/v1/vn/vn-scripts`.

V1 scripts are first-class authored VN resources. Each script has one mutable
draft and immutable published versions. A published version pins:

- one primary VN asset pack;
- the approved manifest snapshot at publish time;
- script defaults and immutable effective policy/generation profile snapshots.

Play sessions pin a published script version. Draft edits do not affect running
or historical sessions.

### Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/templates` | List preview-safe starter templates. |
| `POST` | `/templates/{template_id}/scripts` | Create a script shell with a validated starter draft. |
| `POST` | `/scripts` | Create a script shell. |
| `GET` | `/scripts` | List owned scripts. |
| `GET` | `/scripts/{script_id}` | Read script metadata. |
| `PATCH` | `/scripts/{script_id}` | Update script metadata/status. |
| `DELETE` | `/scripts/{script_id}` | Soft-delete script. |
| `GET` | `/scripts/{script_id}/draft` | Read mutable draft. |
| `PUT` | `/scripts/{script_id}/draft` | Replace whole draft with `if_revision`. |
| `POST` | `/scripts/{script_id}/draft/validate` | Validate draft without publishing. |
| `GET` | `/scripts/{script_id}/draft/diagnostics` | Read author diagnostics. |
| `POST` | `/scripts/{script_id}/publish` | Validate and publish immutable version. |
| `GET` | `/scripts/{script_id}/versions` | List published versions. |
| `GET` | `/scripts/{script_id}/versions/{version_id}` | Read immutable version. |
| `GET` | `/scripts/{script_id}/versions/{version_id}/manifest-snapshot` | Inspect pinned manifest. |
| `POST` | `/scripts/{script_id}/versions/{version_id}/policy/evaluate` | Preflight a version. |

### Starter Templates

Starter templates are backend-owned catalog entries for custom frontends and the
bundled `/vn-scripts` WebUI. `GET /templates` returns only preview-safe metadata:
stable `id`, `label`, `description`, `category`, `recommended_content_rating`,
`required_capabilities`, `preview`, `default_title`, and
`default_description`. Catalog responses must not expose full draft JSON, raw
prompts, internal/debug fields, policy profile IDs, generation profile IDs, or
model/provider settings.

V1 built-ins are:

- `linear_scene`
- `authored_choices`
- `generated_choice_set`
- `scene_update`
- `confirm_gated_generation`

`POST /templates/{template_id}/scripts` creates a normal owned VN script and
stores the starter draft through the same draft replacement and diagnostics path
used by `PUT /scripts/{script_id}/draft`. The response includes both `script`
and `draft`; after creation, clients treat the script like any other authored
script. Validation and publishing remain server-side authority. Unknown template
IDs return `404 template_not_found`.

### Draft Save

Draft saves use whole-draft replacement in V1:

```json
{
  "if_revision": 12,
  "draft": {
    "schema_version": "vn_script_program.v1",
    "title": "Door Under The Archive",
    "primary_asset_pack_id": 7,
    "entry_label": "start",
    "variables": {
      "has_key": { "type": "boolean", "default": false, "public": true },
      "trust": { "type": "integer", "default": 0, "public": true }
    },
    "generation_defaults": {
      "profile_id": "story_default",
      "persist_model_outputs": true
    },
    "labels": {
      "start": [
        { "op": "set_background", "slot_key": "background.archive.default" },
        { "op": "narrate", "text": "The archive door hums in the dark." },
        {
          "op": "choice",
          "id": "door-choice",
          "choices": [
            { "id": "open", "text": "Open it", "target": "open_door" },
            { "id": "wait", "text": "Wait and listen", "target": "listen" }
          ]
        }
      ]
    }
  }
}
```

Stale saves return `409 draft_revision_conflict`.

### Validation

`POST /draft/validate` and `POST /publish` run the same validator. Validation
covers:

- JSON schema shape;
- entry label exists;
- jump and choice targets exist;
- visual slot keys exist in the selected approved manifest;
- BGM/SFX/voice media references are accessible where required;
- variable declarations, assignment types, and condition operands;
- unreachable labels;
- impossible or malformed conditions when statically detectable;
- unsafe or disallowed model-generation settings;
- policy/profile compatibility.

Example validation response:

```json
{
  "valid": false,
  "errors": [
    {
      "code": "jump_target_missing",
      "message": "Jump target label was not found.",
      "path": "$.labels.start[3].target",
      "details": { "target": "missing_label" }
    }
  ],
  "warnings": [
    {
      "code": "label_unreachable",
      "message": "Label is never reached from the entry label.",
      "path": "$.labels.secret"
    }
  ]
}
```

### Publish

Publish request:

```json
{
  "draft_revision": 13,
  "label": "v1",
  "idempotency_key": "publish-script-42-v1"
}
```

Publish response:

```json
{
  "script_id": 42,
  "version_id": 9,
  "version_number": 1,
  "status": "published",
  "asset_pack_id": 7,
  "manifest_snapshot_id": "manifest_snapshot_9",
  "policy_snapshot_id": "policy_snapshot_9",
  "generation_profile_snapshot_id": "generation_snapshot_9",
  "validation": { "valid": true, "errors": [], "warnings": [] },
  "created_at": "2026-05-10T00:00:00Z"
}
```

### Canonical Script Program

Canonical format is JSON opcodes. A future text DSL may compile into this format,
but V1 API truth is JSON.

Opcode groups:

- Flow: `label`, `jump`, `choice`, `return`, `end`.
- Text: `say`, `narrate`.
- Variables: `set`, `increment`.
- Conditions: structured `all`, `any`, `not`, and comparison objects.
- Visuals: `set_background`, `show_sprite`, `hide_sprite`, `show_cg`,
  `clear_visuals`, transition hints.
- Audio hooks: `play_bgm`, `stop_bgm`, `play_sfx`, `voice_cue`.
- Model generation: generation-capable opcodes or scene blocks using named
  generation profiles. Any scene may request model expansion when allowed by the
  script version defaults, session overrides, and active policy profile.
- Random: seeded random opcodes whose resolved values are persisted in runtime
  events.

Visual opcodes reference assets by slot key, not item ID. Published versions pin
the manifest snapshot so slot-key resolution remains deterministic for that
version.

Script variables are strongly typed and declared in the program schema. V1
scripted stories support choices only for player input; free text remains in
Freeform and model-generated Story modes.

Conditions are canonical structured objects:

```json
{
  "all": [
    { "var": "has_key", "op": "eq", "value": true },
    { "var": "trust", "op": "gte", "value": 2 }
  ]
}
```

## VN Play Runtime API

Canonical prefix: `/api/v1/vn/vn-play`.

Session modes:

- `freeform`: open-ended character chat with VN presentation.
- `story`: model-generated Story/CYOA flow.
- `scripted_story`: authored script runtime pinned to a published script version.

### Setup And Session Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/setup-options` | Backend-owned setup selectors and warnings. |
| `POST` | `/sessions` | Create a session. |
| `GET` | `/sessions` | List sessions. |
| `GET` | `/sessions/{session_id}` | Read one session and current scene. |
| `PATCH` | `/sessions/{session_id}` | Update mutable metadata/settings. |
| `DELETE` | `/sessions/{session_id}` | Soft-delete session. |

`GET /setup-options` should cover characters, ready asset packs, published script
versions, policy warnings, defaults, and empty states. For `scripted_story`,
clients should be able to discover script/version readiness without calling
script internals.

`POST /sessions` rules:

- `freeform` and `story` require character plus asset pack.
- `scripted_story` requires `script_version_id`.
- `scripted_story` inherits the script version's pinned manifest snapshot.
- Policy evaluation runs before creation and may block or warn.

### Freeform And Model Story Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/sessions/{session_id}/story/start` | Start fresh model Story session. |
| `POST` | `/sessions/{session_id}/turn` | Freeform text or model Story choices/actions. |
| `POST` | `/sessions/{session_id}/retry-last-turn` | Retry failed accepted turn. |

Story start is a stable backend-owned command for fresh `story` sessions. It
replaces private `custom_action` startup conventions.

Story start request:

```json
{
  "client_scene_version": 0,
  "idempotency_key": "story-42-start"
}
```

The backend builds the opening input from session settings, title, character,
world-book context, seed, policy, and asset-pack state. The request does not
expose private prompt plumbing.

Story start response uses the normal turn response shape:

```json
{
  "turn_request_id": 5,
  "status": "completed",
  "scene_version": 1,
  "replayed": false,
  "session": { "id": 42, "mode": "story", "scene_version": 1 },
  "current_scene": {
    "session_id": 42,
    "scene_version": 1,
    "location_key": "archive",
    "visible_choices": [
      { "id": "open", "text": "Open the archive door" },
      { "id": "wait", "text": "Wait and listen" }
    ],
    "warnings": []
  },
  "events": [],
  "warnings": []
}
```

Story start errors:

- `story_start_not_allowed`
- `story_already_started`
- `stale_scene_version`
- `turn_in_progress`
- `idempotency_key_conflict`
- `model_failed`
- `parse_failed`

Starting a Story session does not create a branch node. Branch nodes are created
when the user selects a visible choice.

### Scripted Story Runtime Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/sessions/{session_id}/script/advance` | Execute until next stop point. |
| `POST` | `/sessions/{session_id}/script/choices/{choice_id}` | Select visible scripted choice. |
| `POST` | `/sessions/{session_id}/script/regenerate` | Explicitly regenerate a model expansion/fork. |
| `GET` | `/sessions/{session_id}/script/state` | Read public interpreter state. |
| `GET` | `/sessions/{session_id}/script/debug-state` | Owner-gated diagnostics. |

Script runtime commands require `client_scene_version` and `idempotency_key`.
Runtime commands are synchronous request/response. They persist events and
failures before returning.

`POST /script/advance` executes from the current command cursor until one of:

- visible choice;
- model output completion;
- save boundary;
- script end;
- runtime error;
- configured command limit.

Public script state exposes:

- spoiler-safe progress token;
- optional public scene/chapter key;
- public variables only;
- visible choices;
- current visual/audio cues;
- current scene state;
- script version metadata.

It does not expose raw labels, command indices, private variables, hidden branch
conditions, or interpreter stack details. `debug-state` can expose current label,
command cursor, stack frames, last opcode, and author diagnostics behind
ownership/admin checks.

Model expansions inside scripts may occur from any scene when allowed by policy
and generation profile. Generated narration, dialogue, choices, and scene beats
are persisted and replayed by default. Explicit regeneration creates a new fork
or event lineage and never silently rewrites history.

### Shared Runtime Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/sessions/{session_id}/events` | Ordered event history, optionally branch-filtered. |
| `POST` | `/sessions/{session_id}/checkpoint` | Create low-level checkpoint. |
| `GET` | `/sessions/{session_id}/checkpoints` | List checkpoints. |
| `POST` | `/sessions/{session_id}/restore` | Restore checkpoint. |
| `GET` | `/sessions/{session_id}/branches` | Compatibility branch rows. |
| `GET` | `/sessions/{session_id}/branch-navigation` | Backend-derived branch navigation. |
| `POST` | `/sessions/{session_id}/branches/{branch_id}/restore` | Restore branch target. |
| `POST` | `/sessions/{session_id}/save-slots` | Create user-facing save slot. |
| `GET` | `/sessions/{session_id}/save-slots` | List session save slots. |
| `GET` | `/sessions/{session_id}/save-slots/{slot_id}` | Read save slot. |
| `PATCH` | `/sessions/{session_id}/save-slots/{slot_id}` | Update label/metadata. |
| `DELETE` | `/sessions/{session_id}/save-slots/{slot_id}` | Delete save slot. |
| `POST` | `/sessions/{session_id}/save-slots/{slot_id}/restore` | Restore save slot. |

Checkpoints are low-level restore points. Save slots are user-facing
per-session snapshots built on checkpoint semantics and may include label,
thumbnail/preview references, scene summary, branch/script position, and created
timestamp.

Save slot create:

```json
{
  "label": "Before opening the archive",
  "scene_version": 6,
  "idempotency_key": "session-42-save-before-door"
}
```

## VN Policy API

Canonical prefix: `/api/v1/vn/vn-policy`.

V1 policy is configurable. Local/self-hosted defaults can rely on metadata,
trust, review, and audit gates. Hosted or multi-user deployments can enable
stricter moderation profiles.

### Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/evaluate` | Evaluate pack/script/session/action. |
| `GET` | `/profiles` | List usable policy profiles. |
| `GET` | `/profiles/{profile_id}` | Read policy profile. |
| `POST` | `/profiles` | Admin create profile. |
| `PATCH` | `/profiles/{profile_id}` | Admin update profile. |
| `DELETE` | `/profiles/{profile_id}` | Admin disable/delete profile. |
| `GET` | `/generation-profiles` | List usable generation profiles. |
| `GET` | `/generation-profiles/{profile_id}` | Read generation profile. |
| `POST` | `/generation-profiles` | Admin create generation profile. |
| `PATCH` | `/generation-profiles/{profile_id}` | Admin update generation profile. |
| `DELETE` | `/generation-profiles/{profile_id}` | Admin disable/delete generation profile. |

`POST /evaluate` lets custom frontends preflight asset packs, script drafts,
published script versions, session setup payloads, runtime actions, and TTS
requests.

Example:

```json
{
  "target_type": "script_draft",
  "target_id": 42,
  "policy_profile_id": "local_default",
  "context": {
    "content_rating": "general",
    "generation_profile_id": "story_default"
  }
}
```

Response:

```json
{
  "decision": "warn",
  "profile_id": "local_default",
  "reasons": [
    {
      "code": "character_safety_unknown",
      "severity": "warning",
      "message": "Character safety metadata is unknown.",
      "requires_acknowledgement": true
    }
  ],
  "blocked": false,
  "requires_acknowledgement": true,
  "remediation": [
    "Add character safety metadata or acknowledge the warning for this session."
  ]
}
```

### Character Safety Metadata

Policy profiles define fail-open, warn, or fail-closed behavior for absent,
unknown, conflicting, or imported character safety metadata. V1 defaults are:

| Situation | `general` local default | `suggestive`/`mature` local default | hosted/strict profile |
| --- | --- | --- | --- |
| Missing age/status metadata | Warn and require acknowledgement | Block until metadata is completed | Block |
| Unknown or ambiguous metadata | Warn and require acknowledgement | Block until clarified | Block |
| Conflicting card/import metadata | Block | Block | Block |
| Imported metadata without trusted provenance | Warn and require acknowledgement | Warn or block by profile | Block unless trusted |

The policy decision records the metadata source, trust level, acknowledgement
requirement, and any user/admin acknowledgement ID. Runtime and publish endpoints
repeat authoritative policy evaluation; `/evaluate` is advisory preflight only.

Generation profiles include:

- provider/model routing;
- structured-output capability;
- temperature and token defaults/bounds;
- allowed content ratings;
- trust and review requirements;
- max choices and branch depth;
- max model expansion scope;
- TTS permission and persistence bounds;
- audit/logging mode.

Scripts reference named generation profile IDs, not raw provider/model strings.

## VN Audio API

Canonical prefix: `/api/v1/vn/vn-audio`.

V1 reserves this namespace for VN-scoped TTS, but the current implementation does
not register the VN audio router. Capabilities discovery must report
`enabled_modules.audio=false` and `features.tts_jobs=false` until these endpoints
exist. Clients must not call `/api/v1/vn/vn-audio` unless capabilities reports it
enabled; they should use the existing `/api/v1/audio` endpoints for generic TTS.

When implemented, this namespace is TTS-only. BGM and SFX cues reference existing
media/generated-file IDs from script opcodes and do not need a VN audio manager.

### Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/tts/jobs` | Create authoring or runtime TTS job. |
| `GET` | `/tts/jobs` | List VN TTS jobs. |
| `GET` | `/tts/jobs/{job_id}` | Read job status. |
| `POST` | `/tts/jobs/{job_id}/cancel` | Cancel job. |
| `GET` | `/tts/outputs/{output_id}` | Read output metadata. |
| `GET` | `/tts/outputs/{output_id}/preview` | Serve authenticated preview. |
| `GET` | `/tts/outputs/{output_id}/content` | Serve authenticated content. |
| `DELETE` | `/tts/outputs/{output_id}` | Soft-delete or detach output. |

Create job request:

```json
{
  "scope": "script_pregen",
  "script_id": 42,
  "script_version_id": 9,
  "line_ref": {
    "label": "start",
    "command_index": 1,
    "speaker": "Mira"
  },
  "text": "The archive door is awake.",
  "voice_profile_id": "mira_default",
  "generation_profile_id": "story_voice_default",
  "persist_output": true,
  "idempotency_key": "script-42-v9-voice-start-1"
}
```

Response:

```json
{
  "job_id": "job_456",
  "status": "queued",
  "scope": "script_pregen",
  "persist_output": true,
  "output_id": null,
  "warnings": []
}
```

Outputs can be transient or persisted subject to generation profile policy.
Persisted outputs become generated-file records with VN metadata. Transient
outputs are still served through authenticated content endpoints while available.

## Data Ownership And Storage Boundaries

- VN asset pack metadata: per-user `ChaChaNotes.db`.
- VN script metadata, drafts, versions, manifest snapshots: per-user
  `ChaChaNotes.db` through a VN scripts repository/module.
- VN play sessions, events, scene state, branches, checkpoints, save slots:
  per-user `ChaChaNotes.db`.
- VN TTS job/output metadata: per-user `ChaChaNotes.db` through a VN audio
  repository/module.
- Generated image and audio bytes: AuthNZ generated-file storage.
- Policy/generation profile global definitions: admin-owned config/database
  boundary consistent with existing admin/RBAC infrastructure.
- Immutable policy/generation profile snapshots used by published scripts and
  sessions: per-user `ChaChaNotes.db` when tied to user-owned runtime state;
  global/admin snapshots remain in the admin-owned boundary.

Cross-store references to generated files are application-validated. VN APIs must
validate owner, source feature, media type, and policy before serving content.
Retention and cleanup code must treat published manifest snapshots, sessions,
checkpoints, save slots, and persisted audio output rows as live references until
the owning metadata is deleted according to the feature's retention policy.

## End-To-End Flows

### Offline Assets To Model Story

1. Client calls `GET /api/v1/vn/vn-capabilities`.
2. Client creates pack under `/api/v1/vn/vn-assets/packs`.
3. Client applies matrix, previews prompts, starts generation, polls generation.
4. Client reviews and approves items.
5. Client checks readiness and manifest.
6. Client calls `/api/v1/vn/vn-play/setup-options`.
7. Client creates `story` session.
8. Client calls `/api/v1/vn/vn-play/sessions/{session_id}/story/start`.
9. Client submits visible choices through `/api/v1/vn/vn-play/sessions/{session_id}/turn`.
10. Client uses branch navigation, checkpoints, and save slots as needed.

### Authored Script To Scripted Story

1. Client creates script shell.
2. Client replaces draft with canonical JSON.
3. Client validates and reads diagnostics.
4. Client publishes version, pinning manifest snapshot.
5. If `GET /api/v1/vn/vn-capabilities` reports `features.tts_jobs=true`, the
   client optionally pre-generates TTS lines through `/api/v1/vn/vn-audio`;
   otherwise it skips VN-scoped TTS and may use generic `/api/v1/audio`.
6. Client creates `scripted_story` session with `script_version_id`.
7. Client advances script runtime until choices/end.
8. Client selects choices through script-specific endpoints.
9. Client restores checkpoints/save slots or explicitly regenerates model
   expansions when desired.

### Policy Preflight

1. Client evaluates candidate setup/script/audio request through `/api/v1/vn/vn-policy/evaluate`.
2. Client displays warnings/acknowledgement requirements.
3. Mutating endpoint repeats authoritative policy evaluation.

## Testing And Verification Plan

Design verification:

- Markdown path/link sanity for referenced docs and endpoints.
- `git diff --check`.
- Bandit is not applicable for this docs-only spec; record skip rationale.

Implementation plans cut from this spec should include:

- schema tests for request/response/error shapes;
- API tests for idempotency, including multipart upload byte-hash conflicts,
  request replay, conflicts, auth, ownership, and pagination;
- script validation unit/property tests;
- runtime replay tests for model output persistence and seeded random results;
- runtime action-request recovery tests for duplicate in-flight requests,
  abandoned leases, stale scene versions, and stored terminal responses;
- publish/session snapshot tests proving admin profile edits do not mutate
  existing script versions or sessions;
- cleanup blocker tests for manifest snapshots, sessions, checkpoints, save
  slots, branch restore targets, and persisted VN TTS outputs;
- policy profile tests for allow/warn/block decisions;
- character safety metadata tests for missing, unknown, conflicting, and imported
  metadata across local-default and strict profiles;
- VN TTS job lifecycle tests;
- migration docs/OpenAPI checks for canonical `/api/v1/vn/vn-*` routes.

## Migration From Existing Routes

The target API design supersedes:

- `/api/v1/vn-assets`
- `/api/v1/vn-play`

with:

- `/api/v1/vn/vn-assets`
- `/api/v1/vn/vn-play`

This spec does not require temporary aliases or an old-to-new path mapping
endpoint. If implementation chooses to keep old routes during a transition, that
is compatibility work outside this target API design, requires explicit separate
approval, and should be documented separately.

## vNext Extension Map

vNext candidates:

- realtime image generation during play;
- session export/import;
- script export/import;
- public/community marketplace or cross-user sharing;
- multiplayer/co-op sessions;
- SSE/WebSocket subscriptions for sessions and jobs;
- full media timeline, lip sync, camera keyframes, animations, and audio mixing;
- built-in inventory, stats, relationship meters, timers, and achievements;
- first-class text DSL authoring source;
- patch-based or collaborative script editing;
- multiple VN asset packs per published script version;
- presigned direct storage URLs where a deployment explicitly supports them.

V1 reserves typed extension hooks for gameplay systems but does not define them as
built-in runtime resources.

## Risks And Open Questions

1. Route migration is breaking. Existing clients using `/api/v1/vn-assets` and
   `/api/v1/vn-play` need explicit migration documentation.
2. Script validation can grow quickly. V1 should keep validation strict for the
   canonical JSON model and avoid accepting free-form DSL text as source truth.
3. Model generation inside scripts risks nondeterministic replay. Persisted output
   and explicit regeneration/fork endpoints are mandatory, not optional.
4. Admin generation-profile CRUD touches broader provider policy. Implementation
   should reuse existing admin/RBAC patterns rather than invent VN-only roles.
5. TTS persistence can create storage pressure. Generation profiles need explicit
   persistence and quota constraints.
6. Save slots and checkpoints overlap. The API should keep checkpoints low-level
   and make save slots user-facing aliases/snapshots with separate labels/previews.
7. Content endpoints must never trust script/manifest references alone. They must
   validate ownership and generated-file metadata every time.
