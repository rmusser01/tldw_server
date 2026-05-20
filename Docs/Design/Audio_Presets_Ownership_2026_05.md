# Audio Presets Ownership Decision - May 2026

Status: Accepted for Phase 4 design. Implementation is gated to the preset CRUD stage.
Owner: Audio API + shared WebUI/extension UI.
Last verified against code: 2026-05-19.

## Decision

Reusable TTS and STT presets are per-user server state owned by the Audio API.
They live in the user's Media DB v2 database, under a new audio-domain preset
table, and are exposed through `/api/v1/audio/presets`.

Presets are configuration objects. They are not TTS history rows, STT transcript
rows, generated audio artifacts, transcript artifacts, provider credentials, or
comparison history. Deleting a preset removes only the reusable configuration and
must not delete generated outputs, transcript rows, TTS history, files, jobs, or
comparison runs.

## Evidence Anchors

- Audio routes are aggregated in
  `tldw_Server_API/app/api/v1/endpoints/audio/audio.py`, which includes sibling
  routers such as `audio_tts`, `audio_transcriptions`, `audio_history`,
  `audio_health`, and `audio_voices`.
- TTS history endpoints are already audio-owned in
  `tldw_Server_API/app/api/v1/endpoints/audio/audio_history.py`.
- TTS history persists per-user state through `get_media_db_for_user` and passes
  `str(request_user.id)` to Media DB runtime helpers.
- The Media DB already has a dedicated `tts_history` table and runtime helpers
  under `tldw_Server_API/app/core/DB_Management/media_db/runtime/`.
- The STT capability summary endpoint introduced for Phase 2B exposes model
  capability assumptions from the transcription catalog, health checks, provider
  adapters, and response schema support.

## Backend Owner And Namespace

The implementation owner is the audio endpoint package:

- Router: `tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py`
- Aggregator include: `tldw_Server_API/app/api/v1/endpoints/audio/audio.py`
- Schemas: `tldw_Server_API/app/api/v1/schemas/audio_presets.py` or a clearly
  separated section in `audio_schemas.py` if the project chooses to keep audio
  schemas in one module.
- DB runtime helpers:
  `tldw_Server_API/app/core/DB_Management/media_db/runtime/audio_preset_ops.py`

The route namespace is:

```http
GET /api/v1/audio/presets
POST /api/v1/audio/presets
PATCH /api/v1/audio/presets/{preset_id}
DELETE /api/v1/audio/presets/{preset_id}
POST /api/v1/audio/presets/{preset_id}/validate
```

List endpoints should filter by `kind`, `favorite`, and `is_default`. The
validate endpoint recomputes current readiness/capability state for a preset and
returns warnings; it must not mutate the preset.

## Principal And AuthNZ Rules

All preset endpoints resolve ownership from the authenticated request principal:

- Use `get_request_user` to identify the user.
- Use `get_media_db_for_user` for the user's Media DB v2 database.
- Pass `str(request_user.id)` to DB helpers.
- Do not accept `ownerUserId`, `user_id`, or tenant identifiers from the client.
- In single-user mode, the existing single-user principal owns the preset.
- In multi-user mode, the JWT/API-key principal owns the preset.
- Admin endpoints are not part of the first preset CRUD slice.

Endpoint dependencies should follow audio history's protected endpoint pattern:
`check_rate_limit` plus `TokenScopeGuard` with explicit endpoint ids such as:

- `audio.presets.list`
- `audio.presets.create`
- `audio.presets.update`
- `audio.presets.delete`
- `audio.presets.validate`

## DB Boundary

Store presets in the per-user Media DB v2 database. Do not store them in
ChaChaNotes, browser Dexie/local storage, TTS history, transcript rows, or the
evaluation preset tables.

Create an `audio_presets` table with this logical shape:

```sql
CREATE TABLE IF NOT EXISTS audio_presets (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    favorite BOOLEAN NOT NULL DEFAULT 0,
    is_default BOOLEAN NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL,
    capability_assumptions_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    deleted_at TEXT,
    version INTEGER NOT NULL DEFAULT 1
);
```

Recommended indexes:

- `(user_id, kind, deleted, updated_at DESC)`
- `(user_id, kind, favorite, deleted)`
- `(user_id, kind, is_default, deleted)`
- A unique active preset name per `(user_id, kind)`, enforced
  case-insensitively if the backend can do that safely across SQLite and
  PostgreSQL.

Only one active default preset should exist per `(user_id, kind)`. Enforce this
transactionally when setting `is_default=true`; a partial unique index is useful
where supported, but the endpoint must not rely on SQLite-only behavior.

## Preset Kind Model

Supported v1 kinds:

- `tts`
- `stt`

Reserved kind:

- `speech`

Do not ship `speech` as a first-slice default unless the UI has a real combined
workflow that applies both TTS and STT settings together. Reserving the kind
keeps the API extensible without forcing a mixed preset model into the initial
implementation.

### TTS Config

A TTS preset config may include:

- `provider`
- `model`
- `voice`
- `response_format`
- `speed`
- `lang_code`
- `target_sample_rate`
- `normalization_options`
- safe, allowlisted provider options

It must not include provider API keys, bearer tokens, OAuth tokens, raw secret
headers, or user-supplied base URLs that are not already represented by a
server-side configured provider id.

### STT Config

An STT preset config may include:

- `model`
- `language`
- `task`
- `response_format`
- `timestamp_granularities`
- `diarization`
- segmentation/chunking options exposed by the UI
- streaming/file mode preference when the UI exposes both as a reusable choice
- safe, allowlisted engine options

Prompt-like fields, hotwords, and vocabulary hints can be included only after a
privacy review because they may contain user content rather than reusable setup.

## Capability Assumptions

`capability_assumptions_json` stores the visible capability snapshot that helped
the user save the preset. It should include the capability value and source for
important fields such as health, installed/downloaded state, batch support,
streaming support, diarization, timestamps, segments, language coverage when
known, and response schema support.

This snapshot is advisory. Applying or validating a preset must recompute current
readiness against the live audio readiness/capability endpoints because models,
credentials, downloads, hardware, provider health, and browser support can drift.

## Browser TTS Rule

Browser TTS is a no-setup escape hatch, not a first-class server-backed provider.

The server must not advertise a Browser TTS preset as portable or server
generatable. If the UI allows saving a Browser TTS configuration to server
presets, the preset must be explicitly marked:

- `provider: "browser"`
- `browser_local: true`
- `requires_browser_revalidation: true`
- no server health or server voice guarantee

The preferred first slice is to keep Browser TTS presets local to the current
browser session unless there is a clear UX need to remember it server-side. If
server persistence is allowed, applying the preset must revalidate against the
current browser's Web Speech availability and available voices.

## Migration And Import/Export

Do not automatically migrate existing local comparison rows, TTS history entries,
STT transcript results, generated files, or extension-local settings into server
presets.

After CRUD exists, the UI may offer explicit actions:

- Save current TTS settings as preset.
- Save current STT settings as preset.
- Duplicate preset.
- Save settings from a comparison row as a preset.

Import/export is not part of v1 preset CRUD. Future chatbook or workspace export
may include user-selected presets, but exported presets must strip owner ids,
credentials, local artifact ids, and non-portable browser state. Browser TTS
exports must preserve the non-portable/revalidation flag if export is later
supported.

## Deletion Semantics

Preset deletion is soft delete in the first implementation:

- Set `deleted=true`.
- Set `deleted_at`.
- Exclude deleted presets from normal lists.
- Make deleted presets unavailable for apply/validate.
- Keep generated audio, transcript artifacts, history rows, and comparison rows
  untouched.

Hard purge can be added later as an admin or retention feature if the broader
Media DB retention model needs it.

## Rate Limit And Security Considerations

Preset endpoints should be protected as authenticated user-state endpoints:

- Rate limit list, create, update, delete, and validate.
- Enforce owner scoping on every DB query.
- Reject unknown preset kinds and unknown config keys.
- Enforce a bounded JSON payload size.
- Validate config through Pydantic before writing JSON.
- Never store provider secrets or browser-derived voice internals that cannot be
  safely reused.
- Return structured warnings for unavailable providers/models instead of
  silently rewriting the preset.

## Frontend API Client Responsibilities

The shared `apps/packages/ui` client should expose audio preset methods once the
backend exists:

- `listAudioPresets({ kind })`
- `createAudioPreset(payload)`
- `updateAudioPreset(id, patch)`
- `deleteAudioPreset(id)`
- `validateAudioPreset(id)`

Shared UI components should use a single hook, for example `useAudioPresets`, so
WebUI and extension surfaces do not fork behavior.

The UI must show:

- saved preset name
- kind
- provider/model/voice or engine/model summary
- favorite/default state
- last updated time
- current validation state and warnings
- whether a Browser TTS preset is local and needs browser revalidation

Applying a preset should fill the current form without starting TTS generation or
STT transcription automatically. Users should be able to edit the applied values
before running.

## WebUI And Extension Parity

The WebUI and browser extension must share the same preset API and shared UI
components. The extension should not define a separate local-only preset model
for TTS/STT, because the product decision is per-user server state.

When the server is reachable, both surfaces should list and apply the same user
presets. When the server is unreachable, the UI may keep the current unsaved form
state but should show that server presets are unavailable instead of silently
creating divergent local presets.

## Stage 7 Implementation Gate

Before implementing CRUD, confirm:

- Exact schema module placement.
- Exact Media DB migration pattern for both SQLite and PostgreSQL support.
- Maximum presets per user and maximum config JSON size.
- Whether `speech` remains reserved or ships as a real combined preset kind.
- Whether Browser TTS presets are excluded from server persistence in v1 or
  allowed with the non-portable flags above.

Stage 7 should update the implementation plan file list to use the audio
subpackage path `tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py`.
