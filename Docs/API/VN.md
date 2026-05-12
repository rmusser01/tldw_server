# VN API

The VN API is backend-owned and lives under `/api/v1/vn/vn-*` for custom
frontends. Existing compatibility routers may also expose `/api/v1/vn-play`,
but new clients should discover canonical paths from:

- `GET /api/v1/vn/vn-capabilities`

## Scripted Generation Runtime

Scripted VN play can pause at `generate` opcodes, ask the user to confirm or
cancel the pending generation, store generated revisions, and later regenerate
or activate a previous revision. All command endpoints require:

- `client_scene_version`: the scene version the client rendered.
- `idempotency_key`: a stable caller-generated key for safe retry.

Stale scene versions return `409 stale_scene_version`. Reusing an
idempotency key with different request payload returns
`409 idempotency_key_conflict`. Completed actions replay their stored response.

### Public Commands

- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/confirm`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/cancel`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/regenerate`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/activate`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations?limit=25&offset=0`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions?limit=25&offset=0`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}`

Public responses include stable `public_output`, applied visual summaries,
revision status, profile lineage, and pagination metadata. They do not include
raw prompts, raw model output, parser diagnostics, moderation diagnostics, or
provider payloads.

### Debug Detail

Debug detail is owner/admin-only:

- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/debug`

Moderation-blocked raw output is redacted by default. To reveal it, callers
must explicitly pass:

```text
include_blocked_raw=true&confirm=REVEAL_MODERATION_BLOCKED
```

Debug reads use owner/admin authorization and emit structured warning logs in
single-user or no-audit deployments.

### Example

```json
POST /api/v1/vn/vn-play/sessions/12/script/generation-requests/44/confirm
{
  "client_scene_version": 3,
  "idempotency_key": "session-12-confirm-44-v1"
}
```

```json
POST /api/v1/vn/vn-play/sessions/12/script/generations/7/regenerate
{
  "client_scene_version": 4,
  "idempotency_key": "session-12-generation-7-regenerate-1"
}
```

## Setup Metadata

`GET /api/v1/vn/vn-play/setup-options?mode=scripted_story` returns script
version options with generation metadata for custom frontends:

- `generation_profile_key`
- `generation_profile_snapshot_id`
- `generation_profile_snapshot_immutable`
- `provider_class`
- `max_automatic_generation_batch_count`
- `moderation_required`
- `estimated_cost_class`
- `supported_output_schemas`
- `dynamic_choice_support`
- `scene_update_support`
- `confirmation_required`

Setup warnings include missing or unavailable generation profile snapshots and
incompatible generated output schemas.
