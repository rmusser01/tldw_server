# VN Platform API

This document covers cross-cutting VN platform endpoints under `/api/v1/vn`.

## Capabilities

`GET /api/v1/vn/vn-capabilities` returns a route-aware discovery payload for custom frontends. Clients should use it before enabling optional VN modules instead of assuming every planned route is installed.

Important fields:

- `resources`: canonical VN namespace paths.
- `enabled_modules`: booleans derived from registered FastAPI routes.
- `features`: route-derived feature flags such as `asset_generation`, `scripted_story`, `story_start`, and `tts_jobs`.
- `limits`: pack, slot, choice, and runtime timeout bounds.
- `route_migration`: the canonical `/api/v1/vn/vn-*` namespace and legacy paths it supersedes.

Example:

```json
{
  "schema_version": "vn_capabilities.v1",
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
  }
}
```

`/api/v1/vn/vn-audio` is reserved for a future VN-scoped TTS module. Current clients must only call VN audio routes when both `enabled_modules.audio` and `features.tts_jobs` are `true`; otherwise use the existing `/api/v1/audio` APIs directly.

## VN Script Starter Templates

VN script starter templates are exposed under the existing scripts resource:

- `GET /api/v1/vn/vn-scripts/templates`
- `POST /api/v1/vn/vn-scripts/templates/{template_id}/scripts`

The catalog endpoint returns preview-safe metadata only. Each item includes a stable `id`, `label`, `description`, `category`, `recommended_content_rating`, `required_capabilities`, `preview`, `default_title`, and `default_description`. It intentionally omits full draft JSON, raw prompts, provider/model settings, and policy or generation profile overrides so custom frontends can display the catalog without becoming a second source of truth.

Built-in V1 template IDs are `linear_scene`, `authored_choices`, `generated_choice_set`, `scene_update`, and `confirm_gated_generation`.

Create-from-template accepts the same script metadata as normal script creation, including `primary_asset_pack_id`, optional policy/generation profile IDs, generation profile maps, and content rating. The response contains:

```json
{
  "script": { "id": 12, "title": "Linear Scene", "status": "draft" },
  "draft": {
    "script_id": 12,
    "revision": 1,
    "draft": { "schema_version": "vn_script_program.v1" },
    "diagnostics": { "valid": true, "errors": [], "warnings": [] }
  }
}
```

After creation, the script is a normal authored script. Clients should use the standard draft, validation, diagnostics, and publish endpoints for further editing.

## Policy Profiles

VN policy definitions are global server configuration stored in the AuthNZ database. This lets admins create one profile that is visible to all API clients and custom frontends. Per-resource effective policy snapshots remain in the owning user's ChaChaNotes database so script/session/asset history preserves the exact settings used at creation time.

### Built-in Profiles

- `local_default`: local/self-hosted default. Missing or ambiguous character safety metadata warns for general/teen content and blocks mature content.
- `strict_hosted`: fail-closed hosted profile. Missing, ambiguous, conflicting, or imported-untrusted character safety metadata blocks.
- `story_default`: default generation profile for structured VN story turns.

### Endpoints

- `GET /api/v1/vn/vn-policy/profiles`
- `GET /api/v1/vn/vn-policy/profiles/{profile_id}`
- `POST /api/v1/vn/vn-policy/profiles` admin only
- `PATCH /api/v1/vn/vn-policy/profiles/{profile_id}` admin only
- `DELETE /api/v1/vn/vn-policy/profiles/{profile_id}` admin only; disables metadata only
- `GET /api/v1/vn/vn-policy/generation-profiles`
- `GET /api/v1/vn/vn-policy/generation-profiles/{profile_id}`
- `POST /api/v1/vn/vn-policy/generation-profiles` admin only
- `PATCH /api/v1/vn/vn-policy/generation-profiles/{profile_id}` admin only
- `DELETE /api/v1/vn/vn-policy/generation-profiles/{profile_id}` admin only; disables metadata only
- `POST /api/v1/vn/vn-policy/evaluate`

List endpoints use offset pagination with the standard `pagination` object plus legacy top-level `limit`, `offset`, `total`, `has_more`, and `next_offset` fields.

## Policy Evaluation

`POST /api/v1/vn/vn-policy/evaluate` performs a non-mutating policy preflight for session setup, script draft, runtime turn, or TTS requests.

Request shape:

```json
{
  "target_type": "session_setup",
  "target_id": null,
  "policy_profile_id": "local_default",
  "context": {
    "content_rating": "general",
    "character_safety": {
      "metadata_status": "missing"
    }
  }
}
```

The V1 evaluator accepts target-less preflight. Requests with `target_id` fail closed with `target_resolution_unavailable` until each target type has an authoritative server-side owner resolver.

Omitted `character_safety` is treated as `missing`, not adult/allowed. Known metadata statuses are `adult`, `minor`, `missing`, `unknown_or_ambiguous`, `conflicting`, and `imported_untrusted`. `minor` metadata is allowed for non-mature content ratings and blocks mature ratings.

Runtime gates resolve built-in policy profile IDs server-side. Custom policy profile IDs must be evaluated from a resolved policy definition or immutable snapshot supplied by the session setup path; a custom ID without a resolved definition fails closed with `policy_profile_unresolved`.

Response shape:

```json
{
  "decision": "warn",
  "profile_id": "local_default",
  "target_type": "session_setup",
  "target_id": null,
  "blocked": false,
  "requires_acknowledgement": true,
  "reasons": [
    {
      "code": "character_safety_missing",
      "severity": "warning",
      "message": "Character safety metadata is missing.",
      "requires_acknowledgement": true
    }
  ],
  "remediation": [
    "Add character safety metadata or acknowledge the warning for this request."
  ]
}
```

All policy endpoints return stable VN error envelopes using codes such as `invalid_request`, `permission_denied`, and `not_found`.
