# BYOK (Bring Your Own Keys) User Guide

BYOK lets multi-user deployments store per-user provider keys, with optional org/team shared keys, while keeping server defaults available as a fallback.

## Requirements

- `AUTH_MODE=multi_user`.
- `BYOK_ENABLED=true`.
- `BYOK_ENCRYPTION_KEY` set (base64-encoded 32-byte key).
- Optional: `BYOK_ALLOWED_PROVIDERS` allowlist (comma-separated).
- Optional: `BYOK_ALLOWED_BASE_URL_PROVIDERS` allowlist for BYOK `base_url` overrides.
- Optional: `BYOK_LAST_USED_THROTTLE_SECONDS` (default `300`) to throttle runtime `last_used_at` updates.

Example key generation:

```bash
python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
```

## OpenAI OAuth First-Time Setup

For the first-time OpenAI account-linking flow (operator prerequisites + user walkthrough), see:

- [OpenAI OAuth First Time Setup](./OpenAI_OAuth_First_Time_Setup.md)

## Anthropic + Claude Code/SDK Setup

For Anthropic setup with Claude Code/Claude SDK and current support limits (API-key based today, no Anthropic OAuth BYOK flow yet), see:

- [Anthropic Claude SDK Setup](../Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md)

## Provider Allowlist

BYOK respects an allowlist at both write time and runtime:

- If a provider is **not** allowlisted, new BYOK keys are rejected and stored keys are ignored at runtime.
- Server default keys still apply when BYOK is disallowed.
- `/api/v1/users/keys` shows stored-but-disallowed providers as `source=disabled`.

Default allowlist (commercial providers):

```
anthropic, bedrock, cohere, custom-openai-api, custom-openai-api-2, deepseek,
elevenlabs, google, groq, huggingface, mistral, moonshot, openai, openrouter,
qwen, voyage, zai
```

## Credential Resolution Order

Resolution is per request and normalized to lower-case provider names:

1. User BYOK key
2. Team shared key (if any)
3. Org shared key (if any)
4. Server default key (from `.env` or `Config_Files/config.txt`)

If the provider requires auth and no key resolves, the API returns:

```json
{
  "error_code": "missing_provider_credentials",
  "message": "Provider '<name>' requires an API key."
}
```

## Credential Fields

Credential fields are validated per provider. By default, unknown providers only
allow `org_id` and `project_id`, and `base_url` is only allowed for providers
listed in `BYOK_ALLOWED_BASE_URL_PROVIDERS` and when the caller is a trusted
principal (admin/service).

Notes:

- Empty strings are rejected.
- Allowed fields must be provided explicitly; they are not inherited from server defaults.
- `base_url` overrides the provider `api_base_url` when permitted.
- `org_id` / `project_id` are applied when not `null`.
- Unsupported fields return `400`.

## User Endpoints

Create or update a key:

```
POST /api/v1/users/keys
```

Request:

```json
{
  "provider": "openai",
  "api_key": "sk-...",
  "credential_fields": { "base_url": "https://api.openai.com/v1" },
  "metadata": { "label": "personal" }
}
```

Response:

```json
{
  "provider": "openai",
  "status": "stored",
  "key_hint": "1234",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

List key status:

```
GET /api/v1/users/keys
```

Response item fields:

- `source`: `user | team | org | server_default | none | disabled`
- `has_key`: `true` if the user stored a key

Test a key:

```
POST /api/v1/users/keys/test
```

Delete a key:

```
DELETE /api/v1/users/keys/{provider}
```

## Org/Team Shared Keys (Manager Role)

Org/team owners, admins, and leads can manage shared keys. Global admins are allowed.

Org endpoints:

```
POST   /api/v1/orgs/{org_id}/keys/shared
GET    /api/v1/orgs/{org_id}/keys/shared
POST   /api/v1/orgs/{org_id}/keys/shared/test
DELETE /api/v1/orgs/{org_id}/keys/shared/{provider}
```

Team endpoints:

```
POST   /api/v1/teams/{team_id}/keys/shared
GET    /api/v1/teams/{team_id}/keys/shared
POST   /api/v1/teams/{team_id}/keys/shared/test
DELETE /api/v1/teams/{team_id}/keys/shared/{provider}
```

Shared keys apply to all members in that org/team and show as `source=shared` in user listings.

## Admin Endpoints

Admins can list and revoke user keys, and manage shared keys across org/team scopes:

```
GET    /api/v1/admin/keys/users/{user_id}
DELETE /api/v1/admin/keys/users/{user_id}/{provider}

POST   /api/v1/admin/keys/shared
GET    /api/v1/admin/keys/shared
POST   /api/v1/admin/keys/shared/test
DELETE /api/v1/admin/keys/shared/{scope_type}/{scope_id}/{provider}
```

## Auditing and Updates

- `last_used_at` updates on successful provider calls.
- Runtime updates are throttled by `BYOK_LAST_USED_THROTTLE_SECONDS`.
- `keys/test` endpoints update `last_used_at` only if the tested key matches the stored key.
- Responses never include plaintext keys; `key_hint` shows the last 4 characters only.

## Key Rotation

Rotate BYOK encryption keys using the maintenance helper:

1. Set `BYOK_ENCRYPTION_KEY` to the new primary key.
2. Set `BYOK_SECONDARY_ENCRYPTION_KEY` to the old key.
3. Run the rotation helper (dry-run first recommended):

```bash
python -m Helper_Scripts.AuthNZ.rotate_byok_keys --dry-run
python -m Helper_Scripts.AuthNZ.rotate_byok_keys
```

After a successful run, remove `BYOK_SECONDARY_ENCRYPTION_KEY` to enforce the new key.

## Error Codes

- `400` invalid payloads.
- `503` missing provider credentials (`error_code=missing_provider_credentials`).
- `403` BYOK disabled or provider disallowed.
- `404` deleting a key that does not exist.
- `401/403` for `/keys/test` when provider rejects credentials.
- `502` for provider test-call failures.

## Metrics (Local-Only by Default)

Metrics appear in `/metrics` and `/api/v1/metrics` when metrics are enabled:

- `byok_resolution_total`
  - Labels: `provider`, `source`, `allowlisted`, `byok_enabled`
- `byok_missing_credentials_total`
  - Labels: `provider`, `operation`, `allowlisted`, `byok_enabled`

Metrics are local-only unless you configure external exporters.
