# Virtual Keys API

Virtual Keys are API keys with scoped access and LLM usage budgets. They can be associated with Organizations and Teams.

Base path: `/api/v1/admin`

## Create Virtual Key

POST `/api/v1/admin/users/{user_id}/virtual-keys`

Request body (application/json):
```json
{
  "name": "lab-chat-key",
  "description": "Ephemeral key for lab UI",
  "expires_in_days": 30,
  "org_id": 1,
  "team_id": 5,
  "allowed_endpoints": ["chat.completions", "embeddings"],
  "allowed_providers": ["openai"],
  "allowed_models": ["gpt-4o", "text-embedding-3-small"],
  "budget_day_tokens": 100000,
  "budget_month_tokens": 2000000,
  "budget_day_usd": 5.0,
  "budget_month_usd": 100.0
}
```

Response 200 (application/json):
```json
{
  "id": 42,
  "key": "tldw_R4nd0mlyG3n3rat3d...",
  "key_prefix": "tldw_R4nd0...",
  "name": "lab-chat-key",
  "scope": "read",
  "expires_at": "2025-12-01T00:00:00Z",
  "created_at": "2025-11-01T00:00:00Z",
  "message": "Store this key securely - it will not be shown again"
}
```

Notes:
- The `key` value is only returned once at creation/rotation time.
- `allowed_endpoints` supports the logical endpoint IDs listed below.

Example (cURL):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/users/123/virtual-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -d '{
    "name": "lab-chat-key",
    "allowed_endpoints": ["chat.completions", "embeddings"],
    "allowed_providers": ["openai"],
    "allowed_models": ["gpt-4o"],
    "budget_day_tokens": 100000,
    "budget_month_usd": 50.0
  }'
```

## List Virtual Keys for a User

GET `/api/v1/admin/users/{user_id}/virtual-keys`

Response 200 (application/json):
```json
[
  {
    "id": 42,
    "key_prefix": "tldw_R4nd0...",
    "name": "lab-chat-key",
    "description": "Ephemeral key for lab UI",
    "scope": "read",
    "status": "active",
    "created_at": "2025-11-01T00:00:00Z",
    "expires_at": "2025-12-01T00:00:00Z",
    "usage_count": 3,
    "last_used_at": "2025-11-02T10:00:00Z",
    "last_used_ip": "203.0.113.10"
  }
]
```

## Enforcement

- Endpoint allowlists: enforced for LLM endpoints (`/api/v1/chat/completions`, `/api/v1/embeddings`). Requests to disallowed endpoints receive 403.
- Budgets: evaluated from `llm_usage_log` daily/monthly. Exceeding a budget returns 402 with details. The response shape (`error`, `message`, `details`) is a stable contract.
- Optional allowlists: set `llm_allowed_providers` and/or `llm_allowed_models` (on the key via admin SQL for now). If present, middleware enforces provider via `X-LLM-Provider` header and model parsed from JSON request body when available.

```http
HTTP/1.1 402 Payment Required
Content-Type: application/json

{
  "error": "budget_exceeded",
  "message": "Virtual key budget exceeded",
  "details": {
    "over": true,
    "reasons": ["day_tokens_exceeded:1200/1000"],
    "day": {"tokens": 1200, "usd": 0.55},
    "month": {"tokens": 20000, "usd": 12.34},
    "limits": {
      "is_virtual": true,
      "llm_budget_day_tokens": 1000,
      "llm_budget_day_usd": 5.0,
      "llm_budget_month_tokens": 20000,
      "llm_budget_month_usd": 100.0
    },
    "principal": {
      "principal_id": "user:1:api_key:42",
      "kind": "api_key",
      "user_id": 1,
      "api_key_id": 42,
      "org_ids": [],
      "team_ids": []
    }
  }
}
```

### Using the virtual key

- Use `X-API-KEY: <virtual_key>` header for API-key auth.
- Optional provider header: `X-LLM-Provider: <provider>` when provider allowlists are enabled.

Chat example (cURL):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <VIRTUAL_KEY>" \
  -H "X-LLM-Provider: openai" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Embeddings example (cURL):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <VIRTUAL_KEY>" \
  -d '{
    "model": "text-embedding-3-small",
    "input": ["test phrase"]
  }'
```

## Admin: Orgs and Teams (selected)

- POST `/api/v1/admin/orgs` - create organization
- GET `/api/v1/admin/orgs` - list
- POST `/api/v1/admin/orgs/{org_id}/teams` - create team
- GET `/api/v1/admin/orgs/{org_id}/teams` - list
- POST `/api/v1/admin/teams/{team_id}/members` - add member
- GET `/api/v1/admin/teams/{team_id}/members` - list

These endpoints assist grouping users and associating Virtual Keys with org/team scopes.
Example (cURL):
```bash
curl -X GET http://127.0.0.1:8000/api/v1/admin/users/123/virtual-keys \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

## Scoped JWTs for Workflows (Virtual Keys)

In addition to API keys, the server supports short-lived, scoped JWTs ("virtual keys") intended for internal automation such as Workflows schedules.

- Mint token (admin; multi-user):
  - POST `/api/v1/workflows/auth/virtual-key`
  - Body: `{ "ttl_minutes": 60, "scope": "workflows", "schedule_id": "<optional>" }`
  - Returns: `{ token, expires_at, scope, schedule_id }`

- Mint token (self-service; multi-user):
  - POST `/api/v1/auth/virtual-key`
  - Body (examples):
    - Basic: `{ "ttl_minutes": 30, "scope": "workflows" }`
    - With constraints:
      ```json
      {
        "ttl_minutes": 30,
        "scope": "workflows",
        "allowed_endpoints": ["scheduler.workflows.run_now", "evals.create_run"],
        "allowed_methods": ["POST"],
        "allowed_paths": ["/api/v1/scheduler/workflows", "/api/v1/evaluations"],
        "max_calls": 10,
        "max_runs": 5,
        "not_before": "2025-01-01T00:00:00Z"
      }
      ```
  - Returns: `{ token, expires_at, scope, schedule_id }`

Scope enforcement
- Endpoints can opt-in to scope checks. For example:
  - `scope=workflows` required when a scoped token is presented for evaluation run creation.
  - For scheduler endpoints such as `POST /api/v1/scheduler/workflows/{schedule_id}/run-now`, when a token contains a `schedule_id` claim, it must match the path `schedule_id`.
- Admin tokens (role=admin) bypass scope checks.
- API key and single-user flows are unaffected.

Endpoint allowlists and quotas
- Tokens may include optional claims enforced where enabled:
  - `allowed_endpoints`: list of logical endpoint IDs; requests outside the list are rejected (403).
  - `allowed_methods`: allowed HTTP methods.
  - `allowed_paths`: list of allowed URL path prefixes.
  - `max_calls`: process-local, best-effort cap on number of invocations with the token.
  - `max_runs`: same as `max_calls` but counted for endpoints that mark the action as a run.
- Note: For hard multi-instance budgets and LLM token budgets, prefer Admin Virtual API keys with DB-backed enforcement. Scoped JWT quotas are lightweight and process-local.

Scheduler integration
- To automatically mint a per-run token for schedules and inject it into workflow steps, set:
  - `WORKFLOWS_MINT_VIRTUAL_KEYS=true`
  - `WORKFLOWS_VIRTUAL_KEY_TTL_MIN=15`
  The scheduler adds a `secrets.jwt` bearer to the run context; adapters (e.g., webhook) use it when headers do not already specify auth.

Helper CLI
- Mint and export a scoped token for the scheduler service:
```bash
python -m Helper_Scripts.AuthNZ.mint_virtual_key \
  --user-id 1 --username tldw-admin --role admin \
  --scope workflows --ttl-minutes 30 --print-export
# or write to dotenv
python -m Helper_Scripts.AuthNZ.mint_virtual_key \
  --user-id 1 --username tldw-admin --role admin \
  --dotenv tldw_Server_API/Config_Files/workflows.env
```

Workflows webhook adapter auth fallbacks
- When an HTTP step omits explicit Authorization or X-API-KEY headers, the adapter will:
  1) Use `secrets.jwt` as `Authorization: Bearer ...`, or
  2) Use `secrets.api_key` as `X-API-KEY: ...`, or
  3) Fall back to `WORKFLOWS_DEFAULT_BEARER_TOKEN` or `WORKFLOWS_DEFAULT_API_KEY` if provided.
- Optionally validate fallback auth once per run by setting `WORKFLOWS_VALIDATE_DEFAULT_AUTH=true` (Base URL: `WORKFLOWS_INTERNAL_BASE_URL`).

## Self-service Virtual API Keys (X-API-KEY)

Users can mint constrained API keys for themselves that enforce the same endpoint/method/path and quota rules as scoped JWTs.

- POST `/api/v1/users/api-keys/virtual`
- Body (examples):
  ```json
  {
    "name": "wf-runner",
    "expires_in_days": 7,
    "allowed_endpoints": ["scheduler.workflows.run_now", "evals.create_run"],
    "allowed_methods": ["POST"],
    "allowed_paths": ["/api/v1/scheduler/workflows", "/api/v1/evaluations"],
    "max_runs": 10
  }
  ```
- Returns the key value once; store it securely. Use `X-API-KEY: <key>`.

Enforcement
- Endpoints with scope enforcement also apply rules for API keys:
  - allowed_endpoints checked against logical endpoint IDs listed above.
  - allowed_methods/allowed_paths verified per request.
  - max_calls/max_runs apply as best-effort process-local caps.
  - Voice-management routes (`audio.voices.*`) are counted under a separate `voice_call` counter type, isolated from the default `call` counter used by `audio.speech`.
- LLM budgets on Virtual API keys (`budget_day_tokens`, etc.) continue to apply for LLM endpoints.

## Endpoint IDs and Paths (allowlists)

Use these logical IDs in `allowed_endpoints`. When you prefer path-based allowlists, match the associated path prefix.

- chat.completions → POST `/api/v1/chat/completions`
- rag.search → POST `/api/v1/rag/search`
- audio.speech → POST `/api/v1/audio/speech`
- audio.voices.upload → POST `/api/v1/audio/voices/upload`
- audio.voices.encode → POST `/api/v1/audio/voices/encode`
- audio.voices.list → GET `/api/v1/audio/voices`
- audio.voices.get → GET `/api/v1/audio/voices/{voice_id}`
- audio.voices.delete → DELETE `/api/v1/audio/voices/{voice_id}`
- audio.voices.preview → POST `/api/v1/audio/voices/{voice_id}/preview`
- audio.transcriptions → POST `/api/v1/audio/transcriptions`
- audio.translations → POST `/api/v1/audio/translations`
- audio.stream.transcribe → WS `/api/v1/audio/stream/transcribe`
- media.ingest → POST `/api/v1/media/ingest-web-content`

## WebSocket STT Enforcement

The real-time STT endpoint (`/api/v1/audio/stream/transcribe`) enforces virtual-key constraints during the handshake:

- Authorization: Bearer (scoped JWT) or `X-API-KEY` (virtual API key) accepted in headers.
- `allowed_endpoints` must include `audio.stream.transcribe` if present.
- Optional `allowed_paths` should include `/api/v1/audio/stream/transcribe` when used.
- Quotas (`max_calls` / `max_runs`) are enforced via DB-backed counters across instances.
  - On quota exhaustion, the server sends an error frame and closes with code 4403.

## Quota Buckets (per-day caps)

For per-day call quotas without adding new tables, you can attach a simple bucket to counters:

- Scoped JWTs: add claim `"period": "day"` alongside `max_calls` or `max_runs`.
- Virtual API keys: put `"period": "day"` inside `metadata`.

With `period=day`, the quota counter is keyed by the current UTC date internally. This provides a daily rolling cap per token/key without additional schema changes.
