# Resource Governor

Centralized rate limiting and concurrency control with policy-based configuration, DB/file-backed stores, and optional Redis backend. This module enforces request, token, and stream limits across endpoints and integrates with FastAPI via helpers and optional middleware.

## Overview & PRD
- Design PRD: `Docs/Product/Completed/AuthNZ-Refactor/Resource_Governor_PRD.md`
- Architecture decision: `Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md`
- Example policies YAML: `tldw_Server_API/Config_Files/resource_governor_policies.yaml`

## New endpoints checklist (RG-aware design)

- When adding new API endpoints, first wire authentication/authorization via `get_auth_principal` together with `RequirePermission(...)` / `RequireRole(...)` (and `require_service_principal()` for service-only routes). Authorization should be claim-first; do not gate new behavior on `AUTH_MODE` or `is_single_user_mode()` / `is_multi_user_mode()` checks.
- For endpoints that are latency/cost-sensitive or user-facing (chat, audio, embeddings, evaluations, MCP, workflows, tools, media ingestion, etc.), decide whether they should be governed by Resource Governor:
  - If yes, add or reuse a policy in the RG policy store (`resource_governor_policies.yaml` or DB-backed `rg_policies`) and ensure there is a corresponding `route_map` entry keyed by path/tag (for example `/api/v1/chat/* → chat.default`).
  - Where feasible, add or extend tests so RG-enforced behavior is covered at the HTTP level (200/429 behavior, `Retry-After` and `X-RateLimit-*` headers, and token/stream caps).

## Policy Store Selection (env)
- `RG_POLICY_STORE`: `file` (default) or `db`
- `RG_POLICY_PATH`: path to YAML when using `file` store (defaults to `tldw_Server_API/Config_Files/resource_governor_policies.yaml`)
- `RG_POLICY_RELOAD_ENABLED`: `true|false` (default `true`)
- `RG_POLICY_RELOAD_INTERVAL_SEC`: reload interval in seconds (default `10`)

## Backend Selection (env)
- `RG_BACKEND`: `memory` (default) or `redis`
- `RG_REDIS_FAIL_MODE`: `fallback_memory` (default) | `fail_closed` | `fail_open`
- `REDIS_URL`: Redis connection URL (used when backend=redis). If unset, defaults to `redis://127.0.0.1:6379`.
- Determinism (tests/dev): `RG_TEST_FORCE_STUB_RATE=1` prefers in‑process rails for requests/tokens when running Redis backend, stabilizing burst vs steady retry‑after behavior in CI.

## DB Policy Store Bootstrap

Configure env for DB store and AuthNZ:
- `RG_POLICY_STORE=db`
- `AUTH_MODE=multi_user`
- `DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME`

Option A — Python bootstrap (no HTTP):
```
python - << 'PY'
import asyncio
from tldw_Server_API.app.core.Resource_Governance.policy_admin import AuthNZPolicyAdmin

async def main():
    admin = AuthNZPolicyAdmin()
    await admin.upsert_policy("chat.default", {"requests": {"rpm": 120, "burst": 2.0}, "tokens": {"per_min": 60000, "burst": 1.5}}, version=1)
    await admin.upsert_policy("embeddings.default", {"requests": {"rpm": 60, "burst": 1.2}}, version=1)
    await admin.upsert_policy("audio.default", {"streams": {"max_concurrent": 2, "ttl_sec": 90}}, version=1)
    print("Seeded rg_policies (DB store)")

asyncio.run(main())
PY
```

Option A2 — Seed DB from YAML (no HTTP):
```
# Seeds missing policy_ids referenced by route_map (default)
python -m tldw_Server_API.app.core.Resource_Governance.seed_db_from_yaml

# Seed all YAML policies
python -m tldw_Server_API.app.core.Resource_Governance.seed_db_from_yaml --all
```

Option B — Admin API (HTTP):
1) Obtain an admin JWT (login as admin in multi-user mode).
2) Upsert policies via API:
```
curl -X PUT \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"requests": {"rpm": 120, "burst": 2.0}, "tokens": {"per_min": 60000, "burst": 1.5}}, "version": 1}' \
  http://127.0.0.1:8000/api/v1/resource-governor/policy/chat.default

curl -X PUT \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"requests": {"rpm": 60, "burst": 1.2}}, "version": 1}' \
  http://127.0.0.1:8000/api/v1/resource-governor/policy/embeddings.default
```

Verify snapshot and list:
```
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:8000/api/v1/resource-governor/policy?include=ids
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:8000/api/v1/resource-governor/policies
```

## Sample Policy YAML (route_map merging)

Provide a YAML at `RG_POLICY_PATH` to supply/override `route_map` while using DB-backed policies. Example:
```
schema_version: 1
defaults:
  fail_mode: fail_closed
  algorithm:
    requests: token_bucket
    tokens: token_bucket

policies:
  chat.default:
    requests: { rpm: 120, burst: 2.0 }
    tokens:   { per_min: 60000, burst: 1.5 }
    scopes: [global, user, conversation]
  embeddings.default:
    requests: { rpm: 60, burst: 1.2 }
    scopes: [user]

route_map:
  by_tag:
    chat: chat.default
    embeddings: embeddings.default
  by_path:
    "/api/v1/chat/*": chat.default
    "/api/v1/embeddings*": embeddings.default
```

Notes
- When `RG_POLICY_STORE=db` is active, the loader merges the file’s `route_map` into the snapshot containing DB policies. File `route_map` takes precedence on conflicts.
- IMPORTANT: When `RG_POLICY_STORE=db` is active, YAML `policies:` are not used at runtime. Every `policy_id` referenced by `route_map` must exist in the DB store (`rg_policies`) or ingress enforcement will fail closed for that route (because missing request limits default to deny).
- Durable daily caps can be specified with `daily_cap` under any category and are enforced via `ResourceDailyLedger` (e.g., `tokens.daily_cap`, `workflows_runs.daily_cap`).
- Proxy/IP scoping (env):
  - `RG_TRUSTED_PROXIES`: comma-separated CIDRs of reverse proxies
  - `RG_CLIENT_IP_HEADER`: trusted header name (e.g., `X-Forwarded-For` or `CF-Connecting-IP`)
- Metrics cardinality (env): `RG_METRICS_ENTITY_LABEL`: `true|false` (default `false`)
- Test mode precedence: `RG_TEST_BYPASS` overrides Resource Governor behavior when set; otherwise falls back to `TLDW_TEST_MODE`

## Simple Middleware (default‑on)

- Resolution order: path-based mapping (`route_map.by_path`) first, then tag-based mapping (`route_map.by_tag`). Wildcards like `/api/v1/chat/*` match by prefix.
- Entity derivation: prefers authenticated user (`user:{id}`), then API key id/hash (`api_key:{id|hash}`), then trusted proxy IP header via `RG_CLIENT_IP_HEADER` when `RG_TRUSTED_PROXIES` contains the peer; otherwise falls back to `request.client.host`.
- Behavior: performs a pre-check/reserve for the `requests` category before calling the endpoint and commits afterwards. On denial, sets `Retry-After` and `X-RateLimit-*` headers. On success, injects accurate `X-RateLimit-*` headers using a governor `peek` when available.
- Scope: this middleware enforces **requests only**. Categories such as `tokens`, `streams`, `jobs`, and `minutes` require explicit endpoint-level RG reserve/commit plumbing (for example chat/embeddings tokens, audio streaming concurrency).
- Enabled automatically whenever `RG_ENABLED=1`.

Headers on success/deny:
- Deny (429): `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining=0`, `X-RateLimit-Reset` (seconds until retry).
- Success: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` computed via peek for the `requests` category.

## Diagnostics

- Capability probe (admin): `GET /api/v1/resource-governor/diag/capabilities`
  - Returns: `backend`, `real_redis`, `tokens_lua_loaded`, `multi_lua_loaded`, `last_used_tokens_lua`, `last_used_multi_lua`.
  - Route is gated by admin auth; in single‑user mode admin is allowed by default.
  - To avoid minute-boundary flakiness, Redis backend maintains an acceptance‑window guard for requests; `RG_TEST_FORCE_STUB_RATE=1` prefers local rails during CI.

## Testing

### Real Redis (optional)
- Optional integration tests validate the multi-key Lua path on a real Redis.
  - Set one of: `RG_REAL_REDIS_URL=redis://localhost:6379` (preferred) or `REDIS_URL=redis://localhost:6379`
  - Run: `pytest -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_lua.py`
  - The `real_redis` fixture verifies connectivity (no in-memory fallback) and skips if Redis unavailable.

### Middleware
- Middleware tests run against tiny stub FastAPI apps and don’t require full server startup.
- Useful env toggles during manual experiments:
  - `RG_ENABLED=1`
- Tests:
  - `pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py`
  - `pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_tokens_headers.py`
  - `pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_enforcement_extended.py`

## Operator examples

- Local dev (file policies, memory backend):
  - `RG_ENABLED=1`
  - `RG_POLICY_STORE=file`
  - `RG_POLICY_PATH=tldw_Server_API/Config_Files/resource_governor_policies.yaml`
  - `RG_BACKEND=memory`
  - `RGSimpleMiddleware` is attached automatically on startup.
- Production with DB policy store + Redis:
  - `RG_ENABLED=1`
  - `RG_POLICY_STORE=db` (AuthNZ `rg_policies` as source of truth)
  - `RG_BACKEND=redis`
  - `RG_REDIS_URL=redis://user:pass@redis-host:6379/0`
  - `RG_REDIS_FAIL_MODE=fail_closed` for strict ingress enforcement, or `fallback_memory` to prefer availability on non-critical paths.
- Route-map driven ingress:
  - Keep `route_map.by_path` in `resource_governor_policies.yaml` aligned with your primary ingress routes (for example `/api/v1/chat/* → chat.default`, `/api/v1/embeddings* → embeddings.default`).
  - Enable `RG_ENABLED=1` so `RGSimpleMiddleware` resolves `policy_id` from the route map and emits standard `Retry-After` / `X-RateLimit-*` headers on 429 responses.
