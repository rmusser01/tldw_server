# Why am I getting 429s (or 402s)?

`tldw_server` has several independent layers that can reject a request for "too many" or "not enough budget." They don't share a switch, so the fix depends on which one fired. Match the response body (or message) you got against the table below, then tune the knob in that row.

See also: [ADR-057](../ADR/057-resource-governor-safety-net.md) for why the Resource Governor (RG) ingress layer is a generous per-entity safety net by default, and [Env_Vars.md](Env_Vars.md#resource-governor-unified-rate-limiting) for the full RG env var reference.

| Response | Layer | Where it lives | How to tune |
|---|---|---|---|
| `{"error":"rate_limited","policy_id":"...","retry_after":N}`, HTTP 429 | RG ingress (`RGSimpleMiddleware`) | `Resource_Governance/middleware_simple.py` | Edit the named policy's `requests` block in `tldw_Server_API/Config_Files/resource_governor_policies.yaml` (hot-reloads within `RG_POLICY_RELOAD_INTERVAL_SEC`, default 10s), or set `RG_ENABLED=false` to disable RG everywhere. |
| `Rate limit exceeded (ResourceGovernor policy=chat.default); retry_after=Ns`, HTTP 429 | Chat token bucket (`_maybe_enforce_with_rg_chat` / the `/chat/completions` RG reserve) | `Chat/rate_limiter.py`, `api/v1/endpoints/chat.py` | Edit `chat.default`'s `tokens` block (`per_min`, `burst`) in `resource_governor_policies.yaml`. |
| `Rate limit exceeded for resource: <resource>`, HTTP 429 | RBAC resource-aware limiter (`enforce_rbac_rate_limit`) | `api/v1/API_Deps/auth_deps.py` | Edit the resource's `rate_limit_class` (`requests_per_min`, `burst`) under `rate_limit_classes` in `tldw_Server_API/Config_Files/privilege_catalog.yaml`, or the per-user/per-role override in the `rbac_user_rate_limits` / `rbac_role_rate_limits` tables. |
| `Rate limit exceeded for endpoint: <path>` (or `Auth rate limit exceeded for endpoint: <path>` on auth routes), HTTP 429 | `auth_deps.py` ingress fallback (`check_rate_limit` / `check_auth_rate_limit`) | `api/v1/API_Deps/auth_deps.py` | `AUTH_DEPS_FALLBACK_RATE_LIMIT` (default 120/min, `check_rate_limit`) or `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT` (default 30/min, `check_auth_rate_limit`). This fallback only runs when RG did **not** already resolve a policy for the request (`request.state.rg_policy_id` unset); since the RG `default` policy now covers nearly every `/api/` route, you should rarely see this — if you do, the route is likely outside `/api/` or RG is disabled. |
| `{"error":"limit_exceeded","category":...,"message":...}`, HTTP 402 (soft block) or 429 (hard block) | Billing plan limits (multi-user) | `api/v1/API_Deps/billing_deps.py` | `LIMIT_ENFORCEMENT_ENABLED` (default `true`) turns billing quota enforcement off entirely; otherwise raise the org's plan limits. |
| `{"error":"budget_exceeded","message":"Virtual key budget exceeded",...}`, HTTP 402 | Virtual-key LLM budget guard | `AuthNZ/llm_budget_guard.py`, `AuthNZ/llm_budget_middleware.py` | `LLM_BUDGET_ENFORCE` (default `true`) turns virtual-key budget enforcement off; otherwise raise the key's `llm_budget_day_tokens` / `llm_budget_day_usd` / `llm_budget_month_tokens` / `llm_budget_month_usd` limits. |
| `Media ingestion concurrency limit reached.`, HTTP 429 | Media ingestion job concurrency (RG `jobs` category) | `Ingestion_Media_Processing/persistence.py` | Edit `media.default`'s `jobs.max_concurrent` (default 2) in `resource_governor_policies.yaml`. |
| `{"status":"quota_exceeded","message":"Transcription quota exceeded (daily minutes)",...}`, HTTP 402 | Audio daily-minutes quota | `Usage/audio_quota.py`, `api/v1/endpoints/audio/audio_transcriptions.py` | Set `AUDIO_TIER_LIMITS_JSON` to raise the caller's tier's daily-minutes cap (JSON object mapping tier names to partial overrides). |
| `Provider rate limit exceeded.` (surfaced as a `ChatRateLimitError` / provider error) | The upstream LLM or TTS provider itself | `LLM_Calls/error_utils.py` (`privacy_safe_chat_error`) | Not a local knob — this is the provider's own rate limit. Check the provider's dashboard/plan, or switch providers/models. |

## Memory-backend buckets are per worker process

The `memory` RG backend (`RG_BACKEND=memory`, the default) keeps rate-limit buckets in each uvicorn worker's own process memory — a bucket is not shared across workers. Worker count is set by `UVICORN_WORKERS` (read directly by `tldw_Server_API/scripts/run_server_guarded_mcp.py`, and passed through by `Dockerfiles/Dockerfile.prod` and the `Dockerfiles/docker-compose.*.yml` files, which default to 2–4 workers depending on the compose profile). With N workers, a policy's effective limit for one caller is roughly N times its configured `rpm`, spread unevenly across whichever worker(s) handle that caller's requests.

If you need an exact, worker-independent limit, set `RG_BACKEND=redis` (with `REDIS_URL` pointing at a shared Redis) so every worker enforces against the same counters.

## Still stuck?

- `GET /api/v1/resource-governor/policy?include=ids` (admin) lists the loaded RG policy IDs.
- `GET /api/v1/resource-governor/diag/capabilities` (admin) reports the active RG backend.
- The RG ingress response always includes the `policy_id` that denied the request — match it against `tldw_Server_API/Config_Files/resource_governor_policies.yaml` (or the DB-backed `rg_policies` table when `RG_POLICY_STORE=db`).
