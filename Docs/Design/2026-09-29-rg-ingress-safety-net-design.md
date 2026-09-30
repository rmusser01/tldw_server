# Resource Governor ingress: enforce tag policies and make the defaults a safety net

- **Date:** 2026-09-29
- **Status:** Draft for owner review
- **Backlog:** TASK-13395 (tag policies never enforced). Implementation tasks are filed after this spec is approved.
- **Scope:** Spec 1 of 2. This spec covers the ingress governor. Spec 2 (usage-quota posture) is described under [Out of scope](#out-of-scope).

## Owner decisions (2026-09-29)

1. **Enforce `route_map.by_tag` policies** (TASK-13395 option a).
2. **Default posture is a safety net.** Governance stays on for every install. Buckets are per user, per API key, or per IP. There are no server-wide shared buckets, and defaults sit well above normal WebUI use. Strict limits remain only against auth brute force and expensive operations. Single-user and multi-user installs get the same defaults.
3. **Two specs, ingress first.** Usage quotas (billing plan limits, audio minutes, media bytes per day, chatbooks, storage) are Spec 2.
4. **Resolution approach A.** Resolve path first, then tags through a served-route index, then a catch-all default.

## Problem

Self-hosters report being rate limited quickly. The ingress governor (`RGSimpleMiddleware`) is on by default everywhere (`Config_Files/config.txt [ResourceGovernor] enabled = true`), and it has the following problems.

- **Server-wide shared buckets at per-user rates.** Fourteen policies include the `global` scope: `core.default`, `health.default`, `chat.default`, four `mcp.*` policies, and seven `authnz.*` policies. `governor.py` then charges one bucket shared by every user and every anonymous client.
  - `core.default` covers about 40 route families (notes, llm, config, users, jobs, sync, kanban, reading, outputs, connectors, admin, …) at 120 rpm with burst 1.2.
- **Identity collapses behind a proxy.** The middleware runs before auth. `deps.derive_entity_key` keys cookie-session traffic by client IP. In the Docker quickstart that IP is the Next.js container, so all WebUI users share it. ADR-044 fixed this only for single-user cookie sessions on policies without an ip or global scope.
- **Tag policies are dead.** `_derive_policy_id` runs before routing, so `scope["route"]` is unset and the `by_tag` branch never matches. On FastAPI ≥ 0.137 it would also miss include-time tags. About 57 routes are tag-only, and those are ungoverned.
  - About 659 more routes have neither a path mapping nor a tag mapping (manuscripts, workspaces, vn-*, guardian, slides, acp, scheduled-tasks, …).
  - The coverage audit (`coverage_audit._route_is_mapped`) and the startup audit (`startup_resource_governor._audit_route_map_coverage`) both count tag matches as protected.
- **Permanent 429s:**
  - An unknown policy is treated as `rpm 0`, and every request is denied (`governor.py`). MCP tool admission uses `mcp.<category>`, and most categories have no policy, so those tools are always denied.
  - When an entity's scope is not listed in a policy's scopes, no bucket exists and the request is denied. For example, an anonymous request to a `[user, api_key]` policy gets 429 instead of 401. This is the ADR-044 bug class.
  - A chat token reservation larger than the bucket's capacity (`per_min × burst`) can never be admitted.
- **Other defects:**
  - Auth-sensitive endpoints are charged twice for the same policy and entity: once by ingress and once by `auth._reserve_auth_rg_requests`.
  - A hot reload never resizes existing buckets. `_get_bucket` sets capacity only when it first creates a bucket, so raising a limit needs a restart.
  - `RG_ENABLED=false` does not turn everything off. The governor is always built, and several paths call it without checking the switch: auth reservations, media job concurrency and bytes per day, the workflows daily cap, and embeddings. Two call sites read the env var directly and ignore `config.txt` (`auth_deps.py`, `Evaluations/config_validator.py`).
  - The YAML carries sections the loader ignores (`defaults`, `hot_reload`, `metadata`, `route_map.by_route`).
  - `Docs/Operations/Env_Vars.md` documents the wrong defaults, and the RG README documents an `RG_TEST_BYPASS` that does not exist.

UAT has hit these limits five times; each time the fix went into the frontend while "preserving backend governance". See `Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md`, `MIGU_VOICE_FOLLOWUP_2026_09_06.md`, TASK-12918 and TASK-13211.

## Goals

1. A stock install, single-user or multi-user, never returns a governor 429 during normal WebUI, extension, or chat use by one person.
2. One user cannot exhaust another user's budget. Two cookie users behind one proxy IP are governed separately.
3. Every `/api/` route is governed by exactly one resolvable policy. The audits report the policy the middleware actually enforces.
4. No configuration can produce a permanent 429.
5. One switch turns governance off everywhere.

## Non-goals

- Usage quotas and plan limits (Spec 2).
- Merging the per-module governor instances. Chat, embeddings, MCP, audio, evals, character and web scraping each build their own.
- Exact limits across uvicorn workers. Memory buckets are per process; use the Redis backend for exactness.
- Websocket governance. The middleware skips non-HTTP scopes.
- The `rbac_rate_limit`, kanban, and `auth_deps` fallback limiters. With a catch-all policy, every `/api/` route is marked as governed, so the `auth_deps` fallback no longer fires.

## Design

### 1. One policy resolver

Add `resolve_policy(path: str, method: str) -> str | None` in a new module, `Resource_Governance/policy_resolver.py`. The middleware, the coverage audit, the startup audit and the RG diag endpoints all call it. The audits count a route as protected exactly when the resolver returns a policy the loader defines.

**Resolution order:**
1. **`by_path`.** Glob patterns in YAML order; the first match wins. The matching code is unchanged. This keeps the strict `authnz.*` and `chatbooks.*` policies ahead of any tag.
2. **`by_tag`.** Look the request up in the route index (below).
   - If the path and method match a served route, take its merged tags from the innermost to the outermost. The innermost is the router closest to the endpoint. The first tag present in `by_tag` wins.
   - Routes are matched in served order, the same order Starlette uses.
3. **`default`.** Any other path under `/api/` resolves to a new `default` policy.
4. **Ungoverned.** Everything else (docs, static files, the frontend) returns None.

The two hardcoded heuristics (`/api/v1/chat/` → `chat.default`, `/api/v1/audio/` → `audio.default`) are removed. `by_path` and `default` already cover those routes.

**Route index.** A new `RouteIndex` is built from `Utils.fastapi_routes.iter_served_routes(app.routes)`. For each served API route it stores:
- the compiled regex of the served path (`starlette.routing.compile_path`),
- the methods,
- the merged tags.

The index is built lazily on the first request and cached on `app.state`. It is rebuilt when:
- the policy loader publishes a new route map, or
- the app's route table changes. FastAPI 0.141 tracks changes in the private `APIRouter._get_routes_version()`. The index uses it behind a `getattr` guard, next to the other private-state reads in `Utils/fastapi_routes.py`, and falls back to `len(app.routes)` if it is missing. `tests/Utils/test_fastapi_routes.py` fails loudly if a FastAPI upgrade moves it.

### 2. Identity: charge the principal, not the proxy

This generalises the ADR-044 preflight to every auth mode and every policy.
- When a request carries credentials (a session cookie, `Authorization: Bearer`, or `X-API-KEY`), the middleware calls `get_auth_principal(request)` before deriving the entity.
  - The resolver caches its `AuthContext` on request state, and endpoint auth reuses it. This adds no database work compared with the current flow.
  - `derive_entity_key` then returns `user:<id>`, or `api_key:<id>` for key-only principals.
- **Invalid or expired credentials are not rejected here.** The middleware falls back to the `ip:` entity and lets the route's own auth return 401. This replaces ADR-044's early error response.
- Anonymous requests are charged to `ip:<client>` (honouring `RG_TRUSTED_PROXIES` and `RG_CLIENT_IP_HEADER`, which is unchanged).
- **A scope list never yields "no bucket".** If a policy does not list the entity's scope, the governor charges a per-entity bucket keyed by that entity (`entity` semantics). A policy's scopes decide which bucket kinds exist, never whether a request can pass at all.

### 3. Safety-net defaults

Changes to `resource_governor_policies.yaml`:
- **Remove `global`** from every policy except `authnz.magic_link.email`. That one caps outbound email for the whole server, which is a genuinely shared resource. Operators who want an aggregate cap can add `global` back.
- **Add `default`:** `requests` at 600 rpm, burst 2.0 (capacity 1200), scopes `[user, api_key, ip]`.

| Policy | Today (rpm / burst) | New (per entity) |
|---|---|---|
| `default` (new) | — | 600 / 2.0 |
| `core.default` | 120 / 1.2, global | 600 / 2.0 |
| `chat.default` requests | 120 / 2.0, global | 300 / 2.0 |
| `chat.default` tokens | 60k/min × 1.5, global | 1,000,000/min × 1.5 per user (runaway-loop guard; providers enforce their own limits) |
| `character_chat.default` | 60 / 1.0 | 300 / 2.0 |
| `embeddings.default`, `workflows.default`, `watchlists.default` | 60 | 300 / 2.0 |
| `mcp.default`, `mcp.ingestion` | 60 / 1.0, global | 300 / 2.0 |
| `mcp.read`, `mcp.search` | 120, global | 600 / 2.0 |
| `authnz.default` (`/auth/me`, `/auth/refresh`, login) | 60 / 1.0, global | 300 / 2.0 |
| `research.default`, `evals.default` (ingress) | 30 / 1.0 | 120 / 2.0 |
| `rag.default` | 120 / 1.2 | 300 / 2.0 |
| `authnz.forgot_password`, `reset_password`, `resend_verification`, `magic_link.request`, `mfa.*` | as today, global | as today, per entity only |
| `chatbooks.*`, `health.default`, `media.*`, `flashcards`, `quizzes`, `prompt_studio`, `audio.default` | as today | as today (`health` loses `global`) |

These numbers are starting points. The WebUI replay test (see [Testing](#testing)) is the acceptance bar, and the numbers may be raised during implementation to pass it.

### 4. Permanent-429 fixes

- **Unknown policy.** `governor.reserve` resolves an unknown policy ID to `default`, logging a warning once per ID, instead of denying.
  - A startup check and a unit test enumerate every referenced policy ID: `by_path`, `by_tag`, the `RG_*_POLICY_ID` defaults, the MCP categories in `MCP_unified/tool_execution/security.py`, and module constants. They require each one to exist in the loaded policies.
- **MCP categories.** `_maybe_enforce_with_rg_mcp` uses `mcp.<category>` when that policy is defined and `mcp.default` otherwise.
- **Scope mismatch.** Handled as described in §2.
- **Hot reload.** When the loader publishes a new snapshot, the memory governor updates `capacity` and `refill_per_sec` on existing buckets for changed policies and clamps current tokens to the new capacity. The Redis backend already reads limits per call; this is verified by a test.
- **Auth double charge.** `_reserve_auth_rg_requests` returns early when ingress has already charged this request for the same policy and entity. The middleware records this on request state (`rg_policy_id`, plus a new `rg_entity`). Reservations with a different entity still apply, for example the per-email-hash throttle.
- **Oversized token reservation.** In the token category, a reservation larger than the bucket's capacity is admitted when the bucket is full, and it drains the bucket to zero. Otherwise it waits for a full bucket, so `retry_after` is computed against capacity. It is never denied permanently.

### 5. One switch

- `config.rg_enabled()` (env `RG_ENABLED`, then `config.txt`, then the default, with the existing test-mode handling) becomes the only way code decides whether governance is on. The direct env reads in `auth_deps.py` and `Evaluations/config_validator.py` move to it.
- When governance is off, **no enforcement path calls the governor**. That covers:
  - ingress,
  - auth reservations,
  - chat and embeddings token reservations,
  - MCP admission,
  - evals and audio concurrency,
  - the workflows daily cap,
  - media job concurrency and ingest bytes per day.

  The governor object may still be built so the diag endpoints report state. Whether those quota values should apply at all is Spec 2's decision; this spec only makes the switch honest.

### 6. Config hygiene and docs

- Remove the ignored YAML sections (`defaults`, `hot_reload`, `metadata`, `route_map.by_route`). The loader logs a warning for any other top-level key it does not consume. `templates` (which holds the YAML anchors the policies reference) and `schema_version` are explicitly allowed.
- Add a new ADR (056): the safety-net posture, the resolution order, the identity rule, and "the `global` scope only for genuinely shared resources". It amends ADR-018 (resolution order, default policy) and ADR-044 (preflight generalised; invalid credentials fall through to IP).
- Correct the RG section of `Docs/Operations/Env_Vars.md` and `Resource_Governance/README.md` (drop `RG_TEST_BYPASS`).
- Add a short self-hoster page, "Why am I getting 429s?". It maps each response shape to the layer that produced it:
  - `{"error":"rate_limited","policy_id":…}` → RG ingress
  - `Rate limit exceeded for resource: X` → rbac
  - `limit_exceeded` 402 → billing
  - `Provider rate limit exceeded` → upstream
  - and so on.

  Each entry gives the switch that tunes that layer.

## Behaviour changes and risks

- **Roughly 716 routes become governed:** about 57 through tags and about 659 through `default`. The limits are generous and per user, so this is a safety net rather than a new constraint. The replay test guards against regressions.
- **The aggregate server-wide cap is gone.**
  - Unauthenticated floods are still bounded per IP.
  - A distributed flood is not bounded in aggregate. That is the reverse proxy's job, and the production reference deployment documents it.
  - Operators can add `global` back in the YAML.
- **Identity resolution in the middleware** now covers multi-user bearer and API-key traffic as well as single-user cookies.
  - The AuthContext cache keeps the cost flat.
  - Invalid credentials now reach the route's 401 instead of a middleware response. This is intentional and tested.
- **Turning RG off now disables** the media, workflows and auth reservations that previously stayed on.

## Testing

Every item below is a pytest test, and each defect fix gets a test that fails before the fix.
1. **Resolver precedence.**
   - Path beats tag.
   - The innermost tag beats an outer tag.
   - A tag route two `include_router` levels deep resolves.
   - Method filtering works.
   - An unmapped `/api/` path resolves to `default`, and a non-API path resolves to None.
2. **Request-level tag enforcement (TASK-13395 acceptance criterion).** An included route with a tag-only policy, no path policy and no heuristic is governed. Exhausting its bucket returns 429 with that tag's `policy_id`.
3. **Audits agree with the resolver.** For every served route of the real app, the coverage audit's classification equals `resolve_policy` plus "the policy is defined".
4. **Isolation.**
   - User A exhausting `core.default` does not affect user B.
   - Two cookie-session users arriving from one proxy IP get separate buckets.
   - An anonymous client is keyed by IP.
5. **WebUI replay.** One user replays a recorded boot-and-navigation request sequence (chat page, notes, characters, media list, settings) at realistic pacing and gets zero 429s. The sequence is taken from the UAT 429 reports and the frontend's startup requests.
6. **Defects, one test each:**
   - An unknown policy falls back to `default`.
   - An undefined MCP category falls back to `mcp.default`.
   - A scope mismatch uses a per-entity bucket.
   - A hot reload resizes a live bucket.
   - An auth endpoint is charged once.
   - An oversized token reservation is admitted when the bucket is full.
   - `RG_ENABLED=false` means zero governor calls (spy on `reserve`).
   - The startup check fails on an unknown referenced policy ID.
7. **Invalid credentials** reach the route's 401 and are charged to the IP bucket.
8. The existing `tests/Resource_Governance` and route-auth and privilege ratchets stay green. Tests that relied on `global` buckets are updated with a note.

## Delivery

Three PRs against `dev`, in order:
1. **Resolver, route index, identity and audits.** This closes TASK-13395. Tests 1–4, 7 and the audit half of 8.
2. **Safety-net defaults and the permanent-429 fixes.** Tests 5 and 6 (except the switch test).
3. **The single switch, config hygiene, ADR-056 and docs.** The switch test in 6.

## Out of scope

**Spec 2, usage-quota posture.** This follows the owner's safety-net decision: quotas are off unless an admin configures them. It covers:
- the OSS billing free plan applied to every multi-user org (300k LLM tokens/month → 402 on chat, 50 RAG queries/day, 60 transcription minutes/month, 1 GB storage);
- audio free-tier minutes (30 per day, in single-user mode too);
- media concurrency (2) and ingest bytes per day (2 GiB);
- chatbooks quotas;
- the per-user storage quota (5 GB);
- the workflows daily cap.
