# Application spine: FastAPI entry, routing, and middleware

This document describes how the server boots and serves: the module-level FastAPI app and its lifespan sequences, the declarative router registry and route-gating policy, v1 vs v2 versioning, the middleware stack and its ordering contract, exception handlers, config resolution, and the CLI surface.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| App | `app/main.py` | module-level `app = FastAPI("tldw API")` (no factory; modes via env: `MINIMAL_TEST_APP`, `ULTRA_MINIMAL_APP`, `_TEST_MODE`), lifespan, middleware registration, exception handlers, control-plane routes, static mounts, `run_server()` |
| Router registry | `app/api/v1/router_registry.py`, `router_groups/{core,content,admin,minimal,conditional}.py` | `register_all_routers` — ~152 declarative `RouterSpec`s (34 core / 102 content / 16 admin) with lazy imports, `route_key` gating, idempotent inclusion keyed on `(id(router), prefix, tags)` |
| v2 | `app/api/v2/router.py` | `api_v2_router` under `/api/v2` (currently user-profiles only) |
| Route gating | `app/core/config.py` | `_route_toggle_policy` / `route_enabled` — config.txt `[API-Routes]` (`stable_only`/`disable`/`enable`/`experimental_routes`) is runtime-authoritative; env overrides honored only in test mode |
| Lifespan | `app/services/lifespan_startup_sequence.py`, `lifespan_shutdown_sequence.py` | pre-core → core init → PG RLS auto-ensure → job-poller registration; coordinated shutdown with an acquire gate and quiesce |
| Readiness | `app/services/readiness_service.py` | `/health` (public, immutable payload), `/internal/ready` (loopback-only), `/ready` + `/health/ready` (permission-gated), `/metrics` |
| Auth deps | `app/api/v1/API_Deps/auth_deps.py` | dependency-based auth (see [authnz.md](./authnz.md)) — auth is *not* middleware |
| Exceptions | `app/api/v1/utils/exception_handlers.py` | global unhandled handler, request/response validation, client disconnect, public TTS errors |
| Config | `app/core/config.py`, `Config_Files/.env`, `Config_Files/config.txt` | env + INI merge; secrets in `.env`; root `Env_Vars.md` is the auto-generated raw reference (1167 vars) with a curated copy under `Docs/Published/` |
| CLI | `cli/` (`evals_cli.py`, `wizard/cli.py`, commands/) | `tldw-server`, `tldw-evals`, `tldw-setup`, `tldw-admin-webhooks` console scripts |

## Startup (dataflow)

1. Import builds the app; duplicate route `method+path` pairs fail hard at import time and again at lifespan start.
2. Lifespan: pre-core preparation → core component initialization → PG RLS auto-ensure (`ensure_chacha_rls` / `ensure_prompt_studio_rls` when the content backend is Postgres) → owned job-poller registration → heavy-init handles.
3. Routers register through the declarative spec groups; each spec lazily imports its endpoint module, checks `route_enabled(spec.route_key)`, then includes idempotently.
4. Shutdown: owned pollers quiesce, the Jobs acquire gate closes (`set_acquire_gate(True)` so acquisitions return None), the coordinated shutdown coordinator stops workers and DB pools with timing segments.

## Middleware stack (order is the contract)

Registration order (Starlette wraps later registrations around earlier ones):

1. `LLMBudgetMiddleware` first — receive-time admission must run before its JSON body parsing.
2. HTML request guard; `CORSMiddleware` (single-user auto-adds localhost origins when unset; production validates explicitly — `*` with credentials is refused); CSRF (misconfiguration is **fatal**, not degraded).
3. Setup CSP/access guards, `SecurityHeadersMiddleware` (default on, auto in production).
4. Metrics, access log, sandbox-artifact traversal guards, usage logging, trace headers (`X-Trace-Id` / traceparent).
5. ChaCha operation middleware, `DrainGateMiddleware` (control-plane paths stay served during drain), `RequestIDMiddleware` last so it wraps everything.

In test mode most observability middleware is skipped. `_WorkflowTemplateTraversalGuard` and `_SandboxArtifactPathGuard` are also applied.

## Routing and versioning

- The API prefix is `/api/v1` (`API_V1_PREFIX`); scale: 166 endpoint entries (~276 route modules recursive), ~167 schema modules, ~40 dependency modules. New endpoints follow the endpoint-module + schemas + API_Deps pattern.
- Route gating: config.txt `[API-Routes]` at runtime; the default experimental set includes `sandbox, connectors, workflows, scheduler, personalization, jobs, benchmarks, guardian, self-monitoring`; the `mcp` alias expands to `mcp-unified` + `mcp-catalogs`. Env overrides only apply in explicit test mode; tests force-enable a known set.
- `/api/v2` exists as a separate router included idempotently (skipped in ultra-minimal mode).
- Control-plane routes register in **all** modes: `/health` public; `/internal/ready` loopback-only and schema-hidden; readiness/metrics permission-gated.

## Config system

`load_comprehensive_config()` merges env over `Config_Files/config.txt` (INI). Secrets live in `Config_Files/.env` (AuthNZ mode, single-user key, JWT secret). Module-level constants (`API_V1_PREFIX`, `ALLOWED_ORIGINS`, host/port) and the `settings` dict (feature flags, `USER_DB_BASE_DIR`, `DATABASE_URL`, Redis) resolve at import.

## Deployment surface (brief)

Docker-first: ~15 images under `Dockerfiles/` (app, prod, worker — embedding pipeline workers installing `.[multiplayer]`, audio GPU worker, webui/admin-ui) plus compose files for single-user, multi-user Postgres, workers, embeddings, production, proxy. The `Makefile` drives quickstart/production/backup targets; `start-sidecars.sh` waits for `/health`, sets `TLDW_WORKERS_SIDECAR_MODE=true`, and launches worker sidecars from `Docs/Deployment/sidecar_workers_manifest.json`. Reference docs: `Docs/Deployment/Production_Reference_Deployment.md`, `Postgres_Migration_Guide.md`, `horizontal-scaling.md`.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Duplicate method+path routes | Startup fails hard (import and lifespan) |
| CSRF misconfiguration | Fatal raise with guidance — never degraded-open |
| Production CORS `*` with credentials | Refused at validation |
| PG RLS ensure failure | Surfaces via the startup warning registry, can block startup |
| Drain state | Control-plane paths keep serving |

## Verified gotchas

1. There is **no app factory** — a single module-level app; test modes are env-selected subsets, and route registration is import-order sensitive in tests.
2. Middleware order is load-bearing (budget-before-body-parse; request-ID-wraps-all) and documented in-code — preserve registration order when inserting new middleware.
3. Two "Databases" directories: `tldw_Server_API/Databases/` holds SQL bootstrap assets; the repo-root `Databases/` is gitignored runtime data, absent from fresh clones (see [database-layer.md](./database-layer.md)).
4. `Docs/Code_Documentation/Code_Map.md` is partly stale: routers register through the spec registry (not directly in `main.py`), `/api/v2` is undocumented there, and ~120 newer endpoint modules are missing from its list.

## Governing decisions

ADR-019 (`Docs/ADR/019-security-request-edge-middleware.md`), ADR-021 (`021-services-lifecycle-startup-and-shutdown.md`). Current references: `core/Jobs/README.md`, `core/DB_Management/README.md`; `Docs/Code_Documentation/Data_Flow_Atlas.md` is mostly current but presents Redis as a jobs backend — Jobs is strictly SQLite/Postgres.

## Related docs

- [authnz.md](./authnz.md) — the dependency auth layer
- [database-layer.md](./database-layer.md), [jobs-background.md](./jobs-background.md)
