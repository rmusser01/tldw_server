---
id: TASK-13417
title: >-
  rg_route_map_lint loses ~188 routes when the policy loader loads before
  iter_served_routes
status: Done
assignee: []
created_date: '2026-10-02 01:39'
updated_date: '2026-10-03 01:42'
labels:
  - ci
  - rate-limit
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduction (TASK-13399): Helper_Scripts/ci/rg_route_map_lint.py:75-78 calls app = load_app(); loader = default_policy_loader(); asyncio.run(loader.load_once()); then list(iter_served_routes(app.routes)) -- i.e. it instantiates/loads the Resource Governor policy loader BEFORE ever calling iter_served_routes() on this app. In that call order, iter_served_routes(app.routes) (tldw_Server_API/app/core/Utils/fastapi_routes.py:63, via fastapi.routing.iter_route_contexts) returns only 2690 HTTP routes with methods, vs. 2878 when iter_served_routes() is called once (for any reason) BEFORE default_policy_loader()/load_once() run -- the same app object, same app.routes, same process. The ~188 missing routes include the benchmarks/connectors/personalization families mounted by TASK-13399's route_auth_ratchet.py fix; rg_route_map_lint_allowlist.txt still carries by_path entries for all three (by_path /api/v1/benchmarks* / /api/v1/connectors* / /api/v1/personalization* matches no served route) because of this, even though route_auth_ratchet.py's own test (test_force_enabled_routers_are_mounted) proves those routers ARE mounted in the identical app.

Suspected cause: FastAPI >= 0.137's RouteContext._effective_route / scope['fastapi']['effective_route_context'] (see fastapi_routes.py's module docstring) is computed/cached per route the first time something resolves it, and default_policy_loader()'s construction or load_once() must be touching request-independent machinery (schema generation? dependency resolution?) that populates or hydrates this cache incompletely/differently than a cold iter_served_routes() call does -- worth instrumenting iter_route_contexts()/RouteContext directly to see which 188 routes are affected and why.

Reproduction script (standalone, run from repo root with the test venv):
  from Helper_Scripts.ci.route_auth_ratchet import load_app
  from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
  app = load_app()
  from tldw_Server_API.app.core.Resource_Governance.policy_loader import default_policy_loader
  import asyncio
  loader = default_policy_loader(); asyncio.run(loader.load_once())
  len([r for r in iter_served_routes(app.routes) if r.methods])  # -> 2690
vs. calling iter_served_routes(app.routes) once before importing/using default_policy_loader at all -> 2878, stable across repeat calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The lint sees the same served routes regardless of call order
- [x] #2 A test fails if the lint's route set differs from iter_served_routes on a fresh app
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause was import order, not the FastAPI route cache: rg_route_map_lint.main() imported policy_loader before load_app(), which imports core/config.py. Outside pytest that module caches the real config.txt at import, so load_app()'s TLDW_CONFIG_FILE copy was ignored and all 188 routes of default-off routers (benchmarks, connectors, personalization, audiobooks, slack, sandbox, ...) never mounted. Under pytest the config import is lazy, so CI never saw it; the 'iter_served_routes first' observation was a red herring.
Fix: route_auth_ratchet.load_app() calls config.clear_config_cache() when config is already imported, so the built app does not depend on what the caller imported first.
Lint setup moved into load_inputs(). New test test_lint_sees_the_routes_a_fresh_app_serves compares it with a fresh load_app() in subprocesses with the pytest markers dropped; the shipped-map test runs the same way.
Removed the by_path allowlist entries for /api/v1/benchmarks*, /api/v1/connectors* and /api/v1/personalization*. The lint is clean at 2878 routes with no new findings. Lint tests: 20 passed. Bandit: CI helper and test only; scoped run not needed. No docs affected.
Side fix in the same PR: test_rg_metrics_redis_backend used a fixed namespace against the local real Redis and flaked on back-to-back runs; it now uses a uuid namespace.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The RG route_map lint now sees the same 2878 served routes however it is invoked: load_app() clears config.py's import-time cache before building the app. A test compares the lint's route set with a fresh load_app() outside pytest's lazy-config mode, and the three allowlist entries that only existed because routes were invisible are gone.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
