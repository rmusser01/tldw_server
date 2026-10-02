---
id: TASK-13417
title: >-
  rg_route_map_lint loses ~188 routes when the policy loader loads before
  iter_served_routes
status: To Do
assignee: []
created_date: '2026-10-02 01:39'
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
- [ ] #1 The lint sees the same served routes regardless of call order
- [ ] #2 A test fails if the lint's route set differs from iter_served_routes on a fresh app
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
