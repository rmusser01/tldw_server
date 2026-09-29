---
id: TASK-13382
title: >-
  Local venv runs FastAPI outside the declared pin, so route-introspection tests
  diverge from CI
status: Done
assignee: []
created_date: '2026-09-23 18:02'
updated_date: '2026-09-29 15:46'
labels:
  - tooling
  - testing
  - dependencies
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
pyproject.toml pins `fastapi>=0.136.3,<0.137.0` (`:57`). The development venv has **0.141.1** with **starlette 1.6.0** -- outside the declared range. There is no `uv.lock` at the repo root, although `ci.yml`'s `setup-python-deps` lists one in `cache-dependency-path`, so CI resolves from pyproject and gets 0.136.x.

**Why it matters beyond version hygiene.** FastAPI 0.141 / Starlette 1.6 changed `include_router` to put a single private `_IncludedRouter` object into `app.routes` instead of flattening the router's routes into it. So `app.routes` means something different in the two environments, and the suite leans on it heavily:

- 106 test files touch `.routes`
- 74 sites iterate them
- **32 sites use `getattr(route, "path", ...)`** -- the accessor that *masks* the change: an `_IncludedRouter` contributes `None` instead of raising, so every included route becomes invisible

Direct `route.path` fails loudly (AttributeError), which is safe. The `getattr` form does not, and that is the dangerous half: a **negative** assertion over `{getattr(route, 'path', None) for route in app.routes}` passes *vacuously* once the routes it is looking for are invisible. Examples of that assertion shape already in the suite -- these three use direct `.path` and so would error rather than lie, but they show the pattern exists:

- `tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py:468`
- `tests/Services/test_router_groups_contract.py:1447`
- `tests/Services/test_router_groups_contract.py:1580`

**Observed locally:** `tests/Audio/test_audio_router_import_resilience.py::test_audio_router_import_survives_broken_streaming_module` fails on clean dev with `AttributeError: '_IncludedRouter' object has no attribute 'path'`. It is presumably green in CI, which is itself the evidence that the two environments differ.

**Correction this forces.** PR #2997 rewrote `test_gateway_fastapi_package.py`'s route assertion, and its commit message attributed the breakage to the dependency having changed. That reading was wrong: CI was never broken; the local venv is out of spec. The rewrite itself is still valid -- it asserts through `app.openapi()`, which is stable public API on both versions -- but the recorded reason needs correcting so nobody concludes CI had a problem.

Related: TASK-13360, where editable installs resolve to the main checkout. Both are cases of local verification silently not matching CI.

**Decide which way to converge.** Either bring the venv to the pin, or raise the pin to what is actually being run and sweep the 74 route-walking sites. The second is a real project: the `getattr` sites need auditing individually, because a vacuous pass looks identical to a real one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The development venv satisfies the declared FastAPI constraint, or the constraint is raised deliberately
- [x] #2 If the pin is raised, every getattr(route, path) site is audited for vacuous passes, not just the ones that error
- [x] #3 A uv.lock exists, or ci.yml stops referencing one
- [x] #4 PR #2997's attribution of the route change is corrected on the task record
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-28. AC3: resolved by premise, not churn. ci.yml and 7 other workflows list uv.lock under cache-dependency-path, but .github/actions/setup-python-deps' resolve-cache-paths step drops any path that does not exist and falls back to pyproject.toml, so the absent lock is tolerated by design. Removing it from 8 workflows, several pinned by the license-first contract tests, would be churn with no behaviour change. AC4 (attribution correction): #2997's commit message said FastAPI changed include_router; it did not. The route-introspection difference came from this shared venv running fastapi 0.141.1 / starlette 1.6.0 against the declared fastapi>=0.136.3,<0.137.0; CI installs within the pin. AC1 stays open as an owner decision: either bring the shared venv inside the pin (uv pip install 'fastapi>=0.136.3,<0.137.0', but the venv is shared by every worktree and concurrent session) or raise the pin deliberately (Dependabot #2772 proposes <0.142.0), which then triggers AC2's getattr(route, path) audit.

2026-09-29, owner decision: "bump fastapi to latest stable". AC1: pin raised deliberately to fastapi>=0.141.1,<0.142.0 (tests/Security/test_dependency_security_floor.py updated). FastAPI 0.141.1 requires starlette>=0.46 and pydantic>=2.9; the existing starlette>=1.0.1 and pydantic>=2.13.5,<2.14 pins already satisfy it. Verified locally: _IncludedRouter arrived in 0.137.0, iter_route_contexts in 0.138.0.

AC2, the audit. It covered request time as well as app.routes, because scope['route'] is the router-local original route under >=0.137. Production regressions the bump would have shipped, now fixed via new core/Utils/fastapi_routes.py (iter_served_routes, served_route_for_scope):
- VN capabilities (every module reported disabled)
- privilege route registry
- RG coverage audit and the startup route-map audit
- main's startup duplicate-route guard
- metrics endpoint labels (router-local path)
- RG tag routing (include-time tags lost)
- token-scope enforcement (scoped tokens 403 where the checker is attached at include time)
Checked and unaffected: evaluations _promote_static_routes (/health and /metrics are still served before /{eval_id}), media router route copying (no nested includes), get_openapi(routes=...) (fingerprint unchanged), Helper_Scripts/ci/route_auth_ratchet.py (already handles _IncludedRouter).

Test side: about 60 files walked routes. 204 tests failed under 0.141 in 34 files. Vacuous passes found and fixed:
- negative asserts: billing public API removed, realtime websocket disabled, evaluations disabled
- test_placeholder_services_not_routed
- dependency-override loops that matched nothing (document insights/references, notes graph, user capabilities, the Embeddings conftest's auth keys)
- two _route_exists skips (trace context, re-embed schedule) that now run and pass
- include-if-missing fixtures that re-included routers on every call

Verification:
- tests/Utils/test_fastapi_routes.py: 8 tests, incl. request-scope and websocket
- new included-router test in test_coverage_audit.py (fails on the old code)
- all 95 route-walking test files under 0.141, compared with a dev/0.136 baseline: no new failures. Remaining failures match the baseline or are xdist-order-only (test_router_groups_contract: 178/178 alone; MediaDB2 metadata passes alone).
- The privilege snapshot, regenerated with Helper_Scripts/update_privilege_registry_snapshot.py, is byte-identical to dev's under 0.136. The refresh only picks up drift already on dev.
- OpenAPI drift gate: rc=0 under 0.141
- Embeddings suite 681 passed
- bandit (uvx, -ll) clean on the touched core modules

Known local-only failures, identical on dev/0.136:
- route_auth_ratchet reports two audio routes stale (this venv does not mount the audio router; see #3044)
- test_pagination_openapi_contract admin webhook deliveries
- test_privilege_service_sqlite ProfileUserWriteRejected
- test_auth_dependency_contract media leaf RequirePermission

No docs change beyond the helper's module docstring. Dependabot #2772 (to main, <0.142.0) is superseded by this.

CORRECTION 2026-09-29 (Qodo on #3053): 'RG tag routing (include-time tags lost)' is wrong and was removed from the change. RGSimpleMiddleware derives its policy before routing, so scope['route'] is empty on 0.136 and on 0.141 alike, and the served_route_for_scope edit there was a no-op; middleware_simple.py is back to dev's version. The underlying gap is pre-existing: by_tag policies are never enforced, while the coverage and startup audits count them as protected. It is filed as TASK-13395 (owner decision). The new coverage-audit test now uses a by_path mapping, so it tests route visibility and does not assert tag enforcement.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
