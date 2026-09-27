---
id: TASK-13382
title: >-
  Local venv runs FastAPI outside the declared pin, so route-introspection tests
  diverge from CI
status: To Do
assignee: []
created_date: '2026-09-23 18:02'
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
- [ ] #1 The development venv satisfies the declared FastAPI constraint, or the constraint is raised deliberately
- [ ] #2 If the pin is raised, every getattr(route, path) site is audited for vacuous passes, not just the ones that error
- [ ] #3 A uv.lock exists, or ci.yml stops referencing one
- [ ] #4 PR #2997's attribution of the route change is corrected on the task record
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
