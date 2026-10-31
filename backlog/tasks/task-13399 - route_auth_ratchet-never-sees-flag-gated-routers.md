---
id: TASK-13399
title: route_auth_ratchet never sees flag-gated routers
status: To Do
assignee: []
created_date: '2026-09-30 06:33'
labels:
  - ci
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Helper_Scripts/ci/route_auth_ratchet.py sets ROUTE_POLICY_ENV to force-enable benchmarks, connectors and personalization (comment ~lines 37-38). But load_app() clears PYTEST_CURRENT_TEST, TEST_MODE and TLDW_TEST_MODE before importing the app (~148-149), and config.route_enabled only honors ROUTES_ENABLE / ROUTES_STABLE_ONLY under explicit pytest or test mode (app/core/config.py ~3565-3666). So the force-enable does nothing: the ratchet never inspects routers with default_stable=False, and an unauthenticated route there would pass CI. Found during the RG route-map lint review (plan 2026-09-29-rg-ingress-safety-net, Task 11).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The route-auth ratchet builds the app with benchmarks, connectors and personalization mounted and checks their routes
- [ ] #2 A test fails if a force-enabled router is missing from the app the ratchet inspects
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
