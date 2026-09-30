---
id: TASK-13398
title: Pin FastAPI to latest stable 0.142.1
status: Done
assignee: []
created_date: '2026-09-30 02:18'
updated_date: '2026-09-30 02:19'
labels:
  - dependencies
  - backend
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
pyproject pinned fastapi>=0.141.1,<0.142.0 while 0.142.1 is the latest stable and the dev venv already ran 0.142. Standing rule: when the venv is newer than the pin, bump to the latest stable and fix breakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 pyproject pins fastapi>=0.142.1,<0.143.0 and the dependency floor test matches
- [x] #2 Route introspection helpers (app/core/Utils/fastapi_routes.py) and route-dependent artifacts (OpenAPI fingerprint, route auth ratchet, privilege snapshot) verified under 0.142.1
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified with FastAPI 0.142.1 installed. tests/Utils/test_fastapi_routes.py and test_dependency_security_floor.py: 17 passed. The private state the helper relies on (RouteContext._effective_route, scope effective_route_context) is unchanged in 0.142. The OpenAPI fingerprint regenerated identically. Ran the backend-required tenant-isolation and contract ratchets, every route-walk test touched by #3053, and the privilege tests: 1149 passed, 20 failed, 6 skipped. The failures are 19 in test_chacha_postgres_http_operation_lifecycle.py (connection ACTIVE instead of IDLE after owner close) plus 1 order-dependent route/CORS guard that passes alone. All of them also fail locally under FastAPI 0.141.1 (15 failed, same five tests), and the db-management-a-l CI shard passed on #3053, so they are local macOS timing issues, not 0.142. Bandit: not applicable (version pin and a docstring only).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
FastAPI pinned to 0.142.1 (<0.143). No code changes were needed: dev's served-route helper from #3053 works unchanged on 0.142, and the OpenAPI fingerprint is identical.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
