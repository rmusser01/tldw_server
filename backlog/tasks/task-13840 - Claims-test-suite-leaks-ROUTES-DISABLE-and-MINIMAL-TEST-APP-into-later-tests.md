---
id: TASK-13840
title: Claims test suite leaks ROUTES_DISABLE and MINIMAL_TEST_APP into later tests
status: To Do
labels:
- tests
- claims
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Claims/conftest.py has a session-scoped autouse fixture, _claims_suite_minimal_app_env, that sets MINIMAL_TEST_APP=1 and adds media, audio and other routes to ROUTES_DISABLE in os.environ until the test session ends. Any test that runs later on the same pytest worker sees those routes disabled. On dev 7b5ba7aa6d, `pytest -p no:randomly tests/Claims/test_claims_service_fva_analyze.py tests/Services/test_router_groups_contract.py` gives 10 failed, 171 passed (the test_iter_minimal_optional_router_specs_* audio and media tests); run alone, the router contract file passes. Under xdist this makes those router tests pass or fail depending on worker placement. Found while sweeping the TASK-13560 change (PR #3247).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Running a Claims test before tests/Services/test_router_groups_contract.py in the same session leaves the router contract tests passing
- [ ] #2 The Claims suite still gets its minimal app and disabled routes while its own tests run
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
