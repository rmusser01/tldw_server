---
id: TASK-13840
title: Claims test suite leaks ROUTES_DISABLE and MINIMAL_TEST_APP into later tests
status: Done
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
- [x] #1 Running a Claims test before tests/Services/test_router_groups_contract.py in the same session leaves the router contract tests passing
- [x] #2 The Claims suite still gets its minimal app and disabled routes while its own tests run
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: tests/Claims/conftest.py's _claims_suite_minimal_app_env was a session-scoped autouse fixture that set MINIMAL_TEST_APP=1 and added media, audio and audio-websocket to ROUTES_DISABLE until the session ended. It is now function-scoped and uses monkeypatch, as tests/Jobs/conftest.py does, so the settings hold only while a Claims test runs. Every Claims test imports app.main inside the test body, so no wider-scoped fixture needs them earlier. Tests: pytest -p no:randomly tests/Claims/test_claims_service_fva_analyze.py tests/Services/test_router_groups_contract.py gave 10 failed, 171 passed on dev and 181 passed with the fix (AC1); tests/Claims alone 1073 passed, 1 skipped (AC2). tests/Claims then tests/Services, no random order: 13 failed on dev, 4 with the fix. Three of the 4 (test_main_router_contract::test_router_contract_includes_key_paths and two in test_openapi_contracts.py) fail when run alone on dev too; the fourth, test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property, hit Hypothesis's too_slow health check while other suites loaded the machine and passed alone 3 times. Ruff's I001 in the conftest is on dev already.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Claims suite's minimal-app settings (MINIMAL_TEST_APP, the disabled media and audio routes) now hold only while each Claims test runs, so later tests on the same worker see the routes they expect: the reported order goes from 10 failed to 181 passed.
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
