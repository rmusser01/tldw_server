---
id: TASK-15497
title: Test files that reload app.main or app modules in place leak a rebuilt app
  into later tests
status: In Progress
labels:
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15426 PR D's test-isolation fix (2026-11-13). Under pytest-xdist --dist loadfile, a test that reloads app modules in place (importlib.reload) and never restores them leaves later files in the same worker with routes bound to different function objects than the ones they override in app.dependency_overrides, so their overrides are ignored (requests then write to the worker's shared per-user Media DB). PR D fixed the four polluters its wider set hit: tests/DB_Management/test_media_db_api_imports.py (an autouse fixture restores each reloaded module), tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py (reload_app_main() instead of importlib.reload(app_main)), tests/Watchlists/conftest.py (module scope instead of session) and tests/conftest.py client_with_single_user (overrides via monkeypatch.setitem). 34 other test files reload app.main in place the same way (33 outside PR D's wider set; the one inside, under performance/, is skipped without PERF=1). The autouse app-main guard (app_main_isolated, tests/helpers/app_main_state.py) compares module identity only, so an in-place reload passes it unnoticed. Also: tests/Jobs/conftest.py:17 and tests/e2e/conftest.py:16 set environment variables in session-scoped fixtures without restoring them. Details: PR D's investigation notes (debug_pollution_d.md, summarized in the PR body).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The app-main guard detects and undoes an in-place reload of app.main (or every test that reloads it uses reload_app_main()), with a test that fails when a later test sees the reloaded app
- [ ] #2 The 34 in-place reloads are removed or covered by the guard; a grep in a lint test keeps new ones out
- [ ] #3 The session-scoped environment changes in tests/Jobs/conftest.py and tests/e2e/conftest.py are restored, or documented as intended
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
