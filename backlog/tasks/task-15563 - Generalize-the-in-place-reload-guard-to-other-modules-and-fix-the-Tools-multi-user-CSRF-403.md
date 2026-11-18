---
id: TASK-15563
title: Generalize the in-place reload guard to other modules and fix the Tools multi-user
  CSRF 403
status: To Do
labels:
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-15497 (owner decision 2, 2026-11-16). TASK-15497's guard restores only app.main. Other in-place module reloads leak the same way and are not guarded: importlib.reload(config_mod) in about 27 test files, plus audit, auth and collections service modules. Generalize the guard to snapshot and restore any reloaded module (an importlib.reload wrapper, like the TASK-15426 PR D imports fixture). Also take the Tools multi-user CSRF 403 that reproduces on dev after Security, Billing, Config and Meetings run in one process: tests/Tools/test_tools_permissions.py::test_tools_execute_forbidden_without_permission_multi_user[sqlite|postgres] (likely a config or AuthNZ reload leak). Out of scope for TASK-15497; it also baselines, not converts, the 38 app.main reloading files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reload of config_mod, audit, auth and collections service modules in one test no longer changes those modules for later tests (guard or fixture, with a polluter/victim test)
- [ ] #2 Tools test_tools_execute_forbidden_without_permission_multi_user[sqlite|postgres] passes after Security, Billing, Config and Meetings run in one process
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deferred minors from TASK-15497's final review (opus, clean session, 2026-11-17; Ready to merge: Yes, 0 Critical / 0 Important / 7 Minor), to take here:
1. The app_main_isolated guard exits before monkeypatch undoes (monkeypatch is requested first by the http_client_patch_guard plugin). Reload-then-monkeypatch of an app.main global writes the reloaded value back; patching app after a reload leaks the reloaded app; patching app plus a new name with raising=False makes monkeypatch.undo() raise AttributeError (new with TASK-15497). No test does any of these today. Fix: a comment at the guard (patch before reloading), and skip names already gone on restore.
2. tests/Config/test_route_and_cors_guards.py: the module fixture now reloads app.main from config_mod state at module start, so the CORS assertion depends on config_mod.ALLOWED_ORIGINS then; a config_mod reload leak changes it. Fix: set ALLOWED_ORIGINS and reload config_mod inside the fixture, and restore both.
3. tests/lint/test_no_new_inplace_app_main_reload.py: the pattern misses importlib.reload(sys.modules[...app.main]), reload(main), reload(tldw_Server_API.app.main) and aliased _reload(app_main); it over-matches reload(app_main_state) and any reload(importlib.sys.modules[...]); it counts files, not calls.
4. tests/e2e/conftest.py docstring: say the restore is at session end, so TESTING, TEST_MODE, ENABLE_REGISTRATION and REQUIRE_REGISTRATION_CODE stay set for later non-e2e tests in the worker (owner decision 3).
5. TASK-15497 AC #2 text still says 34; the note's 38 is the branch count (the base has 37).
6. AC #3 has no committed regression test; the reviewer rebuilt the env probe outside the repo and it is a real RED/GREEN test (dev fails on ROUTES_STABLE_ONLY, ROUTES_ENABLE, PRIVILEGE_METADATA_VALIDATE_ON_STARTUP). Commit it, e.g. tests/Infrastructure/test_jobs_env_does_not_outlive_module.py as a pytester run.
7. Nits: pytestmark unit on a file whose polluter rebuilds the real app; the victim test passes trivially under shuffle or xdist (the fake-module test is the deterministic one); snapshot_app_main_namespace could return dict[str, object] | None.
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
