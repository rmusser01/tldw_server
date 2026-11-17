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
