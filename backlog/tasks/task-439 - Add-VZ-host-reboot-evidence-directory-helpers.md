---
id: TASK-439
title: Add VZ host reboot evidence directory helpers
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: add host reboot evidence directory validation helpers and preserve helper generation details in ping state. Scope is limited to vz-helperctl.py and its focused tests; no pre/post manifest writing or CLI wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `ensure_host_reboot_evidence_dir(create=True)` returns stable `CheckResult` failures instead of raw filesystem exceptions for broken symlink paths and file-as-parent paths.
- [x] Host reboot evidence directory creation keeps every newly-created path component owner-only (`0700`), including intermediate directories under a private parent.
- [x] Import-time volatile evidence root construction ignores unresolvable `TMPDIR` values instead of failing module import.
- [x] `ping_helper_state()` preserves only string-valued helper details from both client-factory replies and raw helper JSON responses.
- [x] Scope remains limited to Task 1 evidence directory/detail helpers; Task 2 pre/post manifest behavior was not implemented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-05-19 code-quality review fix: added red regression tests for unresolvable import-time `TMPDIR`, broken symlink evidence paths, file parent paths, nested evidence directory modes, and raw helper JSON details filtering.
- Replaced the host-reboot wrapper's direct `mkdir(parents=True)` path with the existing component-by-component private directory helper, while preserving volatile-root rejection before creation and mapping operational filesystem failures to `host_reboot_evidence_dir_not_private`.
- Added safe volatile-root construction so `/tmp`, `/private/tmp`, and `TMPDIR` entries that cannot be resolved are skipped instead of raising during import.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 1 host reboot evidence directory helper hardening in vz-helperctl.py and added focused regressions for the code-quality review findings. Verification: RED focused pytest failed on the current behavior with import-time TMPDIR symlink-loop, broken symlink, file parent, and intermediate-mode failures; GREEN focused pytest passed 7 selected tests; full helperctl test file passed 150 tests with 6 skips; Bandit on the touched script reported 0 results; `git diff --check` passed.
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
