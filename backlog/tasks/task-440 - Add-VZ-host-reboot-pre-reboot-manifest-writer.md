---
id: TASK-440
title: Add VZ host reboot pre-reboot manifest writer
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: add a bounded host reboot preflight manifest writer and tests for `run_host_reboot_pre()`. Scope is limited to pre-phase manifest helpers in vz-helperctl.py and focused tests; no post-reboot comparison or CLI wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add failing focused tests for `run_host_reboot_pre()`, `write_json_private()`, and `ping_state_payload()`.
- [x] Write `host-reboot-pre.json` only after validating/creating the evidence directory through `ensure_host_reboot_evidence_dir(...)`.
- [x] Keep manifest payload bounded to the allowed pre-reboot fields and helper ping metadata.
- [x] Write manifest JSON with owner-only file permissions.
- [x] Do not add post-reboot comparison, CLI wiring, or restored-helper smoke execution.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `HOST_REBOOT_PRE_MANIFEST` and future `HOST_REBOOT_POST_MANIFEST` constants in `vz-helperctl.py`.
- Added `write_json_private()` using a private `0600` JSON write path with stable `host_reboot_manifest_write_failed` failures.
- Added `ping_state_payload()` and `run_host_reboot_pre()` callable helpers only; no `host-reboot-drill` parser wiring was added.
- Verified the red phase first with the focused selector, which failed because the new helper APIs were absent.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented Task 2 pre-reboot manifest helpers and review-fix hardening. The pre helper validates the evidence directory, captures helper ping state through an injectable checker, writes bounded `host-reboot-pre.json` evidence, and now returns non-ok when ping fails while preserving the successful `host_reboot_pre_manifest_written` reason for healthy pings.
- Hardened `write_json_private()` to validate and modify the opened file descriptor only: final-component symlinks are refused via the open path, non-regular fd targets are rejected, existing manifests are set to `0600` before truncation/write, and operational failures keep the stable `host_reboot_manifest_write_failed` reason.
- Verification: focused pytest selector passed after an expected red run (`6 passed`), full helperctl test file passed (`157 passed, 6 skipped`), `py_compile` passed, Bandit on the touched script reported `results=0` and `errors=0`, and `git diff --check` passed.
- Known skips: restored-helper smoke execution, post-reboot comparison, CLI wiring, and real host reboot validation were not run because they are outside Task 2 scope.
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
