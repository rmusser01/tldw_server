---
id: TASK-441
title: Add VZ host reboot post-reboot evidence validation
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
- backlog/tasks/task-441 - Add-VZ-host-reboot-post-reboot-evidence-validation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: add `run_host_reboot_post(...)` support that validates the evidence directory, reads the pre manifest, pings the helper after reboot, compares helper generation details, writes `host-reboot-post.json`, and returns named postflight results. Scope excludes CLI wiring and restored-helper smoke execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Post phase validates the evidence directory with `create=False` and returns stable named results.
- [x] Missing `host-reboot-pre.json` returns `host_reboot_pre_manifest_missing` and does not write `host-reboot-post.json`.
- [x] Malformed or non-object pre manifests return `host_reboot_pre_manifest_invalid` and do not write `host-reboot-post.json`.
- [x] Post phase pings the selected socket through injectable `ping_checker`, coerces ping state, and maps ping exceptions to `helper_ping_failed`.
- [x] Helper generation comparison uses `helper_details.helper_instance_id` and returns ok reasons for changed, matching, or unavailable generation state.
- [x] Post manifest remains bounded to phase, host/path metadata, generation IDs/reason, and ping state payload fields without raw env/stdout/stderr/serial log content.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented `run_host_reboot_post(...)` plus private helpers for safe pre-manifest loading and helper generation comparison.
- Added focused postflight tests for missing pre manifest, malformed pre manifest, helper generation drift, and ping exception handling.
- Verification: focused `host_reboot_post` selector passed; full helperctl test file passed with 163 passed, 6 skipped; Bandit reported zero findings; `py_compile` passed; `git diff --check` passed.
- Scope intentionally excludes Task 4 CLI wiring and restored-helper smoke execution.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 implemented. run_host_reboot_post(...) now validates durable evidence state, safely reads the pre manifest, pings helper status, compares pre/post helper_instance_id generation signals, writes bounded host-reboot-post.json when appropriate, and returns stable named CheckResult entries for human/JSON output.
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
