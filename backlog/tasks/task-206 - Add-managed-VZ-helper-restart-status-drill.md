---
id: TASK-206
title: Add managed VZ helper restart/status drill
status: Done
assignee: []
created_date: '2026-05-10 00:55'
updated_date: '2026-05-10 01:19'
labels:
  - sandbox
  - vz-linux
  - operator-workflow
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1465'
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - tools/macos-vz-helper/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused operator-managed VZ helper lifecycle drill that exercises stop/start/status behavior through the existing vz-helperctl path without introducing launchd automation or host reboot mutation. This follows the VZ Linux crash/reboot recovery posture: helper lifecycle validation should be explicit, operator-owned, log-preserving, and safe to run before broader launchd or reboot drills.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A vz-helperctl-managed drill command or equivalent host-side script validates helper stop/start/status/ping flow without requiring pytest to own the helper process.
- [x] #2 The drill uses the existing private socket pid log and serial-log directory hardening checks and does not weaken helper lifecycle safety contracts.
- [x] #3 The drill has portable unit coverage for command construction and failure cleanup behavior without requiring a real Virtualization.framework host.
- [x] #4 Operator documentation explains when to use the drill and clearly states that launchd-managed restart and host reboot remain manual future work.
- [x] #5 Focused tests plus touched-scope security checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-10-vz-helper-managed-restart-drill.md

Verification: pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q => 93 passed, 1 skipped; git diff --check => pass; Bandit production script => 0 findings; Bandit test file with baseline test skips B101/B108/B404/B603 => 0 findings.

Known skips/blockers: no real VZ VM restart drill was run in this portable slice; real host lifecycle validation remains operator-gated.

Review fix pass: Qodo opened two still-valid threads on missing helper docstrings and overlong _prefixed_results signature.

Review fix verification: pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q => 93 passed, 1 skipped; git diff --check => pass; Bandit helper script => 0 findings; Bandit helper tests with baseline skips B101/B108/B404/B603 => 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a managed vz-helperctl restart-drill command that verifies a helperctl-owned helper is running, stops it through the existing pid/socket lease, starts it again on the same managed paths, and verifies status afterward. Added portable unit coverage and operator documentation clarifying this is a local lifecycle drill, not launchd or host reboot automation.

Review follow-up added docstrings for the new restart-drill helper functions and wrapped the overlong _prefixed_results signature.
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
