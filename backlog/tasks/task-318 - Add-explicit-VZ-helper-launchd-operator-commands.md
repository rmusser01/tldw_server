---
id: TASK-318
title: Add explicit VZ helper launchd operator commands
status: Done
assignee: []
created_date: '2026-05-13 14:42'
updated_date: '2026-05-13 14:52'
labels:
  - sandbox
  - vz-linux
  - operator-workflow
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1485'
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - tools/macos-vz-helper/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit operator-owned launchd scaffolding for the macOS VZ helper lifecycle. The goal is to let operators inspect and invoke launchctl bootstrap, bootout, kickstart, and status/print flows through vz-helperctl without auto-installing services, hiding mutations, or expanding into host reboot automation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 vz-helperctl exposes explicit launchd status/bootstrap/kickstart/bootout actions with dry-run support.
- [x] #2 Mutating launchd actions remain operator-invoked and do not run from plist generation, smoke, status, or startup paths automatically.
- [x] #3 Launchd bootstrap reuses existing plist and private runtime/log/serial directory validation, and writes a plist only when explicitly requested.
- [x] #4 Portable unit tests cover command construction, dry-run behavior, missing plist failures, and explicit write-plist behavior without requiring launchd.
- [x] #5 Operator docs explain the launchd flow and keep host reboot validation out of scope.
- [x] #6 Focused tests, diff check, and touched-scope Bandit pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-13-vz-helper-launchd-operator.md

Verification: python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd' -q -> 6 passed, 93 deselected; python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q -> 98 passed, 1 skipped; git diff --check -> clean; Bandit script/tests -> empty errors/results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added explicit vz-helperctl launchd operator commands for status/bootstrap/kickstart/bootout with dry-run support, explicit write-plist/create-dirs gates, private directory validation reuse, portable unit coverage, and operator documentation that keeps launchd and host reboot validation manual.
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
