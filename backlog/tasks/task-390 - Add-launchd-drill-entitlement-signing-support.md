---
id: TASK-390
title: Add launchd-drill entitlement signing support
status: Done
labels:
- sandbox
- macos
- vz-linux
- operator-workflow
priority: medium
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
- tools/macos-vz-helper/README.md
- tools/macos-vz-helper/macos-vz-helper.entitlements
references:
- https://github.com/rmusser01/tldw_server/pull/1747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the macOS VZ helper launchd drill repeatable by allowing operators to pass an explicit entitlements plist for helper signing before launchd bootstrap. Keep signing opt-in and preserve existing behavior when no entitlements path is supplied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `launchd-drill` accepts an explicit `--entitlements` plist without changing default behavior when omitted.
- [x] #2 `launchd-drill` signs the helper after already-loaded preflight and before launchd bootstrap.
- [x] #3 Failed signing aborts before bootstrap so the drill does not leave a launchd service running from that failure.
- [x] #4 A checked-in local development entitlement template includes `com.apple.security.virtualization`.
- [x] #5 Focused unit tests and real Apple VZ launchd smoke are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented opt-in launchd-drill signing through existing `sign_helper`, with an injectable `signing_runner` for unit tests and JSON subprocess capture. Added `tools/macos-vz-helper/macos-vz-helper.entitlements` as the default dev/operator entitlement template. Updated launchd-drill README examples to pass the checked-in entitlement file and explain signing order.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validation: focused new tests passed; full tools/macos-vz-helper/Tests/test_vz_helperctl.py passed with 133 passed, 1 skipped; real launchd-drill with --entitlements and /private/tmp/tldw-vz-linux-bundle-vmrun/bundle passed 3 selected VZ Linux host E2E tests and booted out cleanly; git diff --check passed; py_compile passed; Bandit reported 0 findings for vz-helperctl.py.
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
