---
id: TASK-196
title: Document VZ Linux crash reboot operator follow-up and fill portable tests
status: Done
assignee: []
created_date: '2026-05-09 22:04'
updated_date: '2026-05-10 00:01'
labels:
  - sandbox
  - vz_linux
  - recovery
  - docs
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1459'
  - 'https://github.com/rmusser01/tldw_server/pull/1464'
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - tldw_Server_API/tests/sandbox/test_vz_linux_runner.py
  - tldw_Server_API/tests/sandbox/test_vz_reconciliation.py
  - tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first follow-up implementation slice for the VZ Linux crash/reboot posture: update operator-facing docs for helper crash/restart/host reboot procedures and add only missing portable tests for preserve-versus-clear behavior. Keep scope host-independent and do not add launchd automation, reboot CI, networking changes, guest protocol changes, or generic repair automation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Operator docs explain helper crash/manual restart and host reboot procedure without implying automatic repair
- [x] #2 Portable tests cover any missing session-control preserve-versus-clear or repair-blocking behavior identified by current-code audit
- [x] #3 No host-gated destructive behavior or launchd bootstrap/kickstart implementation is added
- [x] #4 Verification records focused pytest and docs hygiene checks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current-code audit: existing VZ runner tests already covered helper unavailable/protocol mismatch preservation, absent status replacement, unhealthy replacement, and helper-generation mismatch replacement. Added missing portable coverage for reachable helper truth with owner/runtime/session metadata mismatch; the new test passed without production changes. Added operator docs for helper crash/manual stop, direct or future launchd-managed restart, host reboot procedure, and host-gated CI exclusions. Verification: python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q passed (27 passed); git diff --check passed; Bandit on touched test file with baseline test-file B101/B108 skips returned 0 findings; rg verified crash/reboot doc anchors.

Review fix: addressed Qodo type-hint finding by annotating the new metadata-mismatch test fixture parameters and nested store `**kwargs`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated macOS sandbox operator docs and host-gated CI policy to document helper crash/restart and host reboot recovery procedure without implying automatic repair. Added portable VZ runner coverage proving reachable helper status with mismatched owner/runtime/session metadata clears stale session-control state and provisions a fresh VM. No production code or host-gated destructive behavior was changed.
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
