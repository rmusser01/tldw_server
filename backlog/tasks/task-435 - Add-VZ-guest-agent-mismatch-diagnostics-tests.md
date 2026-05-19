---
id: TASK-435
title: Add VZ guest-agent mismatch diagnostics tests
status: Done
assignee:
- codex
labels:
- sandbox
- vz-linux
- diagnostics
priority: medium
modified_files:
- tldw_Server_API/app/core/Sandbox/vz_guest_agent.py
- tldw_Server_API/app/core/Sandbox/macos_diagnostics.py
- tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py
- tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
- tldw_Server_API/tests/sandbox/test_macos_diagnostics.py
- tldw_Server_API/tests/sandbox/test_vz_linux_runner.py
- tldw_Server_API/app/core/Sandbox/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the VZ Linux lifecycle/recovery hardening track by adding host-independent coverage for guest-agent protocol/version mismatch diagnostics and cleanup behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Guest-agent mismatch conditions are represented in the VZ helper/runner diagnostic contract without requiring a real VZ host.
- [ ] #2 Focused tests cover mismatch handling and verify diagnostics degrade to actionable stale/mismatch state instead of crashing or reusing an unhealthy VM.
- [ ] #3 Implementation keeps changes minimal and aligned with existing sandbox lifecycle docs/tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current VZ Linux runner/helper diagnostics and existing lifecycle tests. 2. Add focused failing test(s) for guest-agent mismatch behavior. 3. Implement minimal diagnostic/status handling. 4. Run focused pytest plus diff/security checks on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented host-independent guest-agent mismatch diagnostics and session reuse hardening for VZ Linux. Verification: focused red tests failed before implementation, then `python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q` passed with 56 tests; `git diff --check` passed; Bandit on touched production Python wrote `/tmp/bandit_vz_guest_agent_mismatch.json` with zero findings.
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
