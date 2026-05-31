---
id: TASK-435
title: Add VZ guest-agent mismatch diagnostics tests
status: Done
assignee:
  - codex
created_date: ''
updated_date: '2026-05-19 04:36'
labels:
  - sandbox
  - vz-linux
  - diagnostics
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the VZ Linux lifecycle/recovery hardening track by adding host-independent coverage for guest-agent protocol/version mismatch diagnostics and cleanup behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guest-agent mismatch conditions are represented in the VZ helper/runner diagnostic contract without requiring a real VZ host.
- [x] #2 Focused tests cover mismatch handling and verify diagnostics degrade to actionable stale/mismatch state instead of crashing or reusing an unhealthy VM.
- [x] #3 Implementation keeps changes minimal and aligned with existing sandbox lifecycle docs/tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current VZ Linux runner/helper diagnostics and existing lifecycle tests. 2. Add focused failing test(s) for guest-agent mismatch behavior. 3. Implement minimal diagnostic/status handling. 4. Run focused pytest plus diff/security checks on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR review follow-up: added classifier docstrings/constants, aligned text coercion with diagnostics, refactored classifier to return the full guest observability payload, typed the new pytest parameters, and added direct classifier regression tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented host-independent guest-agent mismatch diagnostics and session reuse hardening for VZ Linux. PR review follow-up added classifier module/helper docstrings, centralized guest observability parsing in the classifier, aligned non-string workspace-root coercion with diagnostics, added direct classifier regression coverage, typed the new pytest parameters, and checked Backlog AC/DoD items. Verification: `python -m pytest tldw_Server_API/tests/sandbox/test_vz_guest_agent.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q` passed with 58 tests; `git diff --check` passed; Bandit on touched production Python wrote `/tmp/bandit_vz_guest_agent_mismatch_review_fix.json` with zero findings.
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
