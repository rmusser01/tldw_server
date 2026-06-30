---
id: TASK-195
title: Design VZ Linux helper crash and host reboot recovery posture
status: Done
assignee:
  - Codex
created_date: '2026-05-09 21:53'
updated_date: '2026-05-09 22:01'
labels:
  - sandbox
  - vz_linux
  - recovery
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1459'
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - >-
    Docs/superpowers/specs/2026-05-09-vz-linux-helper-crash-host-reboot-recovery-posture-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a focused design-first slice for the next VZ Linux recovery gap: helper crash, launchd-managed helper restart, and host reboot recovery posture. The design should build on existing helper-generation session recovery, manual helper restart drills, diagnostics, and dry-run repair surfaces while avoiding destructive automation, networking changes, generic cross-runtime repair, or vz_macos real execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec defines helper crash, launchd restart, and host reboot as distinct failure modes with explicit ownership truth for helper process identity, VM/session-control rows, image-store state, and guest-agent readiness.
- [x] #2 The design identifies which checks can be covered by portable tests, which require host-gated/manual drills, and which remain operator-documented procedures.
- [x] #3 The design preserves dry-run-first repair semantics and says when session-control rows must be preserved versus cleared.
- [x] #4 The design explicitly excludes networking changes, helper boot path rewrites, guest protocol changes, vz_macos real execution, and generic cross-runtime repair automation.
- [x] #5 The task records design review findings, verification/hygiene checks, and links the resulting spec to GitHub issue #1459.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Writing focused recovery posture spec for helper crash, launchd restart, and host reboot; keeping scope docs/design-only and preserving existing generation-aware reuse plus dry-run-first repair invariants.

Added proposed design spec Docs/superpowers/specs/2026-05-09-vz-linux-helper-crash-host-reboot-recovery-posture-design.md linking issue #1459 and mapping acceptance criteria.

Verification: git diff --check passed. rg/sed verified the spec links issue #1459 and contains helper crash, launchd restart, host reboot, and acceptance mapping sections. Bandit was not run because this slice changed documentation/task metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a focused VZ Linux recovery posture design for issue #1459. The spec separates helper crash/manual stop, direct helper restart, launchd-managed restart, and host reboot; defines ownership truth for helper process identity, live VM state, persisted session controls, image-store state, guest readiness, launchd state, and socket/pid files; preserves fail-closed and dry-run-first repair semantics; and separates portable tests, host-gated drills, and operator-only reboot procedures.
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
