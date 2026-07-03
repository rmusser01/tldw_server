---
id: TASK-12124
title: Design Chat Macros and wrapup command
status: In Progress
priority: Medium
documentation:
- Docs/superpowers/specs/2026-07-03-chat-macros-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved Chat Macros design spec covering a dedicated Chat_Macros module, file-backed macro definitions, Jobs-backed execution, configurable output profiles, and /wrapup as the first bundled macro.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents architecture, data model, execution flow, security, error handling, and tests.
- [x] #2 Spec incorporates review refinements around Jobs payload size, context snapshots, retention, cancellation, model/cost controls, and command-name constraints.
- [x] #3 Backlog task references the written spec and verification notes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Design doc written at Docs/superpowers/specs/2026-07-03-chat-macros-design.md.
- Spec review loop completed: reviewer status Approved; no blocking issues.
- Advisory refinements folded in before commit: explicit branch_strategy schema knob and v1 repeated --question behavior.
- Verification: documentation-only task; no code tests or Bandit run applicable. Local checks verified no TODO/TBD/FIXME placeholders in the spec.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Written design spec for Chat Macros and /wrapup. The spec covers the dedicated Chat_Macros module, file-backed macro definitions, Jobs-backed execution with durable run records, configurable output profiles, context snapshots, ACP/chat-native branch behavior, retention, cancellation, model/cost controls, error handling, guardrails, and tests. Spec review passed with no blocking issues; Bandit is not applicable because this task only touched documentation and Backlog metadata.
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
