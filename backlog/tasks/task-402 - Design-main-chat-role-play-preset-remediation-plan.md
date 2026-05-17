---
id: TASK-402
title: Design main chat role-play preset remediation plan
status: Done
labels:
- chat
- ux
- roleplay
- design
documentation:
- Docs/superpowers/specs/2026-05-17-main-chat-role-play-preset-remediation-design.md
modified_files:
- Docs/superpowers/specs/2026-05-17-main-chat-role-play-preset-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a staged design spec for addressing the audited main /chat role-play preset UX issues: crash/recovery/accessibility, visible state, mobile parity, setup consolidation, saved setups, and compatibility guardrails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents the approved hybrid staged remediation approach.
- [ ] #2 Spec stays scoped to main /chat role-play preset experience.
- [ ] #3 Spec defines stage boundaries, data flow, testing, rollout, and implementation task split.
- [ ] #4 Spec is reviewed before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design approved in-session and reviewed twice by spec-document-reviewer. Both reviews returned Approved with no blocking issues. Non-blocking implementation-planning notes: define generation-style reset target, and include character vs persona as a Stage 6 compatibility test axis. Verification: spec was inspected locally for TODO/TBD placeholders; none were found. Bandit skipped because this task touched documentation and Backlog task metadata only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the staged main /chat role-play preset remediation design spec. The spec covers the approved hybrid six-stage sequence: crash/recovery/accessibility fixes, visible state and terminology cleanup, mobile parity, role-play setup consolidation, saved role-play presets, and compatibility/guardrail tests.
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
