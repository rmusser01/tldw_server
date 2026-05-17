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
Second hardening review completed. Issues found and patched: Stage 1 now requires reproducing the current crash before behavior changes; default-entry wording no longer overfits to one fixture name; terminology changes must update i18n/fallbacks; Stage 3 mobile parity must create reusable entry points rather than throwaway controls; Stage 4 must preserve Stage 3 mobile access; extension sidepanel parity is explicitly out of scope except for avoiding shared-component regressions. Verification: local diff review and placeholder scan completed. Bandit skipped because this task touched documentation and Backlog task metadata only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and hardened the staged main /chat role-play preset remediation design spec. The final spec covers the approved hybrid six-stage sequence and includes implementation anchors, coordination constraints, adapter sequencing, recovery semantics, saved setup eligibility, compatibility test axes, i18n requirements, current-branch reproduction guidance, mobile reuse constraints, and extension scope boundaries.
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
