---
id: TASK-521
title: Plan remaining chat UX rebaseline slices
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-05-27 07:12
labels:
- chat
- ux
- planning
dependencies: []
documentation:
- docs/superpowers/plans/2026-05-27-chat-remaining-ux-rebaseline.md
priority: high
modified_files:
- docs/superpowers/plans/2026-05-27-chat-remaining-ux-rebaseline.md
- apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the formal implementation plan and first focused execution slice for the remaining route-scoped `/chat` UX rebaseline work after cockpit rails were restored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Formal implementation plan saved under docs/superpowers/plans with route-scoped remaining /chat UX slices.
- [ ] #2 Plan preserves restored cockpit rails and CharacterControlRail regression guardrails.
- [ ] #3 Plan identifies exact files/tests for mobile cockpit, toast/error feedback, accessibility cleanup, and follow-up audit.
- [ ] #4 Plan records verification commands and known baseline limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Formal plan saved at docs/superpowers/plans/2026-05-27-chat-remaining-ux-rebaseline.md. It narrows the remaining /chat rebaseline work to: restored rail/handoff guardrails; mobile cockpit density without changing the rail model; inline server-save feedback; remaining rail label duplication; and corrected /chat UX re-audit. Follow-up candidates for sidebar/history and model selector data contracts are split out unless still direct P1 blockers after the focused pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 guardrail inspection completed. Existing Playground.cockpit-regression.guard.test.ts already asserts cockpit shell/context/runtime rail wiring and absence of CharacterControlRail in Playground.tsx. Existing SidepanelHeaderSimple.fullscreen-route.test.tsx already asserts both full-screen and dashboard handoffs open /options.html#/chat. Fresh verification: bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx passed with 2 files and 6 tests. Node localStorage ExperimentalWarning observed as baseline.

Task 2 mobile density TDD started. Added a failing a11y/shell test proving the mobile cockpit panel still used max-h-[42vh], then changed PlaygroundCockpitShell mobile panel caps to max-h-[30vh] and made the mobile panel summary sr-only while preserving aria-describedby text. Added real-server Playwright assertions for mobile rail/composer non-overlap and bounded panel height. Fresh focused Vitest passed: Playground.cockpit-a11y, Playground.cockpit-shell, Playground.cockpit-regression.guard, and SidepanelHeaderSimple.fullscreen-route (4 files, 43 tests). Browser probe at http://127.0.0.1:18015/chat with 390x844 viewport confirmed mobile context panel class max-h-[30vh], panel height about 253px, sr-only summary text present, composer below rails, and no standalone CharacterControlRail text. Real-server backend startup exited before health binding in this sandbox run, so the real-server Playwright grep remains pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Planning task is still in progress while the first implementation slice is being verified.
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
