---
id: TASK-521.3
title: Plan remaining chat UX rebaseline slices
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 09:08'
labels:
  - chat
  - ux
  - planning
dependencies: []
documentation:
  - docs/superpowers/plans/2026-05-27-chat-remaining-ux-rebaseline.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the formal implementation plan and first focused execution slice for the remaining route-scoped `/chat` UX rebaseline work after cockpit rails were restored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Formal implementation plan saved under docs/superpowers/plans with route-scoped remaining /chat UX slices.
- [x] #2 Plan preserves restored cockpit rails and CharacterControlRail regression guardrails.
- [x] #3 Plan identifies exact files/tests for mobile cockpit, toast/error feedback, accessibility cleanup, and follow-up audit.
- [x] #4 Plan records verification commands and known baseline limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Formal plan saved at docs/superpowers/plans/2026-05-27-chat-remaining-ux-rebaseline.md. It narrows the remaining /chat rebaseline work to: restored rail/handoff guardrails; mobile cockpit density without changing the rail model; inline server-save feedback; remaining rail label duplication; and corrected /chat UX re-audit. Follow-up candidates for sidebar/history and model selector data contracts are split out unless still direct P1 blockers after the focused pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 guardrail inspection completed. Existing Playground.cockpit-regression.guard.test.ts already asserts cockpit shell/context/runtime rail wiring and absence of CharacterControlRail in Playground.tsx. Existing SidepanelHeaderSimple.fullscreen-route.test.tsx already asserts both full-screen and dashboard handoffs open /options.html#/chat. Fresh verification: bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx passed with 2 files and 6 tests. Node localStorage ExperimentalWarning observed as baseline.

Task 2 mobile density TDD started. Added a failing a11y/shell test proving the mobile cockpit panel still used max-h-[42vh], then changed PlaygroundCockpitShell mobile panel caps to max-h-[30vh] and made the mobile panel summary sr-only while preserving aria-describedby text. Added real-server Playwright assertions for mobile rail/composer non-overlap and bounded panel height. Fresh focused Vitest passed: Playground.cockpit-a11y, Playground.cockpit-shell, Playground.cockpit-regression.guard, and SidepanelHeaderSimple.fullscreen-route (4 files, 43 tests). Browser probe at http://127.0.0.1:18015/chat with 390x844 viewport confirmed mobile context panel class max-h-[30vh], panel height about 253px, sr-only summary text present, composer below rails, and no standalone CharacterControlRail text. Real-server backend startup exited before health binding in this sandbox run, so the real-server Playwright grep remains pending.

Task 3 server-save feedback TDD completed. Added a failing usePlaygroundPersistence hook test proving first server save still opened notificationApi.success with 'Chat now saved on server'. Removed only that success notification call while preserving serverPersistenceHintSeenRef, setServerPersistenceHintSeen(true), setShowServerPersistenceHint(true), and existing error notifications. Added a real-server Playwright assertion to the mobile send flow that no .ant-notification-notice contains 'Chat now saved on server' and the chat input remains visible. Focused verification passed: bunx vitest run src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx (5 files, 51 tests). git diff --check passed. Real-server Playwright remains pending because the prior backend startup exited before health binding in this sandbox.

Task 4 rail-label cleanup TDD completed. Added a failing cockpit a11y test for duplicated empty assistant copy in the composition preview and runtime rail: both rendered 'No assistant selected' as title/value and repeated detail text. Suppressed only duplicate detail text by omitting identical assistant composition detail and by showing the runtime no-assistant explanatory detail when upstream summary detail repeats the empty label. Focused verification passed: bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundCompositionPreview.test.tsx src/components/Option/Playground/__tests__/playground-composition-preview.test.ts src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx (6 files, 75 tests). git diff --check passed.

Additional Task 4 verification: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false still fails on the unrelated baseline Characters design-system test at src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35 because 'comfortable' is not assignable to GalleryCardDensity. Bandit is not applicable for this frontend TypeScript-only slice.

Task 5 corrected /chat UX re-audit artifact refresh completed. Captured/updated first-time unseeded, desktop cockpit, desktop focus, mobile focus, mobile cockpit context, mobile cockpit runtime, mobile send blocked-state, and sidepanel debug-route screenshots under Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline. Updated Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md with current evidence notes, first-time and power-user walkthroughs, severity-ranked findings, quick wins, larger improvements, ideal workflows, and open questions. Updated evidence.json with current observations: rails present; removed standalone CharacterControlRail absent; provider readiness mismatch blocks first send; exact Send role lookup matches two buttons; sidepanel debug route overflows at 390 px with scrollWidth 420.

Task 5 verification completed. Focused Vitest passed: bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx (5 files, 52 tests). git diff --check passed. TypeScript baseline remains unchanged: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false fails only on the known unrelated CharacterListContent.design-system.test.tsx GalleryCardDensity issue. Port cleanup: the audit-owned 18015 WebUI server was stopped, and final recheck showed no listeners on 18015 or 18016; an intermediate unrelated .worktrees/notes-list-reliability Next dev server was observed on 18016 and left untouched. Bandit skipped because this slice touched Markdown, JSON, PNG screenshots, and frontend/Backlog metadata only; no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the remaining /chat UX rebaseline plan and first execution set: preserved restored rail/handoff guardrails, reduced mobile cockpit density, moved server-save success feedback inline, clarified repeated rail empty-state copy, refreshed the corrected /chat UX audit against the rail-enabled page, and recorded current remaining issues for the next implementation slices. Current highest-priority follow-up is the provider/model readiness mismatch that blocks first send while the runtime rail still reports tldw:gpt-4o as ready.
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
