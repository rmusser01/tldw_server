---
id: TASK-400
title: Implement main /chat mode and session clarity
status: Done
assignee: []
created_date: '2026-05-16 02:20'
updated_date: '2026-05-16 02:27'
labels:
  - chat
  - cockpit
  - webui
  - ux
dependencies:
  - TASK-399
references:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR4 slice for the main WebUI /chat cockpit maturity roadmap. Scope is strictly the main chat page, not the browser-extension sidepanel/sidebar. Clarify cockpit/focus mode, independent rail visibility, and saved/temporary/server-backed conversation state so users understand what is visible, what will survive reload, and which warnings remain important when rails are collapsed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cockpit/focus mode copy explains the current mode and the effect of showing or hiding rails without changing chat state.
- [x] #2 Independent context/runtime rail visibility remains keyboard reachable, persisted, and understandable when one or both rails are hidden.
- [x] #3 Session state distinguishes temporary, local unsaved, local history-linked, server-backed loading, loaded, failed, and recovered/recoverable states in the main /chat cockpit.
- [x] #4 Collapsed rails do not hide important session or degraded warnings because the status strip carries the critical state.
- [x] #5 Focused unit/component coverage and real-server /chat proof cover mode toggles, rail visibility, session status, and collapsed-rail warning visibility.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-05-16-chat-cockpit-mode-session-clarity.md

Slice sequence:
1. Lock session summary and status-strip critical state with failing tests.
2. Propagate session status/title/detail/error into the always-visible status strip.
3. Lock cockpit/focus and rail visibility copy with shell tests.
4. Implement compact mode/rail summary copy without changing existing button names or persisted layout behavior.
5. Expand real-server /chat proof, run focused Vitest, real-server Playwright, git diff --check, design-system verification, and record Bandit skip if no Python files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial inspection found PR4 should build on existing PlaygroundCockpitShell, PlaygroundStatusStrip, PlaygroundContextRail, and buildCockpitSessionSummary. The app already persists layout mode and rail visibility; the gap is clarity and critical-state visibility rather than new behavior.

Created implementation plan at Docs/superpowers/plans/2026-05-16-chat-cockpit-mode-session-clarity.md. Scope remains main /chat only. Impeccable PRODUCT/DESIGN context is absent in this worktree, so this slice will follow the existing cockpit components, design-system tokens, and roadmap copy rather than inventing a new visual direction.

Implementation complete for PR4 mode/session clarity. Added compact cockpit/focus mode summaries to the main /chat shell, propagated session title/status/detail/error into the always-visible status strip, and expanded real-server proof so collapsed rails still preserve critical status visibility. Existing button names, persisted cockpit/focus mode, independent rail visibility, composer behavior, and send behavior were preserved.

Verification recorded:
- Red tests confirmed missing behavior before implementation: status strip did not show session failure details and shell had no mode summary.
- Focused Vitest after implementation: bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose => PASS, 5 files, 41 tests.
- Full real-server Playwright /chat cockpit spec: bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium => PASS, 9 tests in 1.4m against http://127.0.0.1:8000 with configured .env API key and no backend route mocking.
- git diff --check: PASS.
- bun run verify:design-system-state: PASS with existing allowed baseline exceptions.
- Bandit: skipped because this slice touched no Python files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the main /chat PR4 mode and session clarity slice. The cockpit shell now explains cockpit/focus and rail visibility states, while the status strip carries session title/status/error details so important saved/temporary/server-backed state remains visible even when rails are hidden. Added focused unit/component coverage and expanded the live-server Playwright proof for mode summaries, rail collapse, focus mode, degraded state visibility, and preserved /chat behavior.
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
