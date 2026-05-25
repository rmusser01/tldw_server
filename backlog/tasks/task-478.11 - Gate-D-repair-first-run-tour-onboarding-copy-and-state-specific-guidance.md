---
id: TASK-478.11
title: 'Gate D: repair first-run tour, onboarding copy, and state-specific guidance'
status: Done
labels:
- research-workspace
- uat
- gate-d
- onboarding
- copy
- tour
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure/gap: the top tour button and Settings > Replay tour did not show a visible walkthrough overlay during UAT. Earlier design feedback also rejected a separate workspace trust bar, so guidance should be contextual and not add non-core banner clutter.

User goal: as a first-time NotebookLM migrant, understand what a workspace contains, what to do next, where data lives, and how processing failures recover without losing the core work surface.

Scope:
- Fix tour launch/replay behavior or remove dead controls if the tour is not ready.
- Add concise empty/loading/error/partial-success copy in context: add sources, processing/indexing, missing model, selected sources, Studio disabled, failed ingestion, retry.
- Preserve a dense research-oriented layout without extra persistent banners that compete with the core source/chat/Studio panes.
- Ensure local-first/privacy/data ownership messaging is present where users make relevant decisions, not as global clutter.
- Add tests or UI assertions for tour open/replay and key empty/error states.

Acceptance criteria:
- Tour/replay controls either open a visible, navigable tour or are not exposed.
- First-run and error-state copy tells the user the next action and system state without generic marketing language.
- No reintroduction of the rejected workspace trust bar or similar persistent banner clutter.
- CDP/Playwright validation covers first-run empty state, tour/replay, missing model, processing, and failed-source copy where available.

Depends on: should align final terms with TASK-478.3 and TASK-478.7.
Parallelization: can run in parallel with layout/source acquisition after terminology is agreed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-25-task-478-11-research-workspace-onboarding-tour.md
1. Verify tour failure surface.
2. Repair tour launch and replay.
3. Improve contextual state guidance.
4. Validate live with CDP/Playwright.
5. Finalize Backlog, tests, commit, and push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Root cause verified with a live backend/WebUI CDP run: Research Workspace tour controls were setting `activeTutorialId`, but the Next WebUI shell did not mount the shared `TutorialRunner`, so no Joyride overlay could render.
- Mounted the shared tutorial runner in `apps/tldw-frontend/components/layout/WebLayout.tsx` behind the existing `!hideHeader` shell condition. Did not mount `TutorialPrompt` and did not add a persistent workspace trust banner.
- Replaced stale chat empty-state copy that referred to adding sources in the "left panel" with responsive Sources pane/tab wording from `source-location-copy.ts`.
- Added contextual empty-source storage guidance in the Sources pane: files are stored in the configured local or self-hosted server and processing status appears in that pane.
- Added regression coverage for WebLayout runner mounting, source-location copy helpers, ChatPane empty copy, and SourcesPane empty guidance.
- Live CDP validation confirmed first-run Start tour and Settings > Replay tour both render a visible Joyride tooltip/overlay; empty state includes local/self-hosted copy, uses Sources pane wording, retains missing-model guidance, and does not contain workspace-trust or left-panel copy.
- Screenshot artifacts:
  - `/private/tmp/task47811-first-run-tour.png`
  - `/private/tmp/task47811-settings-replay-tour.png`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Mounted the Research Workspace walkthrough runner in the WebUI shell and tightened first-run/empty-state copy without adding a global trust banner. Tour controls now open a visible guided walkthrough, and empty states point users to the Sources pane/tab with local/self-hosted storage guidance at the source-addition decision point.
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

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `bun run test:run -- __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-location-copy.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.design-system.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx` from `apps/tldw-frontend`: 4 files, 23 tests passed.
- Live CDP/Playwright against `http://localhost:3000/research-workspace` and backend `http://127.0.0.1:18002`: first-run Start tour and Settings > Replay tour each produced `tooltipCount: 1` and `overlayCount: 1`; empty-state copy asserted `hasWorkspaceTrustText: false`, `hasLeftPanelCopy: false`, `hasLocalServerCopy: true`, `hasSourcesPaneCopy: true`, and `hasMissingModelCopy: true`.
- `./node_modules/.bin/tsc --noEmit --pretty false` from `apps/tldw-frontend`: failed on pre-existing unrelated TypeScript errors in `CharacterControlRail.tsx` and several `e2e/*` specs; no TASK-478.11 file appeared in the reported failures.
- Bandit: not run, frontend-only changes.
<!-- SECTION:VERIFICATION:END -->
