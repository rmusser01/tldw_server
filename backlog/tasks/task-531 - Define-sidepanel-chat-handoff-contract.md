---
id: TASK-531
title: Define sidepanel chat handoff contract
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 20:32'
labels:
  - chat
  - extension
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address /chat UX rebaseline F8 by making the browser-extension sidepanel full-screen chat handoff contract explicit and test-covered. Scope stays limited to sidepanel chat launch/handoff into /chat, including whether draft/page/thread state is route-only or visibly preserved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Header full-screen handoff opens /options.html#/chat and exposes route-only copy stating sidepanel draft/current-page/unsaved chat state stay in the sidepanel.
- [x] #2 ControlRow full-app handoff preserves only /chat route intent, including role-play query parameters when active, and exposes the same route-only state-transfer contract.
- [x] #3 Focused regression tests cover the handoff route and accessible contract copy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: focused Vitest failed on missing SidepanelHeaderSimple accessible handoff label/description and missing ControlRow accessible description. GREEN: bunx vitest run src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx --reporter=verbose passed 2 files / 6 tests. TypeScript: default bunx tsc --noEmit --pretty false OOMed before diagnostics; retry with NODE_OPTIONS=--max-old-space-size=8192 failed only on known unrelated CharacterListContent.design-system GalleryCardDensity baseline. git diff --check passed. Bandit skipped because touched code is frontend TS/TSX, locale JSON, docs, and Backlog markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the sidepanel-to-WebUI chat handoff as route-only for this release. The header full-chat affordance now names /chat in WebUI and states that draft/current-page/unsaved sidepanel chat state stays in the sidepanel. The ControlRow full-app affordance exposes the same contract while preserving active role-play route parameters. Updated the rebaseline audit to move F8 from unresolved ambiguity to explicit route-only behavior.
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
