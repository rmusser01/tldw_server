---
id: TASK-304.7
title: Implement Research Studio no-source progressive disclosure
status: Done
assignee: []
created_date: '2026-05-12 20:44'
updated_date: '2026-05-12 22:26'
labels:
  - implementation
  - research-studio
  - webui
  - studio
dependencies:
  - TASK-304.6
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-source Studio begins with source readiness guidance before generation controls
- [x] #2 Source readiness CTA focuses or opens the Sources pane
- [x] #3 Mobile no-source CTA switches to the Sources tab
- [x] #4 Slides and Audio settings are hidden before source/output intent
- [x] #5 Generation actions reappear when selected sources exist
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current no-source StudioPane rendering, source focus callbacks, and mobile tab tests.
2. Add failing tests for source readiness ordering, CTA callback behavior, mobile tab switching, settings hiding, and action restoration with sources.
3. Implement a focused source-readiness component or inline state, wire onRequestSources from WorkspacePlayground, and hide subordinate settings/actions while no source is selected.
4. Run focused WorkspacePlayground/StudioPane tests and diff hygiene.
5. Update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the no-source Studio state by replacing work-product and output generation controls with source-readiness guidance when no sources are selected.

Wired StudioPane onRequestSources through WorkspacePlayground to focusWorkspacePane('sources'), which opens the Sources pane on desktop and switches to the Sources tab on mobile.

Focused Vitest: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage3.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx (58 passed).

CDP smoke: /research-studio?tab=studio desktop showed source readiness and zero Executive Brief, More outputs, Slides Settings, or Audio Settings controls; mobile Open Sources tab activated Sources and showed the no-sources empty state.

Bandit skipped because touched implementation is frontend TypeScript/TSX only; documentation update not relevant for this UI behavior slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Studio now starts no-source users with source-readiness guidance instead of disabled generation controls, hides Slides/Audio settings until sources exist, and routes the readiness CTA to the Sources pane/tab. Generation actions return once selected sources are present.
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
