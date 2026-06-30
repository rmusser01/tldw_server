---
id: TASK-304.8
title: Implement Research Studio returning-user tab persistence
status: Done
assignee: []
created_date: '2026-05-12 23:11'
updated_date: '2026-05-12 23:24'
labels:
  - implementation
  - research-studio
  - webui
  - studio
dependencies:
  - TASK-304.7
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persist last active mobile tab only when no URL tab is present
- [x] #2 URL tab state wins over stored mobile tab state
- [x] #3 Stored valid mobile tab applies when no URL tab exists
- [x] #4 Invalid or unreadable stored state falls back to Chat
- [x] #5 Storage failures are treated as no-ops
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing route-state helper, mobile tab wiring, and storage patterns.
2. Add failing helper and responsive tests for stored tab fallback, URL precedence, invalid storage fallback, and write no-op behavior.
3. Implement guarded Research Studio tab storage helpers and wire mobile tab changes through them without persisting when URL tab state is present.
4. Run focused Vitest and CDP smoke for persisted-tab and URL-override behavior.
5. Update this task with verification, skips, and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented guarded mobile-tab persistence with the versioned key tldw:research-studio:last-mobile-tab:v1. Reads accept only sources/chat/studio and storage failures return null; writes are no-ops on storage errors.

WorkspacePlayground now initializes from URL tab first, then stored mobile tab, then Chat. Mobile tab changes write the stored tab only when the initial URL did not include a valid tab state.

Focused Vitest: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/research-studio-route-state.test.ts src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx (19 passed).

CDP smoke: mobile /research-studio with stored studio opened Studio; /research-studio?tab=chat overrode stored studio; no-URL Studio click wrote studio; URL-tab Studio click preserved existing stored sources.

Bandit skipped because touched implementation is frontend TypeScript/TSX only; documentation update not relevant for this UI persistence slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Studio now remembers the last active mobile tab for returning users when no ?tab= route state is present, while canonical URL tab state remains authoritative and protected from storage overwrites.
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
