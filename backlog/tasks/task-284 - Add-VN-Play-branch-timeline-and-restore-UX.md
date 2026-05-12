---
id: TASK-284
title: Add VN Play branch timeline and restore UX
status: In Progress
assignee: []
created_date: '2026-05-12 03:03'
updated_date: '2026-05-12 03:04'
labels:
  - vn-play
  - webui
  - branch-navigation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1592'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API-related/VN_PLAY_API.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1592: add a player-facing VN Play branch timeline/navigation surface in the WebUI so Story/CYOA users can inspect branch history and safely restore/resume branches using backend-owned branch navigation APIs. Keep branch semantics server-authoritative; do not reconstruct branch state from raw events or add frontend-owned branching rules. The main checkout is dirty, so this task is implemented in .worktrees/vn-play-branch-timeline-1592 on branch codex/vn-play-branch-timeline-1592.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend API helpers and types cover existing VN Play branch navigation and guarded branch restore endpoints without duplicating backend branch semantics.
- [x] #2 Story sessions show player-facing branch history/path information with active branch status, useful empty states, and no reliance on debug-only inspector details.
- [x] #3 Branch restore/resume controls call backend guarded restore APIs with idempotency keys and show loading, stale, in-progress, and recoverable error states.
- [x] #4 Existing checkpoint creation/restore, generated-choice play, and inspector workflows continue to work.
- [x] #5 Focused frontend tests cover branch timeline rendering, active branch state, restore payloads, recovery states, and no-branch empty states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan of record: Docs/superpowers/plans/2026-05-12-vn-play-branch-timeline-restore-ux.md

Stages:
1. Add typed branch navigation and branch restore frontend API helpers.
2. Add BranchTimelinePanel with active path, restore target buttons, warnings, and empty states.
3. Wire VNPlayWorkspace to load branch navigation and call guarded branch restore.
4. Update docs, task notes, and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan for GitHub issue #1592 after verifying PR #1590 was merged and issue #1587 closed. Worktree: .worktrees/vn-play-branch-timeline-1592. Baseline frontend VN tests could not run before dependency setup because vitest is not installed in the fresh worktree; install/link dependencies before implementation verification.

Implemented typed frontend VN Play branch navigation and branch restore helpers, BranchTimelinePanel, and VNPlayWorkspace wiring against backend-owned branch navigation APIs. The workspace now keeps branch semantics server-authoritative, refreshes branch navigation with session collections, and handles guarded branch restore with idempotency keys plus recoverable stale/in-progress states.

Verification:
- `bun run test:run __tests__/vn-play/vnPlayApi.test.ts __tests__/vn-play/BranchTimelinePanel.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/SceneStage.test.tsx __tests__/vn-play/vnPlayRuntime.test.ts` passed: 5 files, 54 tests.
- `bun run lint -- components/vn-play/BranchTimelinePanel.tsx components/vn-play/VNPlayWorkspace.tsx __tests__/vn-play/BranchTimelinePanel.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts` exited 0 with existing repo-wide warnings only.
- `git diff --check` exited 0.
- Bandit skipped because this slice touched TypeScript/React, Markdown, and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added backend-owned VN Play branch navigation support to the frontend: typed branch navigation/restore API helpers, a Story branch timeline panel with active-path and restore-target rendering, and workspace wiring that refreshes branch navigation alongside session collections.

Branch restore controls now call the guarded backend restore endpoint with scene-version and idempotency protection, update the selected session from the restore response, and surface stale/in-progress restore conflicts as recoverable play-state messages. Documentation now notes that custom frontends should use the backend branch-navigation read model rather than deriving branch state from raw events.
<!-- SECTION:FINAL_SUMMARY:END -->
