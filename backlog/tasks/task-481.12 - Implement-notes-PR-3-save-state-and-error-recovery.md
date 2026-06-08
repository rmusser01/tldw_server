---
id: TASK-481.12
title: Implement notes PR 3 save state and error recovery
status: Done
labels:
- notes
- ux
- webui
- frontend
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 3 from the notes UX remediation plan: make dirty, saving, saved, failed, conflicted, and offline queued save states clear and recoverable without losing edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-3-save-state-and-error-recovery
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR 3 in the stacked notes UX worktree after PR 2 commit `b07c120ca4`.

Changed files:
- `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`: offline save now queues the local draft without setting the top-level server-saved indicator or server-saved timestamp.
- `apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx`: visible save status now prioritizes `saving` and `error` over `dirty`, so in-flight and failed saves are announced and expose retry.
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx`: adds failed-save recovery and in-flight duplicate-submit coverage.
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx`: asserts offline queued saves are not labeled as server-saved and do not update server-saved revision metadata.

Verification:
- RED: `NotesManagerPage.stage41.offline-drafting-sync.test.tsx` failed because an offline queued save rendered `notes-save-status` with `data-state="saved"`.
- RED: `NotesManagerPage.stage1.editor-reliability.test.tsx` failed because failed and in-flight saves both rendered `data-state="dirty"` instead of `error` / `saving`.
- GREEN: `./node_modules/.bin/vitest run src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage9.stale-version-warning.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage34.keyword-partial-save-warning.test.tsx src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx --maxWorkers=1 --no-file-parallelism` passed: 5 files, 16 tests.
- Browser verification against local Next dev server with mocked API: edit/save/reload preserved saved content; mocked failed save showed `data-state="error"`, Retry was visible, draft content remained visible, and retry posted successfully (`postCount: 2`).
- `git diff --check` passed.
- Bandit skipped/not applicable: PR 3 touched frontend TypeScript/React and Backlog/plan metadata only.

Known notes:
- Browser verification required temporary local Ant Design symlink repair because the tracked symlink target is absent from this dependency install; the symlink was restored before final status.
- Dev server required unsandboxed port binding approval after sandbox port bind returned `EPERM` in prior PR2 verification; PR3 reused the approved local dev-server flow.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 3 completed: `/notes` save status now distinguishes dirty, saving, failed, saved, and offline queued states without labeling local-only queued drafts as server-saved. Failed saves preserve draft content and expose retry; in-flight saves show saving state and duplicate submits remain blocked.
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
