---
id: TASK-474
title: Fix notes image attachment save path
status: Done
labels:
- bug
- notes
- frontend
priority: high
modified_files:
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage4.revision-attachments.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability-followup.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage9.stale-version-warning.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage34.keyword-partial-save-warning.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix user-reported failure saving a note after attaching an image. Suspected root cause: frontend notes save sends expected_version as a query parameter while the notes update endpoint requires the expected-version header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A regression test covers saving note content after an image attachment is inserted.
- [x] #2 Frontend note update requests send the optimistic lock in the format accepted by the backend.
- [x] #3 Targeted frontend tests pass.
- [x] #4 Touched scope is checked with Bandit where applicable, or documented if no Python scope changed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: NotesManagerPage sent expected_version as a query parameter on PUT /api/v1/notes/{id}, but the backend requires expected-version as a header. Image attachment upload succeeded and inserted markdown, but the next save hit the broken existing-note update path.

Fix: send existing-note save and offline draft sync updates to /api/v1/notes/{id} with the expected-version header. No Python files were touched, so Bandit is not applicable for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed notes image attachment save failure by aligning existing-note save requests with the backend optimistic-lock header contract. Added regression coverage for saving inserted image attachment markdown and updated related notes save/conflict/offline tests. Verification: bunx vitest run the six touched NotesManagerPage test files -- 6 files, 17 tests passed. Touched-file git diff --check passed. Repo-wide git diff --check still reports an unrelated pre-existing trailing whitespace issue in Docs/Design/Agents.md.
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
