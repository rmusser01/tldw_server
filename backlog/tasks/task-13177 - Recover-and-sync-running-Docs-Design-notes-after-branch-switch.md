---
id: TASK-13177
title: Recover and sync running Docs Design notes after branch switch
status: In Progress
assignee: []
created_date: '2026-09-05 16:04'
updated_date: '2026-09-05 16:10'
labels:
  - docs
  - recovery
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve running design notes from the working directory and GitHub Desktop stash after switching to stale local dev; reconcile with current origin/dev and publish without losing either side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Original saved notes and stash versions are backed up outside the checkout
- [x] #2 Recovered notes preserve local and remote content
- [ ] #3 Reconciled notes are committed and verified on remote dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Back up and inventory current files, Desktop stash, and editor recovery sources. 2. Reconcile notes in an isolated checkout based on current origin/dev. 3. Verify preservation, commit and push, and document safe local editing state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recovered saved Sublime Text note buffers in addition to the Desktop stash and current disk. Reconciled 32 note filenames; 27 differ from origin/dev, including seven new files. Every nonblank line from each source and remote is retained in source order; only trailing whitespace/extra end-of-file blank lines normalized. Existing RAG_Links.md receives the # RAG Links.md additions. Complete raw versions and editor undo records are preserved under /Users/macbook-dev/Documents/tldw-notes-rescue-20260905-090102. Original main checkout, old dev history, and Desktop stash remain intact. Verification: preservation assertions passed; diff whitespace check passed; Bandit scans documentation scope (no Python source changes). Application test suite not applicable to note recovery. Remote push verification pending; live editor-only text beyond saved session snapshots cannot yet be confirmed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
