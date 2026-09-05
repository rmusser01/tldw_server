---
id: TASK-13177
title: Recover and sync running Docs Design notes after branch switch
status: In Progress
assignee: []
created_date: '2026-09-05 16:04'
updated_date: '2026-09-05 16:15'
labels:
  - docs
  - recovery
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2885'
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

Recovery commit ac1ded68080655dcaf32b34927095964c4d69aac was pushed and verified with git ls-remote on origin/codex/notes-rescue-20260905. PR #2885 targets dev. GitHub rejected direct dev push (required pull request, required status checks, frontend license policy check). Merge remains pending CI and requester-owned Change summary per repository policy. Original checkout still uses stale dev to avoid replacing open editor files; requester was asked to confirm all open notes are saved before local checkout reconciliation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recovered additions to 27 running design-note files, including seven unpublished note files from editor/session recovery, and verified line preservation across 32 reconciled filenames. Published exact commit ac1ded68080655dcaf32b34927095964c4d69aac on GitHub and opened PR #2885 for dev. Durable raw and reconciled backups are in /Users/macbook-dev/Documents/tldw-notes-rescue-20260905-090102. Work remains In Progress: required CI and human Change summary gate before merge; local checkout update waits for confirmation that current editor buffers are saved.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
