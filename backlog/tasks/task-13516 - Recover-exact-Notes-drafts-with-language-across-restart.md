---
id: TASK-13516
title: Recover exact Notes drafts with language across restart
status: In Progress
created_date: 2026-10-07 03:08
dependencies:
- TASK-13513
- TASK-13515
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 07:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep unsaved edits recoverable across navigation, rejected saves and process restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Body and source language share one canonical draft revision, including language-only mutation, conflicts and explicit null reset.
- [ ] #2 An empty rejected draft recovers exactly after navigation and restart without claiming a successful save.
- [ ] #3 Recovery is scoped to authority/workspace/note and removed only after its matching complete save; older acknowledgements preserve newer edits.
- [ ] #4 Storage/quota failures prevent discard-producing navigation and show a truthful error; existing storage ownership is reused without diagnostics persistence.
- [ ] #5 Targeted recovery/session tests cover profile isolation, logout, restart and actual copy/history owners touched by this change.
- [ ] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (C); Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (S), existing programme decisions
Reason: implement approved scoped durable-draft recovery and serialized-language ownership; extend the existing decision before storage changes, without a second authority/store.
1. Read task5 brief and actual repo-local task/instructions/lessons/ADRs; trace canonical state, real note_session_port and LibraryScreen wiring, device-state migration/connection owner, and browser offline queue.
2. Add meaningful failing language-only, exact empty-draft restart, older-ack/newer-revision, scope/logout and storage-failure controls before production changes.
3. Carry source/language together through captured draft mutation, save and conflict reload/overwrite; distinguish omission from explicit null and preserve unknown future values.
4. Reuse the existing device-state DB with a dedicated scoped draft table and exact matching-revision cleanup; integrate actual session/navigation owner so storage failure cannot discard a draft. Qualify old schema upgrade and relevant existing consumers; no canonical Notes migration or diagnostics persistence.
5. Extend the existing browser offline queue version/snapshot with language/revision, legacy unknown-mode gating and quota/authority controls. No second storage mechanism or editor adoption.
6. Run named affected C state/session/recovery/device-store and real-port controls plus U structured-draft/authority/offline selections and whole W typecheck; scoped static/security/format comparisons against captured bases, no full sweep.
7. Document actual copy/history ownership, evidence and limitations, self-review and commit explicit C/S unit files separately. Keep In Progress pending independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented versioned language-bearing recovery in the existing authority-scoped browser offline queue. Source and language share revisions, empty/rejected drafts survive navigation and reload, older/incomplete acknowledgements and quota failures retain pending state, and actual Notes page conversion/duplication handlers are source-safe. ADR-065 amended. Final affected Vitest selection: 22 passed; whole frontend TypeScript check passed. Immutable-base ESLint/formatter comparison adds no findings; unrelated pin-order test failure reproduced at immutable base. Remains In Progress for controller review; full evidence and limitations in F5 report.
F5 fix round 1: queue completion validates current note/selection before using editor state; ordinary save and queue sync exclude each other in both orders. Fresh drafts use known null/default mode while loading/legacy remain gated; legacy recovery borrows metadata only from its matching version. Ruling31 pauses automatic error replay until Save/edit requeues, with stable-error and retry controls. Fresh known-null on an incapable server remains locally pending, not Saved. Browser RED5 plus retry RED1 reproduced; affected structured/authority/actual-page coverage passed 30 tests and whole W typecheck passed. Scoped ESLint adds no findings against eeed576b4f. Remains In Progress for independent review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
