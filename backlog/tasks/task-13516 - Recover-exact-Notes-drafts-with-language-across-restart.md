---
id: TASK-13516
title: Recover exact Notes drafts with language across restart
status: Done
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
updated_date: 2026-10-07 08:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep unsaved edits recoverable across navigation, rejected saves and process restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Body and source language share one canonical draft revision, including language-only mutation, conflicts and explicit null reset.
- [x] #2 An empty rejected draft recovers exactly after navigation and restart without claiming a successful save.
- [x] #3 Recovery is scoped to authority/workspace/note and removed only after its matching complete save; older acknowledgements preserve newer edits.
- [x] #4 Storage/quota failures prevent discard-producing navigation and show a truthful error; existing storage ownership is reused without diagnostics persistence.
- [x] #5 Targeted recovery/session tests cover profile isolation, logout, restart and actual copy/history owners touched by this change.
- [x] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
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
Implemented coherent language-bearing recovery in the existing authority-scoped browser queue (v3), with matching source/revision cleanup, quota/authority fences and stable retained-error replay until Save/edit. Untouched fresh defaults carry narrowly validated provenance and retain ordinary legacy-create compatibility; explicit/null/future metadata remains capability-protected. Actual New/reset and duplicate handlers preserve loading ownership and known/unknown mode/provenance. Governing ADR065 amended before serialization. Evidence: final post-format structured/authority/actual-page trio53; whole frontend typecheck exit0 recorded; no introduced scoped ESLint, direct changed-region/test formatting evidence. Initial task review and two scoped rounds are clean. Inherited whole-file/static debt and baseline pin-order fixture limitation remain disclosed; no external live-server/OS-kill or downstream formatter adoption claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
F5 complete after independent review and two scoped fixes. Exact authority-scoped drafts, legacy creation and duplicate provenance are covered by final53; whole frontend typecheck exit0. ADR065 applies. Known inherited static/formatter debt and later live/editor qualification gates are recorded in the F5 report and ledger.
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
