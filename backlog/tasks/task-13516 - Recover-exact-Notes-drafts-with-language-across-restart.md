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
updated_date: 2026-10-07 09:57
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
- [x] #7 Body-only edits during first legacy create or queue replay can subsequently save exact newer source against acknowledged identity/version without a spurious portable-language capability veto; explicit language mutations and ambiguous baselines remain protected.
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
F5 fix round3/5 — controller source-level follow-up after independent round2 closure.
ADR required: yes
ADR path: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (existing); C ADR194 unchanged
Reason: complete the existing source-bound acknowledgement/recovery contract and Ruling32 ordinary legacy compatibility without a new store or state owner.
1. Read task-5-fix3-findings.md and previous report/rulings; trace save and offline replay plus actual snapshot/result/queue consumers.
2. Add meaningful mounted deferred legacy POST controls with body-only newer editing for ordinary Save and queue replay; demonstrate the specific spurious follow-up veto before correction.
3. Correct existing captured acknowledgement/baseline/rekey ownership narrowly so source/version/identity and known/unknown language remain associated while explicit language intent stays protected. No arbitrary existing-null downgrade or metadata borrowing; raise a concrete boundary ambiguity before deviating.
4. Run amended structured-draft, authority-races and actual-page offline-drafting selections plus final whole W typecheck, with scoped immutable-base static/formatter evidence. Do not repeat unchanged C or Monaco native checks without a concrete related change.
5. Append exact RED/GREEN/final commands and output/self-review to task-5-report.md, update concise task notes and commit explicit S unit files. Keep In Progress pending scoped round3 re-review.
Ruling34 (progress.md and task-5-fix3-findings.md): use the existing complete source/identity/version acknowledgement check regardless of language presence; capability/language equality stays portable-only. Baseline association uses only the complete implicit-create reply and leaves explicit language edits protected. Governing existing ADR065 must describe this conservative acknowledgement boundary before changing it; minimal/body-less replies retain pending recovery.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented coherent authority-scoped recovery in the existing v3 browser queue with exact source/revision cleanup and stable retained-error retry. Ordinary Save and queued legacy creation now bind complete acknowledgement language/identity/version to the preserved newer body; absent metadata stays unknown and explicit language intent remains protected. All mutation replies require matching complete source/identity/version before cleanup (Ruling34); portable capability/equality checks remain separate. The actual setter/queue/ack share raw nonempty1..100-codepoint language validation without normalization. Actual New/reset/duplicate ownership and quota/authority fences remain intact. Existing ADR065 governs. Final amended structured/authority/actual-page trio89 passes; whole frontend typecheck exit0 recorded; no introduced scoped lint debt and changed-region formatting verified. Three scoped rounds are clean. Inherited static/formatter debt and later live/OS-kill/editor release gates remain explicit; no new storage/runtime owner.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
F5 complete after independent task review and three scoped fixes: exact recoverable drafts and source-bound legacy acknowledgements qualified by final89 and whole frontend typecheck exit0. ADR065/Rulings31-34 apply; inherited debt and downstream live qualification limits remain recorded.
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
