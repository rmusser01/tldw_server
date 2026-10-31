---
id: TASK-15187
title: Retire sidepanel draft copy work with its source owner and view
status: Done
labels:
- chat
- ux
- extension
- bug
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved Stage4 root repair. ControlRow bounded copy awaits page references, package storage and opening without retaining its source composer/account/history lifetime. Reuse the existing real composer owner/readiness, H1 and draft IDs/address/revision. Retire stale source work and feedback; retain original draft. No package provenance/portable-history authority or new transfer protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copy is unavailable before actual owned draft readiness and captures the real composer/history source fence.
- [x] #2 Held page-reference reads and package creation cannot open a destination after source owner/view/input retirement; the original source draft remains.
- [x] #3 Late open outcomes and old completion cleanup cannot publish feedback or clear a newer pending copy after source retirement.
- [x] #4 Actual ControlRow behavior and production Form guard checks plus affected tests/client gates/reviews pass; provenance/native gates remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest dev7ba48 integrated at2f7fbb8653; destination repair13512 complete at9bb516873d/ac925515a3. The producer ControlRow has one production caller in Sidepanel form. Apply the smallest source guard through this shared caller and fence every awaited boundary. Existing local package service and source retention are reused. Search found parent13457 but no duplicate bounded source task; highest openPR ID13518. New local uncommitted13513 record collided with another openPR and was rolled back before implementation;13519 is canonical.
Failure-first evidence: /private/tmp/chat-handoff-source-red.log (5 actual ControlRow failures), chat-handoff-source-storage-red.log (3 real storage-service boundary failures), chat-handoff-source-form-red.log (6 production capture failures). Root cause: Copy used a surface-wide boolean and did not carry actual composer readiness/owner, H1 selection fence or input revision through awaited page/package/open work. Implemented required production Form wiring and attempt identity; package service checks the captured fence after cleanup/write/read-back and removes its unique unpublished record when retired. Existing shared consume/open helpers are reused; source text is never cleared. Current checks and two independent read-only reviews are running. Task numbering corrected to 13519 after live local/dev/open-PR maximum13518; the erroneous, uncommitted new13513 record was removed before product edits. No other task13513 record was changed.
Review found and reproduced pre-rerender New/ABA/input and H1 epoch recapture races; final guard binds rendered source/selection. Receipt rerender failure reproduced then repaired by retaining completed attempt ownership with pending:false. Final20suites600pass at /private/tmp/chat-handoff-source-final-regressions.log; fresh WebUI types and extension compile pass; no added lint across8TSfiles; diff clean. Both independent final source reviews clear. Latest remote dev7ba48 remains integrated. Master design checkpoint updated and own completed plan retired. Bandit N/A: TS/TSX/docs only; no Python source changed. Native physical storage/tab IO cannot be undone once issued; full Form/native/package/provenance/cross-origin/receipt/D7 gates remain in parent13457.
Renumbered from TASK-13519 to TASK-15187 on 2026-10-31 (TASK-13960): TASK-13519 now names only "Deep RAG pipeline efficiency (reranker singleton, parallel expansion, dedup banding)" (backlog/tasks/task-13519 - Deep-RAG-pipeline-efficiency-reranker-singleton-parallel-expansion-dedup-banding.md). Before 2026-10-31 both tasks used TASK-13519; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bound sidepanel Copy to restored, rendered composer/account/history/input ownership. Fence page, package and open boundaries; retire stale work without affecting newer copies or clearing source text. Reuse shared storage/open helpers and remove unique unpublished retired copies.600 related tests pass, both clients compile/typecheck, no added lint, two independent reviews clear. Parent all-ten remediation remains In Progress.
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
