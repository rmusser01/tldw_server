---
id: TASK-15280
title: Retain interrupted retry ownership through recovery lookup
status: In Progress
labels:
- chat
- ux
- history
- recovery
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved all-ten root continuation against latest dev3ca1ff. Incoming interrupted-history retry reads recovery/admission asynchronously before choosing a history cursor, without retaining the rendered owner/draft/view. Reuse the existing H1 fence and draft revision before lookup, recheck before choose, and capture the new fence after its own choice for dismissal/submission. Scope is shared retry behavior and meaningful tests; no new recovery protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A delayed recovery/admission lookup cannot move a replacement H1 view or submit after New, account, draft or same-chat cursor retirement.
- [ ] #2 An admitted current retry still chooses the preceding point, dismisses only current recovery and submits once with the existing admission.
- [ ] #3 Failure-first regression, appropriate shared checks, both clients, no added lint and independent review pass; task and commit record limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Found by authorized independent reviewer while merging latest dev into TASK13522. Callers route through regenerateLastMessage in shared useChatActions. ADR required:no, existing ADR049/050 ownership/fence contracts; no Python authored, Bandit not applicable. Parent13457 remains In Progress.
Failure-first recovery/admission and own held choice tests red6, stale invocation red2, obsolete Compare validation red1 after truthful compareModeActive fixture. Fence captured at render protects invocation; awaited lookup/admission and synchronously captured own-choice fence protect continuation; validation moved after retirement check. Current admitted Retry submits once; actual H1 with declared storage/native I/O and real account event covers retirement. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_interrupted_retry_13523.md; ADR049/050 reused; Bandit N/A TypeScript. Final full shared matrix/client gates pending.
Superseded for ID collision: canonical work record is now TASK-13457.2; archive this historical local record.
Renumbered from TASK-13523 to TASK-15280 on 2026-10-31 (TASK-13960): TASK-13523 now names only "WebUI perf batch W3: storage indexing and data structures" (backlog/tasks/task-13523 - WebUI-perf-batch-W3-storage-indexing-and-data-structures.md). Before 2026-10-31 both tasks used TASK-13523; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
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
