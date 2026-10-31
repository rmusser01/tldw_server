---
id: TASK-15281
title: Preserve unknown assistant save acknowledgement as failed
status: In Progress
labels:
- chat
- ux
- persistence
- recovery
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved root continuation while merging latest dev3ca1ff. Shared chatModePipeline marks generation complete before assistant persistence and maps a rejected/lost assistant acknowledgement to submitted, then normal keep may retry the write automatically. Use the existing acknowledged resultId boundary to distinguish completed generation from saved completion, preserve generated-unsaved review and bound creation without replaying an unknown append. Cover actual ordinary lost ACK plus persisted/lost receipt and acknowledged/offscreen controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed or lost assistant save acknowledgement returns failed and retains the bound original input/result for review without automatic duplicate append.
- [ ] #2 An acknowledged reply remains submitted even if later local cleanup fails; interrupted, Stopped and offscreen retention continue to work.
- [ ] #3 Failure-first checks, affected shared suites, both clients, no added lint and independent review pass; task and commit record verification limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Found by authorized independent source reviewer; existing actual normalChatMode regression reproduces returned submitted despite rejected assistant append. ADR required:no, existing ADR049/050 and existing resultId receipt; no protocol/new dependency. Bandit N/A authored TypeScript. Parent13457 still In Progress.
Plan Docs/Design/IMPLEMENTATION_PLAN_chat_assistant_save_ack_13524.md. Failure-first: actual ordinary lost ACK persistedfalse/true + shared receipt/helper controls red4; acknowledged cleanup follow red1. Minimal fix uses existing resultId after acknowledged exact-ID settlement before cleanup; complete outcome requires receipt; normal follows known saved result before partial keep. Keep account-retirement precedence because caller success effects are not all owner-fenced; explicitly signal-doubled account control and actual offscreen local retention/reopen controls under verification. Bandit N/A authored TypeScript.
Actual acknowledged navigation control exposed completed-tombstone ordering: recover dismissed the record before finish could retain the settled ID. Reassessed source end-to-end; reused resultId as the sole acknowledgement marker, removing redundant settled boolean, and retained complete+settled_message_id through existing ordered record writes until follow/reopen. Account retirement remains failed for caller safety but retains its known saved old-owner receipt. Independent reviewer agrees this avoids repeat settlement. Control verifies actual local saved row/follow and reopen without another assistant.
Superseded for ID collision: canonical work record is now TASK-13457.3; archive this historical local record.
Renumbered from TASK-13524 to TASK-15281 on 2026-10-31 (TASK-13960): TASK-13524 now names only "WebUI perf batch W4: HTTP architecture" (backlog/tasks/task-13524 - WebUI-perf-batch-W4-HTTP-architecture.md). Before 2026-10-31 both tasks used TASK-13524; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
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
