---
id: TASK-15188
title: Keep unqualified WebUI handoff references out of active history
status: Done
labels:
- chat
- ux
- webui
- extension
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved Stage4 continuation: current WebUI fragment receiver first tries origin-local history IDs and directly sets payload serverChatId despite absent source account/server/workspace provenance. A foreign local read failure also blocks transferable draft handling. Remove these unqualified references from history authority, preserve destination selection, explain explicit destination history selection, and keep transferable draft handling independent. Reuse current payload/service and destination library; do not invent a portable-reference contract or enable unsupported native capabilities. Full fragment conflict/owner/settings/receipt work remains parent13457.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Incoming origin-local IDs are not looked up as destination history or allowed to block transferable drafts.
- [x] #2 Payload serverChatId is not directly assigned to active history; destination ownership remains with existing qualified selection.
- [x] #3 Actual receiver behavior and affected shared tests/client gates/reviews pass; broader fragment review/native/backend limitations are recorded.
- [x] #4 A visible explanation names the actual Chats sidebar and tells users to select a conversation available here; it does not imply foreign history exists at this destination.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red-first actual Playground integration: /private/tmp/chat-webui-handoff-reference-red.log (unqualified local/server reference behavior) and chat-webui-handoff-foreign-draft-red.log (exact transferable draft blocked by failed foreign local lookup). Removed both fragment authority paths; parser still exposes hints only for status. Independent review corrected conceptual Chat Library copy to the actual visible Chats sidebar label and identified coordinator assertion of legacy unsafe server-ID behavior. Updated receiver/coordinator coverage, officially synced two intended English keys and restored unrelated public locale bytes. No new portable identity/ownership/transfer protocol. Broader fragment settings/document/context/draft conflict/lifetime work remains parent13457.
Final3suites116pass at /private/tmp/chat-webui-reference-reviewed.log; fresh WebUI types/extension compile/no-added lint3TSfiles/locale byte and agreement/diff checks pass. Both independent final source reviews clear. Production search finds reference hints only in status UI; qualified H1 and settings-return mechanisms remain separate. Current remote dev7ba48 verified unchanged and included. Master checkpoint updated; own completed plan retired. Bandit N/A TS/TSX/JSON/docs. Actual Playground root/coordinator tests use declared surrounding IO and mocked Form; full fragment owner/readiness/conflict/settings/document/receipt/native/D7 gates remain parent13457.
Renumbered from TASK-13520 to TASK-15188 on 2026-10-31 (TASK-13960): TASK-13520 now names only "WebUI perf batch W0: baseline harness" (backlog/tasks/task-13520 - WebUI-perf-batch-W0-baseline-harness.md). Before 2026-10-31 both tasks used TASK-13520; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unqualified fragment local-history lookup and direct server-ID selection. Transferable draft handling cannot be blocked by source-local IDs, while destination selection retains its existing authority. Added accurate history-not-transferred status naming the real Chats sidebar.116 affected tests pass, both clients compile/typecheck, no added lint, official English locale sync verified, two reviews clear. Broader fragment/native/all-ten gates remain open.
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
