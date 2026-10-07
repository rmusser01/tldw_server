---
id: TASK-13528
title: Explain and retry chat draft account verification
status: Done
labels:
- chat
- ux
- extension
- drafts
dependencies:
- TASK-13457
- TASK-13527
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the silent unknown-owner composer state in the approved all-ten chat UX program. Existing account verification pending/error/Retry is dropped by composer adapters; recovery is disabled with no route to retry and an old saved receipt can outlive authority loss. Expose the existing status/action on both deployed forms through shared recovery UI, keep persistence/import/send gates intact, and retire stale save acknowledgements at their shared source.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing owner pending/error/ready and Retry travel through actual shared composer and WebUI adapter into both deployed recovery surfaces; pending is described, failure offers a usable Retry even while recovery is disabled, and ready removes the notice.
- [x] #2 Unknown-owner verification failure and Retry preserve current same-config unsaved text without owned storage or copy consumption; success enables restoration/saving and explicit copy review without automatic import.
- [x] #3 Known-owner transient rechecks remain ready; explicit credential/account changes, authorization loss and old held generations retain existing clearing/retirement fences.
- [x] #4 Draft saved acknowledgement cannot remain or revive across disabled persistence, unknown owner or replacement owner; current successful persistence still acknowledges normally.
- [x] #5 Failure-first actual hook/component/caller evidence, affected regression checks, both client type gates, no-added diagnostics, source/public English locale conversion and two independent reviews are recorded.
- [x] #6 Master design and Backlog records preserve ADR assessment, proof boundaries and remaining program gates; the work is locally committed with the parent In Progress.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest origin/dev26ae4fd freshly fetched and included in the TASK13527 checkpoint. ID13528 exceeds remote highest13515, all50 open PR highest13526 and local13527. Duplicate search owner verification found related completed session/UI work but no equivalent current shared composer repair; broad TASK13457 remains parent. Governing design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Temporary plan Docs/Design/IMPLEMENTATION_PLAN_chat_owner_verification_13528.md. ADR assessment: no new ADR; ADR049 scoped conversation authority, ADR050 owner/storage lifetime fencing and ADR059 tracking apply. Reuse useChatDraftOwner verificationState/retryVerification and ChatDraftRecovery; add no network endpoint, registry, protocol, persistence system or dependency. Native extension/backend/assistive-technology journeys remain outside component fixture evidence.
Read-only native check now reaches actual worktree Next dev /chat in a background IAB tab; existing older backend8000 is in /private/tmp/chat-workspace-real-uat-latest-dev-20260929, so this is not latest-dev backend or successful-generation proof. Actual failed verification shows status+enabled Retry, recovery/Send disabled; Retry retains the entered marker text and shows no saved receipt. New retry button44px and no document horizontal overflow at320x740 and390x844. A separate existing native root is confirmed on width change: textarea44px client height,64px scroll height, overflow hidden at320 clips a wrapped line; trace/fix will be a separate task after this checkpoint. Initial stopped preview/config/navigation failures were recovered with existing quickstart config; temporary browser error page cannot be bound by the Browser Use data-URL policy. Native screenshots displayed, full save/load/roleplay and packaged extension/AT gates remain open. Combined regression currently18suites483pass with1preexisting shell-contract load failure; portable path and exactHEAD syntax fixture corrections in progress, no assertions dropped.
Final verification:491/491 in19 suites; both client types pass; all12 authored TS/TSX files have zero added ESLint findings. Authored formatting, official English asset/public conversion and diff checks pass. Both independent reviewers clear the bounded source and evidence. Shared red20/37 controls; recovery red2/1; Form red1; adapter/call-site red4/3. Portable shell fixture failed unchanged at03a902f8e2; after root correction the exact checkpoint reproduced two obsolete syntax-contract failures with six controls, and current8/8 retain the guarded submit/module assertions. Bandit N/A for authored TS/TSX/JSON/docs. Evidence/ADR assessment and retired plan stages preserved in the master design. Native retry failure/layout proof is bounded: current worktree frontend, older existing backend, no latest-dev save/load/generation or packaged worker/AT proof. Shared textarea width-clipping root remains a separate next task; parent13457 In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Forwarded the existing account status/Retry through the shared composer into both WebUI and extension recovery UI, without changing owner resolution or persistence/import gates. Retired stale saved receipts at their shared source. Added causal behavioral checks, repaired exact-checkpoint verification fixtures and passed491checks plus both client types and independent reviews. Local checkpoint includes task/master evidence; parent program remains In Progress.
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
