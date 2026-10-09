---
id: TASK-13568
title: Keep chat recovery actions acknowledged and preserve owned focus
status: In Progress
labels:
- chat
- webui
- extension
- accessibility
- ux-audit
dependencies:
- TASK-13278
- TASK-13567
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the source-confirmed retained-action gaps and native keyboard Stop focus: shared Review fires clipboard writes without acknowledgement/error feedback, Dismiss can remove the focused control, and actual native Stop removes its focused button and leaves BODY active. Preserve journal records, exact text, current source/account/view guards and explicit operation intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copy confirms only an acknowledged clipboard write and exposes an actionable failure without losing exact recoverable text.
- [x] #2 Dismiss handles acknowledged removal and failures without unhandled rejection, replay, or wrong-owner journal mutation.
- [ ] #3 Keyboard Stop and removed Dismiss controls return focus to an appropriate remaining control or composer only while the initiating source and user focus remain owned.
- [ ] #4 Actual full/compact integration and native headless controls qualify focus/acknowledgement while physical AT and packaged limits remain explicit.
- [ ] #5 Closing, skipping, or finishing a chat tutorial returns keyboard focus only to its still-owned initiating control or remaining composer without stealing a newer user focus.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent all-ten TASK13458 remains In Progress. Both authorized independent source reviewers confirm shared Review action seams; actual current13278 buildStop12441 leaves BODY active. Existing13566 Latest/Branch ownership/focus stays credited. Use existing composer registry/native disclosure and source/H1 fences. ADR049/050 ownership/explicit operation lifecycle,059 tracking; no new durable architecture or dependency expected. Design/plan and causal tests required before implementation.
Native13270 final buildJ_E3ie_TkxILUHn1UmyPL real Close now exits correctly/exact draft retained, but removes the focused tour control and leaves BODY active. Add the same bounded owned-return requirement to this focus task; do not reopen completed tour host/Close state logic. Native proof /private/tmp/chat-tour-host-13270-native-close-final.log.
2026-10-09: Continuing after verified TASK-13271 commit98ec98004f. Both independent source reviews confirm Dismiss has only entry-time ownership checks, fire-and-forget UI promises and Retry ignoring cleanup outcome. Use existing HistoryOperationOptions/assertLease inside the existing transaction; return boolean committed acknowledgement and compose captured controller epoch/view/profile/native-or-local lease with caller source guard. Throw after a retired write to roll back. Keep proven completion cleanup optional/offscreen-compatible. Await raw exact clipboard bytes with current-attempt feedback; failed or refused Dismiss retains shown text and never triggers Retry dispatch. UI-only Stop wrappers leave internal abort semantics unchanged. Tour focus handoff is ephemeral and excluded from persisted progress; capture before closing launchers, suppress competing Help restore and return only on explicit current Close/Skip/Finish. Broader palette focus source/ABA and hidden-target checks belong here after13271 preserves the chosen destination. No new durable API/storage architecture rule; ADR049/050 govern ownership,059 tracking, per searched ADR index. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_recovery_ack_focus_13568.md; master accepted all-ten design remains governing. No new dependency, global focus framework or participant claim.
2026-10-09 incremental acknowledgement implementation: shared dismissHistoryTurnRecovery now returns a committed boolean, checks the existing operation lease around profile/read/write awaits, and reuses the existing abort-aware transaction helper through transaction acknowledgement. Retirement after write throws inside the transaction; missing/refused/failed cleanup cannot authorize Retry. Controller captures epoch/view/owner/bookmark namespace/local-or-native lease plus caller source; catches failures while retaining recoveries. Retry consumes ACK and reports an actionable failure without dispatch. Review awaits exact raw clipboard bytes, validates captured live owner/view/source/H1 before access or cancelling prior work, tracks one current attempt, retains text on rejected write/removal and cannot clear newer work. Acknowledged removed Dismiss focus uses the existing composer registry and yields to newer focus/pointer/input, editor/dialog, visibility, source/H1 or unmount. 84 connected suites/2513 pass; final reset dependencies pass59 Review cases; both compilers pass;8 scoped TS baselines0introduced; AST-equivalent owned formatting. Both independent source reviewers clear after their pending-commit and expired-owner entry findings were fixed. DB rollback/pending-commit evidence remains declared transaction doubles; native IndexedDB/clipboard/full/compact remains Stage3. Bandit N/A authored TS only, no repo Python/dependency changes. AC1/2 checked for this bounded implementation; AC3/4/5 and all broad acceptance remain open while Stop/tour/palette/native work continues. Plan Stage1Complete/Stage2InProgress. No automatic replay or whole-program completion claim.
2026-10-09 explicit UI Stop increment: adjacent useExplicitStopFocus reaches full Form/Playground status/runtime/Message/Compare and compact legacy/Message plus all three composer variants. Event forwarding preserves the exact initiating button; textarea Control+Enter and raw queue/internal cancellation remain focus-neutral. Captured source/H1/account/controller/input guards plus monotonic focus/pointer/input and pending identity cancel old returns; only focused removed control after owned idle can request the existing composer registry. 91 connected suites/2610 pass; final fixture layout-effect refinement and actual normal first-bind/aborted-prebind controls separately qualified. Both compilers pass;12 TS scopes0introduced static, AST-equivalent owned formatting. Transport independent source reviewer CLEAR. H1/account settlement in23 focused hook tests is declared doubles; actual provider raw Stop negative is full+compact. Native keyboard Stop and tour/palette remain unqualified Stage2/3; AC3/4/5 stay open, broad parent unchanged. Bandit N/A authored TS only.
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
