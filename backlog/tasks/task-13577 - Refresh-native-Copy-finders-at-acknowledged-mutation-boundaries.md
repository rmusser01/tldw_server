---
id: TASK-13577
title: Refresh native Copy finders at acknowledged mutation boundaries
status: Done
labels:
- chat
- ux
- root-cause
- d7
dependencies:
- TASK-13576
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue accepted D7 root from TASK13576. Full and Compact pass an existing finder callback into createBranchMessage, but native Copy never notifies it; acknowledged child/message writes and fallible recovery/opening must preserve original account ownership and honest partial/unknown results. Reuse the shared cache helper and scoped native owner; do not add replay, polling or a receipt protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Copy refreshes existing Full and Compact finders after qualified acknowledged changes without reload or duplicate dispatch.
- [x] #2 Missing, failed, partial, foreign, retired-account and held read/journal outcomes preserve trust and existing recovery semantics.
- [x] #3 Actual caller controls, affected tests, both types, static/security/hooks and current native evidence are recorded with broader limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Existing ADR049 governs original scoped ownership/read authority, ADR050 native fork storage and ADR059 tracking. Read-cache notification changes no public API, persistence authority or durable architecture rule. Master13458/all ten remain open. Two already-authorized independent source reviewers trace the original ACK and lease boundaries; no edits or runners delegated.
Causal shared handler/actual Full/Compact RED12/583 controls precedes fix; final four connected suites865/865 pass. Existing independent draftAccount is captured before awaits and qualified by full serverChatMirrorOwnerKey; native view lease is not a refresh guard. Known candidate refresh precedes candidate journal, exact original owner/operation/candidate legacy_completed or partial refresh precedes final journal. Missing/unobserved/foreign final results and recorded replay do not grant extra refresh. Reviewer found concrete awaited active-GET latency/recovery regression: actual QueryObserver RED2; notifier now starts independently with caught failures, original live account closure retained through cache cancellation. Final controls cover active GET not blocking journals, account retirement across held cancels, replacement-current A while old A remains false, New/unmount and recorded partial replay. New real-account fixture originally returned unknown fork_dispatch_unresolved with zero create because its old table engine used row.id; corrected to actual forkOperations compound key. Diagnostics are excluded as product failures; SDK/scope IO remains declared. Bandit N/A TS-only; no new ADR/protocol/dependency. Both final types/static/build/native/hooks pending. Current native fresh source Send succeeds; stale assumed provider reply text wait and pre-confirm zero-request Copy click are excluded, no replay.
TASK13577 verification: final15 connected1405/1405, both types, six TS baseline240/current240/zero introduced findings, two bounded source CLEAR and current webpack/tokens/budgets593.2/795.4 pass. Current public Full/Compact Copy of declared plain native fixtures yields one child/two exact ACK rows each, primed finder refresh, original-ID reopen and qualified cold capture with blank input/no new creation/message writes. Original5 canonical responses and both source responses preserved. Generated ordinary metadata refusal before child creation remains separate lossless Copy root; public mobile overlay handoff/provider pairing/count clarity and original D7/full matrix remain open. Native/harness limits in closure ledger. Bandit N/A TS-only. Latest dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db remains included. Normal hooks/commit pending.
Normal exact-owned hooks pass without bypass; all bounded stages complete. Own completed plan archived privately and retired per repository process. Both current Copy children and original/source canonical responses are preserved. Master13458/all ten remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native Copy now starts scoped, nonblocking finder refresh at validated candidate and original final result boundaries before fallible journaling. Original independent account lifetime survives view adoption and retires on ABA/unmount. Final15 connected1405 tests, both types, zero introduced static findings, bounded independent reviews, current build/budgets, normal hooks and native Full/Compact plain Copy/finder/reopen/cold qualify the fix. Original5 and both source canonical responses are unchanged. Generated/roleplay lossless Copy and mobile/provider/count/full D7/package/AT/study acceptance remain separate required work; no Python authored, Bandit N/A.
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
