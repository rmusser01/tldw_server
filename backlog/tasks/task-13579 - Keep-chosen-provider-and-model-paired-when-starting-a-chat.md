---
id: TASK-13579
title: Keep chosen provider and model paired when starting a chat
status: Done
labels:
- chat
- ux
- root-cause
- setup
dependencies:
- TASK-13578
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue accepted chat UX root queue: public Compact New keeps the selected model but clears an explicit Custom provider to Default. Preserve the chosen pair through the existing Full/shared and Compact reset paths while clearing conversation-specific overrides. Qualify actual resolver behavior and do not claim a failed native request from UI evidence alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starting a new Full or Compact conversation preserves the current chosen provider/model pair while clearing old conversation instructions and custom request/scoped overrides.
- [x] #2 Refused or retired New does not mutate settings; explicit full reset and Default/no-model paths retain their existing behavior.
- [x] #3 Connected source controls, client types/static/security/hooks and native public setup checks qualify the bounded repair without replay or unintended provider/message writes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; ADR025 provider routing, ADR049 original source ownership and ADR059 tracking govern. Repair aligns retained model with its original provider choice and changes no routing/persistence/public protocol. Scope existing model reset and its only two production callers (useClearChat Full/shared, useMessage Compact), real store/caller controls and setup UI. Private13578-provider-pair-{red,after}.json qualifies Custom to Default plus retained local-uat-chat, not failed routing. Accepted master queue13458 remains open; current latest dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db ancestor. No source edit until causal controls.
Pair causal RED5/81 precedes minimal store seed and two New caller fixes; connected5 suites367/367 pass including actual Compact custom wire, live old-callback provider, Default/global scope, subscriber replacement, qualified model precedence and unseeded saved tab restore. Both authorized source reviews bounded CLEAR. Guarded10-file formatting AST unchanged. Wider14-suite pass642/643 has one inherited coordinator testSave0, reproduced exact HEAD. Three diagnostic attempts paused: broadcurrent; exactHEAD; corrected actualForm/H1 placement+partial actual Scene reader stillSave0. Research actual service-prompts/saved-normal Scene adapters and real Full harness; next angle inspect actual submitted outcome and original SDK source contract, never relax production admission or count inherited failure as paired behavior RED. Coordinator fixture adds current model getState IO; source/baseline restoration all completed. Wider verification/types/build/native/hooks/commit remain; all-ten remains open.
Final14 connected643/643, both types, scoped static28/28 zero introduced; current production build/tokens593.2KB shared600/795.4KB chat900 pass. Corrected inherited original standalone SDK fixture uses actual owner/model/strict Scene source, independent command predicate, exact mirror ACK and awaited scoped save; no native fixture authority fabricated. Both public native New pair controls pass, Compact clears synthetic headers/body, Full intentionally saves chosen local preference; zero completion/chat mutation. All9 canonical responses28 rows exact, baselines untouched. Guarded coordinator formatter retains tested source after AST mismatch; unrelated useClear formatting reduced AST-equally. Python source untouched/Bandit N/A. Private13579 proofs/logs and closure ledger updated; normal hooks/closure pending.
Normal exact-owned hooks pass without bypass. Three bounded plan stages complete; own completed plan archived privately and removed. Original SDK baseline fixture limits, unsafe formatter candidate rejection, native setup preference save, nine canonical preservation and broader outstanding queue are explicit in closure ledger. No Python source authored; Bandit N/A. Incremental commit follows, no PR/push/merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
New Full/shared and Compact conversations now retain the live chosen provider alongside the retained model through one atomic existing reset. Previous conversation instructions/request/scoped overrides clear, full reset and saved-destination restoration stay unseeded, and provider prefix precedence remains. Causal RED controls precede repair; final14 connected643 tests, both types, zero introduced static findings, two bounded source reviews, current build/tokens/budgets, normal exact-owned hooks and native public New controls qualify it. Nine original canonical responses28 rows unchanged. Inherited coordinator fixture repaired using original declared SDK/account/Scene/mirror contracts; no native transport claim. Broader all-ten matrix remains open.
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
