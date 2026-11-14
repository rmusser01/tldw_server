---
id: TASK-13612
title: Repair PR3229 backend CI contract ratchets
status: Done
labels:
- chat
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair concrete backend/precommit/shard blockers without weakening native tenant isolation or invalid-history markers. Record the AuthNZ accounting ledger isolation decision, make explicit audit filters non-truthy, include roleplay tests in all CI matrices, update the stale placeholder assertion, and remove reported trailing whitespace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AuthNZ aggregate accounting exemption is documented/tested distinctly from owner-RLS chat intents.
- [x] #2 An explicitly supplied audit user filter is always scoped and parameterized; obsolete predicate baseline removed.
- [x] #3 Roleplay resume contract enters each matching CI shard; no coverage ignore added.
- [x] #4 Empty failed content retains invalid_content metadata and client correlation; whitespace gates pass.
- [x] #5 Focused red/green checks, security/style, independent review and PR update recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment:no newarchitecture; clarify proposedADR070 existingglobalaccounting authority; acceptedADR050 chatintent RLS unchanged. Userauthorizedallremainingfixes/review/rebase/merge PR3229. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_ci_contracts_13612.md.
Causalratchets3fail29pass11.14s; zeroAuditFilter2fail9deselect2.23s. Green188pass105skip13warnings63.44s; placeholder1pass67deselect7.43s; shardguard5090files0newuncovered across5 matchingmatrices. ActualprodBandit0/HEAD0/errors0; Ruff4/HEAD4 inherited exact; scopedformat/compile/diffclear. Freshboundedreview clear Critical/Important; prosecorrectedunlockedresolution/frozenpoolattribution. Record Docs/Reviews/CHAT_CI_CONTRACTS_13612_2026_10_12.md. WholePRfrontend/H2 gatesstillopen.
Reviewer minorwordingcorrection incorporated: exactkeyresolvecommitted/unlocked; frozenpoolattribution, activeBillingmembership/accountUUID user/admintotals. Normal13filehooks pass no bypass; manualRuff4/HEAD4 exactinherited/Bandit0. Allboundedstages complete; onlyownplanretires /private/tmp/IMPLEMENTATION_PLAN_chat_ci_contracts_13612_completed.md. Exactcommit/push PR3229 follows; no overallreview/CIpassing/mergeclaim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed actual CI contract roots with explicit trusted aggregate ledger isolation, zero-safe parameterized audit scope, roleplay test assignment in5matrices, invalid-content marker assertion and reportedwhitespace.188pass105skip plusplaceholder1pass/shard5090no gaps; causal controls/security/stylebaseline/normalhooks/boundedreviewclear. WholePRremainingfrontend/H2 workstillopen.
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
