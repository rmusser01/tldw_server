---
id: TASK-13532
title: Diagnose broad Research Workspace verification failures
status: In Progress
labels:
- knowledge
- research
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Own the exact StudioPane literature Export CSV assertion that failed in the 2156-case run but passed in its isolated same-code rerun. Diagnose with bounded owning evidence; distinguish fixture/readiness cause from product behavior and avoid a timeout-only waiver.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The exact failed and isolated-pass receipts and assertion have a durable owner without assuming canonical-dev inheritance or a product bug.
- [x] #2 A bounded owning diagnosis records the missing-button transition and repairs a proved cause, or documents the exact unresolved condition and next check without weakening assertions.
- [x] #3 Verification, remaining qualifications, PR linkage and final summary accurately describe the resolved or still active scope.
- [x] #4 The exact current migrated-identity restore failure has a bounded owning diagnosis and demonstrated repair or a recorded unresolved condition and next check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final whole-branch review M1: /private/tmp/knowledge-capture-final-review.md; canonical Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md:85. Owning file apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx:1889–1934, case exports structured literature tables as CSV and JSON without advertising XLSX. Broad receipt knowledge-capture-task6-ui-final-receipt.json SHA a41c069994b14887591ce32bfe1789447527f8f0471ba8ee723db73d34445b36:2155pass1fail/2156/exit1; isolated knowledge-capture-task6-literature-final-named-recheck-receipt.json SHA1a00f27ed052b7a53e891da94e7f3c092ed2165db8911544d244d22f34cb95fc:1pass32nonselected/exit0 at same be4 code. Historical Done562/12450 and generic12116 are not reopened or assigned. ADR required:no; diagnosis of an existing test/fixture contract, no new durable rule. ID13532 exceeds max13531 across currentHEAD/origin-dev/52openPRbranches.
Bounded diagnosis at FIX_BASEf1c4c5bb6333ee3c8e0ba2979857ccb1ce16d023 plus capture-only repairs: unchanged StudioPane.literature-workproducts.test.tsx full owning33/33pass/exit0; /private/tmp/knowledge-final-wave-m1-owning-receipt.json and log SHA f4d80baef8bc54ee28b6eddcc566b1bd4afbe9148c7575eb1e96101f15bff60f. Temporary observation-only MutationObserver saw no dialog0ms, modal+Loading output viewer414ms, CSV/JSON1243ms. Existing View→React.lazy ArtifactModalContent/DataTableArtifactViewer→parseMarkdownTable→export controls traced. Original broad failure DOM truncates before modal, so whether module settled or parse returned null is unknown. No timing/order/inheritance/product root cause proved; no product/fixture/timeout/assertion edit. Next check complete modal DOM and module settlement during a bounded reproduction with original broad concurrency/order around this owning file, then repair only demonstrated cause. Full2155/1/2156 exit1 and isolated1pass32nonselected receipts preserved in initial notes/canonical artifact. Remain In Progress; qualification alone is not a demonstrated merge-blocking product defect.
Ruling26 current broad exit1:2249passed1failed/2250 in99files; CSV literature33passed. New exact failure ResearchWorkspace.stage3.test.tsx > restores migrated server identity and sources before creating a workspace on reload: restoreServerWorkspace has zero calls; expanded DOM shows Unable to restore alert, not loading. Log /private/tmp/knowledge-final-wave-broad-current.log SHAe1b7e625f8a3f3a8bc26dc62719ab97093b7039b6e1e7c2b27f350030127b190; full source/config fingerprints /private/tmp/knowledge-final-wave-broad-inputs.json. This task now owns both exact verification conditions as one bounded diagnosis unit; original CSV cause remains unresolved. Ruling26/root explicitly prohibit a second broad replay. Bounded actual restore fixture trace and named reproduction follow; repair only a proved cause without weakening security/restore assertions.
Ruling26 final: exact prior99-file scope/config with DEBUG_PRINT_LIMIT50000 returned2249passed1failed/2250 exit1 in325.233s, log SHAe1b7e625f8a3f3a8bc26dc62719ab97093b7039b6e1e7c2b27f350030127b190. CSV owning33 passed. Exact restore case also fails alone; observation-only actual-restore wrapper proves useWorkspaceStore.getState is not a function. Selector-only fixture lacked real getState/current workspaceSnapshots; only those fields corrected, all assertions retained. Focused full owning38 + restore neighbor36 =74pass/exit0, log SHA3878c9efe25fe7370273010dbf041233a165d18f8714cbd9531db21de71a2f44. No production change after broad; source/config hashes and sole fixture delta recorded in canonical artifact. Actual broad remains failed; no second broad replay. TASK13532 retains historical CSV condition unresolved and records restore fixture issue repaired separately.
Verification and the minimal migrated-restore fixture repair are published in PR3213 https://github.com/rmusser01/tldw_server/pull/3213. Historical broad-only CSV cause remains InProgress with exact failed/isolated-pass receipts and bounded next check; no second broad replay, timing diagnosis or product-cause claim. Final summary and remaining qualifications unchanged.
2026-10-08 publication reconciliation: PR3213 https://github.com/rmusser01/tldw_server/pull/3213 Task8 commit283849d37ca444a8a5b4a178268ec7742c7a8348 independently spec/quality Approved,0Critical0Important, inherited warning Minor only; review /private/tmp/knowledge-pr3213-broad-ci-review.md SHA256 60e8d105e4cc52aab9121e3eaa4e02f04b6fd120b7d153e98764d1e7f63d7b61. Root matched43 final manifest fingerprints, actual RED5/GREEN CI486+58/default544, normal hooks/commit and all25 retained canonical fields. Three finite test-local allowlists plus native generated artifacts, no production/security/workflow/assertion relaxation. Prior seven required SUCCESS statuses at4de are historical; fresh publication-head seven statuses and the two previously failing broad shards remain merge checks. Requester-written Change summary is now supplied, saved verbatim and readback-verified SHA2566a0770a5536192e448f06deaf7b72220756612b872dd2e602e894136fc3ced44; this supersedes earlier human-summary-pending wording. Hosted Linux/amd64 WebUI production run37768510155 job113282687581 at4de passed actual595.0/600 shared and874.5/900 route budgets; complete log SHA25609c8d30c6572534742aab5956baea056884f5252eaf9abf10c28f26607451a09. Product bytes unchanged through Task8. Original historical failures/public429/default/native/CSV qualifications remain; no claim of fresh hosted acceptance or merge. ADR026/042/066 govern; no new durable decision.
Fresh owning33pass and actual historical-predecessor+owning120pass; complete modal/module observations establish cold import dependency, not original cause. House act+vi.dynamicImportSettled exists. CSV+JSON test downloads JSON only. Corrective stage tests delayed real module readiness and CSV payload without timeout increase/assertion weakening/retroactive cause claim. Original failed receipts retained.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Historical broad-only Export CSV missing-button condition remains open with exact receipts and next full-modal/module-settlement probe. The separate migrated-identity restore failure was diagnosed as a missing getState/current-snapshot test double and repaired; owning38+restore36=74 pass with every assertion retained. Original broad failures remain recorded, production inputs unchanged, and no second broad replay or whole-scope green claim. PR3213 contains the bounded fixture repair.
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
