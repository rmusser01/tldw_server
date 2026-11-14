---
id: TASK-13599
title: Validate original typed retention operation receipts
status: Done
labels:
- chat
- storage
- correctness
dependencies:
- TASK-13595
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved H2 retention result boundary before production retention admission. NativeForkStore currently validates committed fork receipts but returns committed retention with receipt=None. Reuse existing strict immutable retention wire models and canonical hashing to bind original request digest/owner/scope/operation/source and exact reference map. Duplicate reference identities must refuse. No public retention route, byte readiness, source fence admission, canonical bundle or capability is qualified by this bounded structural result work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real store causal controls reject missing, corrupt or mismatched retention receipts instead of reporting blind committed success; valid original typed identity resolves without source recapture and preserves immutable mapping.
- [x] #2 Canonical retention digest binds semantic original owner/scope/source/fence/ordered reference identities and revisions; duplicate references refuse and existing fork behavior remains passing.
- [x] #3 Supported connected checks, matched Ruff/Bandit/compilation, independent reviews and exact-owned hooks pass; unchanged canonical API contract and broader public/physical acceptance limits are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing ADR049/050 and proposed070 plus approved H2 design §§8–9/Task2.3 govern original retention identity and receipt shape. This unit does not mint a readiness assertion: actual retention publication/source fence, live claim/bytes/role-context manifest and public/client consumers remain unqualified. Root sole runner; supported3.12, official no-Docker PG fixture, source/tests frozen during runner. Two original reviewers only inspect source/evidence; no new browser/profile or baseline writes.
Both independent pre-edit design/source reviews confirm private structural scope/no new ADR under049/050. Tighten originalsource==storedsource==target before target lifecycle mutation can erase substituted-missing target result. Repeated outputasset/revision values remain legal only with identical frozen hash/representation/fidelity; per-reference role/context flags may differ. Projectionversion binding explicit; opaque outputrevision no +1/source recapture/creation-op repinning. Exact tuple and three stages in own plan; actual-store typed-row seed controls are explicitly not actual retention publication or byte readiness.
Initial RED38fail/22officialPGskips contains two test-only setup diagnostics: workspace conversation omitted required scope_type so houseAPI normalizedglobal, and houseexecute_query returnscursor rather thanlist. Correct scope_type and actualfetch helpers, production unchanged, then rerun causal controls. Initial genuine source roots:19unrefused receipt/alias negatives,15missing digest helper,duplicate request accepted and untyped result positives; diagnostics not counted as acceptance.
Corrected causalRED37fail1pass22officialPG skips, green38/22. Final formatted eight-file connected323pass205skip11warnings137.04s; collection/source metadata proves201unavailableofficialPG +4SQLitePGdrivertransaction controls, no passingtests rerun. FourASTformat exact, matchedfilenameRuff0/0/new0, threeproductionBanditbaseline/current0findings/0errors, supportedfourfilecompile0. Bothindependent sourceCLEAR. Normalcanonicalexport/fingerprint/pathunion unchanged2129paths3301schemasSHA3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691; logger-prefixed diagnostics caused initialanchoredgrepexit1 afterexportersuccess, actualartifacts/markersverified. Review Docs/Reviews/CHAT_RETENTION_RECEIPT_IDENTITY_13599_2026_10_12.md records all testsetup/style/metadata diagnostics and boundedstructural fixturelimits. No actualretentionpublisher/claims/bytes/manifestreadiness/publiccapability, server/browser/profile/auth or baselinechange. Allbroadergates open; normalexact-ownedhooks/localcommit and ownplanretirement follow.
Nine initial exact-owned normal repository hooks0 withoutbypass. All3AC/all6DOD and own3stagescomplete; owncompletedplan alone retiresprivate. Manual matchedRuff/Black/Bandit checks remain meaningful because normalhookRuff/Black filters skipthesepaths. Finaleight-file metadata hooks and exact-owned localcommit follow; master/public/readiness/physicalgates remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared private result no longerreports blind committedretention: strict immutable originalrequest/receipt/digest/owner/scope/op/projection/source-target/refmap validation precedes lifecycle mutation; lawful compatiblealiasrefs retained. Causal/connected/property/static/security/unchangedcontract and tworeviews qualify structuralidentity only. Actualpublication/claims/bytes/readiness/public/physical/mastergates stayopen; exact-ownedhooks/localcommit follow.
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
