---
id: TASK-13601
title: Read canonical owned native attachment context
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the approved H2 private canonical context read prerequisite after explicit role storage. Preserve the original opaque bundle manifest revision, refuse ambiguous legacy roles/order and validate exact owned live references and claims under workspace/conversation locking without reversing operation lock order. Public routes, context mutations, retention publication and rich readiness remain separate qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict immutable canonical context preserves ordered exact reference and asset identities and opaque original manifest revision.
- [x] #2 Private reader requires owned active native conversation/workspace and exact valid role, ordering, claim/candidate/quota descriptors; wrong or ambiguous context fails without writes or byte I/O.
- [x] #3 Shared claim validation preserves existing byte-reader behavior and lock order; causal and connected checks, security/static verification and two independent reviews are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; ADR-049/050 and accepted H2 design govern owner-scoped context. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_native_context_read_13601.md. Scope private fork-backed retained references and known empty native context; retention-backed claims, unavailable markers, context CAS/source fences/public rollout remain pending.
Private strict context reader/model implemented; no writes/bytes/publicready. Original opaque manifest token, all explicit reference roles/disabled/shared/gapped ordering; refuse nullrole/ambiguousasset/fractionalrevision/bundle/projection/owner/quota mismatches. Workspace-convo-candidate-claim order, operation provenance readonly committed/gone; sharedbyteread descriptor AST preserved. Causal27/27PG; green59/59; fractional/projection additional red. Connected361/238+2fixture errors excludedoverallgreen; realtests moved to registered existingfixture, finalaffected88/81 and finalsharedrestore1/1 pass. Ruff4file0/0, actual2prodBandit0/0/errors0/newcodeformat/compile clear; fullOpenAPI2129/3301/pathunion unchanged. Both final sourceCLEAR. Record Docs/Reviews/CHAT_NATIVE_CONTEXT_READ_13601_2026_10_12.md. PhysicalPG/retentionpublisher/CAS/public gates remainopen.
All3AC/6DOD complete in private metadata scope. Initial9file exactnormal hooks0, no bypass; own3stage plan complete and onlyown plan retired /private/tmp/IMPLEMENTATION_PLAN_chat_native_context_read_13601_completed.md. Final8file hooks/localcommit follow. Canonical mutation, retention-backed claims/source fences/publication, publicroutes/readiness, restoredbyteadmission and physicalPG remainunqualified.
Final8file normal exactowned hooks0/no bypass; fresh dev a744577 with0 missingcommits. Only8owned code/test/review/tracking files for localcommit; no PR/push/merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Add strict immutable canonical native context metadata reader and reuse original byte-claim validation without reversing operation locks or widening byte admission. Preserves original opaque revision, exact ordered/shared/disabled roles and restoredgone key; rejects malformed/foreign/stale descriptors. Finalaffected88pass81skip plus sharedrestore1pass1skip, static/security/publiccontract checks and two source reviews clear. Ten-suite run with setup errors explicitly excluded; existing fixture registration corrected. Broader H2/UX gates remain open.
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
