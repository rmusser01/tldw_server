---
id: TASK-13602
title: Remove reviewed native reference from context atomically
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved private remove_from_context CAS prerequisite. Match original manifest/reference/asset revisions under canonical context locks, exclude only the selected reference, advance manifest and history fences atomically, return canonical metadata and preserve bytes/shared claims/quota/receipts. Replace/restore refuse until original-hash and verified replacement publication are qualified; public routes/admission remain pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact reviewed removal changes only selected reference context participation and reference revision, plus opaque manifest/history fences, and returns canonical context.
- [x] #2 Stale/foreign/closed/invalid requests and caller failures write nothing; same-claim peer references, bytes, claims, quota and creation receipts remain unchanged.
- [x] #3 Causal/connected checks, scoped static/security and two independent reviews qualify the bounded mutation; unsupported replacement/restoration and public/physical PostgreSQL limits remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required no: ADR-049/050 and accepted H2 govern atomic context mutation. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_native_context_remove_13602.md. No new revision column, original hash invention, byte release, source job cancellation or publicready. Original-hash mapping and physical replacement qualification remain separate.
Private remove-only CAS implemented with strict revalidation/exact3fences, safe canonicalreaderlocks, selectedref flag/rev, opaque UUID bundle fence andexistinghistory advance, canonicalresult. Local savepoint contains everytailboundary evencaught errors/parentcallerwork; outerrollbackpreserved. Bytes/peerrefs/claims/quota/receipts/gonekeys remain. Causal13/13PG, green49/44, finalconnected177/177skip13warnings86.41s and finalstrengthenedcaller3/3PG1.93s after setupUUID diagnostic excludedCAS. Manual2file Ruff0/0/newmethodformat/AST/compile and actual1prodBandit0/0/errors0, both finalsourceCLEAR. Review Docs/Reviews/CHAT_NATIVE_CONTEXT_REMOVE_13602_2026_10_12.md. Replace/restore/sourcefence/retentionpublisher/publicadmission/physicalPG remainunqualified.
All3AC/6DOD complete in bounded private remove-only scope. Initial7exactfile normalhooks0/nobypass; own3stagescomplete, onlyown planretired /private/tmp/IMPLEMENTATION_PLAN_chat_native_context_remove_13602_completed.md. Final6filehooks/localcommit follow; broaderparent/public/physical gates remainopen.
Final6file normalexacthooks0/no bypass; freshdev stilla744577 with0missingcommits. Only6ownedfiles forlocalcommit; no PR/push/merge. Masteracceptance remainsopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Atomically remove only the reviewed native reference from prompt context under exact manifest/reference/asset CAS, preserving bytes/shared references/claims/quota/receipts. Rotate opaque manifest and advance existing history fence; local savepoint contains everytailfailure and preserves unrelated caller work. Finalconnected177pass177skip plus correctedcaller3pass3skip, manualstatic/security/compile and bothsourceCLEAR. Replace/restore and public retention/send remain unqualified.
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
