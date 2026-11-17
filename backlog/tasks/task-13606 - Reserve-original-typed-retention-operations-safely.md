---
id: TASK-13606
title: Reserve original typed retention operations safely
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the approved private typed retention reservation transaction. Validate exact frozen request/digest/owner, resolve original keys before current-source admission, freeze source identity at INSERT and classify concurrent winners before source reads. New stale reservation rolls back locally; existing pending/committed/terminal identity remains immutable. No claims, quota, lease, bytes or ready/public capability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed frozen request and original source identity reserve once; changed owner/scope/digest/request refuses and replay never rechecks mutable source.
- [x] #2 Genuinely new source admission uses operation-first locks and exact incarnation-bound revision; failed/caught admission preserves caller work and existing preparing keys.
- [x] #3 Causal and connected checks, concurrency controls, security/static and two source reviews are recorded with official PostgreSQL skips and pending publication/Sync boundaries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required no new: ADR049/050 and proposed071 govern immutable keys and source identity. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_retention_reservation_13606.md. Metadata reservation only; lease/quota/claims/publication/Sync/publicready remain pending.
Typedmetadatareservation strictrequest/digest/owner/source binding atINSERT; operationfirst ONCONFLICT winner reclassifies committed/terminal beforesource; preparingresume checkssource/project andstale preserveskey/gen. Genericsourceindependent pendinglookup unchanged. Local savepointfailednew admission preservescallerwork. Causal10/10PG, green10/10, projectionred1/1PG then finalconnected152/137skip12warnings50.88s. RealSQLiteconcurrency +officialPGforcedabsencecontrol, seededcommitwinner/tombstonefixtures honest. TwofileRuff0/0/actual1prodBandit0/0/errors0/newmethodASTformat/compile andbothsourceCLEAR. Review Docs/Reviews/CHAT_RETENTION_RESERVATION_13606_2026_10_12.md. No lease/quota/claim/byte/ready/publicgrant.
All3AC/6DOD boundedmetadatareservation done. Initial7normalexacthooks0/nobypass; own3stages complete, onlyown planretired /private/tmp/IMPLEMENTATION_PLAN_chat_retention_reservation_13606_completed.md. Final6hooks/localcommit follow; broadparent remainsopen.
Final6normalexacthooks0/no bypass; freshdev stilla744577 with0missingcommits. Only6ownedfiles forlocalcommit; no PR/push/merge. Broadmaster remainsopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reserve strict original retention request/source metadata once, resolving original committed/terminal keys before mutable source and reclassifying insertion-conflict winners. Preparing resume checks source/projection without rebasing stale keys; pending lookup remains independent. Local savepoint preserves caller work. Final152pass137skip/manualstaticsecurity/compile/bothsourceCLEAR. No lease/quota/claims/bytes/ready/publicpublication granted.
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
