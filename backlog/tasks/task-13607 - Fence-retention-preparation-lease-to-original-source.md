---
id: TASK-13607
title: Fence retention preparation lease to original source
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Grant retention attempts after typed original-source reservation in the same owner transaction, using time after lock waits and an outer savepoint. Preserve committed/terminal recovery, pending leases and candidate reconciliation. Reject fractional generation at the shared seam; no quota/claims/publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed lease resolves saved outcomes first and checks original source before generation, capturing clock after admission with local rollback containment.
- [x] #2 Active lease stays pending; expired candidates require reconciliation and old or malformed generations cannot create new work.
- [x] #3 Causal/connected/concurrency/rollback tests, static/security and two reviewers qualify private lease with physical PG/publication limits explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required no new: ADR049/050 and proposed071. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_retention_lease_13607.md. No candidate mapping/prepared bytes/quota/claims/readiness/public permission from lease.
Typedsource reservation+postwait clock+genericlease inouter savepoint, savedresult skipssource/clock/claim. Sharedstoredgenerationint>=0 fixesactual0.5→1.5. Pending/reconcile/actualemptycleanup→nextgen/oldworkerdeny preserved. Causal5/5PG, green21/21, targeted8/8, final160/162skip73.57s +explicitclockpoison saved2/2PG2.40s. Ruff3file0/0/actual1prodBandit0/0/errors0/ASTformat/compile andbothsourceCLEAR. Record Docs/Reviews/CHAT_RETENTION_LEASE_13607_2026_10_12.md. No quota/chargedbytes/claims/publication/publicready.
All3AC/6DOD boundedlease boundary done. Initial8normalexacthooks0/nobypass; own3stages complete/onlyown planretired /private/tmp/IMPLEMENTATION_PLAN_chat_retention_lease_13607_completed.md. Final7hooks/localcommit follow; broadparent remainsopen.
Final7normalexacthooks0/no bypass; dev advanced to2a8ffa duringchecks (2commits notification/schema/fingerprint), separateintegration next. Only7ownedfiles forlocalleasecommit; no PR/push/merge. No latestdev integratedclaim yet.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fence retention lease to original typed source under outer rollback containment with time after admission. Preserve saved recovery, pending/reconciliation and old-generation fencing; reject fractional stored generation at shared seam. Final160pass162skip+clockproof2pass2skip/manualstaticsecurity/compile andbothsourceCLEAR. No quota/prepared-byte/claim/publication/publicready grant.
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
