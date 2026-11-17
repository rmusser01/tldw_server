---
id: TASK-13692
title: Align admin router contract with canonical billing mount
status: Done
labels:
- testing
- ci
- admin
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current full CI main-routing shard reports322passes and one stale billing prefix expectation. Remote dev already mounts billing under /api/v1/admin and forbids legacy public /api/v1/billing. Correct only the obsolete contract expectation, retain actual canonical/public-absence endpoint tests and qualify the complete affected shard without changing production routes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce stale route expectation and verify canonical current-dev mount and forbidden public alias
- [x] #2 Minimal contract update preserves billing route tags and existing security oracles
- [x] #3 Affected routing shard, canonical Billing checks, Bandit/lint and independent reviews pass before exact-head CI and merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-13458, PR3229. ADR assessment: no new durable rule; honor existing admin-only billing mount and OSS public billing removal, plus ADR059 canonical task edits. Evidence: e822CI job116188106582 expected/api/v1 vsactual/api/v1/admin; production admin.py byte-identical to remote6cb, canonical Billing tests assert3adminroutes present and allpublicbilling absent. Plan1reproduce/inspect source contracts;2one literal expectation fix;3qualify connected322+1routing group and canonical route tests, independent exacthead review. Collect current CI remaining results before publication.
OriginalRED1failed matches actualCI. Minimalone-literal correction only: expectedbillingprefix/api/v1/admin; productiongroup unchangedfromcurrentdev. Focused3testsPASS includeactual3adminbillingroutespresent andalllegacypublicroutesabsent. B independentreadonlyreview confirms governingcontract. Full7-file323case mainrouting qualification running. Bandit1926identicalbaselinefindings0introduced0errors. Ruffcurrent/baseline checked underidenticalpath/config; Blackwholefile alreadyuncleaninHEAD, formatterdeltaidenticalexceptintendedliteral, no unrelated11k-lineformatting rewrite.
Complete exact7filemainrouting group323PASS167.77s, plus3actualbilling mount/public-absence checksPASS11.18s. B independent review confirms1literalcorrectness; no productionroutechange. Bandit1926baseline0introduced0errors, Ruff0baseline/0current; wholefileBlackalreadyuncleanonHEAD and no addedformattingdelta. Source ready; final combinedA/Breview and exactnewheadCI/merge follow collectedPostgreSQL failures.
Independent reviewers A and B clear the one-literal fixture correction; production mount unchanged. Exact main-routing shard 323 passed; focused contract/canonical-path/no-public-alias 3 passed. Ruff 0 new; existing whole-file Black delta unchanged; Bandit 1926 baseline findings unchanged, zero introduced, zero scan errors. Commit 9a881a05d2. Publication batches with TASK-13693/13694; PR merge still blocked by current-head full CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected billing router contract expectation to the canonical /api/v1/admin mount; retained admin authorization and removal of public billing aliases. Exact 323-case routing shard and focused real-route assertions pass.
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
