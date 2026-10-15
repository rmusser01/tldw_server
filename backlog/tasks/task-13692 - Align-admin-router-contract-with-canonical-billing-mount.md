---
id: TASK-13692
title: Align admin router contract with canonical billing mount
status: In Progress
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
- [ ] #1 Reproduce stale route expectation and verify canonical current-dev mount and forbidden public alias
- [ ] #2 Minimal contract update preserves billing route tags and existing security oracles
- [ ] #3 Affected routing shard, canonical Billing checks, Bandit/lint and independent reviews pass before exact-head CI and merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-13458, PR3229. ADR assessment: no new durable rule; honor existing admin-only billing mount and OSS public billing removal, plus ADR059 canonical task edits. Evidence: e822CI job116188106582 expected/api/v1 vsactual/api/v1/admin; production admin.py byte-identical to remote6cb, canonical Billing tests assert3adminroutes present and allpublicbilling absent. Plan1reproduce/inspect source contracts;2one literal expectation fix;3qualify connected322+1routing group and canonical route tests, independent exacthead review. Collect current CI remaining results before publication.
OriginalRED1failed matches actualCI. Minimalone-literal correction only: expectedbillingprefix/api/v1/admin; productiongroup unchangedfromcurrentdev. Focused3testsPASS includeactual3adminbillingroutespresent andalllegacypublicroutesabsent. B independentreadonlyreview confirms governingcontract. Full7-file323case mainrouting qualification running. Bandit1926identicalbaselinefindings0introduced0errors. Ruffcurrent/baseline checked underidenticalpath/config; Blackwholefile alreadyuncleaninHEAD, formatterdeltaidenticalexceptintendedliteral, no unrelated11k-lineformatting rewrite.
Complete exact7filemainrouting group323PASS167.77s, plus3actualbilling mount/public-absence checksPASS11.18s. B independent review confirms1literalcorrectness; no productionroutechange. Bandit1926baseline0introduced0errors, Ruff0baseline/0current; wholefileBlackalreadyuncleanonHEAD and no addedformattingdelta. Source ready; final combinedA/Breview and exactnewheadCI/merge follow collectedPostgreSQL failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
