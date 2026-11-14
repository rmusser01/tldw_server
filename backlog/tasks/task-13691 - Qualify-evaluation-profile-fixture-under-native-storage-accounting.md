---
id: TASK-13691
title: Qualify evaluation profile fixture under native storage accounting
status: Done
labels:
- chat
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The full PR3229 product-evaluations-unit shard has351passing tests and one profile fixture failure: it deliberately makes get_storage_service unavailable even though profile storage accounting now requires an authoritative native ledger. Keep production fail-closed behavior and evaluation cost-key assertions; provide the smallest declared storage I/O fixture, cover nonzero native/legacy accounting, and qualify the complete affected shard before publishing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the actual profile fixture failure and preserve production native accounting failure behavior
- [x] #2 Evaluation quota assertions pass with declared authoritative storage data and no reintroduced cost keys
- [x] #3 Affected full shard, security/lint checks and two independent source supplements pass before exact-head CI and merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-13458; PR https://github.com/rmusser01/tldw_server/pull/3229. ADR assessment: existing ownership/quota rules and ADR059 apply, no new durable architecture decision. Plan:1reproduce/inspect actual UserProfiles storage contract;2minimal fixture with real boundaries and preserved cost assertions;3qualify complete affected evaluations shard and independent reviews, collect remaining current CI results before another push. CI RED e822 job116188103543 RuntimeError storage not needed for this test.
Original localRED19cases:1failed18pass matches CI acquisition RuntimeError. Smallest fix only test fixture: stdlib AsyncMock spec actualStorageQuotaService, canonical live quota8MB/legacy.25MB/native.5MB/total.75MB vs stale raw1/0; assert exact user7/read-only request and output totals while retaining3original evaluation-cost assertions. A/B independent read-only supplements clear this successful-read fixture; no production fallback change. Focused21cases pass including2existing UserProfile live/fallback tests. Full exact nonUTC evaluations shard qualified; final style uses explicit AssertionError for the new storage condition following existing profile tests, preserving31 baseline Bandit assertions with0introduced/0scanerrors. Final exact shard rerun and remaining current CI collection underway before another push.
Final exact CI-style product-evaluations-unit shard352PASS58.98s underAmerica/Los_Angeles, Python3.12, explicit asyncio/timeout plugins and originalnotjobs/note2e marker; earlier complete352PASS70.84s. Focused21PASS. ExplicitBlack/Ruff pass, normal2file hooks pass; Bandit31identicalbaselinefindings0introduced0scanerrors. A/B clear successful-read fixture semantics and unchanged production fail-closed fallback. Remaining currente822CI results collected before publication; all7requiredgreen, onlyknown oldfixture failure remains so far. Parent13458/13690 retain exactnewheadCI/authorizedmerge gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced obsolete unavailable-storage test setup with a standard-library AsyncMock constrained to StorageQuotaService. The profile consumes same-user read-only live data with distinct nonzero legacy/native totals, verified against stale input counters, while all3evaluation cost-key assertions remain. No production change. Original19-caseRED1fail18pass; focused21pass; completeCI-style nonUTC shard352pass58.98s. Black/Ruff/normalhooks pass; Bandit31baseline0introduced0scanerrors. Both independent reviewers clear source; fresh exact-head CI and merge remain parent landing conditions.
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
