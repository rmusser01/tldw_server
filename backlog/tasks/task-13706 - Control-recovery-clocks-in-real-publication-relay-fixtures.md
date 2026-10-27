---
id: TASK-13706
title: Control recovery clocks in real publication relay fixtures
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3229 exact-head sync-pc-rest fails before the lease-expiry callback runs when a real publication source query consumes the default100ms recovery budget. Distinguish valid production deadline yielding from the fixtures lease/CAS/durability assertions, reproduce with slow I/O, and use the existing injected clock for non-deadline scenarios without raising production or CI limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Controlled slow source I/O reproduces the exact missing staged receipt with current fixture
- [x] #2 Real lease expiry, row limits, source authentication, CAS and durable recovery assertions remain intact under deterministic recovery clocks
- [x] #3 Connected relay/deadline and actual sync shard tests plus lint/Bandit and independent review qualify the repair
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Published c1894fd5a0884959f208ea8f6cd3d1c626d39832; run38718781704 job116370004279 sync-pc-rest1failed454passed183deselected; prior exact f2 same fixturepasses. Failure line355 expired_receipt is None after resultpending, no logged stageerror. Existing relay clock_ns dependency and separate explicit recovery-budget suite govern. ADR assessment:no new durable rule; existing recovery/lease fences unchanged. Independent source diagnosis A requested; root must confirm controlled slow-I/O reproduction beforefix.
Controlled120ms real publication-source probe reproduces exact expired_receiptNone beforefix (1fail1.20s); sameprobe passes1case1.78s afterthree existing clock_ns injections. Real leaseclock/expiry mutation,row limits,timeout args,authentication,CAS,pending envelope,crash and durableapply oracles unchanged. Exactsync-pc-rest455pass183original deselected53.65s Python3.12, including independent deadline coverage. Ruffpass; no new configuredBlackdrift (existing386line drift retained), Bandit56parent-identicalfindings0scanerrors. Independent sourceBclear; final committed reviews/newheadCI remain PR landinggates. CI assertion alone cannot distinguish all early pending causes; causal probe establishes valid prestage deadline yielding path. TaskIDexception stillpending/unapplied.
Broadened the same controlled slow-source probe across the full6case relay file:5pass1fail exposes identical recovery-budget coupling in corrupt-source persisted-attention test (expectedpoisoned,actualpending beforeclassification). Reopened before extending existing clock injection to that second real-I/O functional fixture; its real corrupt bytes, attention persistence, restart and no-stage assertions must remain intact.
Final fullscope:5existingclock_ns injections+2whycomments across2real-I/O functional fixtures; exact oldsource plus these insertions asserted mechanically, all originaloracles retained. Same120ms probe final6/6pass2.08s; finalactualsync-pc-rest455pass183unchanged deselected51.94s Python3.12, including independent deadline tests. Both A/B source reviewers clear complete5clockscope. Ruff0new; mechanicallyzero incrementalBlackdrift; Bandit56parent-identical0scanerrors. Final committed reviews/current-headCI remain landinggates; IDrenumber exception remains pending/unapplied.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed machine-speed assumptions from real CAS/lease and corrupt-source recovery fixtures via five existing relay-clock injections. Preserved real lease expiry, row limits, corruption/authentication, no-stage, pending-envelope, restart, CAS and durable-state assertions. Controlled slow-I/O red/green covers both root paths;6slow-probe and455exact-shard cases pass with unchanged183deselections. No production deadline or workflow change; final exact-head CI required.
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
