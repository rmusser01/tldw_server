---
id: TASK-13892
title: 'RG reserve: a lost script reply does not count the script''s own lease'
status: Done
labels:
- resource-governance
- redis
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the clean review of #3288 (M3). After TASK-13859, when the lease script or the multi-key rate script runs but its reply is lost (a redis-py TimeoutError after the 5 s socket_timeout, or a connection reset), reserve falls through to the Python path, which counts the members the script just wrote. Leases: with a limit of 1 the request is refused with retry_after=120, and the script's member blocks the user's only slot until it expires; with a limit of 2 the request is admitted holding two slots. Rates: the multi-key script's members are charged again under new member ids and never refunded. Before TASK-13859, dev admitted with no lease and the member was stranded the same way. ADR-078's Consequences record it. Fix: before counting in Python, best-effort ZREM the members planned for that script; they are this reserve's own UUIDs, so removing them is always safe. Needs Redis to stall past socket_timeout, so it is unlikely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a lease or multi-key rate script whose reply is lost, reserve removes that reserve's planned members before the Python fallback counts, with a real-Redis test where the script runs and the reply raises redis.exceptions.TimeoutError
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md (branch fix/rg-redis-unreachable-cooldown, with TASK-13882).
Fixed: when the lease script or the multi-key rate script fails, reserve removes the members it planned for that script, one ZREM per key and best effort, before the Python path counts. They are this reserve's own UUIDs, so removing them is always safe, and a no-op when the script never ran.
Measured on dev (private Redis): with evalsha patched to run the script and then raise redis TimeoutError, a lease at a limit of 1 was refused and the script's lease stayed after release, (allowed, held, after release) == (False, 1, 1); a requests reserve left 2 window members for 1 request. A real lost reply (DEBUG SLEEP 6 landing just before the EVALSHA, 5 s socket timeout) did the same: lease refused (retry_after 60), rate charged twice.
Correction to the description: with a limit of 2 the request never holds two slots, because the Python path re-adds the same member ids. With no other lease it was admitted holding one slot; with one lease held it was refused (retry_after 119). So at a limit of N the request was refused once the user held N-1.
With option (b) of TASK-13882 (owner Decision 1: a PING confirms a timeout): a reply lost to a stall shorter than two socket timeouts does not start the cooldown, so this ZREM reaches Redis. Measured with the real lost reply: lease and rate admitted and counted once, under fallback_memory and fail_closed. In a longer outage the PING fails too, the ZREM and the Python count both skip Redis, and the request gets its fail mode; the script's members then expire (a rate member with its 60 s window, a lease at its TTL).
Tests: test_a_lost_lease_script_reply_does_not_count_its_own_lease and test_a_lost_rate_script_reply_charges_the_request_once (real Redis). RED 2 failed, 29 passed; GREEN 31 passed. Each ZREM's mutant fails its test (plan dry run).
Docs: ADR-081's Follow-up says when this cleanup reaches Redis; ADR-078 is not rewritten (accepted ADRs are not).
Verification: shared with TASK-13882 (same PR): sweep 785 passed, 1 skipped, 2 xfailed; Docs and ratchet 213 passed; ruff and bandit clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
When the lease script or the multi-key rate script fails after it may have run, reserve removes the members it planned for that script before counting in Python. A request whose script reply is lost no longer counts or charges itself twice: a user's only slot goes to that request, instead of being refused and blocked for up to two minutes.
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
