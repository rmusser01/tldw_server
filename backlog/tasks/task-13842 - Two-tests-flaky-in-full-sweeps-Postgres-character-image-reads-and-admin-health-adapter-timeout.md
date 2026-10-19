---
id: TASK-13842
title: 'Two tests flaky in full sweeps: Postgres character image reads and admin health
  adapter timeout'
status: Done
labels:
- tests
- flaky
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while sweeping the TASK-13560 change (PR #3247), each failing in one run and passing in another with no relevant change: tests/Character_Chat_NEW/integration/test_character_chat_images.py::test_character_image_reads_leave_event_loop_and_release_operation[postgres-*] (also flaky on unchanged dev per that PR's plan review), and tests/Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers (failed on the dev base sweep, passed with the change). Both are timing-sensitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each test is reproduced as flaky (or shown stable) with a repeated run, and the cause is recorded
- [x] #2 A flaky test is made deterministic without weakening what it checks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Image test, test_character_image_reads_leave_event_loop_and_release_operation[postgres-*]: flaky, and how often depends on load. Cause: the backend pool has reset=_reset_pooled_connection, so psycopg_pool returns a checkout through a pool worker (rollback, then RESET ROLE / SESSION AUTHORIZATION / ALL and COMMIT) and only then republishes it with _add_to_pool. The test read raw.info.transaction_status from the test thread during that reset and saw ACTIVE (or INTRANS). The request itself hands back INTRANS by design; the pool rolls it back. This is the same race as TASK-13412. Dev counts (9ddc9d83f4): REP_N=4 (24 cases) 24 failed at load 23-33, all at :378 (23 'ACTIVE' == 'IDLE', 1 'INTRANS' == 'IDLE'). Whole file on dev: 5 failed, 103 passed (1120 s, load 60 falling to 31); the 5 failures were 5 of the 6 postgres cases of this test, all at :378. With an injected 0.3 s reset (scratch slow_reset plugin): 6/6 failed at :378. Sqlite cases pass.
Image fix: the test observes the request's checkout after the pool's own cleanup and just before the pool publishes it for reuse, and requires exactly one such publication, IDLE at that moment. psycopg_pool is observed through its _add_to_pool step; the fallback pool (no psycopg-pool) through an ObservedFreeList on _free, after its synchronous rollback, as test_chacha_postgres_http_operation_lifecycle.py does. GREEN: slow_reset 6/6 passed; REP_N=4 24/24 passed (load 19-28); fallback pool 6/6 passed (load 68-80); whole file 108 passed (2541 s, load 56-106). The check still bites: with a reset that raises, the fixed test fails with 'The pool never put the request's checkout back into service'.
Admin test, test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers: flaky in file order, shuffled order and isolation, and how often depends on CPU load. Cause: the test gave the health check 0.05 s, and the endpoint's asyncio.wait_for covers the whole test_provider call, including 5-50 ms of synchronous pre-dispatch work on the event loop (the real load_server_config_snapshot() at 3-10 ms, the rest of test_provider, event-loop scheduling, and in this test one cold AuthNZ Settings() build of about 4 ms). CPU load stretches that work past 50 ms: failing dev runs were cancelled after 175 and 231 ms, before dispatch, so pool.active_count was 0 at :1481. Admission semaphores were full at test start (no leak). Dev counts (9ddc9d83f4): isolated REP_N=50 4/50 and 10/50 failed, all at :1481, at load 50-58 (review: 7/50 and 6/50 at load 47-56); file order 6/20 (load 49-58); shuffled seeds 1001-1020 2/20 (seeds 1003, 1011; load 49-57). With a 0.15 s injected pre-dispatch delay (scratch slow_predispatch plugin): fails at :1481 every time.
Third test, same cause and file, fixed under this task: test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom (0.05 s at :960) failed at :973 (call_count 0 == 1) and :995 (recovery healthy 2 == 3). It never builds Settings (the test stubs resolve_byok_credentials), and failing runs were cancelled after 56-159 ms of pre-dispatch work. Dev: file order 4/20 (:973 x1, :995 x3), shuffled 0/20; slow_predispatch fails at :973. Fix for both: a 0.5 s budget, about 2x the worst pre-dispatch latency the plan observed (231 ms). It is still sub-second, so api_timeout still rounds up to 1, and the blocked adapters (2 s / 5 s, released only after the timed-out call) still outlast it, so every assertion keeps its subject. Not changed: test_individual_admin_provider_timeout_stays_prompt_but_retains_worker_admission also uses 0.05 s, but admin_test_llm_provider has no outer wait_for and its clock starts inside _run_bounded_health_call (admission_deadline, byok_testing.py:259), after pre-dispatch, so it cannot hit this race (0 failures in 43 planning-time runs).
Deviation from the plan (Review M2 left it at 1.0 s): the sibling's first-batch guard asyncio.wait_for(task, timeout=1.0) is now 1.5 s. With a 0.5 s budget the first batch lasts the whole budget (scratch probe: 0.513-0.540 s in 50 runs at load ~30; with the 0.05 s budget it lasted about 0.06 s), so the guard's headroom fell from about 0.95 s to about 0.46 s. With the plan's code the sibling failed 1 of 20 file-order runs (load 45-48) with TimeoutError at that guard while one provider was still polling global admission (byok_testing.py:269), a failure dev never showed (dev fails earlier, at :973/:995). Deterministic: a scratch plugin that blocks the event loop for 0.6 s from 0.45 s into the batch fails the sibling at the guard with 1.0 s and passes with 1.5 s. 1.5 s restores about 1 s of headroom, close to dev's, and stays well below the 5 s the blocked adapter holds an unbounded call. The recovery guard (1.0 s) is unchanged: a healthy recovery batch took at most 0.043 s and does not wait out the budget. Mutation: with the health timeout made ineffective (timeout 30 s for the endpoint's wait_for and await_bounded_daemon_with_timeout, via a scratch plugin), both fixed tests fail (sibling: TimeoutError at the 1.5 s guard, now :972; main: 'healthy' == 'error' at :1484).
GREEN (final code): slow_predispatch 2 passed; main test isolated REP_N=50 50/50 (load 18-20; 50/50 at load ~50 with the plan's code); file order 20/20 and shuffled 20/20 (load 18-38); admin runtime file plus test_admin_llm_providers_health.py and test_byok_validation_boundary.py 155 passed. Stress, two file-order loops of 20 side by side, twice: 38/40 (load 19-23) and 39/40 (load 28-33). Residual failures, both load: (a) in the first stress set both loops' run 11 failed at the 1.5 s guard at the same instant; both processes logged nothing between 17:27:34.26 and their teardown at 17:27:35.738 (identical to the millisecond), a machine-wide freeze of about 1.5 s; (b) in the second set the main test failed once at :1485 (pool.active_count 0 == 1) after about 0.5 s of cold pre-dispatch (config load and Settings build, with 190 ms gaps between consecutive log lines at load ~33). That is the same failure as dev's at :1481, which dev hits at 4-10 of 50 isolated runs. No in-test wall-clock budget is fully load-independent; an event-loop stall longer than 0.5 s before dispatch, or longer than about 1 s across the sibling's budget, still fails these tests.
pytest-randomly is not installed, so -p no:randomly is a no-op; shuffled order came from a scratch collection hook seeded 1001-1020. Bandit (uvx): image file B101 x40 (dev 39; +1 is the new assert); admin file B101 x142, B105 x8 (same as dev). Ruff clean.
ADR: not needed; a test-only change that creates no durable policy (Docs/ADR/README.md). Governing: ADR-005 (Bandit on the touched paths), ADR-059 (backlog-py), ADR-004 (the owner's Change summary on the AI-authored PR).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both tests were flaky, plus a third with the same cause. The Postgres image test read a returned connection while psycopg_pool was still resetting it on a pool worker; it now reads the connection when the pool republishes it, and must find it IDLE then. The admin health tests gave a 50 ms budget to a call that first does 5-50 ms of synchronous pre-dispatch work, which CPU load stretched past 50 ms (up to 231 ms observed), so the call was cancelled before dispatch; a 0.5 s sub-second budget keeps every assertion and the api_timeout round-up, and the multi-provider test's first-batch guard moved from 1.0 s to 1.5 s to keep its headroom over that budget. With injected delays that reproduce each race, dev fails every time and the fixed tests pass; repeated, isolated, shuffled and stress runs are recorded in the notes. Docs: not relevant (test-only). Known: the 0.5 s budget can still be beaten by an event-loop stall longer than 0.5 s before dispatch (1 of 80 stress runs at load ~33, against 4-10 of 50 isolated on dev), and a machine-wide freeze of about 1.5 s failed the multi-provider test once; the individual-provider 0.05 s test was left alone because its clock starts after pre-dispatch.
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
