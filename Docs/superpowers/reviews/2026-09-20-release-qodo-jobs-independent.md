# Independent review of Qodo Jobs repairs

Scope: current working diff in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`, reviewed read-only. No repository changes performed.

## Finding: [P1] A replaced Jobs lease can steal a still-live scheduled execution

At `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py:170-178`, `_stale_execution_claim_id` accepts a prior claim as stale whenever its recorded Jobs lease is replaced or ends. That does not establish that the original executor stopped. The consumer uses the returned stale claim at lines 263-275 and starts the executor at line 416. The worker's lease-duration floor at `tldw_Server_API/app/services/agent_task_jobs_worker.py:68-85` has no accompanying renewal or lease-loss cancellation. `asyncio.wait_for` can also exceed its timeout while cancellation unwinds. A reaper or explicit Jobs release/reacquisition therefore allows the new lease to enter the executor while the old coroutine is still live; the terminal claim fence only rejects the old result afterward.

Reproduced with real Jobs and per-user Scheduled Tasks databases, a registered executor blocked on an event, public `release_job`, and a new `acquire_next_job` while the first handler remained active:

```
OVERLAPPING_EXECUTOR_CALLS: 2
FIRST_HANDLER_STILL_RUNNING: True
```

Probe: `/tmp/qodo_jobs_overlap_probe.py`; log: `/tmp/qodo-jobs-overlap-probe.log`.

Command (after activating project venv):
`PYTHONPATH="$PWD" TEST_MODE=true python /tmp/qodo_jobs_overlap_probe.py`

Shared with qodo_jobs and parent. qodo_jobs acknowledged and is implementing conservative non-stealable execution claims, with explicit uncertain/recovery behavior. Suggested retaining the claim until the original handler releases after stopping, and avoiding a successful Jobs completion for an uncertain changed-lease run. Automatic crash recovery requires an explicit verified-stopped recovery transition; lease replacement alone is insufficient.

## Other reviewed behavior

- Terminal persistence errors now propagate as `ScheduledTaskPersistenceError`, and the worker marks that exception retryable using the same configured Jobs manager. No additional concrete retry-completion bug was found on the normal claim-release path.
- Claim acquisition, release, and terminal claim-fence validation use SQLite immediate write transactions; the claim identity guards prevent stale completion or release from overwriting a successor.
- Macro postback enforces identical repository/message database instances and wraps message creation, final marker, and metadata in the outer transaction. SQLite nested transaction handling and BackendManagedTransaction depth handling preserve the outer commit/rollback. The repository's compare-and-set final marker fails on conflicting concurrent posts, propagating rollback. No concrete remaining atomicity or mismatched-transaction issue found.

This report describes the reviewed snapshot before qodo_jobs' follow-up repair; the overlap finding remains open until its replacement logic is inspected or the probe is rerun.

## Follow-up review: non-stealable claims

The updated claim acquisition no longer replaces an existing claim, and nonterminal contention raises `ScheduledTaskClaimBusy`. The worker catches this separately without completing, failing, or releasing the Jobs lease. The new single-cancellation regression correctly verifies that a replaced lease cannot overlap the original executor during its initial cancellation cleanup.

However, the original claim's unconditional release in `handle_agent_task_job`'s `finally` still assumes the outer handler cannot finish before its executor. A **second** cancellation interrupts `asyncio.wait_for` while it is waiting for inner-task cancellation, allowing the outer handler to release its claim even though the executor is still blocked. A replacement lease then acquires the now-empty claim and starts another executor.

Independent reproduction against the updated non-stealable implementation:

```
OVERLAPPING_EXECUTOR_CALLS: 2
FIRST_EXECUTOR_STILL_RUNNING: True
```

Probe: `/tmp/qodo_jobs_repeated_cancel_probe.py`; log: `/tmp/qodo-jobs-repeated-cancel-probe.log`.

Command (activated project venv):
`PYTHONPATH="$PWD" TEST_MODE=true python /tmp/qodo_jobs_repeated_cancel_probe.py`

The probe uses real Jobs and Scheduled Tasks databases and an executor that catches its first cancellation and waits for an explicit stop event. It calls `cancel()` on the handler twice, replaces its Jobs lease, then observes the second executor entering before releasing the first.

**Finding remains open** pending claim retention until actual executor task completion under repeated cancellation. Reported directly to qodo_jobs and parent. No repository edits.

## Final verification — both overlap findings closed

The final implementation retains an explicit `executor_task` Future initialized before entering the claimed-work try/finally, assigns it before awaiting `asyncio.wait_for`, and checks the Future's actual `done()` state before releasing the claim. Before an executor is created, there is no asynchronous suspension between acquiring the claim and installing the cleanup guard. Cancellation before execution either leaves no executor or a completed/cancelled Future and can release safely. Cancellation during execution, including a second cancellation that interrupts wait_for's cancellation drain, leaves a live Future and therefore retains the claim. The completion callback only observes the eventual result/exception; it never releases an uncertain claim.

The real-database repeated-cancellation probe was rerun against the final code with a replacement Jobs lease:

```
REPLACEMENT_BLOCKED_BY_CLAIM: True
EXECUTOR_CALLS: 1
FIRST_EXECUTOR_STILL_RUNNING: True
```

Verification probe: `/tmp/qodo_jobs_repeated_cancel_verify.py`; log: `/tmp/qodo-jobs-repeated-cancel-verify.log`.

Independent focused suite:
`TEST_MODE=true TLDW_TEST_NO_DOCKER=1 python -m pytest --confcutdir=tldw_Server_API/tests -q tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py --tb=short`

Result: **24 passed** (4 pre-existing warnings, plus pytest temporary-directory cleanup warnings). Log: `/tmp/qodo-jobs-independent-final-tests.log`. The suite covers ordinary contention, terminal persistence retry, explicit reconciled recovery, single cancellation with real lease replacement, repeated cancellation retaining a live executor, and busy worker behavior without ACK/fail/release.

**Disposition: both reproduced P1 overlap paths are closed. No remaining actionable issue identified in reviewed scope.** Crash or repeated-cancellation claims intentionally remain blocked until an operator establishes executor quiescence and explicitly reconciles the claim; automatic lease-based crash takeover is deliberately unavailable. No repository edits made by reviewer.
