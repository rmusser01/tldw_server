# TASK-13263.1: Qodo Jobs, Scheduled Tasks, Chat Macros, Personal Context

Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42
Branch left unchanged; no commits, merges, publication, task edits, or shared plan edits.

## Dispositions

- Finding 5 (discussion_r4001308625), valid, fixed. `_finish` raises `ScheduledTaskPersistenceError` when the terminal status write fails, before notifications/health/audit. Worker catches this RuntimeError-derived error and marks it retryable. Existing consumer handles model errors as durable terminal outcomes. Claim cleanup runs only after execution stopped, allowing a subsequent retry after persistence failure.
- Finding 6 (discussion_r4001308626), valid, fixed conservatively after independent review. Running slots acquire an atomic owner-scoped claim in the existing run summary. Each invocation has a random claim ID plus Jobs job ID and SHA-256 lease fingerprint for diagnostics. Claimant Jobs row must be processing and match owner/current lease, using the worker's configured manager instance. Existing claims are NEVER automatically stolen, including after Jobs replaces a lease: a lease change cannot prove the old executor stopped. Terminal writes and release are fenced by claim ID. Busy or orphaned claims raise central `ScheduledTaskClaimBusy`; the worker logs an actionable reconciliation error and performs no complete/fail/release transition, because the delivery may share the original executor's identical Jobs lease. Only terminal slots return deduped success. The live-overlap regression deliberately suppresses executor cancellation while replacing its Jobs lease and verifies the second invocation remains blocked until the first executor stops and releases its claim.
- Finding 7 (discussion_r4001308630), valid, fixed. Request booleans use StrictBool; purge generation uses StrictInt. JSON booleans/string integers/float integers cannot authorize purge, and string/numeric booleans cannot reach runtime/record service methods. Field-level strictness preserves JSON timestamps and arrays. Existing portable canonical integer parsing already rejects boolean and string values and intentionally accepts finite integral JSON numbers, so that separate contract is unchanged.
- Finding 19 (discussion_r4001308627), valid, fixed. Macro message insert, durable marker, metadata writes and dedupe lookup now share the existing chat database transaction; failures roll back visible messages. Repository/message database identity is explicitly checked. Existing nested transaction handling preserves one outer transaction.
- Finding 21 (discussion_r4001308633), valid, fixed. SQLite chatbooks scheduled-work ordering query includes the acquisition owner predicate and bound parameter. Unscoped behavior unchanged.

## Recovery guarantees and limits

The initial automatic stale-lease takeover design was rejected during independent review after a real-DB probe reproduced overlap. The corrected implementation permits no automatic takeover. Ordinary exceptions/cancellation release the claim only when an explicitly retained executor Future is done. Repeated outer cancellation can interrupt `asyncio.wait_for` cancellation draining: in that case the handler retains the claim, attaches an observation-only callback to consume/log the eventual executor outcome, and requires verified-stopped reconciliation. The callback does not release the claim. A terminal-write failure propagates before success reporting, and its stopped attempt can retry after finally-release. An abandoned process claim intentionally remains blocked until verified-stopped operator reconciliation. No arbitrary external-side-effect exactly-once guarantee is asserted.

### Concrete operator recovery procedure

1. Identify the Jobs job ID, definition/slot, owner ID and run ID from the worker reconciliation warning and scheduled-task run record. Pause/quiesce every worker that could execute that scheduled-task domain/owner (including processes on other hosts sharing these databases). Verify the original process/executor has actually stopped. Lease expiry/replacement, elapsed timeout, or a network partition alone is insufficient evidence. If execution cannot be ruled out, leave the claim in place.
2. Using the same configured user database location, read the run through the supported repository API: `sdb = ScheduledTasksDatabase.for_user(user_id=OWNER_ID)` then `run = sdb.get_run(owner_id=OWNER_ID, run_id=RUN_ID)`. Check the exact definition, slot, owner and `run.run_summary["execution_claim"]` diagnostics; retain its exact `id` for the release. Do not edit SQL or clear claims in bulk.
3. After the stopped-execution verification, call `sdb.release_scheduled_task_run_claim(owner_id=OWNER_ID, run_id=RUN_ID, claim_id=EXACT_OBSERVED_CLAIM_ID)`. Its compare-and-swap semantics refuse to clear a different claim. Read the run back to confirm release. A changed claim ID means stop and investigate, not retry with a new guessed ID.
4. Retry/requeue the existing Jobs job through its supported Jobs lifecycle/admin workflow, preserving the original definition and `scheduled_for` slot. For a still-processing job after all workers are stopped, use the configured `jobs_manager_from_env()` manager's `release_job` with the exact recorded worker ID and lease ID, then allow reacquisition after workers resume. If Jobs quarantined it after lease-reclaim budget exhaustion, use the existing administrative requeue workflow. Do not create a new schedule slot to bypass the old claim. Resume workers only after reconciliation is complete and monitor the durable run's terminal status.

Blocked deliveries intentionally leave Jobs acknowledgement untouched; normal lease recovery/budget handling may eventually surface them for operator action. This is a correctness-over-automatic-recovery tradeoff.

## Changed files

- tldw_Server_API/app/api/v1/schemas/personal_context.py
- tldw_Server_API/app/core/Chat_Macros/jobs.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py
- tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py
- tldw_Server_API/app/services/agent_task_jobs_worker.py
- tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py
- tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py
- tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
- tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py

## Verification

All Python commands used `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` first and ran in the stated worktree.

Red regressions observed before production fixes: coercion reaching service, marker failure retaining visible reply, cross-owner scheduled job changing order, swallowed terminal persistence error, duplicate concurrent executor, absent atomic/fenced claims, stale incoming Jobs lease recovery, worker nonretryable persistence classification, and mismatched macro DB identity. Logs: /tmp/qodo-jobs-red.log, /tmp/qodo-jobs-red2.log, /tmp/qodo-scheduled-red.log, /tmp/qodo-lease-red.log, /tmp/qodo-macro-identity-red.log. Initial regression test harness mistakes were corrected before fixes (missing acquisition argument/result field and pytest.fail BaseException portal teardown).

Final command:
`python -m pytest -q tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py`
Result: 136 passed, 6 warnings, 41.22s; /tmp/qodo-jobs-final-tests.log.

`python -m pytest -q tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py`
Result: 45 passed, 4 warnings, 1.53s; /tmp/qodo-jobs-db-tests.log.

After the non-stealable claim correction the combined suite passed 183 tests (6 warnings, 40.68s). A subsequent repeated-cancellation regression exposed an additional release race; the corrected consumer and Scheduled Tasks database suites then passed 69 tests (4 warnings, 7.74s), including the new test. The other 115 tests in the combined run cover unchanged domains. Pytest also reports pre-existing temporary-directory cleanup warnings unrelated to touched tests. Full repository suite and PostgreSQL were not run.

`python -m ruff check` on the seven touched production files: all checks passed. Black applied only changed line ranges (avoiding unrelated formatting churn). `git diff --check`: clean.

`python -m bandit -r` on the seven touched production files, JSON output `/tmp/bandit_qodo_jobs.json`: exit 0, zero findings, zero scan errors. Existing nosec-comment parse warnings only.

## Independent-review correction validation

- Before correction: live cancellation-resistant executor + replaced Jobs lease regression failed, and worker contention acknowledgement regression failed. `/tmp/qodo-scheduled-overlap-red.log`.
- After correction: all 23 scheduled consumer/worker tests passed, including explicit verified-stopped reconciliation, stale incoming lease rejection, live overlap prevention and busy-claim no-ACK behavior. `/tmp/qodo-scheduled-overlap-green.log`.
- Exception classes now live in central `app/core/exceptions.py`, per Qodo 23 and repository policy.
- Combined suite: 183 passed, `/tmp/qodo-jobs-final-tests.log`. Latest corrected scheduled consumer/database suites: 69 passed, `/tmp/qodo-repeated-cancel-green.log`. Repeated cancellation regression was observed failing before the explicit Future/done-state release gate, `/tmp/qodo-repeated-cancel-red.log`.
- Refreshed Ruff and Bandit on all seven production files: Ruff clean, Bandit zero findings/errors; changed-line Black and `git diff --check` clean.
