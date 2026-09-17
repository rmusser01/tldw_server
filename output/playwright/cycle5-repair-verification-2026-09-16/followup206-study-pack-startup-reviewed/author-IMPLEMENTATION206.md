# UAT206 / TASK13260.144 — StudyPack worker startup default

## Outcome and scope

**Author checks:116 passed,0 skipped,4 warnings,2.10s.** Source/test frozen for independent review. Native acceptance criterion3 remains parent-owned and pending: retain queued job2 and load reviewed source without setting a workaround flag.

Production changes only `tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py`: a local StudyPack predicate delegates to existing `should_start_inprocess_worker`; only the StudyPack spec selects it. Unset/blank flag now follows the enabled Flashcards route. Explicitfalse and sidecar suppress in-process startup. A local hard route gate preserves the current disabled-route contract even when the flag is explicitlytrue. Existing policy retains test-mode default-off, while explicittrue may still opt in during tests with an enabled route. No generic predicate/policy, other worker, producer, queue, worker body or lifecycle ownership changes.

The pre-extraction route-default evidence, native job receipt and no-I/O policy comparison remain at `../uat-study-pack-queued-20260917/DIAGNOSIS.md`. No deliberate harness false override caused the regression. Queue admission remains valid for sidecar deployments and is not changed.

## Frozen identities

- Production: `ede2e5dac545c9cccb139aaa2cb1f15c995dcf6f0da47b9184c8d2fc32ff7afb`.
- New `tldw_Server_API/tests/Services/test_study_pack_startup_default.py`: `90e016d87974220f85145a9c992c6780242365e6b68062a1cdc149ef7853e826`.
- Owned manifest: `c0dacddbcf235d7d5fc1ac209d4aa951e656ff9f0e39d91b0116ec364f076657`.

`owned.patch`, `review-snapshot/`, exact production baseline and RED test bytes are retained. Independent AST comparison proves all pre-existing functions except the provider are unchanged, and the provider differs only in StudyPack's enabled predicate. The helper and policy import are the only additions. No existing test expectations were edited.

## Causal RED

New15 cases against original production: **5 expected failures/10 controls passed,0skip,1.24s**. Three predicate failures cover unset/blank flag default and explicittrue sidecar suppression. Two actual bootstrap/catalog/engine cases independently show missing default registration and inappropriate sidecar registration. The failures are boolean/registered-handle assertions, not timeouts or collection errors.

The explicit truth table covers absent/blank/true/false, disabled route including explicittrue, sidecar, and test-mode default/override. The active integration boundary calls the real startup bootstrap, real complete provider catalog, real StudyPack spec, and real lifecycle engine. It filters the collected graph to this one worker to avoid starting unrelated services; only the worker body is replaced by a coroutine waiting on its real stop event. It proves task registration/name, disabled inventory, running task, graceful phase stop, completed non-cancelled task, and idempotent repeated stop.

## GREEN and static verification

```sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_study_pack_startup_default.py tldw_Server_API/tests/Services/test_startup_study_privilege_jobs_pollers.py tldw_Server_API/tests/Services/test_startup_worker_bootstrap.py tldw_Server_API/tests/Services/test_startup_worker_groups.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py -q --tb=short
```

Result116PASS/0skip. This includes the unchanged study/privilege and Notes worker specs, legacy poller tests, active catalog/bootstrap contracts, content workers and existing media startup-policy controls. Exact argument lists in `commands.json`.

One initial adjacent command named a nonexistent worker-policy test file and collected0 tests (exit4). That command log is retained as `invalid-test-path-command.log`; the corrected command above uses the actual media startup-policy suite and passed. It was a test-path harness error, not a product failure.

Ruff0 baseline/current; Bandit production0 findings/0errors, tests0/0 excluding only B101 assertions. New test formatter and scoped whitespace checks pass. Python AST parses; `static-summary.json` includes attribution and results.

## Handoff and limitations

Independent review and native default-worker execution remain pending. Unit tests start only a controlled stop-event coroutine; no app runtime, provider inference, native process, PostgreSQL database, queue mutation or worker flag/profile change was performed. The existing job may encounter a separate provider/persistence failure once a real worker starts; record it separately. No whole-worker-catalog default audit claim. No task/tracker/git/commit actions by this agent.
