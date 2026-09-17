# UAT181 / TASK13260.118 — StudyPack job owner handoff

## Frozen bounded change

`tldw_Server_API/app/services/study_pack_jobs_worker.py` imports `chacha_operation` and puts one literal `with chacha_operation(independent=True)` around the existing acquisition-through-finally body of `handle_study_pack_job`. Validation remains before acquisition. The existing service, result, cleanup order, exceptions, transaction decisions and WorkerSDK loop are unchanged. `ast-equivalence.json` confirms that removing this import and wrapper produces the baseline module AST exactly.

An explicit per-job owner follows the actual cached DB into `asyncio.to_thread`. It returns that job's checkouts, including deferred cleanup after cancellation of an active source read, without adopting a caller's owned, borrowed or legacy connection. The public generation service itself is not wrapped or changed. This is bounded non-HTTP adoption after committed HTTP/maintenance ownership `07e0abf1c4`; it is not whole-application ownership completion.

Owned paths:

- `tldw_Server_API/app/services/study_pack_jobs_worker.py` — SHA256 `f818bd84f9ddebb63ac42d117bd65c6f6a4eb51584b91f4cccf51377c44a98df`.
- `tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py` — SHA256 `554d98227c8884a8ba326f957a021c60df93e196d2c24736bc16d513df1c0ba4`.

No other production files belong to this unit. Separate UAT197 fixes two count-row accesses needed by the actual successful persistence cases; its packet is `.tmp/uat197-repair-20260917`. Sidebar's concurrent Character ownership/schema work is also separate. `source-manifest.json`, `review-snapshot/`, `owned.patch` and the baseline `worker-before.py` pin this unit.

## Causal evidence and test fidelity

The independent retained original probe in `.tmp/uat181-study-worker-lifetime-20260917/REPORT.md` was 1 expected failure / 2 controls / 0 skips. It established an open **IDLE** source-thread checkout after real handler cleanup and before event-loop shutdown. It did not establish an INTRANS lock, pool exhaustion, native StudyPack failure or provider quality.

Permanent controls use the official required PostgreSQL fixture, a real cached `CharactersRAGDB`, real accessor and default maintenance, actual source resolver and `asyncio.to_thread`, real generation prompt/parser and persistence. Only the model call and unused note-only Media handle/provider configuration are fixtures. Pool instrumentation delegates the real get/return methods; observations occur while the loop/executor remain alive. No backend results or cleanup return values are faked.

Retained chronological RED evidence in `.tmp/fresh-uat-recovery-20260916/`:

| Receipt prefix | Result | Interpretation |
| --- | --- | --- |
| `uat181-study-pack-first-red` | 9 failed / 4 passed / 0 skipped, 22.60s | Two real source/model checkout failures; two owned caller rollback failures; four separate197 count failures; one cancellation observer error. Four SQLite controls passed. |
| `uat181-study-pack-cancellation-red` | 1 failed / 12 deselected / 0 skipped, 7.28s | Corrected current-loan observer: query completed but its checkout was not returned before loop shutdown. |
| `uat181-study-pack-borrowed-red` | 2 failed / 15 deselected / 0 skipped, 4.53s | Existing cleanup retired the caller's explicitly borrowed state; later caller command raised ClosedChaChaOperationError. |

The first cancellation observer compared against all prior returns of a pooled raw object; a previous loan of that same object made the assertion invalid. Its log remains retained as a harness failure. The permanent observer captures the return-list index at entry to the current loan, then pauses actual backend execution, cancels the handler, proves no premature return, releases the real query, and waits for return of that exact loan before loop/executor shutdown. It does not infer cleanup from fixture teardown. Additional final legacy caller cases are positive controls, not claimed as initial RED cases.

## Final verification

**29 passed / 0 skipped / 4 warnings / 42.23s** in `uat181-worker-197-counts-green.redacted.log`: 21 new worker cases plus 8 separately owned197 count cases.

The 21 worker cases cover PostgreSQL and SQLite source/model failure; successful real persistence and repeated three-job cache reuse; PostgreSQL success/failure across owned, explicitly borrowed and legacy pending writes with caller-selected commit/rollback; cancellation while the actual source query is in flight. Success assertions check persisted pack/deck/membership/card content. Caller assertions check INTRANS, pending local visibility, independent committed visibility, no caller return, and the final caller decision.

**48 passed / 0 skipped / 9 warnings / 20.63s** in `uat181-study-pack-adjacent-green.redacted.log`, five existing files: worker, generation service, storage, source resolver and source-error fallback. These retain late-persistence rollback, regeneration/supersession, default provider/model behavior, failed Media acquisition cleanup and SDK cancellation coverage. No existing test expectations were changed.

Exact commands, repository root (requires approved local fixture network access):

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-worker-197-counts-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat181-study-pack-adjacent-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py tldw_Server_API/tests/StudyPacks/test_generation_service.py tldw_Server_API/tests/StudyPacks/test_study_pack_storage.py tldw_Server_API/tests/StudyPacks/test_source_resolver.py tldw_Server_API/tests/StudyPacks/test_source_resolver_db_error_fallback.py -q --tb=short
```

Static checks: scoped Ruff 0 current / 0 baseline findings (baseline replay via actual logical filenames); both new test files already Ruff-formatted; four touched Python paths parse/compile without execution; scoped diff check clean. Bandit production (worker + combined shared ChaCha) 0 findings / 0 errors; both new tests 0 / 0 with B101 excluded for assertions only. Bandit emits existing nosec-comment warnings in the large shared file; these are not findings or parse errors. JSON receipts are retained in this packet. No SQL, permission or provider-policy behavior was changed by the worker patch.

## Status and limits

Author tests/static checks complete; exact source/tests frozen for Retry031 independent review. Root owns task status, integration and native acceptance. No runtime, browser, live provider, live profile, tracker, staging or commit action was performed here. No other non-HTTP adopter was changed. Fake generation validates wiring/persistence, not model quality.
