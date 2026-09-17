# Independent StudyPack UAT181 ownership review

TASK13260.118. **Clear: no source changes requested.** The frozen worker and new 21-case regression match the handoff manifest and snapshots after the independent run. Separate UAT197 count controls were executed as prerequisites, but its shared-file delta is not attributed to this ownership patch.

The production delta is one import and a literal independent operation scope around existing database acquisition, actual service call, result construction, and existing cleanup finally. Validation stays before acquisition. The public generation service and SDK remain unchanged. An independent scope avoids taking over a caller's pending connection; it follows the actual cached DB through context-propagating asyncio.to_thread. Existing deferred cleanup protects a still-running source query when the handler is cancelled. No broad commit/rollback was added.

The permanent tests use a real cached DB/accessor, actual source resolver/thread and persistence, delegating pool instrumentation, and observe returns before loop shutdown. Owned, borrowed and legacy caller transaction controls preserve pending visibility and explicit caller commit/rollback on both success and failure. Cancellation measures the current loan rather than an earlier return of the same pooled object. The model call and unused note-only Media handle are fixtures; these are not real-model or native tests.

## Independent verification

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-study-pack-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_study_pack_worker_operation_lifecycle.py tldw_Server_API/tests/DB_Management/test_study_pack_membership_count_backends.py -q --tb=short
```

**29 passed / 0 skipped / 4 warnings / 36.38s**: 21 owned lifecycle cases and 8 separate count cases. Exact receipt: `required-pg-green.redacted.log`. Official disposable PG fixture, existing cluster, no Docker restart and no native data. Fresh Ruff of worker/new test: zero findings. Fresh production Bandit: zero findings/errors. Hash verification: `source-verification.json`.

Reviewed author causal receipts distinguish the initial cancellation observer error from its corrected permanent RED, and the separate197 count failure from checkout lifetime. Original independent diagnosis established an IDLE executor checkout, not an INTRANS lock or native restart blocker. Author adjacent48 is retained and reviewed as author evidence; it was not rerun independently here.

No browser/runtime, source, task, tracker or git edits. This review covers only this adopter; other non-HTTP callers and whole-application ownership remain outside scope. Native acceptance is parent-owned.
