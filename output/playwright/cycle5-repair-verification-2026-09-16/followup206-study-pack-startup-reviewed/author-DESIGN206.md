# UAT206 / TASK13260.144 — restore StudyPack startup default

## Contract and minimal change

Actual main uses the declarative catalog. Its StudyPack predicate currently requires an explicit true flag, unlike the pre-extraction route-based default. Causal history/flag/native receipts remain in `../uat-study-pack-queued-20260917`. The existing `should_start_inprocess_worker` policy is the replacement for only StudyPack. Retain a StudyPack-local route gate to preserve current disabled-route suppression even with explicittrue; do not change generic helper semantics. Existing policy supplies unset/blank route default, explicitfalse, test-mode default, and sidecar suppression. No other worker spec, queue producer, worker body, runtime or profile flag change.

## Stages

1. **RED — Complete.** Explicit flag/route/test/sidecar cases against the actual StudyPack spec; real active catalog→bootstrap→engine registration/stop with only the worker body replaced by a stop-event coroutine. No real DB, provider or runtime worker.
2. **GREEN — Complete.** Add one local predicate delegating to existing policy, bind only StudyPack spec, keep other specs and legacy helper unchanged.
3. **Freeze — Author Complete; independent/native acceptance pending.** Existing study/privilege, bootstrap, catalog and worker-policy controls; Ruff/Bandit/diff/AST attribution and independent review packet. Root owns native job2 completion acceptance after reviewed source; no explicit flag workaround and no resubmission.

Owned production `app/services/startup_study_privilege_jobs_pollers.py`; new targeted `tests/Services/test_study_pack_startup_default.py`. Other tests only if a concretely necessary expectation change is reported. No generic defaults, sidecar enqueue rejection, task/tracker, browser, runtime or git edits.
