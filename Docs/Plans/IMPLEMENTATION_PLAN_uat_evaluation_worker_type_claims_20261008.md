# Evaluation worker atomic type claims

**Task:** TASK-13260.278.18.83.45 (root-owned). Root recorded and read back this separate before-edit unit before this plan.

**Problem:** Recipe and Embeddings A/B workers both use evaluations/default and currently claim any job type. They can lease and fail each other's valid jobs. A/B owns both run and delayed cleanup jobs, so one scalar-only filter cannot represent its existing contract.

**Design:** Add a nonempty immutable `job_types` tuple alongside the existing scalar `job_type` on the normal WorkerSDK loop, manager single-job acquisition and backend command. Reject ambiguity and malformed/duplicate filters before polling or database effects. Bind one allowed-set predicate inside the current atomic candidate selection on SQLite/PostgreSQL, and use the same scope for expired recovery, adaptive duration history and SQLite's future-job ordering precheck. Recipe uses its existing scalar; A/B supplies both existing types to one serial SDK loop. Keep prepared/batch/admin APIs, enqueue/default queues, ordering, lease/CAS, quota/RLS, counters/outbox, startup/shutdown and business handlers intact.

**ADR check:** ADR required: no. [ADR-003](../ADR/003-jobs-vs-scheduler-default.md), [ADR-021](../ADR/021-services-lifecycle-startup-and-shutdown.md) and [ADR-058](../ADR/058-jobs-completion-row-identity.md) govern existing Jobs ownership, lifecycle and locked completion identity. This extends an internal selection filter without changing those rules or any wire/storage/config contract.

**Scope:** Seven production files: Jobs/operations/contracts.py; Jobs/manager.py; Jobs/worker_sdk.py; Jobs/operations/sqlite/lifecycle.py; Jobs/operations/postgres/lifecycle.py; Evaluations/recipe_runs_jobs_worker.py; Evaluations/embeddings_abtest_jobs_worker.py under tldw_Server_API/app/core. New tests/Jobs/test_typed_worker_claims.py plus only the genuinely affected recipe stop-watch fake in tests/Evaluations/test_recipe_runs_jobs_worker.py. This plan only; parent task and other frozen scopes are untouched.

**Qualification limits:** Three declared issue attempts maximum, normal configured plugins, activated existing Python3.12.13, outer600/case300. SQLite plus canonical isolated_test_environment/jobs_pg_dsn on retained PostgreSQL port15492 with no Docker startup. No private DB construction, dependency install, live/profile/API/model/browser/log/Git/proof actions. Root owns independent review, controlled runtime load and actual UAT acceptance.

## Stage 1: Real claim-routing regression

**Goal:** Establish that actual worker entrypoints deliver foreign jobs through real SDK/manager claims.
**Success Criteria:** Setup works and unchanged production fails only observable routing/lease-state assertions.
**Tests:** Same-queue higher-priority foreign work for Recipe and A/B entrypoints; SQLite/PostgreSQL and both existing acquisition modes. Only business-handler seam is replaced, while claim/lease/completion/stop behavior stays real.
**Status:** Complete

## Stage 2: Atomic allowed-set filter

**Goal:** Apply the minimal validated predicate and configure both owning callers.
**Success Criteria:** A/B run+cleanup are processed serially in existing queue order; recipe claims its exact type; foreign rows stay unchanged; all relevant scope and guard controls pass.
**Tests:** Malformed/ambiguous filters; scalar/unfiltered compatibility; owner/domain/queue/time/dependency controls; pause/gate/quota; stale CAS and lease identity; counters/outbox; expired recovery scope; adaptive duration scope; SQLite scheduling precheck; PostgreSQL concurrent/locked claims.
**Status:** Complete

## Stage 3: Qualification and review freeze

**Goal:** Qualify genuinely affected existing tests, lint/security/syntax and hand off frozen source/tests for independent review.
**Success Criteria:** Bounded red/green ledger; no new touched Ruff/Bandit findings versus baseline; exact hashes; root independent review. Root runtime/live gates remain explicitly open.
**Tests:** New module and affected claim/SDK/worker controls; scoped compile/Ruff/Bandit baseline/current including new tests.
**Status:** In Progress

**Attempt1 — fixture/setup failure, not causal red:** 4failed/4skipped/74deselected,35warnings,2.46s pytest/4.693s outer. Test priority0 violated the unchanged supported1..10 CHECK before any claim. PG cases were skipped by the normal RUN_JOBS opt-in gate, which the command omitted. Production remains unchanged. Correct only fixture priorities (foreign1/cleanup2) and enable the existing RUN_JOBS gate for declared qualification. Attempt2 is corrected causal red; attempt3 will combine new green and genuinely affected controls.

**Attempt2 — causal red:**8failed/84deselected,190warnings,24.81s pytest/29.224s outer, natural1/no skips. All actual entrypoint cases on SQLite/official isolated PostgreSQL and both acquisition modes received the foreign valid type (Recipe received embeddings_abtest_run; A/B received recipe_run). No fixture/setup failures. Production changed only after this evidence.

**Pre-test static/operator corrections:** Scoped Ruff found the missing import of existing RECIPE_RUN_JOB_TYPE in the newly typed recipe call; that exact import was added before attempt3. An optional empty-scalar fixture insertion script guard then rejected the formatter's layout before writing that test. No behavior was exercised by the guard and no optional fixture was added. These are not causal-red evidence or additional test attempts. The original two inherited I001 worker import warnings remain unchanged.

**Attempt3 — current green/affected qualification with fixture failures:**111passed/6failed,1685warnings,91.30s pytest/96.355s outer, natural1/no skips. All eight actual routing regressions passed, as did 25 genuinely affected existing cases. Four new scope cases failed before claiming because test queue `other` is rejected by canonical evaluations admission (allowed default/high/low). Two PG concurrency cases initialized fresh managers concurrently inside the worker threads; actual schema DDL deadlocked / broke the synchronization barrier before a claim, rather than exercising the intended acquisition boundary. No production claim defect is established by either fixture failure. Production/test hashes stay frozen pending reassessment; 111 successful cases are retained, with no broad replay planned.

**Three-attempt reassessment / stopped path:** Attempt1 was the priority/gate fixture error; attempt2 established the real eight-case causal red; attempt3 qualified current source and exposed the two new fixture designs above. The declared three-attempt path is complete. Research of existing PG lease-recovery concurrency tests shows they construct each manager serially before submitting only acquire calls to ThreadPoolExecutor. Existing acquisition operation tests likewise use already-initialized managers/connections and preserve a real locked/concurrent claim. Canonical admission already advertises `high` as a supported alternate evaluations queue. The smallest next scope is test-only: use high for the exclusion control and construct managers serially before the unchanged barrier/real claims. Retain all routing/row/identity/ordering assertions, all111 passes and all production bytes. No automatic fourth run; root coordination is pending for a narrowly declared six-case fixture qualification under the standing human all43 scope.


**Separately authorized fixture-only qualification:** Root recorded and read back a distinct before-edit note after the completed three-attempt STOP/reassessment. Under standing human all43 authorization, correct only the unsupported alternate queue to `high` and serially construct the same two real managers before the existing barrier/concurrent claims. No assertions, seven production files or affected stop fake change. Run exactly the six previously unqualified node IDs once with normal plugins/RUN_JOBS, canonical isolated PostgreSQL15492/noDocker, outer600/case300. Preserve the111 passing cases without replay. This is a bounded new fixture scope, not a reset of the original attempt cap.


**Authorized six-case fixture qualification — green:**6passed/183warnings,20.29s pytest/24.691s outer,natural0/no skips. Exactly the six prior fixture-blocked IDs ran once. Together with the retained111 passes,117distinct cases are qualified (92new plus25 genuinely affected existing cases); there was no whole117 replay or single all117-green command. All8 routing regressions are green. The original three-attempt STOP/reassessment and separate before-edit authorization remain part of this ledger. No remaining failure in the declared source/fixture scope.

**Final static qualification:** All9 touched Python files compile in memory under activated Python3.12.13. Scoped Ruff retains exactly the same two inherited I001 worker import findings and introduces none. STATIC Bandit under the existing primary Python3.11.13 environment reports0scanerrors,0newmodulefindings, and122exactly matching baseline normalized-code fingerprints (1B608 in manager,119B101 and2B105 in the existing recipe tests). No new security finding. Reports/comparators were held in memory; no proof files or environment changes.

**Freeze/review:** Seven production hashes and affected stop-fake hash are unchanged after the source qualification freeze. Final newtest SHA256 `bcaae4490d9399fe9189c2b9641c401d5a569028e33d69b58b3d904054f9344e`. Independent reviewer read the complete production changes and full92-case module/stop fake and reported CLEAR before the approved fixture correction; final two-fixture delta review and root review are pending. No runtime/profile/API/model action was performed. Root owns controlled load, worker readiness and actual Recipe UAT acceptance; Stage3 remains In Progress for those open gates.
