# Phase 3.3 Remaining Review Plan

## Goal

Finish the remaining Phase 3.3 work by using a review queue rather than another broad sweep. Each remaining raw-fallback candidate is treated like a review item with a clear label: `approve now`, `defer`, or `reject/out of scope`.

This plan keeps the same conservative-plus rule already used in Phase 3.3: only sanitize branches that are covered by existing tests or by cheap, direct tests added in the tranche. Preserve validation-facing `400/422`, not-found `404`, conflict `409`, existing safe `500` details, public response contracts, and success-path observability unless a focused test explicitly pins the intended behavior.

## Current Baseline

- Worktree: `.claude/worktrees/phase3.3-error-handler-adoption`
- Branch: `worktree-phase3.3-error-handler-adoption`
- Latest local tranche: `02abe602a4` `Phase 3.3: sanitize jobs prune scheduler`
- Dirty state at planning time: only the untracked implementation artifact `Docs/superpowers/plans/2026-04-28-remaining-phase3-3-parallel-implementation.md`
- Recent verification pattern: focused red/green tests, touched-source Bandit, `git diff --check`, and a local checkpoint commit per tranche

## Review Item Format

Use this format before implementing any remaining candidate:

```text
label: approve now | defer | reject/out of scope
source file:
exact branches:
public contract risk:
existing tests reviewed:
new tests needed:
verification command:
why:
```

The source file is the review trace. The whole source-file tranche gets the label, not individual lines scattered across unrelated modules.

## Selection Rules

- Prefer private background workers, maintenance loops, and scheduler branches.
- Prefer branches with no HTTP payload, DB row payload, or admin-visible persisted error changes.
- Prefer branches that can be tested with direct monkeypatch tests and logger stubs.
- Defer giant modules, user-facing endpoint details, persisted failure metadata, egress policy diagnostics, success-path logs, and anything where raw text is part of a visible contract.
- Stop a tranche after three failed attempts and record the blocker in this plan rather than widening scope.

## Approved Next Batches

### Batch A: Workflows Artifact GC

label: `approve now`
source file: `tldw_Server_API/app/services/workflows_artifact_gc_service.py`
exact branches:
- file delete warning when `Path.unlink()` raises
- per-artifact warning when row handling or DB delete fails inside the artifact loop
- outer loop warning when artifact listing or setup fails
public contract risk: low; private worker logs only
existing tests reviewed: no focused service tests found
new tests needed:
- direct async worker test for file-delete failure with fixed warning and `error_type`
- direct async worker test for per-artifact failure that proves artifact ID and raw exception text do not leak
- direct async worker test for outer loop failure that proves raw backend/path text does not leak
verification command:
`python -m pytest -q tldw_Server_API/tests/Services/test_workflows_artifact_gc_service.py`
why:
This is the best next medium tranche. It has three related private-worker branches, limited file scope, and no public API response changes.

### Batch B: Workflows DB Maintenance

label: `approve now after Batch A`
source file: `tldw_Server_API/app/services/workflows_db_maintenance.py`
exact branches:
- Postgres VACUUM failure warning
- SQLite WAL checkpoint skipped debug
- SQLite PRAGMA optimize skipped debug
- SQLite VACUUM failure warning
- SQLite maintenance and outer loop warnings
public contract risk: low; private maintenance worker logs only
existing tests reviewed: no focused service tests found in the initial search
new tests needed:
- direct tests with fake workflow DB/backend objects for SQLite branch failures
- one direct test for outer setup/loop failure
- optional Postgres branch test only if it stays cheap with a fake backend transaction
verification command:
`python -m pytest -q tldw_Server_API/tests/Services/test_workflows_db_maintenance.py`
why:
It is still private-worker territory, but it has more branches and fake object setup than Batch A. Keep it second.

### Batch C: Jobs Metrics Service

label: `approve now after Batch B if still in scope`
source file: `tldw_Server_API/app/services/jobs_metrics_service.py`
exact branches:
- blocking `run_forever()` reconcile warning
- async reconcile loop debug warning
- async SLO gauges loop debug warning
public contract risk: low; private metrics worker logs only
existing tests reviewed: broad search did not surface a dedicated jobs metrics sanitizer file
new tests needed:
- direct loop tests using fake `JobsMetricsService`/`JobManager` pieces and patched wait helpers
- avoid full DB setup unless an existing jobs metrics test fixture already makes it cheap
verification command:
`python -m pytest -q tldw_Server_API/tests/Services/test_jobs_metrics_service.py`
why:
It is a reasonable medium tranche, but it is more likely than Batch A/B to need careful loop control in tests.

## Narrow Conditional Batch

### Batch D: Sync Log-Only Tail

label: `defer until after private-worker batches`
source file: `tldw_Server_API/app/api/v1/endpoints/sync.py`
exact branches:
- `/sync/get` unexpected outer exception log
- transaction rollback/apply-batch unexpected logs
- single-change SQLite/column inspection logs
public contract risk: medium to high
existing tests reviewed: `tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py`
new tests needed:
- direct route-level or processor-level tests for log-only branches
- explicit assertions preserving existing returned error lists where raw backend details are currently part of response behavior
verification command:
`python -m pytest -q tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py`
why:
The remaining sync hits are not all log-only. Some paths build strings that are returned in `errors`, so this should be a separate deliberate sync contract tranche rather than mixed into worker cleanup.

## Deferred Or Out Of Scope For Phase 3.3

label: `defer`
source file: `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`
why:
Remaining branches mix egress policy diagnostics, retry observability, persisted DLQ `last_error`, URL handling, and admin-visible troubleshooting value. This belongs in a webhook/DLQ-specific contract pass.

label: `defer`
source file: `tldw_Server_API/app/services/storage_cleanup_service.py`
why:
The remaining branches span file deletion, TTS history marking, quota/accounting, cycle stats, loop control, and shutdown behavior. It is too broad for the remaining Phase 3.3 pass unless split into a dedicated storage cleanup plan.

label: `defer`
source file: `tldw_Server_API/app/services/outputs_service.py`
why:
Large service with user-facing outputs behavior, persisted output/media error paths, LLM summaries, topic classification, and cache behavior. Needs a separate service-specific review.

label: `defer`
source files: `audio_jobs_worker.py`, `audiobook_jobs_worker.py`, `core_jobs_worker.py`, `jobs_webhooks_service.py`, `connectors_worker.py`
why:
Large worker modules with persisted job states, retry behavior, external provider details, or user-visible failure metadata. Handle later with module-specific plans and broader tests.

label: `defer`
source files: `admin_*_service.py`, `auth_service.py`, `registration_service.py`, `org_invite_service.py`
why:
Administrative/authentication surfaces have security and audit implications. Continue only with endpoint/service contract tests, not opportunistic log edits.

label: `reject/out of scope for Phase 3.3`
source files: `ebook_processing_service.py`, `podcast_processing_service.py`, `xml_processing_service.py`, `enhanced_web_scraping_service.py`
why:
These are ingestion/provider processing modules rather than the remaining conservative error-handler-adoption core. They may be valid sanitizer work, but not part of closing this Phase 3.3 branch.

## Execution Stages

### Stage 1: Batch A (Complete)

Goal: land `workflows_artifact_gc_service.py`.
Success criteria:
- new focused test file covers all three approved branches
- service logs fixed messages plus `error_type`
- no raw file paths, artifact IDs, tokens, or exception text in touched fallback logs
- focused pytest, source-scope Bandit, and `git diff --check` pass
Result:
- Landed in `workflows_artifact_gc_service.py` with direct coverage in `tldw_Server_API/tests/Services/test_workflows_artifact_gc_service.py`.
- Verification passed with `3 passed`, source-scope Bandit clean, touched-source raw warning scan clean, and `git diff --check` clean.

### Stage 2: Batch B (Complete)

Goal: land `workflows_db_maintenance.py` if fake DB setup stays small.
Success criteria:
- direct tests cover SQLite inner failures and outer loop failure
- optional Postgres fake-backend branch if it is cheap
- fallback behavior and stop-event loop semantics unchanged
- focused pytest, source-scope Bandit, and `git diff --check` pass
Result:
- Landed in `workflows_db_maintenance.py` with direct coverage in `tldw_Server_API/tests/Services/test_workflows_db_maintenance.py`.
- Verification passed with `6 passed`, source-scope Bandit clean, touched-source raw warning scan clean, and `git diff --check` clean.

### Stage 3: Batch C (Complete)

Goal: land `jobs_metrics_service.py` only if loop tests remain direct and stable.
Success criteria:
- direct tests cover blocking reconcile warning and async loop warnings
- no real DB setup introduced unless existing fixtures make it cheap
- focused pytest, source-scope Bandit, and `git diff --check` pass
Result:
- Landed in `jobs_metrics_service.py` with direct coverage in `tldw_Server_API/tests/Services/test_jobs_metrics_service.py`.
- Verification passed with `3 passed` for the focused sanitizer file, `3 passed` for adjacent jobs metrics coverage, source-scope Bandit clean, touched-source raw warning scan clean, and `git diff --check` clean.

### Stage 4: Sync Decision (Complete)

Goal: decide whether the sync tail is still Phase 3.3.
Success criteria:
- identify log-only branches versus returned-error/public-contract branches
- either land one narrow log-only sync tranche or explicitly defer sync tail to a Phase 3.4 sync contract plan
Result:
- Landed the isolated `/sync/get` unexpected outer exception log because it is log-only and keeps the existing sanitized `500` response detail.
- Deferred the transaction rollback, batch-apply, single-change SQLite, and column-inspection branches because their raw exception text is coupled to returned `errors` lists or per-change diagnostics and needs a sync contract pass.
- Verification passed with the focused regression, the full sync endpoint error file, source-scope Bandit, source-only raw scan for the `/sync/get` log, and `git diff --check`.

### Stage 5: Closure (Complete)

Goal: close the Phase 3.3 remaining-review queue.
Success criteria:
- all approved-now batches are either landed or documented as blocked after three attempts
- every remaining audited candidate is categorized as defer or out of scope
- final touched-scope pytest and Bandit pass
- `git diff --check` and `git status --short --branch` are clean except intentional untracked planning artifacts
Result:
- Approved-now batches A, B, and C landed with focused coverage.
- The narrow `/sync/get` log-only tail landed; remaining sync processor branches are deferred to a sync contract pass.
- Remaining non-sync candidates in this review queue are categorized as defer or out of scope above.
- Final verification passed with `29 passed` across the remaining-review test sweep, source-scope Bandit clean with `results=[]`, the targeted raw scan clean, and worktree status clean except the pre-existing untracked implementation artifact.

## Commit Strategy

Commit each landed batch separately:

- `Phase 3.3: sanitize workflows artifact gc`
- `Phase 3.3: sanitize workflows db maintenance`
- `Phase 3.3: sanitize jobs metrics loops`
- `Phase 3.3: classify remaining sanitizer candidates`

Do not push unless explicitly requested.
