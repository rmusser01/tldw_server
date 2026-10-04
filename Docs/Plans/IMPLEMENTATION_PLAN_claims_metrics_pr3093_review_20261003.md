# Claims Metrics PR 3093 Review Implementation Plan

> For agentic workers: use superpowers:executing-plans for inline execution and superpowers:requesting-code-review before publication.

**Goal:** Resolve PR #3093 review feedback, preserve approved contracts, verify the rebased branch, and merge after all gates pass.
**Architecture:** Claims keeps aggregation ownership; shared Jobs owns execution and admission. Reuse one successfully initialized Jobs manager per callback, retain legacy date parsing, and preserve metadata-only diagnostic logging.
**Tech Stack:** Python, pytest/Hypothesis, APScheduler, Loguru, SQLite/PostgreSQL, shared Jobs, GitHub CLI.
**Backlog:** TASK-9935.4
**ADR Check:** No new ADR required. This review and rebase preserve the Jobs ownership boundary governed by `Docs/ADR/003-jobs-vs-scheduler-default.md`. Task-format repairs follow `Docs/ADR/059-backlog-py-task-editor-cutover.md`; no durable architecture rules change.

## Stage 1: Validate and Reproduce
**Goal:** Account for eight Qodo findings and existing CI failures.
**Success Criteria:** Behavioral regressions fail before implementation; each review item has a supported disposition.
**Tests:** Date/datetime/invalid legacy inputs, callback manager construction and admission retry identity, safe invalid-owner context, trigger type annotations.
**Status:** Complete

- [x] Read every Qodo finding and the approved redaction and compatibility requirements.
- [x] Rebase cleanly onto latest fetched dev at 4c4f197f68481664c58d4553bbfbb45dae157e28.
- [x] Add tests in `tldw_Server_API/tests/Claims/test_claims_review_metrics_jobs_scheduler.py` and run them red: six expected failures, one date-only control passed.
- [x] Check failed CI logs against current dev: stale line-number baseline, private coercer and published Env_Vars snapshot reproduced; the monitoring anchor warning confirmed from CI and canonical slug generation.

## Stage 2: Scoped Corrections
**Goal:** Restore compatibility and remove redundant successful Jobs initialization.
**Success Criteria:** New regressions pass without changing export behavior or logging raw errors.
**Tests:** Scheduler/Jobs/worker/aggregation/existing-only regression suites and marker-based collection.
**Status:** Complete

- [x] In `app/services/claims_review_metrics_scheduler.py`, reuse `_parse_iso_date(report_date)` and only override the lookback window when parsing succeeds.
- [x] Lazily call `await asyncio.to_thread(claims_jobs.jobs_manager_from_env)` when the Jobs route first needs an uninjected manager; reuse the result for later owners and retries. Keep initialization failures within the existing bounded transient admission retry path.
- [x] Annotate `get_next_fire_time(previous_fire_time: datetime | None, now: datetime) -> datetime | None`.
- [x] Bind operation, window, page number and owner position when rejecting invalid PostgreSQL discovery entries; never bind the raw invalid owner.
- [x] Document aggregation Args/Returns/Raises in `app/core/Claims_Extraction/claims_review_metrics.py`.
- [x] Classify `tests/DB_Management/test_media_db_existing_only.py` as integration and replace private registry snapshots with public factory-boundary assertions plus existing backend identity/pool checks.
- [x] Retain sanitized worker diagnostics; reply to the traceback suggestion citing the approved contract and regression evidence.
- [x] Resolve validated CI blockers with focused tests: shared coercion, single baseline line-number relocation, refreshed published Env_Vars, and canonical/published link repairs. No gates weakened.

## Stage 3: Verify, Publish and Merge
**Goal:** Publish verified fixes and address every review thread before merging.
**Success Criteria:** Relevant tests, compilation, no new lint/security diagnostics, required checks pass, human summary preserved, merge confirmed.
**Tests:** Full scoped Claims/storage/Jobs/services suites including official PostgreSQL fixtures; Bandit on touched production scope; CI ratchets and failing gates.
**Status:** In Progress

- [x] Run scoped regressions and Bandit using the project virtual environment: 1,286 integrated scoped tests passed with zero skips; 255 Config tests, 69 docs/ratchet tests, 86 additional CI gate tests and 65 separate shared Jobs tests passed; 37 integration tests collected by marker; all 14 production files compiled and Bandit reported zero findings/errors. Corrected source/test Ruff scopes and whitespace checks passed.
- [x] Independently review the corrections and address validated findings. The new stop-during-manager-initialization regression failed red and passed after adding the stop check before admission. Reviewer independently confirmed zero admissions after stop, 540 focused tests passed, and no remaining actionable findings.
- [x] Commit scoped corrections as fda51e5d1ae03a763b74b95e4ea7292bd883f2b5 with TASK-9935.4; push with an explicit lease matching the originally fetched PR head da6b00903ea4ac309b09e08ddbbef0261e606b9b.
- [x] Reply within all eight inline review threads and resolve addressed findings. Qodo's refreshed report on that published head has zero active findings; all seven required checks passed. Recheck feedback and gates again after rebasing.
- [ ] Preserve the user-supplied Change summary; merge only the verified head after current dev and merge gates are satisfied.
- [ ] Record merge evidence, complete Backlog, and remove only this completed plan. If external checks require a later continuation, keep the plan/task active and schedule a quiet thread follow-up.

### 2026-10-04 Reverification

- [x] Rebase all nine PR commits cleanly onto latest fetched dev bf8f2ad6a42ad6396376020876a5f6a709ec6b34. Range-diff confirms each rebased commit is patch-equivalent to its original.
- [x] Rebased production head d395457272151acf143a4035c2adf1abc423a995: 1,286 scoped tests passed with zero skips, including official PostgreSQL fixtures. All 14 changed production files compiled; Bandit reported zero findings and zero errors.
- [x] Reverify full Config and docs/ratchets: 324 passed (255 Config and 69 docs/ratchet cases), including strict docs compilation and published-snapshot contracts. Focused corrected-source/test Ruff passed; whitespace checks passed before publication. Verification logs: `/tmp/claims-pr3093-scoped-20261004.log`, `/tmp/claims-pr3093-config-docs-20261004.log`, `/tmp/bandit_claims_pr3093_20261004.json`.
- [x] Publish rebased head 4ac50974825f9cbd663299744f9720bfc11b5d0e using an explicit lease against remote head fda51e5d1ae03a763b74b95e4ea7292bd883f2b5. The human summary remains verbatim; refreshed Qodo reports zero active findings and all eight threads remain resolved.
- [ ] Wait for all required checks on the final published head. The license policy passed; remaining checks are pending. Address the validated task-format gate below before merge.
- [x] User explicitly approved automatic follow-up. The 10-minute thread heartbeat `claims-metrics-pr-3093-follow-up` is active, remains quiet on unchanged state, and stops after verified merge. No CI cancellation, hook bypass or admin merge is authorized.

### Backlog Format Gate

- [x] Reproduce the new latest-dev pre-commit failure with the official `backlog-py task normalize --check`: the same four PR-touched task records are noncanonical. ADR-059 now requires backlog-py for all task edits.
- [x] Normalize only TASK-9935.1, TASK-12993.1, TASK-9935.3 and TASK-9935.4 through the official repository CLI; diff review confirms task text, statuses, criteria, frontmatter and history are preserved. The exact scoped format check passes.
- [x] Verify Backlog format ratchet/tool tests: 149 passed with no skips. Complete PR-scoped pre-commit passes all applicable hooks with the correction staged; its initial unstaged run had stashed and tested the old records. No runtime/test edits, so Bandit is not applicable to this documentation-only correction. Logs: `/tmp/claims-pr3093-backlog-format-20261004.log`, `/tmp/claims-pr3093-precommit-20261004.log`.
- [x] Publish the scoped correction as 91f207b1c4ad2e70be9dbfdf0f1bd5e60d398df6. No runtime/test files changed. The active follow-up uses the latest ADR-059 task editor for all future task edits.
- [ ] Wait for fresh CI/review on the final published head; dev advanced again before the prior head could satisfy all checks.

### Latest Dev Rebase (2026-10-04 20:54 UTC)

- [x] Rebase all eleven PR commits cleanly onto dev c95e41fc62a07e55fd74052023826c71b2a2c789. This base advance only normalizes unrelated Backlog records. Range-diff confirms every PR commit remains patch-equivalent; production, tests, config, scripts, docs and instructions are identical to the previous head.
- [x] Rebased local head 35ef7486fce5bf3dd41c3f315dc7124432fbc404 passes 1,286 scoped RUN_JOBS=1 tests with zero skips, including official PostgreSQL fixtures; 324 Config/docs/ratchet tests; and 149 Backlog tool/format-ratchet tests with zero skips. All five PR-touched task records are canonical. Compilation passes for fourteen production files, focused Ruff passes, and Bandit reports zero findings/errors across all fourteen files.
- [x] Record verification logs: `/tmp/claims-pr3093-scoped-rebase2-20261004.log`, `/tmp/claims-pr3093-config-docs-rebase2-20261004.log`, `/tmp/bandit_claims_pr3093_rebase2_20261004.json`. No new actionable review feedback or unresolved threads; original eight findings remain resolved. ADR-003 and ADR-059 remain governing decisions; no new ADR required.
- [x] Complete staged PR-scoped pre-commit passes every applicable hook, including canonical Backlog format and Python syntax checks. Log: `/tmp/claims-pr3093-precommit-rebase2-20261004.log`.
- [ ] Publish using an explicit force-with-lease against fetched remote head 91f207b1c4ad2e70be9dbfdf0f1bd5e60d398df6.
- [ ] Recheck the final head's required CI, feedback, human summary and latest dev before exact-head merge. Do not cancel CI, bypass hooks, weaken gates or use admin merge.
