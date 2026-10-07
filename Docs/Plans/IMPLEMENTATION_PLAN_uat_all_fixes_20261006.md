# Remaining UAT bugs and PR3084 integration

Task: TASK-13260.278.18.83.45. User authorized remaining repairs then rebase on 2026-10-06.
Candidate starts at dev 17c47c49d857a2a1fd1e58602b704213760a4a85.
ADR required: no. Restore existing migration and retry contracts; retain ADR-003, ADR-059 and ADR-064.

## Stage 1: Repair cold audio reporting and V1 retry disposition
**Goal**: Reporting-first legacy usage migrates before daily/monthly reads; V1 outcomes follow real WorkerSDK/Jobs retry and quarantine disposition, retaining cancellation authority.
**Success Criteria**: Real causal red then green, retained ownership/lease/cancellation/storage boundaries, affected checks and independent source review.
**Tests**: Cold SQLite and official isolated PostgreSQL reporting before admission, repeated migration and finite/null/zero reporting; real Jobs transient ordinary exception then permitted retry, typed retryable exhaustion and configured quarantine then terminal slot/batch and legitimate new Retry; unchanged backend quarantine cutoff policies; disabled SDK retries and PostgreSQL zero-backoff with an explicit cutoff of one.
**Status**: Complete

## Stage 2: Combine and rebase reviewed work
**Goal**: Combine new fixes, committed bf8 repairs, six qualified readers and published fe117 repairs onto fresh current dev, preserving original worktrees and all published behavior.
**Success Criteria**: Exact source/ref and owned patch guards, resolved conflicts without assertion loss, supported combined API export/types, scoped lint/security and final actual/staged review.
**Tests**: Genuinely affected incoming suites and new regressions; preserve VN call_soon_threadsafe/release_observed/direct-await/finally-drain oracle, audio finite quota/cache/DSN cleanup, offline accounting mock and causal guard.
**Status**: In Progress

## Stage 3: Qualification and acceptance
**Goal**: Publish with normal hooks/exact lease and exact human-summary readback; satisfy current-head hosted and real UAT/native/installer gates before merge.
**Success Criteria**: Final candidate checks and actual acceptance evidence; exhausted controls remain stopped unless separately justified by a new repair and authorized bounded scope.
**Tests**: 43 UAT/15 runtime bug acceptance, native first-import and whole-process Prompt shutdown, installer 53/55/56, fresh runtime/client/principal context, applicable queue-platform acceptance. No API restart, queue mutation, model/environment reconstruction or proof bundles.
**Status**: Not Started

## Current qualification

- Local rebase completed: 143282af365cd8b462f842c6b8907d6120ce0ea7 on guarded dev 7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d; 14 committed patches compare equal to the previous range. Five-path owned WIP is byte-identical, and only two incoming docs/task paths changed the committed tree. Unpublished.
- Independent integrated-source review clear; published main/control-plane/Sync/lifetime files, VN loop-release oracle, audio assertions, offline guards/accounting regressions and task history preserved.
- Canonical API export/types and final drift check passed: 2,107 paths, 3,248 schemas, exactly three published descriptions added.
- 133 changed Python files compile. Focused 49-path incoming production and 28-path owned Python Bandit/Ruff comparisons have no new production or unpublished findings; eight dev-relative test assertions are retained published B101 warnings.
- First 27-module VN/storage batch hit the unchanged 600-second outer limit and is unqualified. Smaller affected checks retain the limit; no fourth V1 production revision is authorized.
- Human approved the separate Python setup, frozen current-lock frontend install and persistent heartbeat metadata save. Both installations completed; metadata saving preserves PAUSED status. No API start/restart, build, browser/native acceptance, queue mutation or merge follows from these approvals.
- All broader UAT/native/installer acceptance and retained STOPs remain open. Stage 2 stays In Progress and Stage 3 Not Started.

- New Jobs fixture probe correction: standard asyncio entry-point name replaces duplicate module-path registration; original guards preserved, corrected four-case qualification passed (4 passed), independent review clear. The eight-file run's 271 passes remain qualified within that run; the unrelated fourteen-file timeout is unqualified.

- Five smaller AuthNZ modules passed 226 cases. The first smaller VN/storage batch passed 94 and failed seven terminal-disposition parameters. Reviewed test-only WorkerConfig(retry_on_exception=False) setup at two constructors preserves every assertion; those 21 parameters passed, with clear independent source review, Ruff lint and no new scoped Bandit findings. Production repair remains unchanged and at its three-attempt cap.

- Seven VN generation/receipt/lease modules passed 321 cases; seven backend consumer modules passed 296 with normal Evaluations opt-in. Remaining VN recovery, AuthNZ generated-file and quota/reporting checks subsequently passed; the approved frozen current-lock frontend installation is now complete; affected client qualification follows. No hosted or real UAT/native/installer acceptance is claimed.

- Remaining selected backend batches passed: VN recovery 272, AuthNZ generated-file integration 79, quota/reporting 58, usage/header/ratchet 53; no failures/skips reported in these four runs. Reviewed dev7ba changes only two foreign docs/task paths. Guarded autostash rebase preserved fourteen commits and five-path WIP before the pending test-fix commit.

- The approved Python3.12 application environment disappeared after these runs; no reconstruction or additional application execution. Existing primary Python3.11 is used only for standalone backlog-py (floor3.10), plan editing and static hooks. The new separate Python setup and frozen frontend installation are complete; real UAT/native/installer acceptance and hosted qualification remain open.

- The hook/commit blocker is resolved by the separately approved Python3.12.13 environment with pre-commit4.6.2. Normal five-path hooks and an updated staged prose review still precede commit; no lower-version or hook bypass. All API/build/browser/native/installer/queue caps remain.

- Approved setup outcome: Python installation natural0/251.231s, 130 applicable declared requirements missing0/mismatch0, pip check clean and exact candidate module binding; frozen Bun installation natural0/81.430s, 3,640 packages, unchanged lockfile, corrected dependency versions and identical real React resolution. Affected UI/WebUI unit qualification starts with 116 changed UI modules and two VN frontend modules. This does not accept runtime UAT or authorize further startup/build.

- Normal five-path pre-commit qualification passed naturally0/2.224s with pre-commit4.6.2; syntax/task/whitespace/security and HTTP/legacy-body guards passed. Wizard-only Ruff/Black hooks had no files; prior scoped lint/security remains recorded. Three test-fix bodies are unchanged, and updated prose receives final staged review before commit. Changed UI unit qualification is running separately.
