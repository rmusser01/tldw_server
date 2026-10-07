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
