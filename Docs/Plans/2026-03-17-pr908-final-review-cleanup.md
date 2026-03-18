## Stage 1: Reconfirm Live Review Scope
**Goal**: Pull the current unresolved PR 908 review threads and map them to concrete code/test changes.
**Success Criteria**: Every unresolved thread has an explicit owner file and planned resolution path.
**Tests**: None.
**Status**: Complete

## Stage 2: Patch Repository and Manager Boundaries
**Goal**: Implement the remaining metering, jobs repository, and fair-share fixes.
**Success Criteria**: Repository integrity checks, optional pooling, constructor validation, and fail-closed admission behavior are all in place.
**Tests**: Targeted pytest coverage for metering repositories, jobs repository, and fair-share integration.
**Status**: Complete

## Stage 3: Expand Regression Coverage
**Goal**: Add or update tests for duplicate subscriptions, sync-log indexing, pooled sessions, SQLite transactional fair-share ordering, and repository validation.
**Success Criteria**: New tests fail before the changes and pass after them.
**Tests**: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`, `tldw_Server_API/tests/Jobs/test_jobs_repository.py`, `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
**Status**: Complete

## Stage 4: Verify the Touched Scope
**Goal**: Prove the cleanup with scoped tests and Bandit.
**Success Criteria**: Targeted pytest suite passes and Bandit reports no findings in changed files.
**Tests**: Scoped pytest command plus scoped Bandit run.
**Status**: Complete

## Stage 5: Push and Resolve Review Threads
**Goal**: Update the branch on GitHub and close the remaining PR 908 review threads with concrete replies.
**Success Criteria**: Branch pushed, summary comment posted, and no unresolved review threads remain.
**Tests**: `gh pr view` / GraphQL review-thread check.
**Status**: In Progress
