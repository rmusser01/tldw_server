## Stage 1: Verify Review Findings
**Goal**: Confirm each requested code review finding against the rebased Sync v2 branch.
**Success Criteria**: Critical and Important findings are classified as valid fixes or explicit pushback.
**Tests**: Read-only inspection of the referenced implementation paths.
**Status**: Complete

## Stage 2: Patch Contract And Restore Safety
**Goal**: Align public push/pull request shape with the M1 contract and make restore preview complete beyond the scan page size.
**Success Criteria**: Contract aliases are accepted without breaking existing clients, and restore preview no longer silently truncates domains.
**Tests**: Focused endpoint/model and restore preview regression tests.
**Status**: Complete

## Stage 3: Patch Blob And Workspace Safety
**Goal**: Add production blob-transfer configuration, bounded chunk reads, retry-safe blob completion, dataset-scoped workspace blob reads, and conservative workspace retention blockers.
**Success Criteria**: Blob upload/download and retention semantics avoid data loss or unauthorized omissions.
**Tests**: Focused blob endpoint, factory, workspace blob, and retention regression tests.
**Status**: Complete

## Stage 4: Verify And Publish
**Goal**: Run targeted Sync tests and Bandit, update Backlog, commit, and push the PR branch.
**Success Criteria**: Verification output is recorded and PR branch contains the review fixes.
**Tests**: `python -m pytest tldw_Server_API/tests/Sync` and Bandit over touched Sync v2 production paths.
**Status**: Complete
