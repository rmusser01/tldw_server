# Real PST verification and dev PR (TASK-13377)

## Stage 1: Inspect native parser and PR base
**Goal**: Obtain a pinned public synthetic PST and identify current dev conflicts.
**Success Criteria**: Native parser opens the fixture; baseline failures and merge preview recorded.
**Tests**: Existing installed-parser and strict valid-PST endpoint tests.
**Status**: Complete

## Stage 2: Preserve native PST metadata
**Goal**: Correct observed recipient/date loss with deterministic regressions.
**Success Criteria**: Native datetime values and transport header fallback survive RFC822 conversion; both real tests pass without skips.
**Tests**: Unit metadata regressions, native-parser endpoint tests and attachment/parser regressions.
**Status**: Complete

## Stage 3: Integrate current dev and verify
**Goal**: Preserve current dev behavior and email fixes while resolving conflicts.
**Success Criteria**: Reviewed merge, focused regression suites, Ruff/Bandit/compile checks and dated evidence.
**Tests**: Changed source and integration behavior suites; source provenance kept distinct from benchmark snapshots.
**Status**: In Progress

## Stage 4: Publish PR
**Goal**: Create and attach a reviewable pull request against dev.
**Success Criteria**: Verified commits pushed, PR base dev and URL recorded; human-written Change summary merge gate explicit.
**Tests**: Remote PR metadata and local clean worktree verification.
**Status**: Not Started
