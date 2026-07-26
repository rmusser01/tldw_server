## Stage 1: Integrate branches

**Goal**: Merge current `origin/main` into the `origin/dev`-based release branch.
**Success Criteria**: Both remote tips are ancestors of the release head and all conflicts are resolved deliberately.
**Tests**: Ancestry checks, unmerged-file check, diff review.
**Status**: Complete

## Stage 2: Verify and open release PR

**Goal**: Validate the combined release candidate and open the reviewed PR to `main`.
**Success Criteria**: Focused release/CI/license checks pass, security and diff checks are clean, and the PR targets `main`.
**Tests**: Focused pytest suites, workflow validation, Bandit on touched runtime scope, `git diff --check`.
**Status**: In Progress

## Stage 3: Publish and synchronize

**Goal**: Merge the release PR, cut `v0.1.42`, verify publication, and sync `main` back to `dev`.
**Success Criteria**: The requester-authored Change summary and required checks permit merge; the tag/release point at the release commit; both branches are synchronized.
**Tests**: GitHub check verification, tag/release SHA checks, branch ancestry checks.
**Status**: Not Started
