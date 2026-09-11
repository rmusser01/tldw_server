# TASK-13013.7.43 — integrate the remediation with current dev

Resolve the confirmed local divergence and PR merge conflicts against exact dev
`6c4bdcbc48f4fe4bab7019d59ad8cf962ab240da`. Preserve the completed security fixes,
all 342 policy records, upstream work and the original untracked handoff. Existing
candidate evidence continues to describe its recorded source revision. Work is
local; no push, merge, release, restricted investigation or new vulnerability
disposition is included.

## Stage 1: Capture recoverable baseline
**Goal**: Record old head, target, recovery branch and changed-file inventories.
**Success Criteria**: A recovery reference protects the original series; six overlapping paths are identified; the existing 292-test result remains tied to its original revision.
**Tests**: Verify worktree isolation, clean tracked files, source/policy hashes and target ancestry.
**Status**: In Progress

## Stage 2: Integrate with dev
**Goal**: Rebase the existing series onto the exact dev revision.
**Success Criteria**: Conflicts resolved with both upstream behavior and remediation preserved; no unintended source or evidence changes.
**Tests**: Compare commit series and all non-overlapping file hashes; inspect the six overlapping paths.
**Status**: Not Started

## Stage 3: Verify and retain
**Goal**: Verify the resulting local branch and record the integration limits.
**Success Criteria**: Relevant CI/policy/backend/frontend checks pass; touched-scope lint and Bandit reviewed; independent review complete; evidence and task notes committed.
**Tests**: Focused pure policy and workflow tests, affected Character Chat/Admin tests where required, source/evidence hash checks, conflict-marker/diff checks.
**Status**: Not Started
