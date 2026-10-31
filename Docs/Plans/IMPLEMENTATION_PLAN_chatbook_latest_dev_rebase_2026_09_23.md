# Chatbook draft PR rebase on latest server dev

Tracking: TASK-13261.9. H1 draft PR #2968 is based on `dev`; H2 draft PR #3002 is stacked on H1. The latest fetched `origin/dev` is `158287db30a28450e79fdbc38253168a468be88f`, six commits after the prior H1 integration base `91e8bbf84c25d3afbba2bb53ed06280d44c35307`. Work only in the two Chatbook worktrees. Preserve the untracked H2 production-build evidence directory and all main-worktree UAT changes.

## Stage 1: Audit latest dev
**Goal:** Identify overlapping runtime, migration, route-auth and CI contracts before rewriting branches.
**Success Criteria:** Source and branch pins, changed-file overlap, and affected checks are recorded.
**Tests:** Read-only ancestry/diff inspection; route-auth ratchet contract inspection.
**Status:** Complete

## Stage 2: Rebase both stacked branches
**Goal:** Make H1 descend `origin/dev`, then make H2 descend the rebased H1 head without losing either branch's authored commits.
**Success Criteria:** Clean rebases, preserved H1/H2 tree changes, no UAT artifact changes, and correct merge bases.
**Tests:** `git diff --check`, range/tree comparison, schema-version and conflict-marker inspection.
**Status:** Complete

## Stage 3: Verify and update draft PRs
**Goal:** Qualify affected backend and shared UI behavior, then update both remote draft heads and PR descriptions.
**Success Criteria:** Focused tests and security/static checks pass or baseline limits are recorded; PR #2968 remains based on `dev`; PR #3002 remains stacked on H1; remote head OIDs match local heads.
**Tests:** Latest-dev route-auth ratchet, SQLite/PostgreSQL H1 and H2 migration/operation suites, affected shared UI tests/build, Bandit for any changed Python source, and independent P1/P2 review if conflict resolution changes behavior.
**Status:** Complete
