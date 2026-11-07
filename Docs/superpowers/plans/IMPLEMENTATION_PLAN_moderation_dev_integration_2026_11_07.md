# Moderation Dev Integration (TASK-15417)

ADR assessment: ADR-086 and ADR-021 govern the existing implementation. No new
architecture decision is required for rebasing and publishing the approved work.

## Stage 1: Rebase
**Goal**: Preserve a backup ref and rebase the existing branch onto freshly fetched dev.
**Success Criteria**: No conflicts or lost branch changes; dev is an ancestor.
**Tests**: Inspect conflict resolutions and compare pre/post-rebase changes.
**Status**: In Progress

## Stage 2: Verify and Review
**Goal**: Verify the rebased implementation and independently review integration changes.
**Success Criteria**: Focused backend, packaging, frontend, compilation and security
checks pass, or unrelated baseline limitations are explicitly documented.
**Tests**: Native process containment prerequisites before hostile probes; recorded
focused regression manifest; UI tests/typecheck; touched Python compilation/Bandit;
Backlog metadata and task ID checks; independent conflict/integration review.
**Status**: Not Started

## Stage 3: Publish
**Goal**: Push the verified branch and create a PR against dev.
**Success Criteria**: PR URL recorded with fresh verification and remaining platform
limitations. No merge or auto-merge; human-owned Change summary remains a merge gate.
**Tests**: Confirm PR base/head and clean worktree; inspect initial CI dispatch.
**Status**: Not Started
