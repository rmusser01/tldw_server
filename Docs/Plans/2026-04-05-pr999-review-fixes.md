# PR 999 Review Fixes Implementation Plan

## Stage 1: Confirm Remaining Review Scope
**Goal**: Verify each still-open PR 999 review item against the current `feat/writing-suite-phase2` branch so work only targets real regressions.
**Success Criteria**: Each actionable comment is mapped to a file and either confirmed as needing a fix or ruled out as stale.
**Tests**: Source inspection, import check for `writing_manuscripts.py`, targeted existing test discovery.
**Status**: Complete

## Stage 2: Add Regression Tests
**Goal**: Add focused failing tests for the confirmed review issues before changing production code.
**Success Criteria**: New or extended tests fail for the current branch state and clearly point at the missing behavior.
**Tests**: Targeted `pytest` coverage for manuscript imports/chapter reparenting plus focused Vitest guards for onboarding and writing playground behavior.
**Status**: Complete

## Stage 3: Implement Confirmed Fixes
**Goal**: Update backend and frontend code to satisfy the verified review comments with minimal, project-consistent changes.
**Success Criteria**: Endpoint imports resolve, chapter `part_id` updates work, onboarding handlers are resilient, research state is scene-safe, offline mutations are guarded, and citation styling follows theme variables.
**Tests**: Re-run the stage 2 regression tests after each change set.
**Status**: Complete

## Stage 4: Verify and Harden
**Goal**: Prove the branch is in a releasable state for the touched scope.
**Success Criteria**: Focused test suites pass and Bandit reports no new issues in touched backend paths.
**Tests**: Targeted `pytest`, targeted `vitest`, and `python -m bandit -r` on touched backend files.
**Status**: Complete
