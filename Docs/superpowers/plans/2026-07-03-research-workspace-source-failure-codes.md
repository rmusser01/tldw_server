# Research Workspace Source Failure Codes Plan

## Stage 1: Regression Tests
**Goal**: Capture the missing durable reason-code behavior for failed `workspace_source_ingest` jobs.
**Success Criteria**: Tests fail because the worker exception lacks a stable `failure_code` and status projection hides job `error_code`.
**Tests**: Add focused worker and source-status API regression tests.
**Status**: Complete

## Stage 2: Worker And Projection Fix
**Goal**: Preserve existing source readiness behavior while exposing actionable failure codes for workspace-source jobs.
**Success Criteria**: Workspace-source missing-media failures carry `workspace_source_media_not_found`, and source status returns that as `status_reason` plus `job.error_code`.
**Tests**: Focused pytest tests pass.
**Status**: Complete

## Stage 3: Verification And Closeout
**Goal**: Verify touched scope, run security scan, and update the Backlog task with the final result.
**Success Criteria**: Focused tests, `git diff --check`, and Bandit complete without new findings.
**Tests**: Focused pytest suite and Bandit on touched production files.
**Status**: Complete
