# Manuscript Schema And World Sync Plan

## Stage 1: Verify Current Branch State
**Goal**: Confirm which review findings still require production changes on the current branch.
**Success Criteria**: The manuscript schema and world-info sync trigger SQL are inspected and any already-fixed items are identified.
**Tests**: Source inspection of `writing_manuscript_schemas.py` and `ChaChaNotes_DB.py`.
**Status**: Complete

## Stage 2: Add Regression Coverage
**Goal**: Add focused tests for typed manuscript response payloads and world-info sync log payload contents.
**Success Criteria**: A schema contract test fails for the untyped project response field, and a DB sync-log test proves world-info payload coverage.
**Tests**: Targeted `pytest` on new schema contract coverage plus `test_manuscript_world_plot_db.py`.
**Status**: Complete

## Stage 3: Apply Minimal Production Fix
**Goal**: Replace the untyped project response payload with a typed Pydantic model and leave already-correct trigger SQL unchanged unless a test proves otherwise.
**Success Criteria**: `ManuscriptProjectResponse.settings` no longer uses a raw dict payload and all new tests pass.
**Tests**: Re-run the focused pytest targets from Stage 2.
**Status**: Complete

## Stage 4: Verify Touched Scope
**Goal**: Prove the touched schema/test scope is passing and security-clean.
**Success Criteria**: Focused pytest passes and Bandit reports no issues in the touched Python files.
**Tests**: Targeted `pytest` and `python -m bandit -r`.
**Status**: Complete
