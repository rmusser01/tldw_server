# Manuscript Character Sync Regression Plan

## Stage 1: Verify Live Trigger Scope
**Goal**: Confirm whether the current branch still lacks the reviewer-requested character sync payload fields or change detection.
**Success Criteria**: The exact trigger SQL and character schema are inspected and the remaining real gap is identified.
**Tests**: Source inspection of `ChaChaNotes_DB.py` and `writing_manuscript_schemas.py`.
**Status**: Complete

## Stage 2: Add Regression Coverage
**Goal**: Add a DB-level test that proves manuscript character create/update/undelete operations emit the expected sync payload fields.
**Success Criteria**: A focused pytest case fails before any necessary implementation change and asserts the presence of the synced character fields in `sync_log.payload`.
**Tests**: Targeted pytest on `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`.
**Status**: Complete

## Stage 3: Apply Minimal Fix If Needed
**Goal**: Change production code only if the regression test demonstrates a real mismatch.
**Success Criteria**: Trigger SQL and/or schema are updated only when necessary, with no unrelated refactors.
**Tests**: Re-run the new targeted pytest case.
**Status**: Complete

## Stage 4: Verify Touched Scope
**Goal**: Prove the final touched scope is secure and passing.
**Success Criteria**: Focused pytest passes and Bandit reports no issues in touched Python files.
**Tests**: Targeted `pytest` and `python -m bandit -r`.
**Status**: Complete
