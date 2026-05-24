## Stage 1: Characterize Source Add Job Contract
**Goal**: Add focused tests proving workspace source creation enqueues a user-visible, idempotent `media_ingest` Job and preserves source creation when job submission is unavailable.
**Success Criteria**: Tests fail before implementation because `POST /workspaces/{id}/sources` does not call `JobManager.create_job`.
**Tests**: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py` focused source endpoint tests.
**Status**: Complete

## Stage 2: Enqueue Workspace Source Jobs
**Goal**: Add a narrow helper in the workspace endpoint that submits a `media_ingest/default/workspace_source_ingest` Job after the source row is persisted.
**Success Criteria**: Job payload contains workspace/source/media identifiers, source metadata, requested lifecycle stages, and a deterministic idempotency key.
**Tests**: Focused workspace source endpoint tests pass.
**Status**: Complete

## Stage 3: Verify Projection Integration
**Goal**: Confirm the existing source status projection recognizes the queued source job via payload keys and keeps no `/workspace-playground` alias behavior.
**Success Criteria**: Workspace source status tests pass and route search finds no new legacy route alias.
**Tests**: `test_workspace_source_status_api.py`, route grep.
**Status**: Complete

## Stage 4: Backend Validation And Security
**Goal**: Run backend-focused verification, Bandit on touched Python files, and a live backend HTTP smoke test for source add/status.
**Success Criteria**: Focused tests, Bandit, diff check, and live backend calls complete with recorded output.
**Tests**: Pytest, Bandit, `git diff --check`, live `uvicorn` + HTTP checks.
**Status**: Complete

## Stage 5: Review Feedback Hardening
**Goal**: Address code-review findings around fail-open Jobs dependency construction and source-status job query precision.
**Success Criteria**: Source creation still persists the source row when Jobs manager construction fails, enqueue failures remain non-destructive, and status projection prioritizes `workspace_source_ingest` Jobs before broad legacy `media_ingest` jobs.
**Tests**: Focused workspace source endpoint tests, workspace source status tests, full Workspaces suite, Bandit, scoped diff check, live backend HTTP smoke.
**Status**: Complete
