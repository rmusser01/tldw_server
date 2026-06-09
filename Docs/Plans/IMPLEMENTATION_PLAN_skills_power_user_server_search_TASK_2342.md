## Stage 1: Server-Backed Search Contract
**Goal**: Add a safe `q` list-query contract for Skills so matching is applied before pagination.
**Success Criteria**: `GET /api/v1/skills?q=...&limit=...&offset=...` returns only matching skills, and `total` reflects the filtered result count.
**Tests**: Focused Skills service and API tests that prove a match outside the first unfiltered page is returned.
**Status**: Complete

## Stage 2: Frontend API Wiring
**Goal**: Let the shared frontend API client serialize the Skills search query while preserving existing callers.
**Success Criteria**: `listSkills({ q, limit, offset })` builds the expected query string and omits empty values.
**Tests**: Existing `tldw-api-client.boundary-slices` coverage extended for `q`.
**Status**: Complete

## Stage 3: Skills Manager Workflow
**Goal**: Replace current-page-only filtering with server-backed search in the Skills manager.
**Success Criteria**: Entering a search query requests page 1 with `q`, renders server-returned rows, and uses the filtered `total` for count and pagination.
**Tests**: Existing Skills manager test suite covers search query calls, page reset, and rendering results absent from the initial page.
**Status**: Complete

## Stage 4: Verification and Closeout
**Goal**: Verify the scoped change and document results.
**Success Criteria**: Focused backend/frontend tests pass, Bandit runs on touched Python paths, whitespace diff check passes, and Backlog is updated.
**Tests**: Focused pytest, focused Vitest, Bandit, and `git diff --check`.
**Status**: Complete
