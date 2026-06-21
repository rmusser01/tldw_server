## Stage 1: Contract And Red Tests
**Goal**: Lock down the dry-render contract before implementation.
**Success Criteria**: Backend tests fail because `dry_run` is not yet accepted/returned and fork execution is still invoked; frontend tests fail because the client/modal do not send dry-render requests.
**Tests**: Targeted Skills executor/API tests and SkillPreview/workspace-api Vitest coverage.
**Status**: Complete

## Stage 2: Backend Dry Render
**Goal**: Add `dry_run` to the Skills execute request/result and short-circuit model/tool/fork execution when requested.
**Success Criteria**: Inline and fork skills return rendered prompt metadata with `dry_run: true`; fork output remains null; normal test execution behavior remains unchanged.
**Tests**: `test_skill_executor.py` and `test_skills_api.py` targeted cases.
**Status**: Complete

## Stage 3: Frontend Dry Render Action
**Goal**: Expose a safe "Render prompt only" action next to "Run test" in the Skills preview modal.
**Success Criteria**: The client sends `dry_run: true` only for the render-only action, the existing test-run action sends `dry_run: false`, pending state prevents duplicates, and results identify whether they were dry-rendered or executed.
**Tests**: SkillPreview modal tests and workspace-api client payload tests.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Verify the scoped change and prepare it for review.
**Success Criteria**: Targeted backend tests, targeted frontend tests, `git diff --check`, and Bandit on touched backend scope are recorded; Backlog task has final notes; branch is committed and PR is opened against `dev`.
**Tests**: Targeted pytest, targeted Vitest, Bandit, diff check.
**Status**: Complete
