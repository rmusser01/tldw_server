# PR 1002 Review Remediation Plan

## Stage 1: Verify Review Items and Map Affected Paths
**Goal**: Turn every inline PR comment into a concrete implementation/test item with no ambiguity about scope.
**Success Criteria**: All 11 inline comments are accounted for; related frontend/backend files and test targets are identified.
**Tests**: N/A for this staging step.
**Status**: Complete

## Stage 2: Frontend Type Safety and UX Corrections
**Goal**: Fix the verified UI issues in writing playground and onboarding flows, and replace new `any`-driven response handling with typed service/client usage where the PR introduced it.
**Success Criteria**: `AIAgentTab`, `CharacterWorldTab`, `ResearchTab`, `ConnectionWebModal`, `OnboardingConnectForm`, and writing-playground mood color usage all match the real API shapes and no longer rely on the reviewed `any` casts in touched paths.
**Tests**: Add or update focused Vitest coverage for onboarding skip behavior and writing-playground typed response handling/source guards as appropriate.
**Status**: Complete

## Stage 3: Feedback Hook Reliability Fixes
**Goal**: Make `useWritingFeedback` use the shared request path and prevent stale async updates from toggles/unmounts.
**Success Criteria**: Chat calls route through `bgRequest`; effect cleanup prevents stale state updates; echo character counts are only consumed on successful reactions or otherwise preserved for retry.
**Tests**: Add targeted Vitest tests for request routing and cancellation/retry-sensitive behavior.
**Status**: Complete

## Stage 4: Backend Correctness and SQL Hardening
**Goal**: Fix project settings projection and remove the newly introduced dynamic-SQL/Bandit suppression pattern in the manuscript helper.
**Success Criteria**: Project responses expose parsed `settings`; backend tests cover the regression; touched character queries no longer require `# nosec B608` suppressions and pass Bandit on touched scope.
**Tests**: Add/extend focused pytest coverage for project settings round-trip and character update/list behavior; run Bandit on touched backend paths.
**Status**: Complete

## Stage 5: Verification and PR Thread Closure Prep
**Goal**: Verify the touched frontend/backend scopes and prepare precise thread replies for each resolved comment.
**Success Criteria**: Targeted Vitest, targeted pytest, and Bandit checks complete successfully; each review comment has a corresponding fix or reasoned disposition.
**Tests**:
- `apps/packages/ui` targeted `vitest run ...`
- `python -m pytest ...` on touched manuscript tests
- `python -m bandit -r ...`
**Status**: Complete
