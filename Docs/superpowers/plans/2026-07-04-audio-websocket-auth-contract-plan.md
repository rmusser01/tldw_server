## Stage 1: Current-Base Contract Inventory
**Goal**: Confirm the current `origin/dev` audio WebSocket client and backend auth behavior for TTS, STT, and voice chat.
**Success Criteria**: Remaining query-token audio client paths are identified before implementation.
**Tests**: Read current frontend clients and existing auth tests.
**Status**: Complete

## Stage 2: TDD Coverage
**Goal**: Add focused regression coverage for token-free audio WebSocket URLs and first-frame auth behavior.
**Success Criteria**: Tests fail on current `origin/dev` for missing shared helper or remaining query-token behavior.
**Tests**: Focused Vitest helper/client tests and backend audio WebSocket auth tests.
**Status**: Complete

## Stage 3: Audio Client Remediation
**Goal**: Route browser audio WebSocket URL construction and auth-frame sending through one shared helper.
**Success Criteria**: TTS, STT, and voice chat send auth before config/audio frames and do not put tokens in URLs.
**Tests**: Focused frontend tests pass.
**Status**: Complete

## Stage 4: Backend Contract Coverage
**Goal**: Ensure default query-token rejection and first-frame auth acceptance stay covered for all audio WS routes.
**Success Criteria**: Backend route-level auth helper tests cover TTS, STT, and voice chat.
**Tests**: Focused audio streaming service tests pass.
**Status**: Complete

## Stage 5: Verification And PR
**Goal**: Validate, document residual gaps, and open a draft PR against `dev`.
**Success Criteria**: Tests, Bandit where applicable, diff checks, Backlog notes, and draft PR are complete.
**Tests**: Focused frontend/backend tests, Bandit touched Python scope, `git diff --check`.
**Status**: In Progress
