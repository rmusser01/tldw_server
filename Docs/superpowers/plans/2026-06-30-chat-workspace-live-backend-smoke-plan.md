# Chat Workspace Live-Backend Smoke Implementation Plan

Backlog: TASK-12131
GitHub: #2035, parent #1239

## Stage 1: Harness Audit
**Goal**: Identify existing WebUI Playwright auth, workspace seeding, backend interception, and release-gate patterns.
**Success Criteria**: The new spec follows existing `apps/tldw-frontend/e2e` helper conventions and does not require a developer-local LLM provider.
**Tests**: Read `playwright.config.ts`, `e2e/utils/helpers.ts`, real-backend workflow specs, Chat Workspace components, and Stage 5 release gate.
**Status**: Complete

## Stage 2: Red Browser Coverage
**Goal**: Add focused `/chat-workspace` Playwright coverage that initially fails because no focused spec or release-gate reference exists.
**Success Criteria**: The spec opens `/chat-workspace`, seeds workspace/model/assistant state, stages a source, sends a message, captures workspace scope in backend requests, observes streaming/stop, and verifies error recovery preserves draft/staged context.
**Tests**: Run the new spec directly and confirm the first red failure is missing/unimplemented coverage rather than a harness syntax error.
**Status**: Complete

Red evidence: the first focused Stage 5 guard run failed because `/chat-workspace` did not reference `e2e/smoke/chat-workspace-live-backend.spec.ts`.

## Stage 3: Realistic Backend Fixture
**Goal**: Implement deterministic route handlers that simulate the live backend enough for route-specific browser proof.
**Success Criteria**: Handlers expose health/config/model/persona/chat/RAG endpoints, stream deterministic chunks, delay streaming long enough for Stop generation, and record request bodies/query params for assertions.
**Tests**: Run the new spec and assert request captures include `scope_type=workspace`, `workspace_id`, selected model/persona state, and staged `ragMediaIds`.
**Status**: Complete

Implemented in `apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts`.

## Stage 4: Release Gate Reference
**Goal**: Make Stage 5 release-gate coverage explicitly reference the focused `/chat-workspace` proof instead of relying only on route-load checks.
**Success Criteria**: The release gate contains an auditable pointer to the focused spec for Chat Workspace route proof.
**Tests**: Add/update a small guard or inline constant assertion if the existing release-gate tests can cover the reference without adding brittle text checks.
**Status**: Complete

Stage 5 now requires `/chat-workspace` to reference `e2e/smoke/chat-workspace-live-backend.spec.ts`; route-only no-backend startup probes are narrowly allowlisted for that route.

## Stage 5: Verification and Task Update
**Goal**: Run targeted Playwright/Vitest verification, update Backlog, and report the precise #2035 state.
**Success Criteria**: New focused spec passes; release-gate reference check passes; TASK-12131 records touched files, verification, Bandit applicability, and remaining risk.
**Tests**: Targeted Playwright spec, any guard/unit test changed for the release-gate reference, `git diff --check`, and Bandit only if Python files are touched.
**Status**: Complete

Verification:
- `npx playwright test e2e/smoke/chat-workspace-live-backend.spec.ts --project=chromium` passed.
- `npx playwright test e2e/smoke/stage5-release-gate.spec.ts --project=chromium --grep "Chat Workspace"` passed.
- `git diff --check` passed.
- Bandit not applicable; no Python files touched.
