# Chat Workspace Live Flow Implementation Plan

Backlog: TASK-12130
GitHub: #2031, parent #1239

## Stage 1: Scoped Chat Flow Audit
**Goal**: Identify every `/chat-workspace` send, create, load, and resume path that can touch server chat state.
**Success Criteria**: The implementation targets are narrowed to concrete call sites and tests, with no speculative UI rewrites.
**Tests**: Read existing Chat Workspace, `useMessageOption`, `useChatActions`, server loader, API client, and scoped-history tests.
**Status**: Complete

## Stage 2: Baseline Focused Coverage
**Goal**: Establish the current behavior of the existing focused frontend tests before changing implementation code.
**Success Criteria**: Relevant tests either pass at baseline or any pre-existing failures are documented before edits.
**Tests**: Focused Vitest runs for Chat Workspace panel/page, server chat loading, persona chat creation, and chat action integrations.
**Status**: Complete

## Stage 3: Failing Scoped Persistence Regression Tests
**Goal**: Add tests proving workspace-scoped sends cannot create or resume global chats when a workspace scope is available.
**Success Criteria**: At least one focused test fails before the production fix because the hook drops workspace scope on chat creation.
**Tests**: Extended `useChatActions` persona and character integration coverage to assert scoped `createChat`, scoped character persistence/stream calls, scoped settings sync, workspace normal/RAG server-chat bootstrap, and a global-chat compatibility regression.
**Status**: Complete

## Stage 4: Minimal Scoped Chat Fix
**Goal**: Thread `scope` through the missing `useChatActions` create paths while preserving existing global chat behavior.
**Success Criteria**: Workspace sends create/resume workspace-scoped server chats; global sends keep current API payloads; stale assistant metadata reset behavior remains unchanged.
**Tests**: The new regression tests and existing focused hook tests pass.
**Status**: Complete

## Stage 5: Chat Workspace Acceptance Surface
**Goal**: Confirm model/persona state, streaming stop, error recovery, draft/staged context preservation, and hydration guards meet #2031 acceptance.
**Success Criteria**: Existing or new focused tests cover the acceptance criteria that are feasible without the live-backend fixture from #2035.
**Tests**: Chat Workspace panel/page tests plus any small additions for missing state labels or disabled send behavior.
**Status**: Complete

## Stage 6: Verification and Tracker Update
**Goal**: Run targeted verification, update Backlog, and report the precise state of #2031.
**Success Criteria**: Focused tests pass or failures are documented with concrete blockers; TASK-12130 records touched files and verification.
**Tests**: Focused Vitest suite from Stage 2, plus Bandit only if Python files are touched.
**Status**: Complete
