## Stage 1: Baseline And Contract Mapping
**Goal**: Confirm the current public-share and prototype collaborator entry flow, then define the frontend-only contract mapping needed for Risk Gate 5.
**Success Criteria**: Existing focused PublicShare, prototype workspace route, hook, and client tests pass before changes; the implementation uses frozen Risk Gate 4 categories and frontend state buckets without changing backend semantics.
**Tests**: `bunx vitest run src/components/Option/__tests__/PublicShare.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx --maxWorkers=1 --no-file-parallelism`
**Status**: Complete

## Stage 2: Explicit Collaborator Entry State
**Goal**: Make collaborator entry state explicit and route-scoped so token-only entries never inherit stale owner workspace/session state.
**Success Criteria**: URL share/session tokens replace stale collaborator state; token-bearing route entries do not read owner workspace detail until the exchanged/created collaborator session provides a workspace id; transient passwords remain outside persistent store.
**Tests**: Add failing component/store tests around token changes, stale owner workspace isolation, and password handoff persistence.
**Status**: Complete

## Stage 3: Contract Error Presentation
**Goal**: Map prototype contract error categories and retryability into stable collaborator-entry UI states.
**Success Criteria**: Public link exchange and collaborator branch-session failures show the frozen state buckets for invalid/unavailable links, password-required/rejected, inactive sessions, workspace unavailable, setup failed, and preview unavailable; retry affordances use `retryable`.
**Tests**: Add failing session-view tests that inject structured prototype errors from exchange/session mutations and assert the rendered state and retry behavior.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Verify the frontend slice and record remaining release-gate evidence requirements.
**Success Criteria**: Focused Vitest suite passes, formatting/type checks for touched files pass where practical, Backlog task records verification and any browser/E2E skip that Risk Gate 8 must pick up, and PR references issue #1457.
**Tests**: Focused Vitest command from Stage 1 plus targeted tests added in Stages 2-3.
**Status**: Complete
