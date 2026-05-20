# Character Chat Phase 6 Accessibility, Mobile, And Backend Signoff Plan

## Stage 1: Current-State Gate And Targeted Contracts
**Goal**: Freeze the current Phase 6 release gates against the latest `origin/dev` after PR #1888.
**Success Criteria**: `TASK-454` links this plan, the Character Chat DB health release dependency is recorded as resolved through `TASK-429`, and current focused tests identify any missing accessibility/mobile signoff contracts before production edits.
**Tests**:
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx --config vitest.config.ts`
**Status**: Complete

## Stage 2: Accessibility Contract Hardening
**Goal**: Make the primary Character Chat mode surfaces easier to verify and operate by keyboard and screen reader.
**Success Criteria**: Character Chat sessions, setup, saved setups, readiness/status, and composer role-play controls expose stable region labels, alert/status semantics, focus return behavior, and accessible names for destructive or state-changing actions.
**Tests**:
- Add or extend focused React Testing Library tests for `CharacterChatSessionsPanel`, `RolePlaySetupDrawer`, `SavedRolePlaySetupsPanel`, and the Character Chat mode shell.
- Verify no broad component rewrites or parallel state systems are introduced.
**Status**: Complete

## Stage 3: Responsive And Mobile Signoff
**Goal**: Prove `/chat` Character Chat mode remains usable at narrow, tablet, and desktop widths.
**Success Criteria**: A Playwright signoff spec visits `/chat?mode=character` at 390px, 768px, and desktop, verifies no horizontal overflow, and checks that Character Chat setup, sessions, and composer recovery actions remain reachable.
**Tests**:
- Add `apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase6.spec.ts`.
- Reuse the existing route-overflow helper pattern from `setup-connection-flow.spec.ts`.
**Status**: Complete

## Stage 4: Real Backend Character Chat Signoff
**Goal**: Ensure Phase 6 signoff uses the real FastAPI backend and real frontend path instead of frontend-only simulation.
**Success Criteria**: The existing `character-chat.spec.ts` or a new companion spec verifies character mode entry, selected character restoration, backend `POST /api/v1/chats/` creation, `POST /api/v1/chats/{id}/complete-v2` with `include_character_context: true`, and character-session resume readiness. Provider-unconfigured environments must assert the visible recovery path rather than pretending streaming succeeded.
**Tests**:
- `bunx playwright test e2e/workflows/journeys/character-chat.spec.ts --reporter=line`
- New Phase 6 signoff spec list/load check.
**Status**: Complete

## Stage 5: Browser Verification And Closeout
**Goal**: Record real browser evidence and finish the Phase 6 task as a reviewable PR slice.
**Success Criteria**: Real backend and WebUI are started locally, `/chat?mode=character` is inspected at desktop and narrow widths, focused tests pass, `git diff --check` passes, TypeScript baseline is classified, and Bandit is run only if Python/backend files are touched.
**Tests**:
- Focused Vitest suite from Stages 1-2.
- Playwright Phase 6 spec list or live run depending on backend/provider availability.
- Real backend browser smoke using `AUTH_MODE=single_user` and the project test API key.
**Status**: Complete
