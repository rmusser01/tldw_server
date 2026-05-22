## Stage 1: Server-Chat Resume Continuity
**Goal**: Restore server-backed Character Chat sessions across refresh and direct `/chat?mode=character` entry when no explicit route state is present.
**Success Criteria**: A persisted `serverChatId` without a local Dexie `historyId` restores into the message option store and triggers the existing server chat loader.
**Tests**: Hook/session restore tests for server-only persisted state and route restore decision coverage.
**Status**: Complete

## Stage 2: Route Precedence
**Goal**: Make Character Chat route intent follow the PRD precedence: explicit server chat id wins, explicit character id applies only to fresh state, persisted last session applies only when route/server state is absent.
**Success Criteria**: `/chat?mode=character&chatId=...` loads that chat; `/chat?mode=character&characterId=...` does not overwrite an active or non-empty session.
**Tests**: Route-intent parser tests and Playground coordinator tests for explicit `chatId` and guarded `characterId`.
**Status**: Complete

## Stage 3: Character-Aware Titles
**Goal**: Stop WebUI-created character chats from inheriting the `Extension chat` fallback.
**Success Criteria**: Character chat creation sends a character-aware title and WebUI character-chat source when no explicit title/source exists; generic WebUI persistence uses a WebUI-specific fallback.
**Tests**: Character mode contract test for createChat payload; persistence hook test for selected-character fallback title.
**Status**: Complete

## Stage 4: Recent Session Metadata And Resume CTA
**Goal**: Make recent character sessions distinguishable and quick to resume.
**Success Criteria**: Session rows expose character identity, topic or title, updated age, message count, persistence state, and a primary resume-last action when no chat is active.
**Tests**: CharacterChatSessionsPanel tests for metadata rendering and primary resume action.
**Status**: Complete

## Stage 5: Real-Backend Verification
**Goal**: Verify Phase 8 behavior with the real backend and WebUI.
**Success Criteria**: Focused unit/integration tests pass; `git diff --check` passes; browser walkthrough confirms `/chat?mode=character` resume behavior against the running backend.
**Tests**: Focused Vitest suite, relevant Playwright/real-backend Character Chat checks when available.
**Status**: Complete

## Verification Notes

- `bunx vitest run ../packages/ui/src/utils/__tests__/character-chat-mode-intent.test.ts ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx ../packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts ../packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx` - passed, 25 tests.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py::test_character_chat_flow_sessions_messages_worldbooks -q` - passed.
- Real backend and WebUI verification used `127.0.0.1:8000` and `127.0.0.1:8080` with `SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY`; seeded a real `/api/v1/chats/` character session and verified `/chat?mode=character`, resume, and `/chat?mode=character&chatId=...`.
- `apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs --quiet <touched frontend files>` - passed with the existing Next pages-directory warning.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py -f json -o /tmp/bandit_character_chat_phase8.json` - passed, 0 results.
- `git diff --check` - passed.
- `bunx tsc --noEmit --pretty false` - failed on pre-existing baseline TypeScript errors outside this slice: Media read-along, embeddings recipe config, watchlists, workspace studio, keyboard shortcuts, persona live control, and tier-4 admin e2e fixtures.
