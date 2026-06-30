# Character Chat Real-Backend E2E Profile

Status: Phase 0 contract
Backlog: TASK-428
Related dependency: TASK-429

This runbook defines the real-backend verification profile for first-class Character Chat / role-play work. It is intentionally not a frontend-only mock profile. Unit and component tests may mock API clients, but release signoff for Character Chat must exercise a running WebUI against a running `tldw_server` backend.

## Scope

Use this profile for changes that affect:

- `/chat` Character Chat mode, setup, composer, or session continuity.
- Character selection paths from `/characters` into `/chat`.
- Character chat streaming, retry, regenerate, persistence, or history restore.
- Browser-extension sidepanel behavior that carries character chat state into WebUI.

## Required Environment

- FastAPI backend running at `TLDW_E2E_SERVER_URL` or `TLDW_SERVER_URL`.
- WebUI running at `TLDW_WEB_URL`.
- Single-user API key in `TLDW_E2E_API_KEY`, `TLDW_API_KEY`, or `SINGLE_USER_API_KEY`.
- At least one configured chat-capable provider/model exposed through `/api/v1/llm/providers`.
- A deterministic backend provider path for CI/release evidence. Acceptable options are:
  - a local OpenAI-compatible deterministic provider wired through backend provider configuration;
  - a configured local model provider with a stable smoke prompt;
  - an explicitly documented commercial provider run for manual release evidence.

Do not satisfy this profile by intercepting `/api` requests in Playwright or by returning fake completions from the frontend.

## Commands

From `apps/tldw-frontend`:

```bash
TLDW_WEB_URL=http://127.0.0.1:18080 \
TLDW_WEB_CMD='bun run dev -- -p 18080' \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=<local-api-key> \
bunx playwright test e2e/workflows/journeys/character-chat.spec.ts --reporter=line
```

For the broader `/chat` live-backend cockpit gate:

```bash
TLDW_WEB_URL=http://127.0.0.1:18080 \
TLDW_WEB_CMD='bun run dev -- -p 18080' \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=<local-api-key> \
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

## Required Assertions

Character Chat release signoff must prove:

1. The test creates or selects a real backend character.
2. `/chat` selects that character without losing the selected model.
3. Sending a message creates or reuses a real server chat.
4. The backend request path uses `/api/v1/chats/{chat_id}/complete-v2` for character streaming.
5. The stream payload includes `include_character_context: true`.
6. The payload uses a backend-resolved model/provider pair, not a frontend-only response.
7. The conversation can be found again through character-scoped chat history.

## Current Phase 0 Coverage

- `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts` freezes the character chat creation and stream payload shape at the hook seam.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.chat-debug.test.ts` verifies the API client debug snapshot for `/complete-v2` streaming includes character context.
- `apps/packages/ui/src/hooks/__tests__/useServerChatHistory.test.ts` verifies character-scoped overview and search requests pass `character_scope: "character"`.
- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts` is the browser journey profile to run against a real backend.

## DB Health Release Dependency

Character Chat GA is blocked on TASK-429 unless the release owner explicitly accepts the risk. That task owns the backend recovery path for corrupt per-user `ChaChaNotes` or chat databases:

- identify the affected DB and failure reason;
- document backup, integrity validation, SQLite recovery, and restore;
- keep setup, diagnostics, or recovery UI reachable where safe;
- avoid silent data mutation.

Phase 1 and later UI work may proceed, but Character Chat should not be marked GA until TASK-429 is resolved or release-blocked in the release notes.
