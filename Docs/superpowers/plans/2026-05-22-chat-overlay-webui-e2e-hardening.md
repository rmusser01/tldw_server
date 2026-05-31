# Chat Overlay WebUI E2E Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add browser-level WebUI `/chat` coverage for overlay and tracked assistant identity behavior using the live chat surface and real chat/session APIs.

**Architecture:** Extend the existing Playwright `/chat` workflow coverage with a focused real-server spec that drives the desktop `CharacterControlRail`, creates disposable character/persona fixtures through the API, and verifies continuity through network assertions plus reload checks. Keep the harness additive: prefer local helpers inside the new/updated spec, and only touch shared page objects if the rail interactions genuinely need reuse.

**Tech Stack:** Playwright, existing `apps/tldw-frontend/e2e` fixtures/helpers/page objects, real tldw server chat/persona/character APIs

---

### Task 1: Add failing overlay continuity coverage for `/chat`

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Optional modify: `apps/tldw-frontend/e2e/utils/page-objects/ChatPage.ts`
- Reference: `apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx`

- [x] **Step 1: Write the failing overlay continuity test**

Add a real-server Playwright test that:
- creates one disposable character and one disposable persona
- opens `/chat` on desktop
- applies a character overlay from `data-testid="character-control-rail"`
- sends a message and captures the first created `chat_id`
- changes the overlay to the persona
- reloads and verifies the overlay summary survives
- clears the overlay and verifies later sends still use the same `chat_id`

- [x] **Step 2: Run the new test alone to verify RED**

Run:

```bash
bun run e2e:pw -- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --grep "overlay"
```

Expected: FAIL because the new workflow assertions are not implemented yet, or because helper code for stable rail interaction/chat-id capture is missing.

- [x] **Step 3: Add the minimal helper code needed for the test**

Implement only the helpers required to make the test readable and stable, for example:
- disposable character/persona create/delete helpers using `request`
- rail picker helpers that click `Apply overlay` / `Change overlay`, switch tabs, and choose the named assistant
- API-call capture/assert helpers for `POST /api/v1/chats/`, `POST /api/v1/chats/{id}/complete-v2`, and `PUT /api/v1/chats/{id}/settings`

Keep these local to the spec unless `ChatPage` clearly benefits from a reusable method.

- [x] **Step 4: Re-run the overlay test to verify GREEN**

Run:

```bash
bun run e2e:pw -- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --grep "overlay"
```

Expected: PASS.

- [x] **Step 5: Commit the overlay coverage slice**

```bash
git add apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts apps/tldw-frontend/e2e/utils/page-objects/ChatPage.ts
git commit -m "test: cover chat overlay continuity in webui"
```

### Task 2: Add tracked character/persona rail-start coverage and reload checks

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Reference: `apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx`
- Reference: `apps/packages/ui/src/hooks/chat/effective-assistant-state.ts`

- [x] **Step 1: Write failing tracked-start tests**

Add two tests:
- tracked character start from the rail creates a tracked character chat and reload restores tracked mode
- tracked persona start from the rail creates a tracked persona chat and reload restores tracked mode

Assertions should include:
- correct create payload metadata (`character_id` for character chat, `assistant_kind` / `assistant_id` for persona chat)
- overlay controls hidden once the chat is tracked
- reload keeps the rail in tracked mode and keeps overlay actions unavailable

- [x] **Step 2: Run only the new tracked tests to verify RED**

Run:

```bash
bun run e2e:pw -- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --grep "tracked"
```

Expected: FAIL until the helper flow and assertions are wired correctly.

- [x] **Step 3: Implement the minimal tracked-flow helpers/assertions**

Add only the code needed to:
- start tracked character/persona selection from the rail
- send the first message through each tracked flow
- parse the resulting create request body and completion URL
- assert reload state from the rail UI

- [x] **Step 4: Re-run the tracked tests to verify GREEN**

Run:

```bash
bun run e2e:pw -- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --grep "tracked"
```

Expected: PASS.

- [x] **Step 5: Commit the tracked coverage slice**

```bash
git add apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
git commit -m "test: cover tracked rail starts in webui"
```

### Task 3: Verify, update backlog, and document residual risk

**Files:**
- Modify: `backlog/tasks/task-487 - Add-WebUI-chat-end-to-end-verification-for-overlay-and-tracked-identity.md`
- Modify: `Docs/superpowers/plans/2026-05-22-chat-overlay-webui-e2e-hardening.md`

- [x] **Step 1: Run the focused verification suite**

Run:

```bash
bun run e2e:pw -- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts
source .venv/bin/activate && python -m bandit -r apps/tldw-frontend/e2e -f json -o /tmp/bandit_task487.json
```

Expected:
- Playwright target passes
- related UI/unit guard tests stay green
- Bandit produces no findings in touched Python-parsable files; if it cannot parse the TS-only scope, document that explicitly

- [x] **Step 2: Update the plan status and TASK-487**

Record:
- tests run and results
- touched files
- any live-server preconditions or skipped coverage
- residual risk, especially around environment-dependent real-server flows

- [x] **Step 3: Commit task/plan bookkeeping**

```bash
git add backlog/tasks/task-487\ -\ Add-WebUI-chat-end-to-end-verification-for-overlay-and-tracked-identity.md Docs/superpowers/plans/2026-05-22-chat-overlay-webui-e2e-hardening.md
git commit -m "chore: record chat overlay webui e2e verification"
```

### Verification Notes

- Overlay continuity now uses a pre-seeded plain server chat instead of relying on live model generation; this keeps the browser path focused on character-control behavior and avoids provider startup variability.
- Focused Playwright verification passed:
  - `TLDW_WEB_AUTOSTART=false TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --grep "same conversation while overlay changes|tracked character chat|tracked persona chat" --reporter=line`
  - Result: `3 passed (31.7s)`
- Related UI/unit verification passed:
  - `bunx vitest run src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/hooks/__tests__/useCharacterGreeting.test.tsx src/hooks/__tests__/useMessage.routing-mode.test.ts src/hooks/__tests__/useSelectedAssistant.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/chat/__tests__/useChatActions.overlay.integration.test.tsx src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/hooks/utils/__tests__/messageHelpers.test.ts src/services/__tests__/chat-settings.overlay.test.ts src/types/__tests__/assistant-selection.test.ts`
  - Result: `12 passed`, `102 tests passed`
  - `bunx vitest run src/models/__tests__/ChatTldw.stream-metadata.test.ts`
  - Result: `1 passed`, `1 test passed`
- Related backend verification passed:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -k "overlay or plain or assistant_overlay" -v`
  - Result: `1 passed`
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py -k assistant_overlay -v`
  - Result: `1 passed`
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py -v`
  - Result: `3 passed`
- Bandit verification passed against the touched E2E and backend scope:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r apps/tldw-frontend/e2e tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py -f json -o /tmp/bandit_task487_full.json`
  - Result: exit `0`, no findings; only existing `# nosec` warnings were reported
- Residual risk:
  - The live-server slice is sensitive to shared local rate-limit windows. The disposable fixture helpers now back off on 429s, and the overlay clear step retries transient 429s after reload before treating the flow as failed.
