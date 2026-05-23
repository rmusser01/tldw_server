# PR-1987 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Execution note: user explicitly requested CDP/direct investigation for the critical reload issue, so this was executed directly without subagents.

**Goal:** Address the actionable PR-1987 review findings without widening the overlay/tracked identity scope.

**Architecture:** Keep the fixes tightly local to the existing overlay/tracked hardening path. Add regression coverage for the two real behavior bugs first, then patch the implementation to use explicit per-action mode selection and to keep scratch chat settings opt-in for server-chat linking only.

**Tech Stack:** React, Vitest, Playwright-adjacent WebUI helpers, FastAPI/Pytest for touched backend tests

---

### Task 1: Fix stale assistant selection mode persistence

**Files:**
- Modify: `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/AssistantSelect.tsx`

- [x] **Step 1: Write the failing test**

Add a regression test proving an overlay-triggered selection persists `metadata.selectionMode: "overlay"` and writes overlay settings even when the component starts from its default tracked state.

- [x] **Step 2: Run the targeted test to verify RED**

Run:

```bash
bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
```

Expected: FAIL on the new overlay-mode regression.

- [x] **Step 3: Implement the minimal fix**

Use a local `nextMode` derived from the current selection intent inside `handleSelect`, and use it for metadata, overlay snapshot resolution, and overlay persistence branching instead of the stale state variable.

- [x] **Step 4: Re-run the targeted test to verify GREEN**

Run:

```bash
bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
```

Expected: PASS.

### Task 2: Stop scratch settings from leaking during server-chat hydration

**Files:**
- Modify: `apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`
- Modify: `apps/packages/ui/src/services/chat-settings.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`

- [x] **Step 1: Write the failing tests**

Add regression coverage that:
- `syncChatSettingsForServerChat` does not consume the global scratch key unless explicitly allowed
- server-chat loader hydration for plain chats does not use the scratch-seeding path

- [x] **Step 2: Run the targeted tests to verify RED**

Run:

```bash
bunx vitest run src/services/__tests__/chat-settings.overlay.test.ts src/hooks/__tests__/useServerChatLoader.test.ts
```

Expected: FAIL on the new scratch-leak regressions.

- [x] **Step 3: Implement the minimal fix**

Add an explicit opt-in flag for scratch fallback in `syncChatSettingsForServerChat`, default it to `false`, and only enable it from server-chat-linking flows where the current draft session is being reconciled into a freshly discovered server chat id.

- [x] **Step 4: Re-run the targeted tests to verify GREEN**

Run:

```bash
bunx vitest run src/services/__tests__/chat-settings.overlay.test.ts src/hooks/__tests__/useServerChatLoader.test.ts
```

Expected: PASS.

### Task 3: Remove the test cleanup swallow and dead debug stub, then verify

**Files:**
- Modify: `tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py`
- Modify: `apps/packages/ui/src/hooks/useMessageOption.tsx`
- Modify: `backlog/tasks/task-488 - Address-PR-1987-review-findings-for-chat-overlay-hardening.md`

- [x] **Step 1: Apply the scoped cleanup fixes**

Remove the dead `logE2EDebug` stub from `useMessageOption.tsx` and replace the test cleanup swallow with direct `shutil.rmtree(..., ignore_errors=True)` cleanup.

- [x] **Step 2: Run focused verification**

Run:

```bash
bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/services/__tests__/chat-settings.overlay.test.ts src/hooks/__tests__/useServerChatLoader.test.ts
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -k plain_chat_session_without_tracked_identity -v
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py apps/packages/ui/src/services/chat-settings.ts apps/packages/ui/src/components/Common/AssistantSelect.tsx apps/packages/ui/src/hooks/chat/useServerChatLoader.ts -f json -o /tmp/bandit_task488.json
```

Expected:
- focused Vitest slice passes
- focused pytest slice passes
- Bandit returns no new findings; if TS paths are skipped, document that explicitly

- [x] **Step 3: Update backlog task with verification**

Record touched files, verification commands/results, and any residual risk in `TASK-488`.

### Task 4: Preserve tracked identity across immediate server-chat reloads

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
- Modify: `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx`
- Modify: `apps/packages/ui/src/store/playground-session.tsx`

- [x] **Step 1: Reproduce the reload failure**

The real-server browser flow showed tracked character/persona chats creating successfully, but reload could restore as plain chat or with a generic persona label because tracked identity metadata was not durable before subsequent autosave.

- [x] **Step 2: Add focused regressions**

Added regressions for immediate tracked-character session persistence, persona metadata ordering after `setServerChatId`, and preserving a richer tracked persona snapshot when a later autosave only has generic kind/id metadata.

- [x] **Step 3: Fix the persistence order and snapshot enrichment**

Persist tracked character identity immediately after server chat creation, write persona server-chat metadata after `setServerChatId` clears identity fields, and enrich generic autosave snapshots from the existing persisted tracked selection before saving.

- [x] **Step 4: Verify with unit and real-server browser tests**

Verified with focused Vitest, the broader touched UI slice, and the real-server `/chat` browser cases for overlay continuity plus tracked character/persona reload restoration.
