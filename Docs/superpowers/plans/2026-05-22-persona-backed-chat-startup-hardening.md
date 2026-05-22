# Persona-backed Chat Startup Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Harden Persona-backed ordinary chat startup so selected Personas create and resume normal chat sessions with explicit assistant metadata, default read-only memory behavior, no stale Character metadata, and regression coverage across the WebUI and backend API contracts.

**Architecture:** The current implementation already has the right substrate: `AssistantSelection` represents Character and Persona choices, `ensurePersonaServerChat` creates/reuses Persona-backed server chats, `useChatActions` and `useMessage` call normal chat mode with Persona identity, `useServerChatLoader` restores assistant identity from server chat metadata, and backend chat session schemas/endpoints persist `assistant_kind`, `assistant_id`, and `persona_memory_mode`. This plan closes the remaining PRD Stage 1 contract gaps and the smallest Stage 2 startup/persistence risk without introducing new Persona surfaces.

**Tech Stack:** Next.js/React WebUI package tests with Vitest; FastAPI/Pydantic backend schema/API tests with pytest; existing `tldwClient`, chat session endpoints, and ChaCha conversation storage.

**References:**
- GitHub issue: `#1908`
- Accepted PRD: `Docs/Product/Persona_Backed_Chat_Startup_PRD.md`
- Backlog task: `TASK-474`

**Explicit Non-goals:** Do not touch Buddy animation/runtime, Persona visual packs, Workspace Persona defaults, scheduled Persona work, multi-agent collaboration, broad personalization memory, or design-system backlog tasks.

---

### Task 1: Lock Assistant Selection Contract Coverage

**Files:**
- `apps/packages/ui/src/types/assistant-selection.ts`
- `apps/packages/ui/src/types/__tests__/assistant-selection.test.ts`
- `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`

**Step 1: Write/extend tests first**
- [ ] Add missing `AssistantSelection` normalization cases for Persona inputs with numeric IDs, blank IDs, legacy Character-like values, and invalid object shapes.
- [ ] Confirm the Assistant selector can choose a Persona and can switch between Character, Persona, and none without leaking the previous assistant kind.

**Step 2: Implement only if tests expose a gap**
- [ ] Patch `normalizeAssistantSelection` or selector behavior only for failing contract cases.
- [ ] Keep labels/avatar metadata as display-only; do not add Persona runtime behavior here.

**Verification:**
- [ ] `bunx vitest run apps/packages/ui/src/types/__tests__/assistant-selection.test.ts apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`

---

### Task 2: Fix Persona Server Chat Memory Isolation

**Files:**
- `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
- `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts`

**Current finding:** `ensurePersonaServerChat` computes `personaMemoryMode` before stale chat reset. If the user switches from a Character-backed server chat whose local state contains `serverChatPersonaMemoryMode: "read_write"`, the new Persona chat can inherit `read_write`. The existing test currently encodes this unsafe behavior. The PRD requires explicit `read_only` by default and no implicit durable memory writes.

**Step 1: Write failing test**
- [ ] Update the stale Character-backed chat test so the created Persona chat sends `persona_memory_mode: "read_only"` and returns `personaMemoryMode: "read_only"` even when stale local state had `read_write`.
- [ ] Add a matching stale different-Persona chat case if current coverage only covers Character-to-Persona switching.

**Step 2: Implement minimal fix**
- [ ] Compute effective Persona memory mode after `shouldResetServerChat`, or derive it as `serverChatPersonaMemoryMode` only when the existing server chat is already the same Persona.
- [ ] Preserve `read_write` only when reusing an existing matching Persona-backed chat.
- [ ] Keep `setServerChatCharacterId(null)` on Persona reuse and creation.

**Verification:**
- [ ] `bunx vitest run apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts`

---

### Task 3: Guard First-send Persona Startup Paths

**Files:**
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- `apps/packages/ui/src/hooks/useMessage.tsx`
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`
- Add a focused `useMessage` test only if the legacy path lacks equivalent coverage.

**Step 1: Write first-send regression tests**
- [ ] Cover normal `/chat` first send with selected Persona and stale Character server-chat state.
- [ ] Assert `tldwClient.createChat` receives `assistant_kind: "persona"`, selected Persona `assistant_id`, and explicit `persona_memory_mode: "read_only"`.
- [ ] Assert `normalChatMode` receives `assistantIdentity` for display and `serverChatId`/`historyId` from the Persona-backed server chat.
- [ ] Assert stale `serverChatCharacterId` is cleared and no Character ID is forwarded as Persona state.

**Step 2: Implement only if needed**
- [ ] Route both `useChatActions` and `useMessage` through the fixed helper behavior.
- [ ] Avoid duplicating Persona-specific startup rules outside `personaServerChat.ts`.

**Verification:**
- [ ] `bunx vitest run apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`
- [ ] Run any added `useMessage` focused test file.

---

### Task 4: Strengthen Resume Metadata Loading

**Files:**
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`

**Step 1: Write loader contract tests**
- [ ] Add a Persona resume case where `assistant_kind: "persona"` and `assistant_id` are present while legacy `character_id` is also present; the resolved identity must stay Persona-backed.
- [ ] Add or verify invalid/missing `persona_memory_mode` behavior remains non-escalating: preserve Persona identity, but do not synthesize `read_write`.
- [ ] Confirm legacy Character-only records still fall back to Character identity.

**Step 2: Implement only if needed**
- [ ] Patch `resolveServerChatAssistantIdentity` only for tested gaps.
- [ ] Do not infer Persona identity from `character_id`.

**Verification:**
- [ ] `bunx vitest run apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`

---

### Task 5: Backend Session Metadata Contract Tests

**Files:**
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py`
- Add schema-level tests near existing chat session schema/API tests if endpoint coverage is too broad.

**Step 1: Write backend tests first**
- [ ] Verify Persona chat creation requires `assistant_id`.
- [ ] Verify explicit `persona_memory_mode: "read_only"` is stored and returned by create/detail responses.
- [ ] Verify `persona_memory_mode` is rejected for Character chats.
- [ ] Verify invalid Persona memory mode is rejected by schema/API validation.

**Step 2: Implement only if tests expose a gap**
- [ ] Keep backend permissive about omitted Persona memory mode unless the PRD implementation slice explicitly chooses server-side defaulting.
- [ ] Do not introduce broad memory write behavior in backend chat startup.

**Verification:**
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -k persona`
- [ ] If schema-only tests are added, run that focused test file too.

---

### Task 6: Final Verification And Closeout

**Files:**
- All changed implementation/test files.
- Backlog task for the implementation slice.

**Steps:**
- [ ] Run the focused Vitest commands from Tasks 1-4.
- [ ] Run the focused pytest command from Task 5.
- [ ] Run `git diff --check`.
- [ ] If Python implementation files changed, run Bandit on touched Python paths:
  `source .venv/bin/activate && python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_persona_chat_startup.json`
- [ ] Update the implementation Backlog task with files changed, verification commands, skips, and final summary.
- [ ] Create a PR linked to `#1908` with a human-owned `Change Summary` placeholder.

**Success Criteria:**
- [ ] Selected Persona first-send creates or reuses a normal server chat with `assistant_kind: "persona"` and the selected Persona ID.
- [ ] New Persona-backed chats default to explicit `read_only` and never inherit stale `read_write` from Character or different-Persona chat state.
- [ ] Resuming Persona-backed chats restores Persona identity from server metadata without falling back to Character identity.
- [ ] Backend API contracts store and return Persona assistant metadata and reject invalid Character/Persona memory combinations.
- [ ] No Buddy runtime, Workspace defaults, scheduling, broad memory, or design-system files are modified.
