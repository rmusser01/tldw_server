# Chat Character Overlay And Tracked Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Reconciled on 2026-05-30. This plan is historical and should not be executed as written.

**Goal:** Preserve existing tracked character/persona chats while adding snapshot-based, non-destructive character/persona personality overlays for normal `/chat` conversations, with the main control surface in the existing cockpit runtime/context rails.

**Architecture:** Keep the current tracked chat contract intact and add a separate `assistantOverlay` settings contract for normal chats. Resolve effective assistant mode from tracked chat metadata plus chat settings, route tracked and overlay sends differently, and use the existing runtime/context cockpit rails as the assistant control surface.

**Tech Stack:** React, TypeScript, Next.js route wrappers, shared `@/components` UI package, Plasmo/local storage utilities, FastAPI, Pydantic validation, pytest, Vitest, Testing Library, Browser plugin or Playwright for final UI verification.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md`
- Design Backlog task: `TASK-444`

## 2026-05-30 Reconciliation

Do not create `CharacterControlRail`, `CharacterControlRailSheet`, or a new `character-control` coordinator panel from this plan. The current `dev` branch already includes the durable parts of this plan:

- `assistantOverlay` typing, normalization, sync, and backend validation;
- effective assistant-state resolution;
- snapshot-on-apply overlay helpers;
- runtime/context rail assistant selection, clear, context-source, tracked-start, and reload proof;
- real-server tests asserting `character-control-rail` is absent.

Tasks 1-3 below are retained as historical implementation context. Tasks 4-5 are superseded by the current cockpit runtime/context rail implementation and should not be used to reintroduce a standalone rail.

## Scope Constraints

- Keep existing tracked character chats and tracked persona chats behaviorally unchanged.
- Overlay is **snapshot-on-apply**, not live re-resolution.
- Overlay lives in chat settings, not in tracked chat identity fields.
- `Start tracked character/persona chat` uses tracked defaults; current plain-chat overlay, scene, style, and context do not implicitly carry forward.
- No route split for character chat.
- No removal of the existing composer, left sidebar, knowledge section, or artifacts panel.
- No backend tracked chat contract rewrite.
- Extension parity should reuse the same state contract, but desktop runtime/context rail work lands first.

## File Map

### Existing Files To Modify

- `apps/packages/ui/src/types/chat-session-settings.ts`
  - Add `assistantOverlay` with explicit snapshot semantics.

- `apps/packages/ui/src/services/chat-settings.ts`
  - Normalize, persist, sync, and merge `assistantOverlay`.
  - Preserve local-before-server-chat behavior.

- `apps/packages/ui/src/hooks/useMessageOption.tsx`
  - Remove destructive reset behavior keyed to assistant selection changes.

- `apps/packages/ui/src/hooks/chat/useSelectServerChat.ts`
  - Hydrate tracked state without making global selected assistant the active-chat truth.

- `apps/packages/ui/src/hooks/useMessage.tsx`
  - Route send behavior by effective assistant mode.

- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
  - Inject overlay personality snapshot with deterministic prompt ordering.

- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
  - Keep summary picker behavior intact while supporting runtime/context rail-driven overlay apply.

- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Historical note: the original plan proposed rendering a desktop character-control rail. The reconciled direction keeps this inside the existing cockpit runtime/context rails.

- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Add runtime/context rail entry/status wiring where needed without rehoming the composer.

- `apps/packages/ui/src/store/chat-surface-coordinator.ts`
  - Historical note: the original plan proposed rail panel state. The reconciled direction does not add a standalone character panel.

- `apps/packages/ui/src/routes/sidepanel-chat.tsx`
  - Reuse the same effective-mode and overlay settings contract for extension chat flows.

- `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - Reuse shared assistant state where sidepanel chat exposes assistant controls.

- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - Validate `assistantOverlay` in chat settings payloads.

### New Files To Create

- `apps/packages/ui/src/hooks/chat/effective-assistant-state.ts`
  - Pure resolver for `plain | overlay | tracked_character | tracked_persona`.

- `apps/packages/ui/src/utils/assistant-overlay.ts`
  - Build snapshot-on-apply overlay payloads from full character/persona detail.

Do not create these files from this plan:

- `apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/CharacterControlRailSheet.tsx`

### New Or Modified Tests

- `apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts`
- `apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts`
- `apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts`
- `apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx`
- `apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx`
- `apps/packages/ui/src/store/__tests__/chat-surface-coordinator.test.ts`
- `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.mobile-toolbar.contract.test.ts`
- `tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py`

## Task 0: Execution Setup And Tracking

**Files:**
- Backlog tasks only unless the implementation branch setup requires a note in the design task.

- [ ] **Step 1: Confirm branch and worktree state**

Run:
```bash
git branch --show-current
git status --short
```

Expected:
- You know whether implementation needs a dedicated branch or worktree.
- You do not stage unrelated dirty files.

- [ ] **Step 2: Create implementation Backlog tasks**

Use Backlog MCP or CLI to create:
- parent task: `Implement chat character overlay and tracked identity`
- child task 1: `Add chat assistant overlay settings contract and validation`
- child task 2: `Add effective assistant mode resolver and remove destructive reset`
- child task 3: `Implement overlay send-path and snapshot-on-apply behavior`
- child task 4: `Add desktop character control rail`
- child task 5: `Add sidepanel/mobile parity and final verification`

Expected:
- Each code-edit stage has a task before code changes begin.

- [ ] **Step 3: Link this plan in the active Backlog tasks**

Record:
- `Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md`
- `Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md`

Expected:
- Task records point at both the design spec and the implementation plan.

- [ ] **Step 4: Commit task-tracking changes only if task files changed**

Run:
```bash
git status --short backlog/tasks
git add <exact implementation task files created in Step 2>
git commit -m "chore: track chat overlay implementation tasks"
```

Expected:
- Backlog tracking lands separately from code when practical.

## Task 1: Add Overlay Settings Contract And Backend Validation

**Files:**
- Modify: `apps/packages/ui/src/types/chat-session-settings.ts`
- Modify: `apps/packages/ui/src/services/chat-settings.ts`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Test: `apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts`
- Test: `apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts`
- Test: `tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py`

- [ ] **Step 1: Write failing frontend tests for overlay normalization and local-first persistence**

Add tests for:
- accepting `assistantOverlay.system_prompt_snapshot`
- rejecting malformed overlay payloads at normalization time
- persisting overlay under `scratch` / `local:<historyId>` before `serverChatId` exists
- reconciling local overlay into server-backed settings once a chat id exists

Run:
```bash
bunx vitest run apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts
```

Expected:
- FAIL because `assistantOverlay` is not yet typed/normalized/synced.

- [ ] **Step 2: Write failing backend tests for overlay settings roundtrip and validation**

Add tests for:
- successful `PUT /api/v1/chats/{id}/settings` roundtrip with `assistantOverlay`
- rejection when `assistantOverlay.kind` is invalid
- rejection when `assistantOverlay.system_prompt_snapshot` is the wrong type or oversize

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py -k "assistant_overlay or overlay" -v
```

Expected:
- FAIL because backend validation does not yet know about `assistantOverlay`.

- [ ] **Step 3: Implement the frontend settings contract**

Implement:
- `assistantOverlay` type in `ChatSettingsRecord`
- optional-keys inclusion and coercion in `chat-settings.ts`
- comparable merge behavior so timestamp-only differences do not force unnecessary writes

Keep:
- existing deep-research settings logic untouched
- current `resolveChatSettingsKey()` semantics unchanged

- [ ] **Step 4: Implement backend validation**

Implement:
- shape validation for `assistantOverlay.kind`, `id`, `name`, `avatar_url`, `system_prompt_snapshot`, `updatedAt`
- bounded string checks consistent with the current settings payload style

Do not:
- add `assistantOverlay` to tracked chat identity models
- repurpose `ChatSessionUpdate`

- [ ] **Step 5: Run targeted frontend and backend tests**

Run:
```bash
bunx vitest run apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py -k "assistant_overlay or overlay" -v
```

Expected:
- PASS on both suites.

- [ ] **Step 6: Commit the settings-contract slice**

Run:
```bash
git add apps/packages/ui/src/types/chat-session-settings.ts apps/packages/ui/src/services/chat-settings.ts apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py
git commit -m "feat: add chat assistant overlay settings contract"
```

## Task 2: Add Effective Assistant Mode Resolution And Remove Destructive Resets

**Files:**
- Create: `apps/packages/ui/src/hooks/chat/effective-assistant-state.ts`
- Modify: `apps/packages/ui/src/hooks/useMessageOption.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/useSelectServerChat.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx`

- [ ] **Step 1: Write failing resolver tests**

Cover:
- tracked character resolution from `assistant_kind + character_id`
- tracked persona resolution from `assistant_kind + assistant_id`
- overlay resolution from `settings.assistantOverlay`
- plain resolution when neither exists
- tracked modes winning over overlay when bad mixed data appears

Run:
```bash
bunx vitest run apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts
```

Expected:
- FAIL because resolver does not exist.

- [ ] **Step 2: Write failing hook test for non-destructive overlay changes**

Cover:
- changing overlay does not clear `messages`
- changing overlay does not null `serverChatId`
- changing overlay does not wipe `historyId`

Run:
```bash
bunx vitest run apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx
```

Expected:
- FAIL because the assistant-switch reset path is still destructive.

- [ ] **Step 3: Implement the pure resolver**

Implement:
- one exported resolver with a stable return shape
- no storage, no API calls, no React dependencies

Return at minimum:
- `mode`
- `kind`
- `id`
- `displayName`
- `avatarUrl`
- `systemPromptSnapshot`
- `source`

- [ ] **Step 4: Replace destructive reset logic with conversation-keyed behavior**

Update `useMessageOption.tsx` so resets happen only for actual conversation changes, not for overlay/assistant draft changes.

Update `useSelectServerChat.ts` so:
- tracked chat metadata still hydrates
- global selected-assistant storage stops acting as the loaded chat’s source of truth

- [ ] **Step 5: Run targeted tests**

Run:
```bash
bunx vitest run apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx apps/packages/ui/src/hooks/__tests__/useMessageOption.selected-model-sync.test.tsx
```

Expected:
- PASS, including existing selected-model sync coverage.

- [ ] **Step 6: Commit the resolver/reset slice**

Run:
```bash
git add apps/packages/ui/src/hooks/chat/effective-assistant-state.ts apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts apps/packages/ui/src/hooks/useMessageOption.tsx apps/packages/ui/src/hooks/chat/useSelectServerChat.ts apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx
git commit -m "fix: separate active assistant mode from destructive selection resets"
```

## Task 3: Implement Snapshot-On-Apply Overlay Resolution And Send Routing

**Files:**
- Create: `apps/packages/ui/src/utils/assistant-overlay.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify as needed: `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- Test: `apps/packages/ui/src/utils/__tests__/assistant-overlay.test.ts`
- Test: `apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts`
- Test: `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`

- [ ] **Step 1: Write failing overlay snapshot tests**

Cover:
- persona overlay uses `getPersonaProfile()` detail, not catalog summary data
- character overlay uses full character detail
- stored payload captures `kind`, `id`, `name`, `avatar_url`, and `system_prompt_snapshot`
- later source-record changes do not affect stored snapshot sends

Run:
```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/assistant-overlay.test.ts
```

Expected:
- FAIL because the snapshot builder does not exist.

- [ ] **Step 2: Write failing prompt-ordering tests**

Cover:
- base system prompt resolves first
- overlay snapshot appends second
- actor/scene injection happens after overlay
- web-search system message stays last

Run:
```bash
bunx vitest run apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts
```

Expected:
- FAIL because normal chat mode does not yet encode overlay ordering.

- [ ] **Step 3: Implement the snapshot-on-apply helper**

Implement one helper that:
- accepts `AssistantSelection`
- fetches full detail via existing client methods
- returns a normalized `assistantOverlay` payload for settings persistence

Persona path:
- use `tldwClient.getPersonaProfile(selection.id)`

Character path:
- use `tldwClient.getCharacter(selection.id, { forceRefresh: true })` only if the existing character summary lacks required prompt material

Do not:
- live re-resolve overlay during send
- mutate tracked chat identity fields

- [ ] **Step 4: Route sends by effective mode**

Update `useMessage.tsx`:
- `tracked_character` -> existing `characterChatMode`
- `tracked_persona` -> existing persona-backed normal path
- `overlay` -> `normalChatMode` with `assistantIdentity` and `systemPromptAppendix` from `assistantOverlay`
- `plain` -> existing normal path

Update `normalChatMode.ts` so overlay ordering is explicit and stable.

- [ ] **Step 5: Run targeted tests**

Run:
```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/assistant-overlay.test.ts apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
```

Expected:
- PASS, with persona overlay proving it uses detail fetch rather than summary-only catalog data.

- [ ] **Step 6: Commit the send-path slice**

Run:
```bash
git add apps/packages/ui/src/utils/assistant-overlay.ts apps/packages/ui/src/utils/__tests__/assistant-overlay.test.ts apps/packages/ui/src/hooks/useMessage.tsx apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts apps/packages/ui/src/components/Common/AssistantSelect.tsx apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
git commit -m "feat: add snapshot-based assistant overlay send path"
```

## Task 4: Superseded Standalone Character Control Rail

Do not execute the original desktop-rail slice. Current `dev` intentionally keeps character/persona controls inside `PlaygroundRuntimeInspector`, `PlaygroundContextRail`, and the existing cockpit mobile tabs.

Current proof points:

- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` asserts `CharacterControlRail` is absent.
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` asserts `character-control-rail` has count `0` while proving runtime-rail character/persona selection, clear, tracked start, and reload.
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx` covers the runtime assistant controls that replaced the standalone rail direction.

Future assistant UI work should extend those existing cockpit rail components unless a new product decision explicitly reopens a separate panel.

## Task 5: Superseded Sidepanel/Mobile Rail Sheet

Do not create a `CharacterControlRailSheet`. Mobile `/chat` uses the existing cockpit context/runtime tabs, and the extension sidepanel handoff is currently route-only. Any future sidepanel assistant-state work should start from the shared `assistantOverlay` and effective-assistant-state contract, not from this rail-sheet plan.

Keep these proof gates when touching the assistant workflow:

- focused overlay/settings tests: `chat-settings.overlay`, `chat-settings.sync`, `effective-assistant-state`, `normalChatMode.overlay`, and `assistant-overlay`;
- focused cockpit tests: `PlaygroundRuntimeInspector.first-slice`, `Playground.cockpit-regression.guard`, `Playground.cockpit-controls`;
- real-server proof for runtime rail character/persona selection, clear, tracked start, reload, and absence of `character-control-rail`.

## Done Checklist

- [ ] Tracked character chats still behave exactly as before.
- [ ] Tracked persona chats still behave exactly as before.
- [ ] Plain chats can apply overlay before first send.
- [ ] Overlay changes do not reset thread state.
- [ ] Overlay is snapshot-on-apply, not live re-resolution.
- [ ] Tracked-start actions use tracked defaults rather than carrying plain-chat overlay state forward.
- [ ] Desktop `/chat` exposes runtime/context cockpit rails without replacing existing UI.
- [ ] Sidepanel/mobile surfaces reuse the same state contract.
- [ ] Targeted frontend and backend tests pass.
- [ ] Bandit result recorded.
