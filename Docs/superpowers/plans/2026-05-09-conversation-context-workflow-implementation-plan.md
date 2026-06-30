# Conversation Context Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Conversation Context workflow with client-managed context composition. Blank chats, character chats, workspace chats, worldbooks, and chat dictionaries should share one inspectable client-side composition model in the chat composer.

**Architecture:** The client owns effective context selection, ordering, preview assembly, and send-payload assembly. The server provides reliable composable primitives: chat settings, character/prompt pieces, worldbook matching, dictionary transformation, provider readiness, and validation. Do not add a monolithic backend effective-context endpoint.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes/CharactersRAGDB, `WorldBookService`, `ChatDictionaryService`, Next.js/React, Ant Design popover controls, Tailwind utility classes, Vitest, Playwright/Chromium, pytest, Bandit.

---

## Source Documents

- Design spec: `Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md`
- UX audit: `Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md`
- Planning task: `TASK-186`
- Architecture correction task: `TASK-187`
- Prior spec task: `TASK-185`

## Scope

This plan covers the first implementation tranche needed to make Conversation Context real at the conversation boundary while preserving the intended client/server split:

- Server primitive hardening for chat settings, worldbook processing, dictionary processing, and prompt pieces.
- A client-side context composer that assembles effective preview state and send payloads from server primitives.
- Preview/send parity enforced in the client by using the same composed context object for the inspector and the actual request.
- Chat-scoped context settings that keep worldbooks and dictionaries conversation-scoped, not character-exclusive.
- A composer popover that replaces or evolves the existing `CharacterSelect` placement in `ControlRow`.
- Unit, API, frontend, and browser validation.

This plan intentionally defers broad asset-management work that is adjacent but not required for the first working conversation-boundary workflow:

- Full bulk assignment redesign for dictionaries and worldbooks.
- Full Workspace Playground context-management redesign.
- New DB tables for first-class context attachments. Use existing conversation settings for this tranche unless implementation proves settings cannot support the behavior safely.
- A server-owned all-in-one effective context contract. That is explicitly out of scope for this architecture.

## Current Code Map

Backend files to inspect before editing:

- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - Existing prompt preview: `POST /api/v1/chats/{chat_id}/prompt-preview`
  - Existing settings endpoints: `GET/PUT /api/v1/chats/{chat_id}/settings`
  - Existing send path: `POST /api/v1/chats/{chat_id}/complete-v2`
  - Existing worldbook preview and send-time diagnostics are inline and character-derived.
- `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
  - Existing dictionary processing endpoint: `POST /api/v1/chat/dictionaries/process`
- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
  - Existing worldbook processing endpoint: `POST /api/v1/characters/world-books/process`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - `CharacterChatCompletionPrepRequest`
  - `ChatSettingsUpdate`
  - `ChatSettingsResponse`
- `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
  - Dictionary processing request/response schemas.
- `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py`
  - Worldbook processing request/response schemas.
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
  - `WorldBookService.process_context(text, world_book_ids=None, character_id=None, include_diagnostics=True, ...)`
- `tldw_Server_API/app/core/Character_Chat/chat_dictionary.py`
  - `ChatDictionaryService.process_text(text, dictionary_id=None, return_stats=True, chat_id=...)`
  - Existing recursive settings scanner for dictionary ids.

Frontend files to inspect before editing:

- `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
  - Current `CharacterSelect` call site in the composer controls.
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
  - Existing character/persona selector behavior. Preserve this behavior by composing it into the Conversation Context popover rather than deleting and recreating it.
- `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - Current `ControlRow` wiring, selected character state, context chips, next-gen composer slots, and send gating.
- `apps/packages/ui/src/components/Chat/composer/ChatComposer.tsx`
  - Existing next-gen composer slots; `form.tsx` already passes `facetsSlot` or `bottomBarSlot`.
- `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
  - Existing `getCharacterPromptPreview`, `getChatSettings`, `updateChatSettings`, worldbook processing, dictionary processing.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Monolithic mirrored client methods.
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Endpoint literal guard.
- `apps/packages/ui/src/components/Option/Dictionaries/components/useDictionaryQuickAssign.ts`
  - Existing dictionary assignment writes top-level `chat_dictionary_ids`; new code must read/write compatibly.

## Architecture Contract

Do not implement `POST /api/v1/chats/{chat_id}/context-preview`.

The client should compose effective context using a typed object like:

```ts
export type ConversationContextSource =
  | "request"
  | "explicit_chat"
  | "workspace"
  | "character_start"
  | "character_inherited"
  | "global"

export type ConversationContextAssetKind =
  | "character"
  | "worldbook"
  | "dictionary"
  | "workspace"
  | "provider"

export interface ConversationContextSelection {
  chatId?: string
  characterId?: number | string | null
  worldBookIds: number[]
  dictionaryIds: number[]
  workspaceId?: string | null
  providerId?: string | null
  modelId?: string | null
}

export interface ConversationContextPiece {
  kind: ConversationContextAssetKind
  id?: number | string | null
  name?: string | null
  source: ConversationContextSource
  status: "configured" | "active" | "matched" | "skipped" | "blocked" | "missing"
  content?: string
  diagnostics?: unknown
  warnings?: string[]
}

export interface ConversationContextComposition {
  selection: ConversationContextSelection
  inputText: string
  transformedInputText: string
  pieces: ConversationContextPiece[]
  previewSections: Array<{ name: string; content: string; source: ConversationContextSource }>
  providerMessages: Array<{ role: string; content: string }>
  readiness: "ready" | "partial" | "blocked"
  warnings: string[]
}
```

Server primitives should remain small and composable:

- `GET /api/v1/chats/{chat_id}/settings`
- `PUT /api/v1/chats/{chat_id}/settings`
- `POST /api/v1/characters/world-books/process`
- `POST /api/v1/chat/dictionaries/process`
- Character/persona detail and list endpoints already used by selectors.
- Existing `POST /api/v1/chats/{chat_id}/prompt-preview` may be used only as a compatibility primitive where it returns useful prompt fragments. It must not become the canonical all-in-one context engine.
- If existing endpoints cannot expose a needed prompt fragment without also assembling final context, add a narrow prompt-pieces endpoint. It should return pieces, not an effective context object.

Canonical settings shape for this tranche:

```json
{
  "conversationContext": {
    "world_book_ids": [1, 2],
    "chat_dictionary_ids": [7, 42]
  },
  "chat_dictionary_ids": [7, 42]
}
```

Implementation notes:

- Read both nested `conversationContext.chat_dictionary_ids` and existing top-level dictionary aliases.
- Write the nested canonical shape from the new composer UI.
- Mirror `chat_dictionary_ids` at the top level while Dictionary Quick Assign still uses that key.
- Read `conversationContext.world_book_ids` plus route/request overrides for explicit chat worldbooks.
- Do not attach worldbooks to a character just because a chat selected them.
- Character-attached worldbooks remain inherited context when a character is active.

## Client Composition Rules

- The client owns ordering and precedence:
  - request/route seed
  - explicit chat settings
  - workspace inherited settings, if available
  - character-start or character-attached context
  - global defaults
- The client must call server primitives for domain-specific operations:
  - dictionary transformation
  - worldbook matching
  - prompt or character fragments that are server-owned today
  - provider/model readiness when available
- The client must not reimplement dictionary matching, regex replacement, probability, worldbook keyword matching, or token-budget semantics.
- Dictionary transformation should run before worldbook matching for the candidate user turn so terms such as `EV` can normalize to lorebook keywords such as `Echo Vault`.
- The same `ConversationContextComposition` object must feed:
  - the composer popover preview
  - the send payload
  - any post-send diagnostics captured by the UI
- If a server completion endpoint would also inject dictionaries or worldbooks, the implementation must avoid double application. Prefer a client-composed provider message path; if `complete-v2` is required for persistence semantics, add a narrow pass-through mode that accepts composed provider messages and bypasses server-side context injection.

---

## Task 0: Start The Implementation Task

**Goal:** Create the implementation Backlog task and execution trail before code edits.

**Success Criteria:** Backlog has a task linked to this plan and the approved design spec.

**Tests:** N/A.

**Status:** Not Started.

- [ ] Create a new Backlog task, for example `Implement client-managed conversation context workflow`.
- [ ] Link docs:
  - `Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md`
  - `Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md`
  - `Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md`
- [ ] Set the task to `In Progress`.
- [ ] Record that this plan requires subagent/executing-plan workflow per the header. If runtime policy prevents spawning subagents, record that implementation will use local execution checkpoints instead.

## Task 1: Server Primitive Audit And Hardening

**Goal:** Verify and harden the server primitives the client needs to compose context. Do not build a monolithic effective-context endpoint.

**Success Criteria:** The client can fetch settings, process dictionaries, process worldbooks, and retrieve required prompt/character pieces independently for blank and character chats.

**Tests:** Add targeted backend tests for the primitive endpoints and known failure modes.

**Status:** Not Started.

Files:

- Edit `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Edit `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py` only if dictionary primitive gaps are found.
- Edit `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py` only if worldbook primitive gaps are found.
- Edit `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` only if a prompt-pieces or pass-through schema is needed.
- Add `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py`
- Extend `tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py` if dictionary processing needs list-id or settings behavior.

Steps:

- [ ] Write failing tests that prove the existing primitive surface can support a blank chat with explicit worldbook and dictionary ids.
- [ ] Verify `POST /api/v1/characters/world-books/process` supports explicit `world_book_ids` without a character id.
- [ ] Verify dictionary processing supports the client use case:
  - explicit dictionary id or ids
  - chat-associated dictionaries from settings where currently supported
  - return stats/diagnostics suitable for a preview
- [ ] If dictionary processing only supports one explicit id, add a narrow primitive extension for ordered dictionary ids. Do not add full context composition.
- [ ] Verify `GET/PUT /api/v1/chats/{chat_id}/settings` accepts the canonical `conversationContext` shape and preserves the top-level `chat_dictionary_ids` compatibility mirror.
- [ ] If prompt fragments such as character preset, greeting, or author note cannot be retrieved without final prompt assembly, add a narrow `prompt-pieces` primitive. It should return named pieces and diagnostics only.
- [ ] Return `400` or `404` for invalid context asset ids. Do not let invalid workspace, worldbook, or dictionary ids surface as 500s.
- [ ] Preserve existing prompt-preview and completion behavior for current clients.

TDD tests to add first:

```python
def test_worldbook_process_accepts_explicit_ids_without_character(...):
    ...

def test_dictionary_process_accepts_client_ordered_dictionary_ids(...):
    ...

def test_chat_settings_preserve_conversation_context_shape(...):
    ...

def test_invalid_context_primitive_ids_return_domain_errors(...):
    ...
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py -k "conversation_context or dictionary_process or worldbook_process" -v
```

Expected result:

- New tests fail before hardening where gaps exist.
- New tests pass after primitive hardening.
- No server endpoint returns a composed effective context object.

Commit after this task:

```bash
git add tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py backlog/tasks/<task-file>
git commit -m "feat: harden conversation context primitives"
```

## Task 2: Client Context Composer Core

**Goal:** Add a client-side composition engine that turns route seeds, chat settings, server primitive outputs, and draft text into one `ConversationContextComposition`.

**Success Criteria:** Frontend tests can compose preview sections and provider messages for blank chat, character chat, dictionary-only chat, worldbook-only chat, and combined dictionary-before-worldbook chat.

**Tests:** Vitest tests for pure composition, primitive orchestration, and preview/send parity.

**Status:** Not Started.

Files:

- Add `apps/packages/ui/src/types/conversation-context.ts`
- Add `apps/packages/ui/src/services/conversation-context/conversationContextComposer.ts`
- Add `apps/packages/ui/src/services/conversation-context/conversationContextSettings.ts`
- Add `apps/packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts`
- Add `apps/packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts`
- Edit `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Edit `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Edit `apps/packages/ui/src/services/tldw/openapi-guard.ts` if primitive endpoints are added or changed.

Steps:

- [ ] Define `ConversationContextSelection`, `ConversationContextPiece`, and `ConversationContextComposition`.
- [ ] Add settings normalization:
  - read nested `conversationContext`
  - read old top-level dictionary aliases
  - merge route seeds with chat settings
  - preserve source labels
- [ ] Add primitive client wrappers:
  - `processWorldBookContext`
  - `processDictionary`
  - `getChatSettings`
  - `updateChatSettings`
  - optional `getPromptPieces`
- [ ] Implement `composeConversationContext(input)`:
  - normalize selection
  - call dictionary processing first
  - call worldbook processing using transformed text
  - add character/prompt pieces
  - build preview sections
  - build provider messages
  - emit warnings and readiness state
- [ ] Ensure the composer is testable with mocked primitive functions.
- [ ] Ensure composition returns traceable diagnostics, not opaque strings.

Vitest examples:

```ts
it("composes blank chat with dictionary before worldbook matching", async () => {
  ...
})

it("keeps worldbooks and dictionaries conversation-scoped without a character", async () => {
  ...
})

it("uses the same composition object for preview sections and provider messages", async () => {
  ...
})

it("labels explicit chat and character-inherited worldbooks separately", async () => {
  ...
})
```

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts
```

Expected result:

- Pure composition tests pass with mocked server primitive outputs.
- No test calls a monolithic backend context endpoint.
- Dictionary-before-worldbook ordering is explicit and asserted.

Commit after this task:

```bash
git add apps/packages/ui/src/types/conversation-context.ts apps/packages/ui/src/services/conversation-context/conversationContextComposer.ts apps/packages/ui/src/services/conversation-context/conversationContextSettings.ts apps/packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts apps/packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/openapi-guard.ts backlog/tasks/<task-file>
git commit -m "feat: add client conversation context composer"
```

## Task 3: Composition Hook And Send Integration

**Goal:** Connect the client composer to the chat route so preview and send use the same composed context.

**Success Criteria:** The composer popover and send path consume one `ConversationContextComposition`; no separate preview-only path exists.

**Tests:** Hook tests plus send-payload tests.

**Status:** Not Started.

Files:

- Add `apps/packages/ui/src/hooks/chat/useConversationContextComposition.ts`
- Add `apps/packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx`
- Edit `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Edit `apps/packages/ui/src/services/tldw/domains/chat-rag.ts` if send needs a client-composed path helper.
- Edit backend only if a narrow pass-through mode is needed:
  - `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py`

Steps:

- [ ] Add `useConversationContextComposition` that:
  - waits for a persisted chat id before saving context settings
  - can compose local preview state before persistence from route seeds
  - debounces draft-message recomposition
  - exposes `composition`, `status`, `refresh`, and `saveSelection`
- [ ] Route the composer inspector preview through the hook.
- [ ] Route the send path through the same composition object.
- [ ] Prefer sending client-composed provider messages to the existing OpenAI-compatible chat endpoint when possible.
- [ ] If the product requires `complete-v2` for character-chat persistence semantics, add a narrow request mode such as `client_composed_messages` plus `skip_context_injection`. This mode should forward client-composed provider messages and avoid server-side worldbook/dictionary duplication.
- [ ] Preserve persisted user message text as authored by the user; do not overwrite chat history with dictionary-transformed text.
- [ ] Include client-composition diagnostics in UI state or message metadata where currently supported.

Vitest examples:

```tsx
it("reuses one composition object for preview and send", async () => {
  ...
})

it("does not apply dictionary transforms twice when sending", async () => {
  ...
})

it("allows blank chat send without selected optional context", async () => {
  ...
})
```

Backend test only if pass-through mode is added:

```python
def test_complete_v2_client_composed_messages_bypass_server_context_injection(...):
    ...
```

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts --config vitest.config.ts
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py -k "client_composed or pass_through" -v
```

Expected result:

- Frontend tests prove preview and send share one composition path.
- Backend pass-through tests run only if a backend pass-through path was added.
- No server endpoint assembles final effective context on behalf of the UI.

Commit after this task:

```bash
git add apps/packages/ui/src/hooks/chat/useConversationContextComposition.ts apps/packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/services/tldw/domains/chat-rag.ts tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py backlog/tasks/<task-file>
git commit -m "feat: use client-composed conversation context for chat"
```

## Task 4: Composer Popover Replacing The Character Picker

**Goal:** Replace or evolve the existing chat-composer character picker into a Conversation Context popover, with character selection as one slot inside broader context.

**Success Criteria:** The composer has one visible context control where users can see character, worldbooks, dictionaries, and diagnostics. The old character-selection behavior remains available inside that control.

**Tests:** Component tests for rendering, keyboard flow, selection callbacks, and client-composed preview states.

**Status:** Not Started.

Files:

- Add `apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx`
- Add `apps/packages/ui/src/components/Sidepanel/Chat/conversation-context-utils.ts`
- Edit `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
- Edit `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Add `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx`
- Extend `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx` only if wrapping affects avatar behavior.

Component behavior:

- Trigger:
  - Use a compact icon+status button in the existing `ControlRow` slot.
  - Show selected character name only when space allows; otherwise use tooltip and status badge.
  - Use stable width/height so composer layout does not shift as preview state changes.
- Popover summary:
  - Character slot using the existing `CharacterSelect` implementation or a small extracted subcomponent from it.
  - Worldbook slot showing configured/matched/skipped counts from the client composition.
  - Dictionary slot showing configured/replaced/skipped counts from the client composition.
  - Readiness row: ready, partial, blocked.
- Power-user details:
  - Diagnostics tab or collapsible region for matched worldbook entries, dictionary replacements, token estimates, and warnings.
  - Scope/source labels: explicit chat, workspace, character-start, character-inherited, global, request.
- Editing:
  - Character selection updates the existing `selectedCharacterId` flow.
  - Worldbook/dictionary edits update conversation settings only when a persisted chat exists.
  - If no persisted chat exists, show a non-blocking disabled edit state and allow blank chat/send flow to remain usable.

Steps:

- [ ] Extract the minimum reusable character-picker body from `CharacterSelect.tsx` if direct composition is awkward. Do not rewrite character/persona listing behavior.
- [ ] Add `ConversationContextPopover` with props:
  - `chatId`
  - `draftMessage`
  - `selectedCharacterId`
  - `setSelectedCharacterId`
  - `composition`
  - `compositionStatus`
  - `saveSelection`
  - `scope`
  - `disabled`
  - `className`
  - `iconClassName`
- [ ] Add utility functions:
  - `summarizeConversationContextPieces`
  - `formatContextSourceLabel`
  - `resolveContextReadiness`
- [ ] Replace the direct `CharacterSelect` call in `ControlRow.tsx` with `ConversationContextPopover`.
- [ ] Pass `chatId`, draft message, scope, character props, and composition state from `form.tsx` into `ControlRow`.
- [ ] Preserve mobile hit target sizes and current composer control wrapping.
- [ ] Ensure next-gen composer slots receive the same `composerControlAreaNode`; do not create separate next-gen context UI.

Tests:

```tsx
it("renders character selection as a slot inside conversation context", () => {
  ...
})

it("shows dictionary and worldbook diagnostics from client composition", async () => {
  ...
})

it("does not block blank chat when no context assets are selected", () => {
  ...
})

it("keeps the trigger size stable between loading and ready states", () => {
  ...
})
```

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx --config vitest.config.ts
```

Expected result:

- Existing character selection tests still pass.
- New popover tests pass.
- No test depends on frontend-side dictionary/worldbook matching logic; mocked server primitive outputs drive those states.

Commit after this task:

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx apps/packages/ui/src/components/Sidepanel/Chat/conversation-context-utils.ts apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx backlog/tasks/<task-file>
git commit -m "feat: add composer conversation context popover"
```

## Task 5: Minimal Asset Selection For Worldbooks And Dictionaries

**Goal:** Let users attach or remove worldbooks and dictionaries from a conversation through the new popover without making them character-card exclusive.

**Success Criteria:** A blank chat can carry selected worldbooks and dictionaries; a character chat can carry both explicit chat assets and character-inherited assets with clear labels.

**Tests:** Frontend unit tests plus backend settings primitive tests.

**Status:** Not Started.

Files:

- Edit `apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx`
- Edit `apps/packages/ui/src/hooks/chat/useConversationContextComposition.ts`
- Edit `apps/packages/ui/src/services/conversation-context/conversationContextSettings.ts`
- Add or extend backend tests in `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py`
- Add or extend frontend tests in `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx`

Steps:

- [ ] Add lightweight selectors inside the popover using existing list APIs:
  - worldbooks list/search endpoint already exposed through `tldwClient`
  - dictionaries list/search endpoint already exposed through `tldwClient`
- [ ] Use checkboxes or multi-select controls for worldbooks and dictionaries.
- [ ] Save to conversation settings via the client settings helper.
- [ ] On save, rerun client composition and show server primitive diagnostics.
- [ ] If the selected context is inherited and not explicitly attached, removal should disable or hide the explicit chat override only if current settings support that distinction. Otherwise label it as inherited and do not offer destructive removal.
- [ ] Keep Dictionary Quick Assign compatibility by preserving top-level `chat_dictionary_ids` while the older assignment UI exists.

Regression tests:

```tsx
it("persists selected worldbooks under conversationContext.world_book_ids", async () => {
  ...
})

it("persists selected dictionaries under nested and compatibility keys", async () => {
  ...
})

it("labels character-inherited and explicit-chat worldbooks separately", async () => {
  ...
})
```

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py -k "settings" -v
```

Expected result:

- Blank chat context assets persist in chat settings.
- Character-inherited assets are visible but not mislabeled as character-owned dictionaries.
- Dictionary Quick Assign still recognizes dictionary usage through the compatibility key.

Commit after this task:

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx apps/packages/ui/src/hooks/chat/useConversationContextComposition.ts apps/packages/ui/src/services/conversation-context/conversationContextSettings.ts apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx apps/packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py backlog/tasks/<task-file>
git commit -m "feat: attach conversation context assets from composer"
```

## Task 6: Browser And Reliability Validation

**Goal:** Validate the end-to-end workflow from first-time and power-user perspectives using Chrome-backed automation, without Computer Use.

**Success Criteria:** Browser validation confirms the popover is discoverable, non-blocking, and reliable for blank chat, character chat, worldbook use, dictionary use, and no-provider states.

**Tests:** Existing Playwright/Chromium smoke tests plus a focused Puppeteer or CDP walkthrough artifact.

**Status:** Not Started.

Files:

- Add or extend `apps/tldw-frontend/e2e/smoke/composer-picker-keyboard.spec.ts`
- Add or extend `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`
- Add `Docs/Reviews/assets/2026-05-09-conversation-context-workflow/` artifacts if a Puppeteer/CDP walkthrough is recorded.
- Add `Docs/Reviews/CONVERSATION_CONTEXT_WORKFLOW_VALIDATION_2026_05_09.md` if validation findings are substantial.

Steps:

- [ ] Run unit/API tests from prior tasks.
- [ ] Start backend and frontend with the existing local development commands.
- [ ] Seed one character, one worldbook with an `Echo Vault` entry, one dictionary replacing `EV` with `Echo Vault`, one blank chat, and one character chat.
- [ ] Use Chrome-backed browser automation to walk:
  - first-time blank chat with no optional context
  - first-time character chat with a character selected
  - blank chat with explicit worldbook and dictionary
  - character chat with character-inherited worldbook plus explicit dictionary
  - no-provider state
  - mobile composer viewport
- [ ] Capture screenshots and JSON state for the popover open state and diagnostics.
- [ ] Confirm browser-visible labels do not imply that dictionaries or worldbooks are character-card exclusive.
- [ ] Confirm the UI preview traces to server primitive outputs and the send payload uses the same client composition object.

Run existing browser smoke:

```bash
cd apps/tldw-frontend && npx playwright test e2e/smoke/composer-picker-keyboard.spec.ts e2e/smoke/composer-mobile-viewport.spec.ts e2e/smoke/playground-nextgen-composer.spec.ts --reporter=line
```

Run a focused Chrome/CDP UX audit if adding one:

```bash
cd apps/tldw-frontend && bunx tsx scripts/ux-audit-cdp.ts --route /chat --label conversation-context
```

Expected result:

- Composer popover opens with keyboard and pointer.
- Empty/no-provider states are understandable and recoverable.
- Context preview traces to individual server primitive outputs.
- No visible overlap or layout jump in desktop or mobile screenshots.

Commit after this task:

```bash
git add apps/tldw-frontend/e2e/smoke/composer-picker-keyboard.spec.ts apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts apps/tldw-frontend/e2e/smoke/playground-nextgen-composer.spec.ts Docs/Reviews/CONVERSATION_CONTEXT_WORKFLOW_VALIDATION_2026_05_09.md Docs/Reviews/assets/2026-05-09-conversation-context-workflow backlog/tasks/<task-file>
git commit -m "test: validate conversation context workflow"
```

## Task 7: Documentation And Closeout

**Goal:** Update user-facing and API-facing docs enough that future maintainers understand the client-managed conversation context model.

**Success Criteria:** Docs describe Conversation Context as conversation-scoped, client-composed, and powered by server primitives. They also document dictionary/worldbook order and composer placement.

**Tests:** Documentation review plus command verification.

**Status:** Not Started.

Files:

- Edit `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md`
- Edit `Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md` if published docs are expected to mirror source docs in this repo.
- Edit `Docs/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md`
- Edit `Docs/Published/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md` if published docs are mirrored.
- Optionally add `Docs/User_Guides/WebUI_Extension/Conversation_Context_Guide.md`
- Update Backlog implementation task final summary.

Steps:

- [ ] Document the client-managed composition model.
- [ ] Document the server primitives used by the client.
- [ ] Document dictionary transform order relative to worldbook matching.
- [ ] Document that worldbooks and dictionaries can be attached to conversations without a character.
- [ ] Document that character-attached worldbooks are inherited context, not the only worldbook source.
- [ ] Include short troubleshooting notes for missing provider, missing asset ids, no matches, and empty draft text.
- [ ] Record verification commands and results in the Backlog task.

Final verification commands:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py -k "conversation_context or dictionary_process or worldbook_process or settings" -v
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx --config vitest.config.ts
cd apps/tldw-frontend && npx playwright test e2e/smoke/composer-picker-keyboard.spec.ts e2e/smoke/composer-mobile-viewport.spec.ts --reporter=line
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py -f json -o /tmp/bandit_conversation_context_primitives.json
git diff --check
```

Expected result:

- Pytest, Vitest, browser smoke, Bandit touched-scope scan, and `git diff --check` complete successfully.
- If any suite is skipped due environment, record the concrete blocker in Backlog and the final response.

Commit after this task:

```bash
git add Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md Docs/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md Docs/Published/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md Docs/User_Guides/WebUI_Extension/Conversation_Context_Guide.md backlog/tasks/<task-file>
git commit -m "docs: document client-managed conversation context"
```

## Review Checklist Before Implementation

- [ ] Does the client own effective context composition?
- [ ] Are server responsibilities limited to composable primitives and validation?
- [ ] Does every context piece identify its scope/source?
- [ ] Can a blank chat use worldbooks and dictionaries without selecting a character?
- [ ] Can a character chat show character-inherited and explicitly attached context separately?
- [ ] Does the popover replace/evolve the existing composer character picker rather than adding a second competing picker?
- [ ] Does the client-composed preview include dictionary diagnostics?
- [ ] Does the send path use the same client composition object as preview?
- [ ] Are dictionary transforms and worldbook matching performed by server primitives, not frontend reimplementations?
- [ ] Does the UI remain usable when no provider is configured?
- [ ] Does the UI remain usable before a chat is persisted?
- [ ] Are invalid asset ids handled as user/domain errors, not server errors?
- [ ] Are quick-assign dictionary settings still compatible?
- [ ] Are browser screenshots clean at desktop and mobile widths?

## Known Risks And Mitigations

- Risk: Client composition can drift from server completion behavior if both assemble context.
  - Mitigation: Use one client composition object for preview and send; add a pass-through/bypass mode only if an existing server endpoint would double-apply context.
- Risk: The frontend could accidentally reimplement dictionary or worldbook matching.
  - Mitigation: Treat dictionary and worldbook processing as server primitives and mock those primitive outputs in frontend tests.
- Risk: Conversation settings are flexible JSON and may accumulate alias sprawl.
  - Mitigation: Write one canonical nested shape from new UI and maintain only the existing dictionary compatibility mirror.
- Risk: Dictionary transforms could surprise users if persisted message text changes.
  - Mitigation: Persist original text; use transformed text only for effective provider prompt and diagnostics.
- Risk: Workspace and global inheritance may not have existing helper APIs.
  - Mitigation: Do not fake inherited rows. Implement request, explicit chat, and character-inherited scope first, then add workspace/global follow-ups when real sources are available.
- Risk: The popover could become too dense for first-time users.
  - Mitigation: Default to a compact summary with optional diagnostics. Keep blank chat fast and skippable.
