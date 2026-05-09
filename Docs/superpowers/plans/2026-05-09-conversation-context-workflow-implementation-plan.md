# Conversation Context Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Conversation Context workflow so blank chats, character chats, workspace chats, worldbooks, and chat dictionaries share one inspectable effective-context model in the chat composer.

**Architecture:** The backend owns effective context resolution, prompt preview, dictionary transforms, worldbook matching, diagnostics, and scope precedence. The frontend renders that backend contract in a composer popover that replaces or evolves the existing character picker; it does not reimplement matching or precedence.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes/CharactersRAGDB, `WorldBookService`, `ChatDictionaryService`, Next.js/React, Ant Design popover controls, Tailwind utility classes, Vitest, Playwright/Chromium, pytest, Bandit.

---

## Source Documents

- Design spec: `Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md`
- UX audit: `Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md`
- Backlog planning task: `TASK-186`
- Prior spec task: `TASK-185`

## Scope

This plan covers the first implementation tranche needed to make Conversation Context real at the conversation boundary:

- A backend effective-context preview endpoint for any chat conversation.
- Prompt-preview and send-time parity for worldbook and dictionary diagnostics.
- Chat-scoped context settings that keep worldbooks and dictionaries conversation-scoped, not character-exclusive.
- Frontend client types and a hook for context preview.
- A composer popover that replaces or evolves the existing `CharacterSelect` placement in `ControlRow`.
- Unit, API, frontend, and browser validation.

This plan intentionally defers broad asset-management work that is adjacent but not required for the first working conversation-boundary workflow:

- Full bulk assignment redesign for dictionaries and worldbooks.
- Full Workspace Playground context-management redesign.
- New DB tables for first-class context attachments. Use existing conversation settings for this tranche unless implementation proves settings cannot support the behavior safely.
- A full rewrite of character prompt assembly. Extract only what is needed to remove preview/send drift.

## Current Code Map

Backend files to inspect before editing:

- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - Existing prompt preview: `POST /api/v1/chats/{chat_id}/prompt-preview`
  - Existing settings endpoints: `GET/PUT /api/v1/chats/{chat_id}/settings`
  - Existing send path: `POST /api/v1/chats/{chat_id}/complete-v2`
  - Current worldbook preview and send-time diagnostics are inline and character-derived.
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - `CharacterChatCompletionPrepRequest`
  - `CharacterChatCompletionPrepResponse`
  - `ChatSettingsUpdate`
  - `ChatSettingsResponse`
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
  - `WorldBookService.process_context(text, world_book_ids=None, character_id=None, include_diagnostics=True, ...)`
- `tldw_Server_API/app/core/Character_Chat/chat_dictionary.py`
  - `ChatDictionaryService.process_text(text, dictionary_id=None, return_stats=True, chat_id=...)`
  - Existing recursive settings scanner for dictionary ids.
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py`
  - Existing prompt-preview integration coverage.
- `tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py`
  - Existing dictionary endpoint behavior.

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
- `apps/packages/ui/src/services/tldw/server-capabilities.ts`
  - Optional capability flag surface.
- `apps/packages/ui/src/components/Option/Dictionaries/components/useDictionaryQuickAssign.ts`
  - Existing dictionary assignment writes top-level `chat_dictionary_ids`; new code must read/write compatibly.

## Data Contract

Add a conversation-scoped preview endpoint:

`POST /api/v1/chats/{chat_id}/context-preview`

Request schema:

```python
class ConversationContextPreviewRequest(CharacterChatCompletionPrepRequest):
    world_book_ids: Optional[list[int]] = None
    chat_dictionary_ids: Optional[list[int]] = None
    include_inherited_context: bool = True
    include_diagnostics: bool = True
```

Response schema:

```python
class ConversationContextAsset(BaseModel):
    kind: Literal["character", "worldbook", "dictionary", "workspace", "provider"]
    id: Optional[str | int] = None
    name: Optional[str] = None
    scope: Literal["explicit_chat", "workspace", "character_start", "global", "request", "none"]
    status: Literal["active", "matched", "available", "disabled", "missing", "skipped", "blocked"]
    source: Optional[str] = None
    tokens_estimated: Optional[int] = None
    matched_count: Optional[int] = None
    warnings: Optional[list[str]] = None

class ConversationContextPreviewResponse(BaseModel):
    chat_id: str
    character_id: Optional[int] = None
    character_name: Optional[str] = None
    readiness: Literal["ready", "partial", "blocked"]
    assets: list[ConversationContextAsset]
    sections: list[dict[str, Any]]
    dictionary: Optional[dict[str, Any]] = None
    worldbook: Optional[dict[str, Any]] = None
    warnings: Optional[list[str]] = None
    conflicts: Optional[list[dict[str, str]]] = None
    prompt_preview: dict[str, Any]
```

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
- Read `conversationContext.world_book_ids` plus request overrides for explicit chat worldbooks.
- Do not attach worldbooks to a character just because a chat selected them.
- Character-attached worldbooks remain inherited context when a character is active.

## Effective Context Rules

- Precedence should be visible in the response and UI:
  - request override
  - explicit chat settings
  - workspace inherited settings, if available
  - character-start or character-attached context
  - global defaults
- The first implementation may only support request, explicit chat, and character-inherited context if workspace/global context helpers are not already available. If so, return placeholder asset rows with `status="available"` or `status="skipped"` only when the source is actually known.
- Dictionary transforms must be backend-side. The frontend must not mutate prompt text.
- For send-time behavior, persist the original user message but send the effective transformed user text to the provider. Store dictionary diagnostics in assistant metadata, matching the existing lorebook diagnostics pattern.
- Dictionary processing should run before worldbook matching for the candidate user turn so terms such as `EV` can normalize to lorebook keywords such as `Echo Vault`. Document this order in response diagnostics.
- Preview and `complete-v2` must call the same resolver/helper for dictionary and worldbook resolution.

---

## Task 0: Start The Implementation Task

**Goal:** Create the implementation Backlog task and open a fresh implementation plan execution trail before code edits.

**Success Criteria:** Backlog has a task linked to this plan and the approved design spec.

**Tests:** N/A.

**Status:** Not Started.

- [ ] Create a new Backlog task, for example `Implement conversation context composer workflow`.
- [ ] Add docs:
  - `Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md`
  - `Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md`
  - `Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md`
- [ ] Set the task to `In Progress`.
- [ ] Record that this plan requires subagent/executing-plan workflow per the header. If the runtime policy prevents spawning subagents, record that implementation will use local execution checkpoints instead.

## Task 1: Backend Effective Context Contract

**Goal:** Add schemas and a resolver helper that computes effective conversation context for any chat, independent of whether a character is present.

**Success Criteria:** The backend can return a context preview for blank chats and character chats, including explicit chat worldbooks, character-inherited worldbooks, and chat dictionaries.

**Tests:** New focused pytest coverage should fail before implementation and pass after.

**Status:** Not Started.

Files:

- Add `tldw_Server_API/app/core/Character_Chat/conversation_context.py`
- Edit `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Edit `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Add `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py`

Steps:

- [ ] Add schema classes:
  - `ConversationContextPreviewRequest`
  - `ConversationContextAsset`
  - `ConversationContextPreviewResponse`
- [ ] Add helper functions in `conversation_context.py`:
  - `extract_context_settings(settings: dict[str, Any]) -> dict[str, list[int]]`
  - `merge_context_ids(request_ids, settings_ids, inherited_ids) -> list[ContextSource]`
  - `resolve_conversation_context_preview(...) -> ConversationContextPreviewResponse`
- [ ] Keep the helper dependency-light: pass in `db`, `chat_id`, `conversation`, `settings`, `turn_context`, and draft text rather than importing endpoint globals where avoidable.
- [ ] Reuse `WorldBookService.process_context(..., world_book_ids=..., character_id=..., include_diagnostics=True)` for worldbook matching.
- [ ] Reuse `ChatDictionaryService.process_text(..., return_stats=True, chat_id=chat_id)` for dictionary diagnostics.
- [ ] Implement `POST /api/v1/chats/{chat_id}/context-preview`.
- [ ] Make invalid worldbook or dictionary ids return `400` or `404` with a domain message, not a 500.
- [ ] Ensure blank chats without character ids return `readiness="ready"` unless provider/model state blocks send.

TDD tests to add first:

```python
def test_context_preview_blank_chat_uses_explicit_worldbook_and_dictionary(...):
    ...

def test_context_preview_character_chat_reports_character_inherited_worldbook(...):
    ...

def test_context_preview_dictionary_transform_runs_before_worldbook_match(...):
    ...

def test_context_preview_invalid_context_asset_returns_400_or_404(...):
    ...
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py -v
```

Expected result:

- New tests fail before code.
- New tests pass after implementation.
- No uncaught SQLite integrity or foreign-key errors appear in the response body.

Commit after this task:

```bash
git add tldw_Server_API/app/core/Character_Chat/conversation_context.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py backlog/tasks/<task-file>
git commit -m "feat: add conversation context preview contract"
```

## Task 2: Prompt Preview And Send-Time Parity

**Goal:** Make existing prompt preview and `complete-v2` use the same dictionary/worldbook resolver so preview, send, and diagnostics do not drift.

**Success Criteria:** `prompt-preview`, `context-preview`, and `complete-v2` agree about dictionary transforms, worldbook matches, section order, and diagnostics.

**Tests:** Extend existing prompt-preview and completion tests.

**Status:** Not Started.

Files:

- Edit `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Edit `tldw_Server_API/app/core/Character_Chat/conversation_context.py`
- Edit `tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py`
- Add or extend `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py`

Steps:

- [ ] Replace inline prompt-preview worldbook processing with a call to the new resolver.
- [ ] Keep the old `sections` response shape for `prompt-preview` so existing clients do not break.
- [ ] Add a `dictionary` section or diagnostics block to prompt preview.
- [ ] Update `complete-v2` so the effective user turn uses dictionary-transformed text for provider messages while persisted chat history keeps the original user text.
- [ ] Store send-time dictionary diagnostics in assistant metadata using a sibling field to existing `lorebook_diagnostics`, for example `dictionary_diagnostics`.
- [ ] Ensure lorebook diagnostics still store exactly as before.
- [ ] Add warnings when dictionary transforms or worldbook matching are skipped due missing assets, disabled assets, or empty draft text.

Regression tests:

```python
def test_prompt_preview_includes_dictionary_diagnostics_for_chat_settings(...):
    ...

def test_complete_v2_stores_dictionary_diagnostics_metadata(...):
    ...

def test_prompt_preview_and_context_preview_report_same_lorebook_matches(...):
    ...
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py -k "context_preview or prompt_preview or dictionary_diagnostics or lorebook" -v
```

Expected result:

- Prompt preview still returns existing sections.
- New dictionary diagnostics are present when active dictionaries are configured.
- Existing lorebook prompt-preview tests continue to pass.

Commit after this task:

```bash
git add tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/core/Character_Chat/conversation_context.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py backlog/tasks/<task-file>
git commit -m "feat: unify conversation context diagnostics"
```

## Task 3: Frontend API Types And Hook

**Goal:** Add typed frontend access to the backend Conversation Context contract.

**Success Criteria:** Frontend code can fetch context preview, update canonical context settings, and render stable preview states without using ad hoc payload shapes.

**Tests:** Vitest unit tests for client method, hook state, and utility normalization.

**Status:** Not Started.

Files:

- Add `apps/packages/ui/src/types/conversation-context.ts`
- Add `apps/packages/ui/src/hooks/chat/useConversationContextPreview.ts`
- Edit `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Edit `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Edit `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Optionally edit `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Add `apps/packages/ui/src/hooks/chat/__tests__/useConversationContextPreview.test.tsx`
- Add or extend `apps/packages/ui/src/services/tldw/__tests__/api-client.chat-rag.test.ts`

Steps:

- [ ] Define `ConversationContextPreviewRequest`, `ConversationContextPreviewResponse`, `ConversationContextAsset`, and related literals.
- [ ] Add `getConversationContextPreview(chatId, payload, options?)`.
- [ ] Add `updateConversationContextSettings(chatId, patch, options?)` as a small wrapper around `updateChatSettings`.
- [ ] Ensure the settings wrapper writes:
  - `conversationContext.world_book_ids`
  - `conversationContext.chat_dictionary_ids`
  - top-level `chat_dictionary_ids` compatibility mirror
- [ ] Add endpoint literal to `openapi-guard.ts`.
- [ ] Add hook states:
  - `idle`
  - `loading`
  - `ready`
  - `partial`
  - `blocked`
  - `error`
- [ ] Debounce preview refreshes from draft message changes so typing in the composer does not issue a request per keystroke.
- [ ] Keep preview fetch optional when no persisted chat id exists; show local empty state until the conversation exists.

Vitest examples:

```tsx
it("writes canonical conversationContext settings with dictionary compatibility mirror", async () => {
  ...
})

it("does not fetch context preview until a chat id exists", async () => {
  ...
})
```

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useConversationContextPreview.test.tsx ../packages/ui/src/services/tldw/__tests__/api-client.chat-rag.test.ts --config vitest.config.ts
```

Expected result:

- Hook tests pass.
- Client tests show the endpoint path `/api/v1/chats/{chat_id}/context-preview`.
- Settings update test includes the nested canonical context and top-level dictionary compatibility mirror.

Commit after this task:

```bash
git add apps/packages/ui/src/types/conversation-context.ts apps/packages/ui/src/hooks/chat/useConversationContextPreview.ts apps/packages/ui/src/hooks/chat/__tests__/useConversationContextPreview.test.tsx apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/services/tldw/__tests__/api-client.chat-rag.test.ts backlog/tasks/<task-file>
git commit -m "feat: add conversation context frontend client"
```

## Task 4: Composer Popover Replacing The Character Picker

**Goal:** Replace or evolve the existing chat-composer character picker into a Conversation Context popover, with character selection as one slot inside broader context.

**Success Criteria:** The composer has one visible context control where users can see character, worldbooks, dictionaries, and diagnostics. The old character-selection behavior remains available inside that control.

**Tests:** Component tests for rendering, keyboard flow, selection callbacks, and backend preview states.

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
  - Worldbook slot showing active/matched/skipped counts.
  - Dictionary slot showing active/replaced/skipped counts.
  - Readiness row: ready, partial, blocked.
- Power-user details:
  - Diagnostics tab or collapsible region for matched worldbook entries, dictionary replacements, token estimates, and warnings.
  - Scope/source labels: explicit chat, workspace, character-start, global, request.
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
  - `scope`
  - `disabled`
  - `className`
  - `iconClassName`
- [ ] Add utility functions:
  - `summarizeConversationContextAssets`
  - `formatContextSourceLabel`
  - `resolveContextReadiness`
- [ ] Replace the direct `CharacterSelect` call in `ControlRow.tsx` with `ConversationContextPopover`.
- [ ] Pass `chatId`, draft message, scope, and character props from `form.tsx` into `ControlRow`.
- [ ] Preserve mobile hit target sizes and current composer control wrapping.
- [ ] Ensure next-gen composer slots receive the same `composerControlAreaNode`; do not create separate next-gen context UI.

Tests:

```tsx
it("renders character selection as a slot inside conversation context", () => {
  ...
})

it("shows dictionary and worldbook diagnostics from backend preview", async () => {
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
- No test depends on frontend-side dictionary/worldbook matching.

Commit after this task:

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx apps/packages/ui/src/components/Sidepanel/Chat/conversation-context-utils.ts apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx backlog/tasks/<task-file>
git commit -m "feat: add composer conversation context popover"
```

## Task 5: Minimal Asset Selection For Worldbooks And Dictionaries

**Goal:** Let users attach or remove worldbooks and dictionaries from a conversation through the new popover without making them character-card exclusive.

**Success Criteria:** A blank chat can carry selected worldbooks and dictionaries; a character chat can carry both explicit chat assets and character-inherited assets with clear labels.

**Tests:** Frontend unit tests plus backend settings tests.

**Status:** Not Started.

Files:

- Edit `apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx`
- Edit `apps/packages/ui/src/hooks/chat/useConversationContextPreview.ts`
- Edit `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Add or extend backend tests in `tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py`
- Add or extend frontend tests in `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx`

Steps:

- [ ] Add lightweight selectors inside the popover using existing list APIs:
  - worldbooks list/search endpoint already exposed through `tldwClient`
  - dictionaries list/search endpoint already exposed through `tldwClient`
- [ ] Use checkboxes or multi-select controls for worldbooks and dictionaries.
- [ ] Save to conversation settings via `updateConversationContextSettings`.
- [ ] On save, refetch context preview and show backend-computed status.
- [ ] If the selected context is inherited and not explicitly attached, removal should disable or hide the explicit chat override only if the backend supports that distinction. Otherwise label it as inherited and do not offer destructive removal.
- [ ] Keep dictionary quick assign compatibility by preserving top-level `chat_dictionary_ids` while the older assignment UI exists.

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
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx --config vitest.config.ts
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py -v
```

Expected result:

- Blank chat context assets persist in chat settings.
- Character-inherited assets are visible but not mislabeled as character-owned dictionaries.
- Dictionary Quick Assign still recognizes dictionary usage through the compatibility key.

Commit after this task:

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx apps/packages/ui/src/hooks/chat/useConversationContextPreview.ts apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py backlog/tasks/<task-file>
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
- Context preview matches backend API output.
- No visible overlap or layout jump in desktop or mobile screenshots.

Commit after this task:

```bash
git add apps/tldw-frontend/e2e/smoke/composer-picker-keyboard.spec.ts apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts apps/tldw-frontend/e2e/smoke/playground-nextgen-composer.spec.ts Docs/Reviews/CONVERSATION_CONTEXT_WORKFLOW_VALIDATION_2026_05_09.md Docs/Reviews/assets/2026-05-09-conversation-context-workflow backlog/tasks/<task-file>
git commit -m "test: validate conversation context workflow"
```

## Task 7: Documentation And Closeout

**Goal:** Update user-facing and API-facing docs enough that future maintainers understand the new conversation-scoped model.

**Success Criteria:** Docs describe Conversation Context as conversation-scoped, prompt-preview parity, dictionary/worldbook order, and composer placement.

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

- [ ] Document `POST /api/v1/chats/{chat_id}/context-preview`.
- [ ] Document dictionary transform order relative to worldbook matching.
- [ ] Document that worldbooks and dictionaries can be attached to conversations without a character.
- [ ] Document that character-attached worldbooks are inherited context, not the only worldbook source.
- [ ] Include short troubleshooting notes for missing provider, missing asset ids, no matches, and empty draft text.
- [ ] Record verification commands and results in the Backlog task.

Final verification commands:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_preview.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py -k "context_preview or prompt_preview or dictionary_diagnostics or lorebook" -v
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useConversationContextPreview.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx --config vitest.config.ts
cd apps/tldw-frontend && npx playwright test e2e/smoke/composer-picker-keyboard.spec.ts e2e/smoke/composer-mobile-viewport.spec.ts --reporter=line
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/app/core/Character_Chat/conversation_context.py -f json -o /tmp/bandit_conversation_context.json
git diff --check
```

Expected result:

- Pytest, Vitest, browser smoke, Bandit touched-scope scan, and `git diff --check` complete successfully.
- If any suite is skipped due environment, record the concrete blocker in Backlog and the final response.

Commit after this task:

```bash
git add Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md Docs/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md Docs/Published/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md Docs/User_Guides/WebUI_Extension/Conversation_Context_Guide.md backlog/tasks/<task-file>
git commit -m "docs: document conversation context workflow"
```

## Review Checklist Before Implementation

- [ ] Does every context asset row identify its scope/source?
- [ ] Can a blank chat use worldbooks and dictionaries without selecting a character?
- [ ] Can a character chat show character-inherited and explicitly attached context separately?
- [ ] Does the popover replace/evolve the existing composer character picker rather than adding a second competing picker?
- [ ] Does prompt preview include dictionary diagnostics?
- [ ] Does send-time generation use the same dictionary/worldbook order as preview?
- [ ] Are dictionary transforms performed server-side only?
- [ ] Does the UI remain usable when no provider is configured?
- [ ] Does the UI remain usable before a chat is persisted?
- [ ] Are invalid asset ids handled as user/domain errors, not server errors?
- [ ] Are quick-assign dictionary settings still compatible?
- [ ] Are browser screenshots clean at desktop and mobile widths?

## Known Risks And Mitigations

- Risk: `character_chat_sessions.py` is already large, and deeper prompt assembly extraction could cause regressions.
  - Mitigation: Extract only dictionary/worldbook context resolution in this tranche; keep existing prompt-preview response shape.
- Risk: Conversation settings are flexible JSON and may accumulate alias sprawl.
  - Mitigation: Write one canonical nested shape from new UI and maintain only the existing dictionary compatibility mirror.
- Risk: Dictionary transforms could surprise users if persisted message text changes.
  - Mitigation: Persist original text; use transformed text only for effective provider prompt and diagnostics.
- Risk: Frontend could drift by synthesizing context from separate endpoints.
  - Mitigation: Render backend `context-preview` as the source of truth and use local formatting only.
- Risk: Workspace and global inheritance may not have existing helper APIs.
  - Mitigation: Do not fake inherited rows. Implement request, explicit chat, and character-inherited scope first, then add workspace/global follow-ups when real sources are available.
- Risk: The popover could become too dense for first-time users.
  - Mitigation: Default to a compact summary with optional diagnostics. Keep blank chat fast and skippable.
