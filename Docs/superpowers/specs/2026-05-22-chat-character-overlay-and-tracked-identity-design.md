# Chat Character Overlay And Tracked Identity Design

**Date:** 2026-05-22
**Surface:** WebUI `/chat` and extension sidepanel chat flows
**Status:** Reconciled on 2026-05-30
**Backlog:** TASK-444

---

## 2026-05-30 Reconciliation

This spec's data-model decision remains current: tracked character/persona identity and normal-chat assistant overlay must stay separate.

The original UI direction is superseded. Do not add or reintroduce a standalone `CharacterControlRail`. The current `dev` implementation intentionally routes character/persona controls through the existing `/chat` cockpit rails:

- the runtime rail shows assistant state, selection, clear, Scene Director access, and tracked-start behavior;
- the context rail shows assistant/context sources and clear affordances;
- mobile uses the existing cockpit rail tabs rather than a separate character sheet;
- regression coverage asserts `CharacterControlRail` remains absent from `/chat`.

The implementation plan linked from this task is retained as historical context for the state split and overlay contract, but its standalone rail stage is not executable.

## Goal

Preserve existing tracked character/persona chats while adding a separate, non-destructive way to apply character/persona personality to a normal conversation.

The end state is one chat surface with a clearer contract:

- tracked character/persona chats still behave as tracked chats;
- normal chats can borrow character/persona personality without becoming tracked chats;
- assistant switching in a normal chat does not reset the thread;
- the main character/persona control surface lives in the existing cockpit runtime/context rails, not in starter cards, not in a drawer-first setup flow, not primarily in the composer, and not in a standalone character rail.

## Product Decision

The current code collapses two different concepts into one `selectedAssistant` path:

1. conversation ownership and storage;
2. assistant personality and presentation.

That is the root design error.

This design splits them into separate concepts:

- **tracked identity**: a conversation is owned by a character or persona and should keep current character/persona chat semantics;
- **assistant overlay**: a conversation remains a normal chat, but assistant replies are steered to sound like a selected character or persona.

Those concepts must use different persistence and send paths.

## Required Outcomes

The implementation must support both of these behaviors at the same time:

1. Existing character/persona chats continue to be tracked to the selected character/persona exactly as they are today.
2. A user can add, change, or clear a character/persona on an existing non-character/persona conversation and use it only for assistant personality, without resetting or reclassifying the chat.

## Original Problem

At original authoring time, `/chat` conflated tracked identity with UI selection and send-time routing:

- assistant selection is stored globally in [useSelectedAssistant.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useSelectedAssistant.ts:59);
- tracked chat hydration writes back into that same global selection in [useSelectServerChat.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useSelectServerChat.ts:151);
- assistant changes can clear thread state in [useMessageOption.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessageOption.tsx:265);
- send routing in [useMessage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessage.tsx:2452) treats character/persona as chat-path changes rather than a product-level distinction between tracked ownership and overlay behavior;
- normal-chat prompt assembly in [normalChatMode.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:365) does not use selected assistant identity for personality steering;
- the backend tracked chat path in [character_chat_sessions.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4628) correctly uses conversation identity, but that same model should not be forced onto plain chats.

The result is destructive switching and unclear semantics.

## Implementation Anchors

The design should stay grounded in the current code:

- `/chat` shell and transcript layout:
  - [Playground.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/Playground.tsx:1705)
  - [PlaygroundForm.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx:4962)
- existing chat sidebar:
  - [ChatSidebar.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar.tsx:400)
- existing assistant picker:
  - [AssistantSelect.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/AssistantSelect.tsx:101)
- existing normal-chat prompt assembly:
  - [normalChatMode.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:117)
- existing tracked send routing:
  - [useMessage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessage.tsx:2452)
- existing chat settings pipeline:
  - [chat-session-settings.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/types/chat-session-settings.ts:45)
  - [chat-settings.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/chat-settings.ts:31)
  - [useChatSettingsRecord.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useChatSettingsRecord.ts:14)
- tracked chat backend contract:
  - [chat_session_schemas.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:42)
  - [character_chat_sessions.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4628)

## Non-Goals

- No new starter cards.
- No dedicated Character Chat route.
- No separate Character Chat workspace that replaces the existing `/chat` shell.
- No removal of the existing composer, toolbar, sidebar, knowledge panel, or artifacts panel.
- No forced conversion of normal chats into tracked chats.
- No regression to current tracked character/persona chat storage.
- No broad redesign of chat history navigation.

## Terminology

Use these terms consistently:

| Term | Meaning |
| --- | --- |
| Tracked identity | Conversation ownership stored on the chat record via `character_id` / `assistant_kind` / `assistant_id`. |
| Assistant overlay | A per-conversation personality layer stored in chat settings that does not change conversation ownership. |
| Plain chat | A conversation with neither tracked identity nor overlay. |
| Tracked character chat | A conversation owned by a character. |
| Tracked persona chat | A conversation owned by a persona. |
| Runtime assistant controls | The existing cockpit runtime/context rail controls for assistant identity, selection, clear/change actions, Scene Director access, context visibility, and tracked session actions. |

Do not describe overlay state as "the chat is now a character chat." It is still a normal chat.

## State Model

The current conversation has one effective assistant mode:

- `plain`
- `overlay`
- `tracked_character`
- `tracked_persona`

These modes are mutually exclusive.

Tracked identity and overlay cannot be active at the same time for one conversation. In v1, tracked chats do not layer overlay on top.

### Plain

- no tracked identity;
- no overlay;
- current normal chat behavior.

### Overlay

- no tracked identity;
- `assistantOverlay` exists in conversation settings;
- assistant replies use selected character/persona personality only;
- chat remains classified as a normal conversation.

### Tracked Character

- `assistant_kind == "character"` and `character_id` is set;
- current tracked character chat behavior remains canonical.

### Tracked Persona

- `assistant_kind == "persona"` and `assistant_id` is set;
- current tracked persona chat behavior remains canonical.

## Data Contract

### Tracked Identity

Tracked identity remains on the conversation record and continues to use the current backend contract:

- `character_id`
- `assistant_kind`
- `assistant_id`
- `persona_memory_mode`

This stays defined by:

- [ChatSessionCreate](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:42)
- [ChatSessionResponse](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:147)

### Assistant Overlay

Assistant overlay should live in chat settings, not in tracked identity fields.

Add to [ChatSettingsRecord](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/types/chat-session-settings.ts:45):

```ts
assistantOverlay?: {
  kind: "character" | "persona"
  id: string
  name: string
  avatar_url?: string | null
  system_prompt_snapshot?: string | null
  updatedAt: string
} | null
```

This should be:

- normalized in [chat-settings.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/chat-settings.ts:252);
- accepted by the frontend optional-keys list in [chat-settings.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/chat-settings.ts:31);
- validated in [_validate_chat_settings_payload](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:1017);
- persisted through the existing [update_chat_settings](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:5787) flow.

### Why Settings, Not Chat Identity

Overlay is not conversation ownership. If overlay writes `assistant_kind`, `assistant_id`, or `character_id`, then:

- plain chats will be reclassified as tracked chats;
- tracked-session filters will become inaccurate;
- conversation lists will not distinguish ownership from presentation;
- changing overlay will start acting like a chat replacement operation again.

That is explicitly out of scope for overlay behavior.

## Effective Assistant Resolution

Introduce one derived resolver for the current conversation:

Inputs:

- server chat metadata;
- chat settings;
- optional rail draft selection while the picker is open.

Resolution order:

1. if `assistant_kind == "character"` and `character_id` is present -> `tracked_character`
2. if `assistant_kind == "persona"` and `assistant_id` is present -> `tracked_persona`
3. if `settings.assistantOverlay` exists -> `overlay`
4. otherwise -> `plain`

The resolver should return:

- `mode`
- `kind`
- `id`
- `displayName`
- `avatarUrl`
- `systemPrompt`
- `source: "tracked" | "overlay" | "none"`

This resolver should replace ad hoc assistant checks in:

- [useMessage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessage.tsx:2452)
- [useMessageOption.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessageOption.tsx:265)
- [useSelectServerChat.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useSelectServerChat.ts:151)

## UI Model

The UI must expose the split between tracked identity and overlay directly, rather than hiding it behind one generic assistant picker.

### Runtime Assistant Controls

Use the existing `/chat` cockpit rails as the character/persona control plane.

Placement:

- desktop: inside the existing runtime and context rails rendered by the cockpit shell;
- mobile and narrow layouts: inside the existing cockpit context/runtime tabs;
- extension sidepanel: keep the route-only handoff contract and reuse the shared state model where sidepanel chat exposes assistant controls.

Do not create a separate persistent `CharacterControlRail`, a replacement character workspace, or a drawer-first setup flow for this task.

### Cockpit Sections

The existing cockpit rails should own:

1. current assistant mode and summary;
2. identity picker and clear/change actions;
3. Scene Director entry when a character-backed mode supports it;
4. context source summary and assistant context sources;
5. runtime/provider/model state;
6. tracked recent/session entry points where currently exposed.

### Existing UI Preservation

Keep:

- [ComposerToolbar.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx:110)
- [AssistantSelect.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/AssistantSelect.tsx:101)
- [ChatSidebar.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar.tsx:400)
- [PlaygroundKnowledgeSection.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/PlaygroundKnowledgeSection.tsx:31)

But change their role:

- the runtime/context cockpit rails become the main character/persona control surface;
- the composer remains usable but secondary for identity control;
- `AssistantSelect` becomes the picker the runtime rail invokes, not the workflow container.

## Runtime/Context Rail Actions

Each assistant action must map to one write path.

### Apply Overlay

Available from `plain` or existing `overlay`.

Behavior:

- write `assistantOverlay` to chat settings;
- do not mutate conversation identity;
- do not clear messages, history, `serverChatId`, or `historyId`.

### Change Overlay

- overwrite `assistantOverlay`;
- preserve the same conversation.

### Clear Overlay

- set `assistantOverlay = null`;
- preserve the same conversation.

### Start Tracked Character Chat

- explicit action;
- create or open a tracked character-backed conversation;
- tracked identity lives on the chat record.

### Start Tracked Persona Chat

- explicit action;
- create or open a tracked persona-backed conversation;
- tracked identity lives on the chat record.

### Open Tracked Session

- explicit session navigation;
- does not mutate the current conversation into a tracked chat.

## Send Routing

In [useMessage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessage.tsx:2452), route by effective mode rather than by a single assistant selection:

- `tracked_character` -> current `characterChatMode`
- `tracked_persona` -> current persona-backed `normalChatMode` flow
- `overlay` -> `normalChatMode` with injected personality steering
- `plain` -> current `normalChatMode`

This keeps tracked paths intact while isolating overlay logic to the normal-chat path.

## Overlay Prompt Semantics

Overlay is deliberately narrower than tracked character/persona chat behavior.

### Overlay Should Do

- change assistant display name and avatar;
- change assistant personality through prompt steering;
- persist per conversation;
- survive reload;
- allow change/clear without resetting the thread.

### Overlay Should Not Do

- change conversation ownership;
- use persona durable-memory mode;
- use tracked character greeting-session semantics;
- use tracked character memory rows as if the chat were character-owned;
- reclassify the conversation in tracked-character or tracked-persona session views;
- automatically attach worldbook/lorebook character bindings that assume tracked ownership.

### Overlay Prompt Source

For v1, overlay steering should reuse the selected assistant's prompt material from [AssistantSelection](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/types/assistant-selection.ts:9).

Use:

- `assistantIdentity` for name/avatar in [normalChatMode.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:122);
- `systemPromptAppendix` in [normalChatMode.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:144) and [normalChatMode.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts:488) to inject character/persona behavior instructions.

This is intentionally narrower than the tracked chat completion builder in [character_chat_sessions.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4701).

## Frontend Architecture Changes

### 1. Stop Treating Global Assistant Selection As Active Chat Truth

[useSelectedAssistant.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useSelectedAssistant.ts:59) should no longer act as the canonical active assistant for the conversation.

Keep it only for:

- picker/default memory;
- rail draft selection;
- compatibility migration while the new effective-state resolver rolls out.

Active conversation mode must come from:

- tracked chat metadata; or
- `assistantOverlay` in chat settings.

### 2. Remove Destructive Assistant Switch Resets

The assistant-switch reset block in [useMessageOption.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/useMessageOption.tsx:265) is incompatible with overlay behavior and should be replaced with logic keyed to actual conversation changes, not picker changes.

### 3. Preserve Tracked Hydration Without Reimposing Global Selection

[useSelectServerChat.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useSelectServerChat.ts:151) should continue to hydrate tracked chat metadata, but it should not make global assistant selection the active source of truth for the loaded chat.

### 4. Do Not Add Standalone Character Rail Panel State

The reconciled direction does not require a new `character-control` panel id. Keep assistant controls inside the existing cockpit runtime/context rail ownership model unless a future product decision explicitly reopens a separate panel.

## Backend Architecture Changes

### 1. Preserve Tracked Chat Contract

Do not repurpose [ChatSessionUpdate](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:124) for overlay behavior.

Tracked identity and overlay state must remain separate.

### 2. Validate Overlay Settings

Extend [_validate_chat_settings_payload](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:1017) to validate:

- `assistantOverlay.kind`
- `assistantOverlay.id`
- `assistantOverlay.name`
- optional `assistantOverlay.avatar_url`
- optional `assistantOverlay.system_prompt_snapshot`
- `assistantOverlay.updatedAt`

The settings merge path in [update_chat_settings](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:5787) should remain the persistence mechanism.

### 3. Keep Tracked Completion Path Unchanged

Tracked chats should continue using the current `complete-v2` flow in [character_chat_sessions.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4628).

No overlay feature should alter tracked-chat prompt composition.

## Saved Setups

Saved setups should follow the same split as runtime state.

### Overlay-Applied Setup

Applying a setup to the current plain chat should:

- write `assistantOverlay` when setup identity is intended as overlay;
- apply scene/style/behavior settings through chat settings;
- preserve conversation identity.

### Tracked Setup

Opening a setup as a tracked chat should:

- create or open a tracked character/persona conversation;
- preserve current tracked session semantics.

The default "apply here" behavior for normal chats should prefer overlay, not tracked conversion.

Reuse existing startup-template plumbing where possible:

- [startup-template-bundles.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/startup-template-bundles.ts:12)
- [usePromptTemplates.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts:23)

## Recent Sessions

Tracked recent sessions must remain based on tracked conversation identity only.

Overlayed plain chats should not be grouped or labeled as tracked character/persona sessions simply because they currently use overlay personality.

This keeps:

- character/persona session history accurate;
- plain chat history accurate;
- runtime/context rail actions understandable.

## Extension And Sidepanel Parity

The state model should be shared across WebUI and extension chat surfaces even if the presentation differs.

Requirements:

- tracked chat metadata keeps working in extension handoff;
- `assistantOverlay` persists through shared chat settings flows;
- desktop WebUI can use the persistent runtime/context rails;
- extension sidepanel and mobile surfaces can render the same controls through their existing compact rail/tab patterns;
- neither surface should rely on starter cards or route-level character mode.

## Risks

### Global Selection Leakage

If global selected-assistant storage remains authoritative, overlay or tracked state will leak across chats.

### Prompt Richness Mismatch

Tracked persona chat prompt assembly is richer than a simple `system_prompt` append. That is acceptable for overlay v1 as long as it is explicit that overlay is personality steering, not full tracked persona runtime.

### Existing Contract Mismatch Around Persona Memory Updates

The frontend currently attempts persona memory mode updates from [ConversationTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Settings/tabs/ConversationTab.tsx:860), while backend tracked chat update support is narrower in [character_chat_sessions.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:5667). That inconsistency should not be expanded by overlay work and should be resolved only if the tracked chat controls touched by this effort require it.

## Rollout Stages

## Stage 1: State Split And Persistence

**Goal:** Introduce explicit overlay state without changing tracked chat semantics.

**Scope:**

- add `assistantOverlay` typing and normalization;
- validate overlay in backend chat settings;
- add effective assistant-state resolver;
- stop destructive assistant-switch resets.

**Success Criteria:**

- tracked chats still load as tracked;
- plain chats can store overlay state;
- changing overlay does not clear the thread.

## Stage 2: Send Path Split

**Goal:** Make tracked and overlay chats use different behavior paths.

**Scope:**

- route sends by effective mode in `useMessage.tsx`;
- keep tracked character/persona behavior unchanged;
- inject overlay personality through `normalChatMode`.

**Success Criteria:**

- tracked chats still use current completion flows;
- overlay chats use current conversation id and history;
- overlay visibly changes assistant behavior.

## Stage 3: Cockpit Runtime/Context Rail Integration

**Goal:** Keep the main character/persona control surface in the existing cockpit rails without replacing existing chat UI.

**Scope:**

- wire runtime/context rail actions to overlay writes and tracked session actions;
- keep composer and sidebar intact.

**Success Criteria:**

- desktop `/chat` exposes persistent runtime/context rails for identity controls;
- users can apply/change/clear overlay from the rail;
- users can start or open tracked sessions from the rail.

## Stage 4: Parity And Hardening

**Goal:** Ensure reload, mobile, and extension parity.

**Scope:**

- reload restoration for tracked and overlay states;
- mobile/sidepanel presentation through existing compact cockpit controls;
- recent-session and saved-setup classification fixes.

**Success Criteria:**

- reload restores the correct mode;
- overlay does not become tracked after reload;
- extension flow follows the same state contract.

## Testing

### Unit

- `assistantOverlay` normalization and validation
- effective assistant-state resolution
- settings merge behavior for overlay updates

### Integration

- tracked character chat remains tracked
- tracked persona chat remains tracked
- plain chat with overlay keeps the same conversation id
- changing overlay mid-chat does not clear messages
- clearing overlay mid-chat does not clear messages

### UI

- runtime/context rail apply/change/clear flows
- tracked-session open flows
- saved setup apply-in-place vs tracked-open behavior
- desktop and mobile layout coverage

### Regression

- no starter-card dependency for character/persona control
- no assistant-switch thread reset in normal chats
- tracked recent-session classification unchanged

## Acceptance Criteria

- Existing tracked character chats behave exactly as they do today.
- Existing tracked persona chats behave exactly as they do today.
- A plain chat can apply a character/persona overlay and keep the same chat id, thread, and history.
- Changing overlay mid-chat does not reset the conversation.
- Clearing overlay mid-chat does not reset the conversation.
- Overlay state persists through chat settings and reload.
- Overlay chats are not reclassified as tracked character/persona sessions.
- The main character/persona control surface is the existing runtime/context cockpit rails, while the rest of the `/chat` UI remains in place.
