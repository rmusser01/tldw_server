# Main /chat Cockpit Rail Completion Design

## Scope

This spec defines the remaining staged work needed to make the main WebUI `/chat` page a fully mature cockpit in PR #1582. It is limited to the main `/chat` Playground surface. It excludes the browser-extension sidepanel/sidebar, settings pages, onboarding, character library pages, backend API redesign, MCP Hub management, evaluations, media ingestion, and repo-wide architecture unless a change is directly required for the main `/chat` cockpit experience.

The merge bar is not a first-slice shell. The cockpit must let users configure the current conversation from the main chat window rails: context, prompts, MCP, model/chat settings, character/persona, and turn/runtime state. Focus mode must still collapse the interface back to a chat-first layout.

## Current State Evidence

- `PlaygroundContextRail.tsx` exposes context state, web search, Search & Context entry, active source inventory, clear/remove actions, and temporary chat state.
- `PlaygroundRuntimeInspector.tsx` exposes runtime status, provider/model route, scoped setting summaries, model settings entry, character entry, MCP tools entry, stop/regenerate actions, and message count.
- `Playground.tsx` derives rail state from existing chat state including web search, context files, selected knowledge, media scopes, research context, selected model, selected character, temporary chat, history, and degraded health.
- `PlaygroundForm.tsx` still owns the real deep controls: prompt selector, character/persona selector, MCP popover/settings modal, model/chat settings modal, tools menu, and advanced controls.
- `chat-cockpit.real-server.spec.ts` proves the rails and dialogs are reachable against the real server, but it does not yet prove every rail path completes a configure/save/use workflow.

## Design Principles

1. Main `/chat` only. Do not make sidepanel/sidebar changes for this work.
2. Rails must call existing shared state paths. Do not create parallel cockpit-only state.
3. Durable conversation configuration belongs in the rails; the composer remains optimized for drafting and turn-level actions.
4. Dialogs and popovers are acceptable for deeper editing, but the rail action should open the correct editing surface directly.
5. Real-server tests must use the real running server for merge-critical browser evidence. Do not use mocked server data or `page.route` for that path.

## Stage 1: Rail Information Architecture Cleanup

Goal: make the cockpit rails read as control surfaces rather than status cards.

Left rail sections:

- Context
- Prompts
- Session

Right rail sections:

- Runtime
- Model & Chat
- MCP Tools
- Character / Persona
- Run Controls

Changes:

- Rename the runtime action from "Model settings" to "Model & chat settings".
- Split "Character / Persona" from "Scene Director". Scene Director remains a secondary character-backed scene-context control, not the primary assistant selector.
- Keep focus mode and rail visibility controls as layout affordances, not configuration controls.

Acceptance criteria:

- A first-time user can identify where to configure prompts, context, model/chat settings, MCP, and character/persona without knowing composer internals.
- Returning users can collapse either rail or enter focus mode without losing configured state.

## Stage 2: Prompt Management In The Left Rail

Goal: make prompt selection and prompt context visible and configurable from the same rail that manages other context inputs.

Controls:

- Show current selected system prompt or "No prompt selected".
- Open prompt selector from the left rail.
- Clear selected prompt when a prompt is active.
- Show whether an inline system prompt/custom prompt is contributing context when that state is already available.
- Preserve the existing composer prompt selector and shared prompt state.

Behavior:

- Prompt selection from the rail must update the same `selectedSystemPrompt`, `selectedQuickPrompt`, or prompt state used by the composer.
- Clearing a prompt from the rail must not clear unrelated file, knowledge, media, web, or research context.
- If prompt data is unavailable, show a recoverable disabled state with a reason and keep the composer usable.

Acceptance criteria:

- User can select and clear the active prompt from the left rail.
- The context/status surfaces reflect prompt state before the next send.
- Existing composer prompt workflows continue to work.

## Stage 3: Direct MCP Workflow

Goal: make MCP configurable from the runtime rail without forcing users through an indirect composer popover.

Controls:

- Show MCP health, tool choice, enabled tool count, disabled count, catalog/module filter summary, and strict catalog state.
- Add direct runtime-rail actions:
  - Tool choice: Auto / Required / None.
  - Configure MCP: opens `PlaygroundMcpSettingsModal` directly.
- Keep the existing composer MCP control available.

Behavior:

- Tool choice changes from the rail update the same state used by request construction.
- Configure MCP opens the full settings modal directly, not only the small MCP popover.
- Unavailable or unhealthy MCP states show a reason and do not present enabled controls that cannot work.

Acceptance criteria:

- User can change MCP tool choice from the rail.
- User can open catalog/module/per-tool settings directly from the rail.
- Available and unavailable MCP states are both covered by tests.

## Stage 4: Model & Chat Settings Workflow

Goal: make active model/chat settings clear, scoped, and editable from the runtime rail.

Controls:

- Show provider, model, provider:model route, temperature, context window, max tokens, reasoning effort, and relevant capability hints when available.
- Rename the entry point to "Model & chat settings".
- Open the existing `CurrentChatModelSettings` modal.
- Make the active provider:model scope visible near the action.

Behavior:

- The settings modal continues to own detailed model, conversation, advanced, and scoped provider:model settings.
- The rail summary updates when model/chat settings change.
- Provider:model scoped settings remain isolated across providers and duplicate model IDs.

Acceptance criteria:

- User can understand the active runtime and open the correct editor from the rail.
- Tests prove changed settings remain scoped to the active provider:model.
- Existing composer model selection remains reachable and functional.

## Stage 5: First-Class Character / Persona Flow

Goal: make character/persona selection and inspection a first-class cockpit workflow, not a Scene Director shortcut.

Controls:

- Show current assistant mode: none, character, or persona.
- Show current character/persona name and status.
- Select character/persona from the rail workflow.
- Clear active character/persona when allowed.
- Open relevant character/persona details or settings from the rail workflow.
- Show persona memory mode when persona chat is active.
- Keep Scene Director as a separate action for character-backed chats only.

Behavior:

- The rail must not route primary character/persona configuration through `ActorPopout`.
- Scene Director stays available for character-backed scene context and remains disabled/explanatory for persona chats.
- Character/persona selection updates the same current chat state used by the composer and send pipeline.

Acceptance criteria:

- User can select, inspect, and clear character/persona state from the cockpit rail flow.
- Persona chats do not show Actor as the primary persona settings route.
- Tests cover none, character, and persona states.

## Stage 6: Context Rail Tightening

Goal: ensure the left rail fully answers what will affect the next reply.

Controls:

- Preserve web search toggle, Search & Context launcher, clear actions, and per-source removal.
- Include prompt context from Stage 2.
- Strengthen summaries for file, knowledge, media, research, web, prompt, and system-context contributors where existing state is available.
- Keep one clear empty-state action.

Acceptance criteria:

- User can see all active non-message inputs that affect the next reply.
- User can remove each supported context class from the rail without affecting unrelated context.
- Empty, degraded, and disabled states are explicit and recoverable.

## Stage 7: Real-Server And Component Coverage

Goal: raise verification from reachability to completed configuration workflows.

Component and integration coverage:

- Runtime rail opens direct MCP settings and changes tool choice through shared state.
- Prompt rail selection/clear updates shared prompt state.
- Model & chat settings label, scope summary, and provider:model isolation render correctly.
- Character/persona cockpit flow distinguishes none, character, persona, and Scene Director states.
- Context rail clear/remove actions remain isolated by context type.

Real-server Playwright coverage:

- Open `/chat` against the real running server.
- Use the left rail to select or clear a prompt where real prompt data exists, or assert the recoverable empty state.
- Use the context rail to toggle web search and verify cockpit/status state updates.
- Use the runtime rail to open model/chat settings and persist a harmless setting.
- Use the runtime rail to configure MCP if available, or assert the real unavailable/degraded reason.
- Use the character/persona rail flow with real server state where possible, or assert recoverable empty state.
- Send a real chat turn and verify the cockpit status reflects configured model/context state and either a real assistant response or a recoverable provider/server error.

Acceptance criteria:

- Merge-critical browser evidence uses the real server and no mocked server payloads.
- Tests prove at least one completed state-changing rail workflow per control family: prompts, context, MCP, model/chat, and character/persona.

## Implementation Notes

Likely files:

- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- `apps/packages/ui/src/components/Option/Playground/playground-cockpit-actions.ts`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx`
- `apps/packages/ui/src/components/Common/Settings/CurrentChatModelSettings.tsx`
- Existing character/persona selector components used by the main composer.
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

Do not introduce a new backend contract unless an existing frontend state path cannot support the rail workflow. Prefer exposing shared callbacks or extracting small shared control components from existing composer controls.

## Out Of Scope

- Browser-extension sidepanel/sidebar UI.
- Character library redesign.
- MCP Hub management redesign.
- New provider setup flows.
- Drag-resizable/dockable panel systems.
- Replacing all composer controls.
- Mocked real-server proof.

## Open Questions

- If the real server has no prompt or character/persona data, should the real-server E2E seed minimal data through existing APIs, or should it assert recoverable empty states and leave populated coverage to component/integration tests?
- Should the prompt rail control manage only selected prompt templates first, or also inline/system prompt editing in the first completion pass?
- Which existing character/persona selector component is the safest to reuse in the cockpit rail workflow without importing sidepanel-specific behavior?
