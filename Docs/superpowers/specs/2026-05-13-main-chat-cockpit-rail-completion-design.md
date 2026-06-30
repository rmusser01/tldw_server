# Main /chat Cockpit Rail Completion Design

## Scope

This spec defines the remaining staged work needed to make the main WebUI `/chat` page a fully mature cockpit in PR #1582. It is limited to the main `/chat` Playground surface. It excludes the browser-extension sidepanel/sidebar, settings pages, onboarding, character library pages, backend API redesign, MCP Hub management, evaluations, media ingestion, and repo-wide architecture unless a change is directly required for the main `/chat` cockpit experience.

The merge bar is not a first-slice shell. The cockpit must let users configure the current conversation from the main chat window rails: context, prompts, MCP, model/chat settings, character/persona, and turn/runtime state. Focus mode must still collapse the interface back to a chat-first layout.

## Merge Definition And First-Slice Boundary

This document is the first implementation slice inside PR #1582, but the PR should stay draft until the main `/chat` cockpit is mature enough to merge as a complete experience. Individual stages can be committed and reviewed in sequence, but the merge bar is the complete rail flow: prompt selection, context management, MCP exposure, model/chat settings, character/persona configuration, run controls, degraded warnings, accessibility, responsive behavior, and real-server proof all work from the main chat window.

The first slice must not become a partial replacement for the existing chat page. Any existing composer workflow that is still required for chat must remain present and working until the cockpit rail equivalent is verified. Rail controls should improve discoverability and efficiency without breaking the current chat/roleplay path or focus-mode path.

## Current State Evidence

- `PlaygroundContextRail.tsx` exposes context state, web search, Search & Context entry, active source inventory, clear/remove actions, and temporary chat state.
- `PlaygroundRuntimeInspector.tsx` exposes runtime status, provider/model route, scoped setting summaries, model settings entry, character entry, MCP tools entry, stop/regenerate actions, and message count.
- `Playground.tsx` derives rail state from existing chat state including web search, context files, selected knowledge, media scopes, research context, selected model, selected character, temporary chat, history, and degraded health.
- `PlaygroundForm.tsx` still owns the real deep controls: prompt selector, character/persona selector, MCP popover/settings modal, model/chat settings modal, tools menu, and advanced controls.
- `chat-cockpit.real-server.spec.ts` proves the rails and dialogs are reachable against the real server, but it does not yet prove every rail path completes a configure/save/use workflow.
- Current runtime rail events route `Character` to `ActorPopout` through `OPEN_ACTOR_SETTINGS_EVENT`; that is not an acceptable primary character/persona selection flow for the cockpit.
- Current MCP rail action opens the composer MCP popover first through `OPEN_MCP_TOOLS_EVENT`; the mature cockpit needs a direct path to chat MCP settings when the user chooses Configure MCP.
- Prompt state already exists in the composer path through `selectedSystemPrompt`, `selectedQuickPrompt`, `systemPrompt`, and `promptSummaryLabel`; the rail should expose this state rather than create another prompt model.

## Design Principles

1. Main `/chat` only. Do not make sidepanel/sidebar changes for this work.
2. Rails must call existing shared state paths. Do not create parallel cockpit-only state.
3. Durable conversation configuration belongs in the rails; the composer remains optimized for drafting and turn-level actions.
4. Dialogs and popovers are acceptable for deeper editing, but the rail action should open the correct editing surface directly.
5. Real-server tests must use the real running server for merge-critical browser evidence. Do not use mocked server data or `page.route` for that path.
6. Every rail control must define loading, empty, disabled, degraded, and error states. No enabled-looking control should lead to a dead end.
7. Keyboard operation and focus restoration are part of the design, not a later polish pass.
8. Degraded health should block chat only when the degraded subsystem affects chat. Unrelated degradation should permit sending with explicit warnings.
9. Real-server tests that mutate user-visible settings must restore the original setting or use disposable data created through real server APIs.

## Shared State Ownership

The cockpit should be a better control surface over existing chat state, not a second application inside `/chat`.

- Prompts: use the same prompt template and prompt context state owned by `PlaygroundForm.tsx` and `usePromptTemplates`, including `selectedSystemPrompt`, `selectedQuickPrompt`, `systemPrompt`, and `promptSummaryLabel`.
- Context: use the existing file, knowledge, media, research, web-search, and temporary-chat state already derived in `Playground.tsx`.
- MCP: use `useMcpTools`, the MCP tool store, `useMcpToolsControl`, and `PlaygroundMcpSettingsModal`. The cockpit can control chat exposure and tool choice; MCP Hub remains the authority for server lifecycle, credentials, catalog governance, and policy.
- Model/chat settings: use `useStoreChatModelSettings`, `CurrentChatModelSettings`, and the active provider:model scope set from the selected model key. Settings for duplicate model IDs across providers must stay isolated.
- Character/persona: use the same selected assistant state as the composer and send pipeline. The cockpit must distinguish no assistant, character, and persona states before choosing the editing surface.
- Run controls: use the existing streaming, stop, regenerate, and queued request actions. Rail controls should not bypass the request state machine.

## Cross-Cutting UX Requirements

- Focus: opening a rail dialog or selector should move focus into that surface and restore focus to the triggering rail action when closed.
- Keyboard: all rail controls must be reachable and operable by keyboard, with visible focus states and meaningful labels.
- Responsive: desktop rails may be persistent or collapsible; narrow viewports need an equivalent cockpit access path that does not crowd the composer or hide send controls.
- Visual density: cockpit sections should be dense but scannable. Avoid card-in-card composition and keep summaries short enough for rail widths.
- Warnings: degraded or unavailable states need short cause text and the next available action. For example, `chacha_notes` degraded should show a warning but allow chat if chat runtime checks are healthy.
- Copy: labels should use user-facing concepts (`Model & chat`, `Prompts`, `MCP tools`, `Character / Persona`) and avoid implementation terms unless they are already established in the UI.

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
- Treat rail section ordering as a task flow: choose context and prompts, confirm runtime and assistant, configure tools, then run/monitor.

Acceptance criteria:

- A first-time user can identify where to configure prompts, context, model/chat settings, MCP, and character/persona without knowing composer internals.
- Returning users can collapse either rail or enter focus mode without losing configured state.
- Rail labels match the actual destination. A button labeled Configure MCP opens MCP settings, not an unrelated or intermediate popover.

## Stage 2: Prompt Management In The Left Rail

Goal: make prompt selection and prompt context visible and configurable from the same rail that manages other context inputs.

Controls:

- Show current selected system prompt or "No prompt selected".
- Open prompt selector from the left rail.
- Clear selected prompt when a prompt is active.
- Show whether an inline system prompt/custom prompt is contributing context when that state is already available.
- Preserve the existing composer prompt selector and shared prompt state.
- Keep full prompt library authoring out of this slice unless an existing shared prompt editor can be opened without creating a second CRUD path.

Behavior:

- Prompt selection from the rail must update the same `selectedSystemPrompt`, `selectedQuickPrompt`, or prompt state used by the composer.
- Clearing a selected prompt from the rail must not clear unrelated file, knowledge, media, web, research, model, or character/persona context.
- If inline `systemPrompt` text is active but no prompt template is selected, the rail should identify it as custom/inline prompt context and offer the safest existing way to review or clear it.
- If prompt data is unavailable, show a recoverable disabled state with a reason and keep the composer usable.

Acceptance criteria:

- User can select and clear the active prompt from the left rail.
- The context/status surfaces reflect prompt state before the next send.
- Existing composer prompt workflows continue to work.
- Tests cover selected template, quick prompt, inline custom prompt, and no prompt states where those states are available.

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
- The cockpit controls chat exposure only: tool choice, catalog/module filters, strict catalog state, and personal per-tool availability. It must not imply that the user is starting servers, changing credentials, or editing MCP Hub policy from `/chat`.
- Tool counts must distinguish discovered, executable, disabled-by-user, unavailable, and chat-enabled tools when that state exists. A single "enabled" count is not enough if discovery succeeds but chat cannot execute tools.

Acceptance criteria:

- User can change MCP tool choice from the rail.
- User can open catalog/module/per-tool settings directly from the rail.
- Available and unavailable MCP states are both covered by tests.
- The UI copy makes the boundary with MCP Hub clear without sending the user away for normal chat tool selection.

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
- Recently used or frequent configured models can be surfaced by the selector, but the default list must stay limited to configured usable choices. Search across all known models is a separate discovery mode and must not make unusable models look selected by default.
- Settings summaries should show when a value is inherited/default versus explicitly overridden for the active provider:model scope.

Acceptance criteria:

- User can understand the active runtime and open the correct editor from the rail.
- Tests prove changed settings remain scoped to the active provider:model.
- Existing composer model selection remains reachable and functional.
- Any real-server persistence test must snapshot and restore the changed value, or mutate a disposable/local-only setting.

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
- The workflow must define what happens for fresh chats, existing server chats, temporary chats, default character bootstrap, and persona memory mode before implementation begins.
- Clearing an assistant must be reversible and must not delete the underlying character/persona or erase unrelated conversation context.
- If a reusable shared character/persona selector does not exist, extract the smallest shared selector from the main composer path. Do not import sidepanel/sidebar-specific behavior into the main `/chat` page.

Acceptance criteria:

- User can select, inspect, and clear character/persona state from the cockpit rail flow.
- Persona chats do not show Actor as the primary persona settings route.
- Tests cover none, character, and persona states.
- Tests cover the default-character bootstrap case so the rail does not fight automatic selection on a fresh chat.

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
- Prompt context appears in the same "what affects the next reply" inventory as files, knowledge, media, research, and web state.

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
- Use the runtime rail to open model/chat settings and persist a harmless setting, then restore the original value.
- Use the runtime rail to configure MCP if available, or assert the real unavailable/degraded reason.
- Use the character/persona rail flow with real server state where possible, or assert recoverable empty state.
- Send a real chat turn and verify the cockpit status reflects configured model/context state and either a real assistant response or a recoverable provider/server error.

Real-server data policy:

- Do not use mocked server payloads, `page.route`, or synthetic browser-only state for merge-critical proof.
- Prefer existing real server data when present.
- If populated-state proof requires seed data, create it through existing real APIs with a unique test prefix and clean it up when an API supports cleanup.
- If cleanup is not available or the server has no suitable data, assert the real recoverable empty state in Playwright and cover the populated workflow in component/integration tests.
- Real-server tests should tolerate unrelated degraded subsystems by asserting the warning and continuing when chat itself is usable.

Acceptance criteria:

- Merge-critical browser evidence uses the real server and no mocked server payloads.
- Tests prove at least one completed state-changing rail workflow per control family: prompts, context, MCP, model/chat, and character/persona.
- Tests include keyboard/focus assertions for at least the main rail dialog entry points.

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

Recommended implementation order:

1. Add shared cockpit action/state seams for direct prompt, MCP settings, and character/persona entry points.
2. Update rail IA and labels while preserving all composer controls.
3. Implement one rail workflow at a time with component coverage before broadening real-server E2E.
4. Keep PR #1582 draft until every stage above is either implemented and verified or explicitly moved out by the user.

## Out Of Scope

- Browser-extension sidepanel/sidebar UI.
- Character library redesign.
- MCP Hub management redesign.
- New provider setup flows.
- Drag-resizable/dockable panel systems.
- Replacing all composer controls.
- Mocked real-server proof.
- New visual language or a replacement design system for `/chat`.
- Prompt library CRUD beyond opening or reusing existing prompt selection/editing surfaces.
- MCP server lifecycle, credential, RBAC, or policy management from `/chat`.

## Open Questions

- Which existing character/persona selector component is the safest to reuse or extract for the cockpit rail workflow without importing sidepanel-specific behavior?
- Should inline/custom prompt editing open an existing editor surface, or should the first slice only show and clear inline prompt context?
- Which real-server fixture policy is acceptable for prompt and character/persona proof if the user's running server has no existing data: create-and-cleanup via real APIs, or empty-state Playwright plus populated component tests?
