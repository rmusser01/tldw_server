# Main /chat True Cockpit Controls Gap Design

## Scope

This spec covers only the main WebUI `/chat` page, routed through `apps/tldw-frontend/pages/chat/index.tsx` into the shared `Playground` surface. It explicitly excludes the browser-extension sidepanel, sidepanel sidebar, settings pages, onboarding, character library pages, backend API design, MCP Hub, evaluations, media ingestion, and repo-wide architecture unless a point directly affects the main `/chat` user experience.

The goal is not just to keep the old `/chat` controls reachable. The long-term target is a true cockpit in the main chat window: side rails and status areas should provide direct, reliable controls for the current conversation, model/runtime state, context/RAG/tool state, and session state, while still allowing the user to collapse into a focused chat layout that resembles the current chat-first page.

This document defines the first implementation slice toward that target. It is not approval to build the entire cockpit maturity backlog in one PR.

## Source Evidence

- `PlaygroundCockpitShell.tsx` currently provides a binary `cockpit` / `focus` layout toggle, fixed left/right desktop rails, mobile `details` rails, and a status strip slot.
- `PlaygroundContextRail.tsx` currently summarizes context and opens Search & Context via a custom event. It does not directly manage context, RAG, web search, files, or tool state.
- `PlaygroundRuntimeInspector.tsx` currently shows streaming/ready state, selected model, message count, thread-search state, and buttons that open model and character settings dialogs.
- `Playground.tsx` derives cockpit rail state from `attachedResearchContext`, `webSearch`, `contextFiles`, `selectedKnowledge`, `ragMediaIds`, `temporaryChat`, `serverChatId`, `historyId`, `selectedModel`, `streaming`, and `selectedCharacter`.
- `ComposerToolbar.tsx` remains the primary location for many old `/chat` controls: temporary/saved chat, Search & Context, web search, model/prompt/character controls, MCP, dictation, voice chat, attachments, tools, compare mode, and advanced controls.
- `chat-cockpit.real-server.spec.ts` verifies the running server, degraded-health pass-through, cockpit/focus rail visibility, and reachability of key controls. It does not fully exercise all old `/chat` workflows or true cockpit-side control behavior.
- `/research` has a more console-like pattern: create/list/select, live status, pause/resume/cancel, checkpoint approval, and editable checkpoint forms in one operational surface. The main `/chat` cockpit should borrow the operational density and direct-control model, not the exact research workflow.

## Design Decision

Treat the work as two tiers:

1. Merge-blocking parity and verification: prove that every existing main `/chat` workflow still works in cockpit and focus modes.
2. True cockpit-control completion: turn the main chat rails and status area from summaries/launchers into direct, inspectable, keyboard-accessible controls over multiple slices.

This avoids merging a shell that looks like a cockpit while leaving the real work surface hidden in the composer.

The implementation plan should not attempt every cockpit maturity idea in one oversized patch. Use a narrow first implementation slice:

1. Make cockpit rails control the highest-risk existing state directly: web search, context entry, temporary/saved session state, model/provider summary, model settings, character/persona entry, streaming/error status, and the existing degraded warning.
2. Add independent rail collapse only after the direct controls have one shared source of truth with existing composer state.
3. Keep deeper workflows in the existing dialogs/panels until the cockpit control path is proven by tests.

## First Slice Boundary

The next implementation plan should cover only the first slice below. Anything not listed here is a later cockpit-maturity task unless the user explicitly expands scope.

In scope for the first slice:

- Main WebUI `/chat` only.
- Direct cockpit controls for web search, Search & Context entry, active context summaries, temporary/saved session state, selected model/provider summary, model settings entry, character/persona entry, streaming/error/degraded status, and status-strip state.
- Shared handler/store wiring so rail controls update the same state as existing composer controls.
- Real-server browser verification that proves at least one state-changing cockpit action works without mocked server data.
- Component/integration tests that prove the new rail controls call shared state paths rather than rendering static labels.

Out of scope for the first slice:

- Browser-extension sidepanel/sidebar changes.
- Full replacement of the composer toolbar.
- Full direct-control coverage for compare mode, image generation, voice conversation, advanced parameter presets, MCP execution, or every artifact workflow.
- Drag-resizable or dockable panel systems.
- Full provider-health dashboards outside the current chat turn.
- Repo-wide design-system or backend architecture refactors.

## Shared Control Contract

Cockpit controls must not create parallel state. Each rail/status control should call the same handlers or stores already used by the composer and dialogs. If a state cannot be updated through an existing handler, the first implementation task should expose a shared handler rather than duplicating behavior in the rail.

This applies especially to:

- Model/provider selection and provider:model scoped settings.
- Web search and Search & Context state.
- Attached research context actions.
- Temporary versus saved chat state.
- Character/persona selection and actor settings.
- MCP/tool availability and selected tool mode.
- Streaming, stop, retry, and regenerate actions.

## Merge-Blocking Parity Gaps

### P1: Real-server tests prove reachability more than behavior

Current real-server coverage opens important controls, but it does not prove that the workflows complete. Before merge, add tests that verify behavior for the main `/chat` page against the real running server for merge-critical smoke coverage. Unit and component tests may mock component dependencies, but the merge-critical browser path must not use mocked server data, `page.route`, synthetic server payloads, or sidepanel/sidebar routes.

Required coverage:

- Send a normal chat message and verify the assistant response appears or the expected provider error is rendered recoverably.
- Toggle cockpit/focus mode before and after a send and verify the conversation state survives.
- Select a configured provider:model entry and verify the outgoing chat request uses the model id and provider routing expected by provider-qualified selection.
- Open model settings for two distinct provider:model keys with the same model id where possible and verify settings remain scoped independently.
- Open Search & Context, select or apply real context where the running server has available data, then verify the context indicator and send path reflect it. If the real server has no usable context corpus, verify the empty/recoverable state and keep a component/integration test for the populated state.
- Toggle web search from the main chat UI and verify the state is reflected in the active context/cockpit surface.
- Exercise attachment flow enough to prove file/image selection still reaches the composer state, even if the test uses a small fixture.
- Verify MCP controls in both available and unavailable states. If MCP is available on the real server, include a harmless tool selection/execution or request construction check. If not, verify the disabled/offline state and cover the available state in a component/integration test without pretending it is real-server evidence.
- Verify prompt selector, character/persona selector, tools menu, thread search, artifacts trigger, compare mode, dictation/voice availability messaging, and advanced controls remain accessible in both cockpit and focus layouts.
- Verify mobile focus-first behavior plus opening cockpit controls without losing composer usability.

### P1: Degraded health permits entry but is not chat-aware

The current gate allows `/chat` to enter on HTTP 206 degraded health when degraded entry is enabled. That matches the desired behavior when the degraded subsystem is unrelated to chat. The missing piece is classification: chat should permit unrelated degraded subsystems with warnings, but it should not treat chat-critical degradation the same way.

Recommended design:

- Keep degraded entry for non-chat-critical checks.
- Display a specific warning that names degraded checks.
- Prefer a backend-provided chat readiness/capability flag if available. If not available in this PR, define a small client-side interim allow/block list.
- Test both unrelated degradation and chat-critical degradation behavior.

Interim default:

- Warning-only degraded checks: subsystems unrelated to opening `/chat`, choosing a model, submitting a turn, or rendering the conversation.
- Chat-critical checks: inability to reach the server, auth failure, provider/model metadata unavailable when no cached usable model exists, selected provider unusable, or the chat completion path unavailable.
- Persistence-related degradation should warn but not block temporary chat unless it prevents the conversation from rendering or sending.

### P1: Side rails must control main chat state, not only open old controls

The current rails use summary text and event dispatchers. For merge readiness under the clarified target, rails should include direct controls for the most important active states, with dialogs used for deeper editing only.

Minimum direct cockpit controls for the first implementation slice:

- Context rail: web search toggle, Search & Context open/apply affordance, visible counts for files/knowledge/media, remove or clear actions for active context types where the existing state supports it, and temporary/saved session toggle or status with action.
- Runtime rail: selected model display, model selector entry point, model settings entry point, provider route/status display, character/persona selector entry point, streaming/stop/regenerate state where available, and last error/status recovery action where available.
- Status strip: visible ready/streaming/error/degraded state, active model/provider, context-active state, session persistence state, and message count.

Do not move every old composer action at once. Compare mode, full image generation, voice conversation, and advanced parameter presets can remain composer/dialog workflows in the first slice if they remain reachable and the cockpit exposes accurate status or availability.

## Cockpit Maturity Backlog

The items below describe the broader true-cockpit direction. They are not automatically in scope for the first implementation slice unless they are explicitly repeated in the First Slice Boundary above.

### P1: Layout control is binary

Current cockpit mode shows both rails and focus mode hides both. A true cockpit should support independent rail visibility.

Recommended design:

- Keep `Cockpit` and `Focus` as simple presets.
- Add independent left/right rail collapse controls in cockpit mode. For the first slice, this is secondary to wiring direct controls to shared state.
- Persist the rail visibility state separately from the overall preset in browser storage for the first implementation slice.
- Keep focus mode as the fast chat-only preset.
- Avoid draggable/resizable panels in the first completion pass unless needed after usability testing.

### P1: Context rail is not yet a context work surface

The context rail should answer: what context will affect the next response, how much is it using, and how do I change it?

Recommended direct controls:

- Web search on/off.
- Search & Context open button plus current search/RAG state summary.
- Active attached research context summary with apply/reset/remove/pin actions where already supported by `PlaygroundForm`.
- File, knowledge, and media-scope counts with clear/remove actions when supported by existing state.
- Context budget/truncation warning entry point.
- Prompt/system context indicator.
- Temporary/saved session state and persistence warning.

For the first slice, prioritize web search, Search & Context entry, active context summaries, temporary/saved state, and any clear/remove actions already supported by existing shared handlers.

### P1: Runtime rail is not yet an operational runtime inspector

The runtime rail should answer: what engine is running this turn, what is it doing, and what can I do now?

Recommended direct controls:

- Provider and model as separate readable fields, not only raw selected model.
- Configured/catalog model selector entry point.
- Provider route warning when provider-qualified routing is inferred, ambiguous, or falling back.
- Model settings button scoped to current provider:model.
- Character/persona selector and actor settings button.
- Streaming state with stop action if a stop hook is available.
- Regenerate/retry action when there is a last assistant turn or last recoverable error.
- Usage/cost/token budget indicators where existing data exists.
- MCP/tools availability and selected tool mode summary.

For the first slice, prioritize selected model/provider display, model settings entry, character/persona entry, streaming/error/degraded status, and provider route warnings relevant to the current turn.

### P2: The composer is overloaded as the only real command surface

The composer should remain efficient, but it should not be the only place to control chat state in cockpit mode. The side rails should carry durable state controls, and the composer should focus on drafting, attachments, send options, and immediate turn-level choices.

Recommended design:

- Keep core composer actions available for focus mode.
- In cockpit mode, duplicate or rehome durable state controls into rails without creating conflicting independent state.
- Prefer shared handlers and stores over parallel rail-only state.
- Keep one source of truth for every control.

### P2: Mobile cockpit is only a compact disclosure surface

Mobile currently uses `details` sections for rails. That is acceptable as a first pass, but true cockpit controls should remain usable on mobile.

Recommended design:

- Keep focus as the default mobile preset.
- Use a cockpit drawer or stacked disclosure controls for context/runtime.
- Verify every required direct control is reachable by keyboard/touch and does not occlude the composer.

## Test Coverage Design

### Unit and component tests

- `PlaygroundCockpitShell`: independent rail collapse state, focus preset, cockpit preset, accessibility labels, persistence.
- `PlaygroundContextRail`: web toggle, context counts, clear/remove actions, Search & Context open/apply affordance, degraded/persistence states.
- `PlaygroundRuntimeInspector`: provider/model fields, model selector action, scoped model settings action, character/persona action, streaming/error states, stop/retry/regenerate affordances where supported.
- `PlaygroundStatusStrip`: ready/streaming/error/degraded states, context-active state, session state, model/provider labels.
- Model selector utilities: configured default scope, catalog search scope, recent/frequent ordering, provider:model duplicate model ids, provider-specific settings isolation.

These tests may use mocks for stores, hooks, and component handlers. They should prove the rail controls call the shared control contract, not just render labels.

### Main /chat integration tests

- Cockpit and focus render the same active conversation and composer state.
- Rail controls update the same state as composer controls.
- No sidepanel/sidebar route is used in the main `/chat` tests.
- Keyboard focus returns correctly after rail actions open and close dialogs/panels.
- Mobile starts in focus and can expose direct cockpit controls without losing composer access.

These tests should prefer the main `/chat` route and real shared components. Mocks are acceptable only for narrow browser-host APIs or expensive dependencies that are not the server contract under test.

### Real-server Playwright tests

Use the real server already expected by the project, not mocked data, for merge-critical smoke coverage:

- Health 200 and unrelated health 206 allow chat entry.
- Configured providers and model metadata are fetched from the real server.
- A configured model can be selected and used to submit or attempt a chat turn.
- Degraded warning remains visible but non-blocking for unrelated degraded checks.
- Core cockpit controls complete at least one state-changing action, not just open.

For expensive or environment-sensitive paths such as MCP execution, voice, image generation, and external providers, tests may assert availability/disabled/error recovery if the running server lacks the dependency. The important requirement is that the user-facing state is correct and recoverable.

The real-server submit test should accept either a successful assistant response or a real provider/server error shown in the chat UI, as long as the request path is real and the user gets a recoverable state. It should not replace the real server with fixture responses.

## Implementation Boundaries For The Future Plan

- Do not touch extension sidepanel/sidebar routes for this work.
- Do not duplicate setup/provider rules client-side when the backend already exposes configured provider/model data.
- Do not introduce net-new product workflows. Rehouse and harden existing main `/chat` capabilities into true cockpit controls.
- Do not remove the existing composer controls until the cockpit equivalent is proven and keyboard-accessible.
- Do not rely on mocked data for the merge-critical real-server verification path.
- Do not treat reachability-only assertions as proof of cockpit functionality.
- Do not implement cockpit-maturity backlog items beyond the first slice unless the implementation plan explicitly calls them out and the user approves the expanded scope.

## Likely Files

- `apps/tldw-frontend/pages/chat/index.tsx`
- `apps/packages/ui/src/routes/option-chat.tsx`
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- `apps/packages/ui/src/hooks/playground/useModelSelector.tsx`
- `apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts`
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

## Open Questions

- Backend readiness: can `/api/v1/health` or a chat-specific endpoint expose chat-critical readiness directly? If not, use the interim client-side classification above.
- Persistence scope: browser-local rail visibility is the default for the first slice. Revisit per-user or per-chat persistence only if users need device/session sync.
- Runtime scope: show model/provider route state relevant to the current turn first. Full provider health belongs outside this slice unless required to explain a current chat failure.
- Real-server submit determinism: use the configured model path when available. If provider credentials or local model availability vary, assert the real request plus recoverable UI response rather than requiring a successful model completion in every environment.
