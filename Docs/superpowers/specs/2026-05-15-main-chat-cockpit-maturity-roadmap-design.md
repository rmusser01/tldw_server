# Main /chat Cockpit Maturity Roadmap Design

## Scope

This roadmap covers only the main WebUI `/chat` page. The route is `apps/tldw-frontend/pages/chat/index.tsx`, which dynamically loads `apps/packages/ui/src/routes/option-chat.tsx`, which renders `Playground` from `apps/packages/ui/src/components/Option/Playground/Playground.tsx`.

This excludes browser-extension sidepanel/sidebar work, `/settings`, character library pages, onboarding, backend API design, MCP Hub pages, evaluations, media ingestion, and repo-wide architecture unless the behavior directly affects the main `/chat` page.

The merged cockpit work has achieved parity and recertification. The next bar is UX maturity: `/chat` should behave like a real cockpit where the user can understand and adjust the next run's composition, context, assistant/persona, model/runtime, and tool policy without hunting through unrelated controls.

## Current Evidence

- `PlaygroundCockpitShell.tsx` provides the cockpit/focus shell, desktop rails, mobile rail tabs, rail visibility controls, and status strip slot.
- `PlaygroundContextRail.tsx` currently exposes context status, active context sources, web search, Search & Context, prompt selection, session status, and clear/remove affordances for supported context types.
- `PlaygroundRuntimeInspector.tsx` currently exposes runtime state, provider/model route, Model & Chat settings, MCP tool state, character/persona selection and clearing, scoped model settings, stop/regenerate controls, and timeline count.
- `playground-cockpit-summaries.ts` already normalizes prompt, assistant, MCP, provider-route, and session summaries.
- `PromptAssemblyPreview.tsx` already renders prompt-section/token/conflict data, but it is server-chat gated and lives in settings rather than as a first-class `/chat` composition preview.
- `useConversationContextComposition.ts` and `conversationContextComposer.ts` already provide one composition object for preview/send context transformation.
- `chat-cockpit.real-server.spec.ts` already proves real-server `/chat` cockpit/focus entry, prompt selection/clearing, provider:model settings persistence, MCP state distinction, mobile rails, real character selection, and real persona selection without stubbing backend routes.

## Design Principles

1. **Composition transparency before send.** A user should be able to answer: what prompt/persona/model/context/tools will shape my next message?
2. **Rails are work surfaces, not static summaries.** The left rail owns context/prompt/session awareness. The right rail owns runtime/model/persona/tools awareness.
3. **Single source of truth.** Rail controls must call the same handlers, stores, or dialogs as existing composer/settings controls.
4. **Progressive disclosure.** Dense cockpit state should be visible at a glance, with deeper preview/details behind an inline disclosure or modal.
5. **Focus mode remains first-class.** The cockpit can collapse back to the chat-first experience without losing state.
6. **Real-server proof for merge-critical behavior.** Final claims about prompt/persona/model/MCP/chat behavior require the running server, not mocked payloads.
7. **Provider:model identity is sacred.** Model labels, scoped settings, and persistence must preserve provider-qualified identity.

## Staged PR Roadmap

### PR 1: Context Stack + Prompt/Persona/Model Composition Preview

**Goal:** Make `/chat` answer "what will be sent, under which prompt/persona/model/context/tool policy?" before the user sends.

**Scope:**

- Add a first-class composition preview in the main cockpit.
- Show active prompt, assistant/persona/character, model/provider route, provider:model settings scope, context sources, MCP/tool policy, and approximate prompt/context footprint.
- Reuse existing summary/composition paths before introducing new state.
- Keep context removals/disable actions limited to flows already supported by shared handlers.

**Likely files:**

- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts`
- `apps/packages/ui/src/components/Common/Settings/PromptAssemblyPreview.tsx`
- `apps/packages/ui/src/hooks/chat/useConversationContextComposition.ts`
- `apps/packages/ui/src/components/Option/Playground/ContextFootprintPanel.tsx`
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

**Acceptance:**

- The user can inspect the effective prompt/persona/model/context/tool setup before sending.
- Prompt and assistant selection appear together with the model route and tool policy instead of being scattered across rails.
- Provider:model settings scope is visible and not collapsed into a raw model string.
- The preview handles empty, loading, unavailable, degraded, and populated states.
- No sidepanel/sidebar files are touched.

### PR 2: Rail Information Architecture and Action Hierarchy

**Goal:** Make cockpit rail groups predictable and scannable for first-time and returning users.

**Scope:**

- Reorganize left rail into context stack, prompt management, search/RAG sources, files/media, and session persistence.
- Reorganize right rail into runtime state, model route/settings, assistant/persona, tools/MCP, and recovery controls.
- Tighten headings, status labels, and empty-state copy.
- Preserve existing controls and keyboard names.

**Acceptance:**

- Durable controls are in rails; turn-level draft controls stay in the composer.
- First-time users can identify where to change prompt, persona, model, context, and tools.
- Power users can scan active state without opening every section.

### PR 3: Model/Provider Confidence

**Goal:** Make model selection and scoped settings reliable and understandable.

**Scope:**

- Default model choices to configured usable provider/model entries.
- Preserve search across the broader known catalog as an explicit secondary mode.
- Show recent/frequent configured choices where existing data supports it.
- Make settings persistence/restoration visibly keyed by provider:model.

**Acceptance:**

- The default selector avoids unusable catalog noise.
- Same model id under different providers does not share settings accidentally.
- The runtime rail shows provider route and settings scope clearly.

### PR 4: Mode and Session Clarity

**Goal:** Make cockpit/focus and saved/temporary states explicit without adding friction.

**Scope:**

- Clarify cockpit/focus presets and independent rail visibility.
- Improve saved, temporary, local, server-backed, loading, failed, and recovered session copy.
- Keep focus mode fast and chat-first.

**Acceptance:**

- Users know whether the current conversation is saved and what will happen after reload.
- Collapsing rails does not obscure important warnings.

### PR 5: Error, Degraded, Loading, and Recovery Polish

**Goal:** Turn system state into actionable feedback.

**Scope:**

- Refine status strip priority for ready, streaming, degraded, error, no model, unavailable server, and context-loading states.
- Permit chat under unrelated degraded health with warnings.
- Distinguish chat-critical unavailable states from unrelated degraded subsystem states.
- Improve disabled control reasons and recovery affordances.

**Acceptance:**

- A degraded unrelated subsystem does not block chat.
- Chat-critical failures explain the next recovery action.
- Disabled controls expose why they are disabled.

### PR 6: Mobile Cockpit

**Goal:** Make mobile `/chat` a deliberate cockpit/focus experience, not a squeezed desktop.

**Scope:**

- Validate mobile context/runtime tabs or sheet behavior.
- Ensure composer reachability, focus return, and no occlusion.
- Tune density, sticky status, and tap targets.

**Acceptance:**

- Mobile users can inspect/change prompt, persona, model, context, and tool state without losing the draft.
- Mobile screenshots cover context tab, runtime tab, focus mode, active conversation, and key error/degraded states.

### PR 7: Visual and Copy Polish

**Goal:** Make the cockpit feel mature, restrained, and consistent with the WebUI design system.

**Scope:**

- Reduce visual noise, repeated borders, and unclear status labels.
- Normalize button/icon/toggle treatments.
- Tighten terminology around prompt, persona, character, context, model route, and tools.
- Preserve dense, utilitarian cockpit ergonomics.

**Acceptance:**

- No rail section looks like a generic settings dump.
- Copy uses user-facing terms consistently.
- Design-system state labels and tokens remain the source of truth.

### PR 8: QA Harness and Merge Certification

**Goal:** Make future cockpit regressions hard to miss.

**Scope:**

- Expand focused Vitest coverage for summaries, rails, accessibility, keyboard focus, and responsive behavior.
- Expand real-server Playwright proof for prompt, persona, model settings, MCP states, conversation send, mobile, screenshots, and focus/cockpit transitions.
- Maintain the no-mocked-server rule for merge-critical browser proof.

**Acceptance:**

- The PR series leaves durable tests, not just one-off screenshots.
- Final certification can be checked against issue/task acceptance item by item.

## Quick Wins

- Add a compact "Next message composition" card using existing prompt, assistant, model route, context source, and MCP summaries.
- Show provider:model settings scope directly near the Model & Chat settings button.
- Add loading/unavailable copy for prompt preview instead of hiding it behind saved-chat-only behavior.
- Group context sources into a named stack with consistent active/disabled/degraded states.
- Add focus-return tests for preview disclosures and rail dialogs.

## Larger Redesign Opportunities

- A unified model selector that defaults to configured usable choices while supporting catalog-wide search.
- A richer context stack with token budget, source ordering, per-source enable/disable, and conflict warnings.
- A composition preview that merges prompt assembly, persona/character injected context, RAG/context transforms, model settings, and tool policy into one inspectable model.
- A mobile cockpit sheet that can become the shared pattern for other operational pages.

## Risks

- Duplicating state between rails, composer, and settings would create contradictory cockpit behavior.
- Prompt preview is currently server-chat gated; broadening it must handle temporary/local chats honestly.
- Context composition can be expensive or asynchronous; preview must avoid stale or blocking states.
- Provider:model display improvements must not mutate the actual routing string.
- MCP availability varies by server. Tests must distinguish populated, empty, unavailable, and degraded states.

## Open Questions and Defaults

- **Can prompt preview work before a server-backed chat exists?** Default: show local known prompt/persona/model/context summary immediately; keep server-only detailed prompt assembly labeled as unavailable until a server chat id exists.
- **Should PR 1 include full context token accounting?** Default: use existing token/footprint data where available; do not invent exact token counts when the app only has approximations.
- **Should rail composition preview replace `PromptAssemblyPreview`?** Default: no. Reuse or wrap it, but keep compatibility with settings surfaces until the cockpit preview is proven.
- **Should MCP unavailable and empty be separate?** Default: yes. Empty means reachable but no chat-enabled tools. Unavailable means MCP state cannot be used or inspected.
- **Should sidepanel inherit any changes?** Default: no. Any shared component changes must be validated for main `/chat`, but no sidepanel/sidebar behavior is a deliverable here.
