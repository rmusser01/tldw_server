# Main /chat Context Stack and Composition Preview Plan

> **For agentic workers:** This is PR 1 of the post-merge main `/chat` cockpit maturity roadmap. Keep the scope to the main WebUI `/chat` page. Do not modify the browser-extension sidepanel/sidebar.

**Goal:** Add a first-class Context Stack plus Prompt/Persona/Model Composition Preview to the main `/chat` cockpit so users can inspect the effective setup for the next message before sending.

**Implementation Status:** Complete for this first slice. Verified with focused Vitest coverage, the design-system product-state guard, and the full real-server `/chat` cockpit Playwright spec.

**Architecture:** Keep `Playground` as the coordinator. Build a small, testable summary layer from existing prompt, assistant, provider route, context-source, MCP, and conversation composition data. Rails remain presentational and call shared handlers. Do not create a parallel prompt/model/context state system.

**Tech Stack:** React 18, TypeScript, i18next, existing WebUI design-system tokens, Vitest with Testing Library, Playwright real-server checks.

---

## Scope Lock

In scope:

- Main WebUI `/chat` only.
- `apps/tldw-frontend/pages/chat/index.tsx` through `apps/packages/ui/src/routes/option-chat.tsx` and `Playground`.
- Context Stack card in the cockpit context rail.
- Composition Preview that shows active prompt, character/persona, model/provider route, provider:model settings scope, context sources, MCP/tool policy, and available token/context footprint data.
- Empty/loading/degraded/unavailable/populated states.
- Tests proving the preview uses current state and does not rely on mocked real-server proof.

Out of scope:

- Browser-extension sidepanel/sidebar.
- Full model selector redesign.
- Backend API redesign.
- New MCP execution features.
- Full drag/drop context ordering.
- Exact tokenizer-backed token accounting unless an existing API already provides it.
- Replacing every composer control with rail controls.

## Carry-Forward Decisions

- Degraded health should allow chat when the degraded subsystem is unrelated to chat, with warnings.
- The model selector should default to configured usable choices, but that selector redesign is not part of PR 1 unless required to display the current provider:model scope.
- Each provider:model keeps its own settings. Any preview display must preserve provider-qualified identity.
- Prompt management/selection belongs in the left rail for the main `/chat` cockpit.
- Character/persona selection and state belong in the cockpit rails for the main `/chat` page, not in a sidepanel/sidebar deliverable.

## Current Evidence

- `/chat` route: `apps/tldw-frontend/pages/chat/index.tsx` dynamically imports `@/routes/option-chat`; `option-chat.tsx` renders `<Playground />`.
- Context rail: `PlaygroundContextRail.tsx` already renders context status, sources, prompt selection, Search & Context, web search, and session status.
- Runtime rail: `PlaygroundRuntimeInspector.tsx` already renders provider/model route, Model & Chat settings, MCP state, character/persona selection, scoped settings, stop/regenerate, and timeline.
- Summary helpers: `playground-cockpit-summaries.ts` already builds prompt, assistant, MCP, provider-route, and session summaries.
- Prompt assembly: `PromptAssemblyPreview.tsx` already normalizes server prompt preview sections, token budget, warnings, conflicts, and examples.
- Context composition: `useConversationContextComposition.ts` already keeps preview/send composition aligned for transformed input and provider messages.
- Real-server E2E: `chat-cockpit.real-server.spec.ts` already covers prompt selection/clearing, model setting persistence/restoration, MCP state distinction, mobile rails, character selection, and persona selection against a running server.

## Stage 1: Define the Composition Summary Contract

**Goal:** Create a small typed summary contract for the next-message setup without changing UI behavior yet.

**Files:**

- Add or update `apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts`
- Update `apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts` only if existing summary builders need shared exports.
- Add `apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts`

**Steps:**

- [ ] Write tests first for a `buildPlaygroundCompositionPreviewSummary` helper.
- [ ] Cover no prompt/no assistant/no model/no context.
- [ ] Cover selected prompt + persona + provider:model + MCP available.
- [ ] Cover selected character + context sources + MCP unavailable.
- [ ] Cover degraded context/tool states without treating them as active success.
- [ ] Include provider:model settings scope as a distinct field from display model.
- [ ] Implement the helper using existing summary types where possible.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-composition-preview.test.ts
```

**Expected Outcome:** The cockpit has one tested data shape for what the next message will use, without UI churn.

## Stage 2: Add the Context Stack Card

**Goal:** Make the left rail present context as an ordered stack with clear states and actions.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx` or add `PlaygroundContextRail.composition-preview.test.tsx`

**Steps:**

- [ ] Write tests for an explicit "Context stack" region.
- [ ] Show prompt, assistant/persona/character, web search, files, knowledge, media, research, and MCP/tool policy entries when present.
- [ ] Show empty state when no extra context is active.
- [ ] Preserve existing per-source open/remove controls.
- [ ] Preserve prompt selection and clear behavior.
- [ ] Use active/disabled/degraded/unavailable labels consistently.
- [ ] Keep the rail compact; avoid nested cards inside cards.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx
```

**Expected Outcome:** The left rail makes active context legible as a stack instead of scattered summaries.

## Stage 3: Add the Composition Preview UI

**Goal:** Add an inspectable preview of effective prompt/persona/model/context/tool setup.

**Files:**

- Add `apps/packages/ui/src/components/Option/Playground/PlaygroundCompositionPreview.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx` only if model/tool summary data needs a compact cross-link.
- Add `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundCompositionPreview.test.tsx`

**Steps:**

- [ ] Write component tests first.
- [ ] Render top-level rows for prompt, assistant, model route, settings scope, context, and tools.
- [ ] Render a disclosure for details instead of showing long prompt/context text by default.
- [ ] Show loading/error/unavailable states for server-backed prompt assembly.
- [ ] For temporary/local chats, show local known summary and label server-only prompt assembly as unavailable.
- [ ] If wrapping `PromptAssemblyPreview`, keep its server-chat requirement honest in copy.
- [ ] Return focus to the disclosure/trigger after opening and closing any dialog.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundCompositionPreview.test.tsx
```

**Expected Outcome:** Users can inspect the effective next-message setup before sending without opening multiple unrelated settings panels.

## Stage 4: Wire Playground State Into the Preview

**Goal:** Feed the preview with real `/chat` state from `Playground`.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`

**Steps:**

- [ ] Build the preview summary from existing selected prompt, quick prompt, system prompt, assistant summary, selected model/provider route, context sources, MCP summary, and scoped setting summaries.
- [ ] Use existing `buildCockpitProviderRouteSummary`, `buildCockpitPromptSummary`, `buildCockpitAssistantSummary`, and `buildCockpitMcpSummary` where possible.
- [ ] Ensure selected provider:model is passed through as a scope key, not only display text.
- [ ] Pass current `serverChatId` and settings fingerprint only where needed by `PromptAssemblyPreview`.
- [ ] Do not move the source of truth out of existing hooks/stores.
- [ ] Add integration tests that selected prompt/persona/model/context changes update the preview.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx
```

**Expected Outcome:** The preview reflects the same live state used by send and settings flows.

## Stage 5: Real-Server Proof

**Goal:** Prove the preview works with the running server and existing `/chat` workflows.

**Files:**

- Update `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

**Steps:**

- [ ] Extend the existing real prompt/model/MCP test to assert composition preview content after prompt selection.
- [ ] Extend real character test to assert the character appears in the Context Stack and Composition Preview.
- [ ] Extend real persona test to assert the persona appears in the Context Stack and Composition Preview.
- [ ] Assert provider:model settings scope remains visible after model settings save/restore.
- [ ] Assert MCP populated, empty, unavailable, or degraded states are labeled distinctly based on the real server state.
- [ ] Capture updated desktop screenshot for populated composition preview.
- [ ] Do not use `page.route`, synthetic server responses, or sidepanel/sidebar routes.

**Verification:**

```bash
TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=<real-key> bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

**Expected Outcome:** The cockpit preview is proven with the real server for prompt, persona/character, provider:model settings, MCP state, and screenshots.

## Stage 6: Mobile and Focus Regression

**Goal:** Ensure the new preview does not break focus mode, rail collapse, or mobile cockpit tabs.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Update `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

**Steps:**

- [ ] Verify focus mode hides rails but preserves selected composition state.
- [ ] Verify returning to cockpit restores preview content.
- [ ] Verify mobile context/runtime tabs can expose the preview without occluding the composer.
- [ ] Add screenshots for mobile context preview and mobile focus after preview interaction.
- [ ] Add keyboard focus assertions for preview disclosure and modal/dialog returns.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts
```

**Expected Outcome:** The preview improves cockpit comprehension without reducing focus-mode or mobile usability.

## Stage 7: Final Verification and Handoff

**Goal:** Certify PR 1 as a mature, bounded enhancement.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-composition-preview.test.ts src/components/Option/Playground/__tests__/PlaygroundCompositionPreview.test.tsx src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts
```

```bash
TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=<real-key> bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

```bash
git diff --check
```

**Definition of Done:**

- [ ] Composition preview is visible and useful on main `/chat`.
- [ ] Context Stack clearly shows active, disabled, degraded, unavailable, and empty states.
- [ ] Prompt/persona/character/model/tool state appears in one preview.
- [ ] Provider:model settings scope is visible and preserved.
- [ ] Real-server E2E proves prompt, persona/character, model settings, MCP state, and screenshots.
- [ ] Mobile and focus regressions are covered.
- [ ] No sidepanel/sidebar files are part of the PR.
