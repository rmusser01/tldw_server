# Main /chat Cockpit P-Series Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all P0, P1, and P2 work from issue #1646 for the main WebUI `/chat` cockpit rails in draft PR #1582.

**Architecture:** Keep `/chat` as the existing `Playground` route, with cockpit rails as views over the same state used by the composer and send pipeline. Extract small summary/state helpers where needed, but do not create a parallel chat control system or move work into the app sidebar or browser-extension sidepanel.

**Tech Stack:** Next.js page shim, shared React UI package, Zustand/storage-backed chat state, React Query, Ant Design dialogs/dropdowns already in use, Vitest + Testing Library, Playwright real-server E2E.

---

## Scope And Merge Gate

This plan is only for the main `/chat` page:

- Route shim: `apps/tldw-frontend/pages/chat/index.tsx`
- Route body: `apps/packages/ui/src/routes/option-chat.tsx`
- Main implementation: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`

Do not implement app sidebar, browser-extension sidepanel/sidebar, settings-page redesign, character-library redesign, MCP Hub lifecycle/policy, media ingestion, evaluations, or backend architecture unless a `/chat` workflow cannot be completed without a narrow contract fix.

PR #1582 stays draft. It is not ready for review or merge until:

- P0 is implemented, verified, and explicitly approved by the maintainer.
- P1 is implemented, verified, and explicitly approved by the maintainer.
- P2 is implemented, verified, and explicitly approved by the maintainer.

Merge-critical proof must use the running server. Do not use mocked payloads, `page.route`, or Computer Use for proof. Component tests may mock hooks and stores, but real-server Playwright tests must hit the live API.

## File Structure

Modify these existing files as needed:

- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`: wire canonical cockpit state into left/right rails and status strip.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`: prompt/context/session rail UI, clear actions, empty/degraded states.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`: runtime/model/MCP/assistant/run-control rail UI.
- `apps/packages/ui/src/components/Option/Playground/playground-cockpit-actions.ts`: event payloads for rail-launched shared surfaces.
- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`: shared character/persona selector behavior only where needed for `/chat` rail workflows.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`: only for shared modal/listener state that the rails launch, especially return-focus and MCP/model settings.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx`: focus return and unavailable/degraded close behavior if needed.
- `apps/packages/ui/src/hooks/useMessageOption.tsx`: only if assistant state synchronization must be fixed at the shared chat state boundary.
- `apps/packages/ui/src/hooks/playground/useMcpToolsControl.ts`: source of MCP availability/summary state.
- `apps/packages/ui/src/hooks/playground/useModelSelector.tsx` and `apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts`: model selector scope/search behavior if P0 model proof exposes gaps.
- `apps/packages/ui/src/store/model.tsx` and `apps/packages/ui/src/store/model-settings-scope.ts`: provider:model scoped settings only.
- `apps/packages/ui/src/assets/locale/en/playground.json` and `apps/packages/ui/src/public/_locales/en/playground.json`: new cockpit copy keys.

Prefer creating these focused helper files instead of expanding `Playground.tsx` further:

- Create: `apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts`
  - Responsibility: pure helpers for assistant, prompt, MCP, and session/context rail summaries.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts`
  - Responsibility: state precedence, no stale summaries, user-facing labels, provider:model scope display.

Extend these tests:

- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
- `apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts`
- `apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx`
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

## Shared Commands

Run focused tests after each task:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --reporter=verbose
```

Run the real-server proof only against the real running server:

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=$TLDW_E2E_API_KEY bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

Always run:

```bash
git diff --check
```

If touching Python/backend code, run Bandit on the touched backend scope. Most planned work is TS/TSX/Playwright/docs, so record Bandit as not applicable unless backend files change.

---

### Task 0: State Contract Helpers And Guard Tests

**P-Series:** P0 foundation

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts`
- Create: `apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md`

- [ ] **Step 1: Write failing helper tests**

Cover:

- `selectedAssistant` character wins over legacy `selectedCharacter` when both are present.
- Legacy `selectedCharacter` still hydrates a character summary when `selectedAssistant` is null.
- Persona summary includes persona mode and does not expose Scene Director.
- Prompt summary prefers selected prompt title/name over raw id when a prompt record exists.
- Inline custom prompt summary is distinct from quick prompt and selected prompt.
- MCP summary has unavailable, loading, degraded/unhealthy, empty, and available states.
- Provider:model summary keeps provider-qualified route distinct from API model id.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts --reporter=verbose
```

Expected: FAIL because the helper file does not exist.

- [ ] **Step 2: Implement pure summary helpers**

Create helper exports with small pure functions:

```ts
export function buildCockpitAssistantSummary(input: {
  selectedAssistant: AssistantSelection | null | undefined
  selectedCharacter: { id?: string | number; name?: string | null } | null | undefined
  personaMemoryMode?: "read_only" | "read_write" | null
}): RuntimeAssistantSummary

export function buildCockpitPromptSummary(input: {
  selectedSystemPrompt: string | null | undefined
  selectedSystemPromptRecord?: { id?: string; title?: string; name?: string } | null
  selectedQuickPrompt: string | null | undefined
  systemPrompt: string | null | undefined
}): PlaygroundPromptSummary

export function buildCockpitMcpSummary(input: {
  hasMcp: boolean
  healthState: string
  toolsLoading: boolean
  discoveredCount: number
  chatToolCount: number
  disabledReason?: string
}): RuntimeToolSummary
```

Keep helpers UI-free. Return labels/details that existing rail components can render.

- [ ] **Step 3: Wire helpers into `Playground.tsx`**

Replace inline assistant, prompt, MCP, and provider/model summary construction where possible. Keep side effects in `Playground.tsx`; helpers only compute summaries.

- [ ] **Step 4: Run focused helper and cockpit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 5: Update TASK-295 note and commit**

Record that state contracts are now represented by pure helper tests.

```bash
git add apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts apps/packages/ui/src/components/Option/Playground/Playground.tsx "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "test(chat): lock cockpit state contracts"
```

---

### Task 1: P0 Character / Persona Rail Workflow

**P-Series:** P0

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/playground-cockpit-actions.ts`
- Modify: `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- Modify: `apps/packages/ui/src/hooks/useMessageOption.tsx` only if canonical state sync is incomplete.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [ ] **Step 1: Write failing rail tests**

Add tests for:

- Runtime rail shows `Clear assistant` when character selected.
- Runtime rail shows `Clear assistant` when persona selected.
- Clear calls the canonical `setSelectedAssistant(null)` path and does not leave legacy `selectedCharacter` displayed.
- Character state opens selector on Characters tab.
- Persona state opens selector on Personas tab.
- No assistant state opens selector on Characters tab.
- Scene Director only appears for character mode.
- Persona mode explains Scene Director unavailability.
- Focus returns to the triggering rail button after selector close and clear.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --reporter=verbose
```

Expected: FAIL on missing clear, tab-targeting, or focus behavior.

- [ ] **Step 2: Add runtime rail assistant actions**

Extend `PlaygroundRuntimeInspectorProps` with:

```ts
onClearAssistant?: () => void
onInspectAssistant?: () => void
assistantSelectTab?: "character" | "persona"
```

Use existing button vocabulary. Do not add a new character-management surface inside `/chat`.

- [ ] **Step 3: Wire canonical state in `Playground.tsx`**

Use `selectedAssistant` as the rail-facing source. For legacy `selectedCharacter`, only use it as fallback hydration. Clear must call the canonical assistant setter and clear any incompatible legacy character mirror through the existing shared state path.

- [ ] **Step 4: Fix selector tab and return-focus payload**

Extend assistant-select event detail if needed:

```ts
export type AssistantSelectOpenDetail = {
  tab?: AssistantSelectTab
  source?: string
  returnFocusSelector?: string
}
```

`AssistantSelect` should return focus to `returnFocusSelector` when it closes after selection or Escape.

- [ ] **Step 5: Add inspect/manage path**

If the selected assistant is a character, use existing character/actor settings or character-library route only as an inspect/manage escape hatch. If the selected assistant is a persona, use the existing persona route only as an inspect/manage escape hatch. Keep Scene Director character-only.

- [ ] **Step 6: Verify component and persona send tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 7: Extend real-server proof**

In `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`, add a P0 assistant flow:

- Create or reuse disposable real character data if the running API/UI supports it.
- Open the runtime rail selector from `/chat`.
- Select the assistant.
- Verify rail and composer agree.
- Send a real chat turn.
- Clear assistant from rail.
- Verify rail and composer agree no assistant is selected.

If real data cannot be created or listed, assert the real empty state and record the blocker in TASK-295. Do not treat empty-state-only proof as P0 complete without maintainer approval.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground apps/packages/ui/src/components/Common/AssistantSelect.tsx apps/packages/ui/src/hooks/useMessageOption.tsx apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): complete cockpit assistant rail workflow"
```

---

### Task 2: P0 Prompt Rail Workflow

**P-Series:** P0

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/playground-cockpit-actions.ts`
- Modify: `apps/packages/ui/src/utils/prompt-select-events.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx` only for shared prompt selector return-focus behavior.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [ ] **Step 1: Write failing prompt rail tests**

Cover:

- Selected library prompt shows user-facing title when the prompt record exists.
- Selected library prompt does not show raw ID as primary rail label.
- Quick prompt and inline custom system prompt are distinct.
- Clear prompt clears selected template, quick prompt, and inline system prompt.
- Clear prompt does not clear files, knowledge, media, research, web search, assistant, model, or MCP state.
- Selector close and clear return focus to the prompt rail trigger.
- Empty/loading/error prompt states have visible, recoverable copy.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose
```

Expected: FAIL where title lookup/focus/empty states are missing.

- [ ] **Step 2: Feed prompt record into rail summary**

Use the existing prompt library query and selected prompt record logic from `PlaygroundForm.tsx` as the pattern. Do not duplicate Dexie reads in every rail component. Prefer a helper input:

```ts
selectedSystemPromptRecord?: { id?: string; title?: string; name?: string } | null
```

- [ ] **Step 3: Extend prompt selector event for focus return**

If needed:

```ts
export type PromptSelectOpenDetail = {
  source?: string
  returnFocusSelector?: string
}
```

Close, Escape, select, and clear should return focus to the launching rail button.

- [ ] **Step 4: Add manage/inspect prompt path**

Use existing prompt-library navigation or existing prompt surface where available. The rail should not contain a second full prompt editor.

- [ ] **Step 5: Verify prompt isolation**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 6: Extend real-server prompt proof**

In `chat-cockpit.real-server.spec.ts`:

- Create or reuse a disposable real prompt through the running server/UI if available.
- Select it from the `/chat` rail.
- Send a real chat turn and verify the request reflects prompt context.
- Clear it from the rail and verify prompt context is absent.

If no real prompt can be created or listed, assert the recoverable empty state and record the blocker in TASK-295.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground apps/packages/ui/src/utils/prompt-select-events.ts apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): complete cockpit prompt rail workflow"
```

---

### Task 3: P0 Model & Chat Rail Workflow

**P-Series:** P0

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Common/Settings/CurrentChatModelSettings.tsx` only for return-focus or scoped-setting close behavior.
- Modify: `apps/packages/ui/src/store/model.tsx`
- Modify: `apps/packages/ui/src/store/model-settings-scope.ts`
- Test: `apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- Test: `apps/packages/ui/src/hooks/playground/__tests__/modelSelectorUtils.test.ts`
- Test: `apps/packages/ui/src/hooks/playground/__tests__/useModelSelector.capabilities.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [ ] **Step 1: Write failing scoped model tests**

Cover:

- Provider:model summary shows the active scoped route.
- Inherited/default settings are visually different from explicit overrides.
- A harmless setting override persists for one provider:model without leaking to another route with the same model id.
- Closing `CurrentChatModelSettings` returns focus to the rail trigger.

Run:

```bash
bunx vitest run apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx --reporter=verbose
```

Expected: FAIL for missing inherited/explicit summaries or focus return.

- [ ] **Step 2: Add explicit scope summaries**

Extend rail setting summaries to include `scope` or `source` if needed:

```ts
type RuntimeSettingSummary = {
  label: string
  value: string
  source?: "default" | "override"
}
```

Render inherited/default values quietly and explicit overrides more prominently, without adding decorative noise.

- [ ] **Step 3: Verify selector default scope behavior**

Keep default selector list to configured usable choices. Broader search can discover all models, but choosing a model must preserve provider-qualified identity and scoped settings.

- [ ] **Step 4: Verify real request routing**

In real-server Playwright:

- Select or preserve a configured usable provider:model.
- Open rail model settings.
- Change a harmless local setting where safe.
- Close settings and verify focus return.
- Send a real turn.
- Assert request payload sends the bare model id plus the correct provider route when the UI route is provider-qualified.

- [ ] **Step 5: Run focused tests**

```bash
bunx vitest run apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/hooks/playground/__tests__/modelSelectorUtils.test.ts apps/packages/ui/src/hooks/playground/__tests__/useModelSelector.capabilities.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground apps/packages/ui/src/components/Common/Settings/CurrentChatModelSettings.tsx apps/packages/ui/src/store/model.tsx apps/packages/ui/src/store/model-settings-scope.ts apps/packages/ui/src/hooks/playground apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): prove cockpit model settings scope"
```

---

### Task 4: P0 MCP Rail Workflow

**P-Series:** P0

**Files:**
- Modify: `apps/packages/ui/src/hooks/playground/useMcpToolsControl.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx`
- Test: `apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [ ] **Step 1: Write failing MCP state tests**

Cover:

- Runtime rail does not hardcode MCP as available.
- Unavailable MCP shows unavailable reason and disables impossible actions.
- Unhealthy MCP shows degraded/offline reason.
- Loading MCP shows loading state.
- Empty MCP shows no tools available.
- Available MCP shows discovered/chat-enabled counts where available.
- Tool choice updates shared request-construction state.
- MCP settings close returns focus to rail trigger.

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx --reporter=verbose
```

Expected: FAIL for hardcoded available rail state.

- [ ] **Step 2: Extend MCP control hook outputs**

Add a rail-friendly derived summary to `useMcpToolsControl` or to the new summary helper:

```ts
{
  state: "available" | "unavailable" | "disabled" | "degraded"
  label: string
  detail: string
  counts: {
    discovered: number
    chatEnabled: number
    executable?: number
  }
}
```

- [ ] **Step 3: Wire runtime rail to real MCP state**

Pass actual `hasMcp`, `mcpHealthState`, `mcpToolsLoading`, `discoveredMcpTools`, `chatMcpTools`, `mcpToolCounts`, and disabled reason from the composer path into `PlaygroundRuntimeInspector`.

- [ ] **Step 4: Disable dead-end actions**

When MCP is unavailable or unhealthy, the rail should expose the reason and avoid enabled buttons that open empty/dead-end surfaces. Keep a route to settings only when it can help the user recover.

- [ ] **Step 5: Verify request path**

Add/extend tests proving rail `toolChoice` changes feed the same state used by request construction. Do not add a cockpit-only tool-choice state.

- [ ] **Step 6: Run focused tests**

```bash
bunx vitest run apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 7: Extend real-server MCP proof**

Real-server Playwright should complete an MCP settings flow if available. If MCP is unavailable or degraded on the running server, assert the real unavailable/degraded state and verify the chat flow remains usable.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/hooks/playground/useMcpToolsControl.ts apps/packages/ui/src/components/Option/Playground apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): derive cockpit MCP rail from real tool state"
```

---

### Task 5: P0 Real-Server Gate And Maintainer Approval

**P-Series:** P0 gate

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md`

- [ ] **Step 1: Run P0 focused component/store tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 2: Run P0 real-server Playwright**

Run:

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=$TLDW_E2E_API_KEY bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

Expected: PASS, with no route interception and at least one real chat completion attempt.

- [ ] **Step 3: Record evidence**

Add a TASK-295 note with:

- Commit range.
- Focused test results.
- Real-server Playwright command and outcome.
- Whether assistant/prompt/MCP used populated real data or a recorded blocker.
- Confirmation that PR #1582 remains draft.

- [ ] **Step 4: Ask maintainer for P0 approval**

Do not mark P0 complete yourself. Ask the maintainer to approve or reject P0.

- [ ] **Step 5: Commit tracking evidence**

```bash
git add "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "docs(chat): record cockpit p0 verification"
```

---

### Task 6: P1 Context And Session Rail Workflow

**P-Series:** P1

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/useSelectServerChat.ts` only if stale session state is proven there.
- Modify: `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` only if stale server-chat state is proven there.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [x] **Step 1: Write failing context/session tests**

Cover:

- Context rail lists files, knowledge, media, research, web search, prompt, and assistant/worldbook context where applicable.
- Each clear/remove action clears only its context class.
- Session switch clears or updates stale rail summary before next send.
- Temporary/server/local/history-linked labels match actual state.
- Loading, empty, disabled, degraded, and error states are visible and recoverable.

- [x] **Step 2: Implement missing context summaries and stale-state guard**

Use helper functions for summary construction. Avoid making `PlaygroundContextRail` responsible for state derivation.

- [x] **Step 3: Verify session switching**

Add tests that simulate switching from a context-heavy session to an empty/different session. The rail must not show old context after the switch settles.

- [x] **Step 4: Extend real-server context proof**

Real-server proof must cover web search and at least one available real context class. If no real context class is available, record the blocker.

- [x] **Step 5: Run tests and commit**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx --reporter=verbose
git diff --check
git add apps/packages/ui/src/components/Option/Playground apps/packages/ui/src/hooks/chat "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): complete cockpit context session rail"
```

---

### Task 7: P1 Run Controls, Recovery, And Degraded Health

**P-Series:** P1

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx` only if recovery state is not surfaced.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx` or existing readiness tests only if readiness behavior changes.
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [x] **Step 1: Write failing run-control tests**

Cover:

- Stop is visible and enabled only while streaming.
- Regenerate is visible and enabled only when an assistant response exists and no turn is running.
- Disabled controls explain why an action is unavailable.
- Provider/server error states show recoverable next action.
- Unrelated degraded health permits `/chat` immediately with warnings.
- Chat-blocking readiness blocks or disables send clearly.

- [x] **Step 2: Implement disabled/recovery states**

Keep actions wired to existing `stopStreamingRequest` and `regenerateLastMessage`; do not bypass the request state machine.

- [x] **Step 3: Prove degraded-but-chat-allowed**

Use existing readiness events and real-server health behavior. Do not mock health in merge-critical Playwright proof.

- [x] **Step 4: Run tests and commit**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx --reporter=verbose
git diff --check
git add apps/packages/ui/src/components/Option/Playground apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): harden cockpit run recovery states"
```

---

### Task 8: P1 Keyboard, Focus, And Mobile Workflow Parity

**P-Series:** P1 gate

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: shared selector/modal files only where focus return is owned there.
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [ ] **Step 1: Write failing focus/mobile tests**

Cover:

- Keyboard opens/closes assistant selector, prompt selector, model settings, and MCP settings.
- Focus returns to launch trigger after close, Escape, select, clear, and disabled action.
- Mobile cockpit tabs can complete assistant select/clear, prompt select/clear, model settings, MCP settings, web search, and focus-mode return.
- Mobile composer is not covered by rail panels or dropdowns.

- [ ] **Step 2: Implement focus-return plumbing**

Prefer event detail `returnFocusSelector` for global rail-launched surfaces. Prefer local refs for direct clear/remove actions.

- [ ] **Step 3: Fix mobile clipping or overlay issues**

If dropdowns are clipped inside mobile rail panels, adjust portal/container behavior or panel overflow only for the affected shared surface. Do not solve by moving workflows to the app sidebar.

- [ ] **Step 4: Run P1 focused tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 5: Run P1 real-server Playwright**

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=$TLDW_E2E_API_KEY bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
```

Expected: PASS, including mobile workflow parity.

- [ ] **Step 6: Record P1 evidence and ask approval**

Add TASK-295 note with P1 evidence and ask maintainer for P1 approval. Do not mark P1 complete yourself.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): complete cockpit keyboard mobile workflows"
```

---

### Task 9: P2 IA, Copy, Visual QA, Screenshots, And PR Closeout

**P-Series:** P2 gate

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/playground.json`
- Modify: `apps/packages/ui/src/public/_locales/en/playground.json`
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md`

- [ ] **Step 1: Write visual/copy guard tests where practical**

Cover:

- Rail section order is Runtime, Model & Chat, MCP Tools, Character / Persona, Scoped Settings, Run Controls where that order is applicable.
- Disabled/degraded copy does not imply unrelated degraded subsystems block chat.
- Composer and rail duplicate controls remain state-consistent.
- Locale keys exist in both app and extension public locale mirrors.

- [ ] **Step 2: Polish IA and copy**

Keep product UI restrained and dense. Do not add hero treatments, decorative cards, or marketing copy. Each rail section should answer one of:

- What affects the next reply?
- What runtime/model/tool state is active?
- What can I safely change now?
- What is unavailable and why?

- [ ] **Step 3: Resolve composer/rail duplication intentionally**

Do not remove composer controls unless rail workflow equivalence is proven by P0/P1 tests. De-emphasize duplicates only where it reduces confusion and keeps power-user speed.

- [ ] **Step 4: Capture real-server screenshots**

Use terminal Playwright or an approved browser tool, not Computer Use. Capture:

- Desktop cockpit with actual conversation and visible prompt/context/model/MCP/assistant state.
- Desktop focus mode.
- Mobile cockpit tabs.
- Mobile focus mode.
- Degraded-but-chat-allowed state if the running server exposes degraded health.

Store temporary screenshots under `/private/tmp` unless the maintainer explicitly wants tracked artifacts.

- [ ] **Step 5: Run final focused verification**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx apps/packages/ui/src/store/__tests__/model.scoped-settings.test.ts --reporter=verbose
```

Run:

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=$TLDW_E2E_API_KEY bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --reporter=line
git diff --check
```

- [ ] **Step 6: Record P2 evidence and ask approval**

Add TASK-295 note with:

- P0/P1/P2 status.
- Focused test results.
- Real-server Playwright results.
- Screenshot paths.
- Any known non-chat CI baseline issues.
- Confirmation PR #1582 remains draft until maintainer says otherwise.

Ask maintainer to approve P2. Do not mark PR ready or merge.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground apps/packages/ui/src/assets/locale/en/playground.json apps/packages/ui/src/public/_locales/en/playground.json apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts "backlog/tasks/task-295 - Complete-main-chat-cockpit-functionality-in-PR-1582.md"
git commit -m "feat(chat): polish cockpit p2 readiness"
```

---

## Final Verification Before Any Ready-For-Review Discussion

Only after maintainer approval for P0, P1, and P2:

- [ ] Confirm PR #1582 is still draft.
- [ ] Run focused Vitest command from Task 9.
- [ ] Run real-server Playwright command from Task 9.
- [ ] Run `git diff --check`.
- [ ] Check PR #1582 comments and review threads with `gh pr view` and GraphQL review-thread query.
- [ ] Record final evidence in TASK-295 and PR comment.
- [ ] Ask maintainer whether to keep draft or mark ready.

Do not mark PR #1582 ready without explicit maintainer instruction.
