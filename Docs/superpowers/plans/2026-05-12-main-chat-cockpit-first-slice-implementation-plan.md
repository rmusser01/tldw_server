# Main /chat Cockpit First Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the main WebUI `/chat` cockpit rails and status strip from mostly passive summaries into first-slice direct controls for the existing chat state, while preserving the current composer workflows and focus mode.

**Architecture:** Keep `Playground` as the layout/state coordinator. Add a narrow cockpit action bridge for state owned inside `PlaygroundForm`, and pass controlled props/callbacks for state already available in `Playground`. Rails stay presentational and never maintain separate copies of model, context, session, or runtime state.

**Tech Stack:** React 18, TypeScript, i18next, existing `useMessageOption`/`usePlaygroundPersistence` state, Vitest with Testing Library, Playwright real-server smoke coverage, Next.js WebUI.

---

## Scope Lock

This is only the first implementation slice from `Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md`.

In scope:

- Main WebUI `/chat` only.
- `apps/tldw-frontend/pages/chat/index.tsx` through the shared `Playground` surface.
- Direct controls for web search, Search & Context entry, active context summaries, temporary/saved session state, selected model/provider summary, model settings entry, character/persona entry, streaming/error/degraded status, and status-strip state.
- Tests proving rail controls call the same state paths used by existing composer/dialog controls.
- Real-server browser verification without mocked server data for the merge-critical smoke path.

Out of scope:

- Browser-extension sidepanel/sidebar changes.
- Full replacement of `ComposerToolbar`.
- Full direct-control coverage for compare mode, image generation, voice conversation, advanced parameter presets, MCP execution, or every artifact workflow.
- Drag-resizable or dockable panel systems.
- Full provider-health dashboards outside the current chat turn.
- Repo-wide design-system or backend architecture refactors.

## Carry-Forward Decisions

- Degraded health should permit `/chat` immediately when the degraded subsystem is unrelated to chat, with visible warnings. Chat-critical failures must still be shown as blocking or recoverable errors.
- The model selector should default to configured usable provider/model choices. This first slice exposes the cockpit entry point and selected provider/model state; catalog-wide search, recent/frequent ordering, and selector redesign are later selector work unless touched by this slice.
- Each `provider:model` must keep its own settings. If this slice changes the model settings entry path, add or preserve coverage that proves the settings path remains scoped to the current provider-qualified model identity.

## Current Evidence

- `/chat` routes through `apps/tldw-frontend/pages/chat/index.tsx` into `Playground`.
- `Playground.tsx` already derives rail state from `attachedResearchContext`, `webSearch`, `contextFiles`, `selectedKnowledge`, `ragMediaIds`, `temporaryChat`, `serverChatId`, `historyId`, `selectedModel`, `streaming`, `messages`, `threadSearchOpen`, and `selectedCharacter`.
- `PlaygroundContextRail.tsx` currently renders context/session summaries and dispatches `tldw:open-knowledge-panel` directly.
- `PlaygroundRuntimeInspector.tsx` currently renders ready/streaming, model, character, message count, and dispatches `tldw:open-model-settings` plus `tldw:open-actor-settings` directly.
- `PlaygroundForm.tsx` already listens for `tldw:open-actor-settings`, `tldw:open-model-settings`, and `tldw:open-knowledge-panel`.
- `ComposerToolbar.tsx` already receives shared handlers for temporary chat, Search & Context, and web search.
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` already proves live-server entry and cockpit/focus reachability, but it does not prove enough state-changing cockpit behavior.

## Design Rules

- Do not introduce rail-local copies of state.
- Prefer typed callbacks where the parent already owns the state.
- Use a tiny shared event bridge only where the existing owner is inside `PlaygroundForm`.
- Keep all existing composer controls reachable in cockpit and focus modes.
- Add direct rail controls incrementally. Do not move every toolbar command into the rails in this slice.
- Real-server Playwright coverage must not use `page.route`, mocked payloads, synthetic server responses, or sidepanel/sidebar routes.

## Task 1: Centralize Cockpit Actions

**Goal:** Replace ad hoc string event dispatches with a shared, typed cockpit action module that can be used by rails and listened to by `PlaygroundForm`.

**Files:**

- Add `apps/packages/ui/src/components/Option/Playground/playground-cockpit-actions.ts`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Add `apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts`

**Steps:**

- [ ] Write the action module tests first.
  - Assert `openSearchAndContext({ tab: "search" })` dispatches the existing `tldw:open-knowledge-panel` event with the same detail shape.
  - Assert `openModelSettings()` dispatches the existing `tldw:open-model-settings` event.
  - Assert `openActorSettings()` dispatches the existing `tldw:open-actor-settings` event.
  - Assert `toggleWebSearchFromCockpit()` dispatches one new event for `PlaygroundForm`.
  - Assert `setTemporaryChatFromCockpit(true)` dispatches one new event with `{ next: true }`.
- [ ] Implement `playground-cockpit-actions.ts`.
  - Export event name constants for the three existing events.
  - Export event name constants for `tldw:cockpit-toggle-web-search` and `tldw:cockpit-set-temporary-chat`.
  - Export dispatch helper functions.
  - Guard all dispatches for server-side rendering with `typeof window === "undefined"`.
- [ ] Replace inline dispatch strings in `PlaygroundContextRail.tsx` and `PlaygroundRuntimeInspector.tsx` with helpers from the new module.
- [ ] In `PlaygroundForm.tsx`, replace existing listener strings with event constants.
- [ ] In `PlaygroundForm.tsx`, add listeners for the new cockpit web-search and temporary-chat events.
  - Web-search listener calls the existing `handleToggleWebSearch`.
  - Temporary-chat listener calls the existing `handleToggleTemporaryChat(next)` path, not raw persistence state.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts
```

**Expected Outcome:** Cockpit event wiring is centralized, existing open-dialog behavior still uses the same events, and new rail-owned toggles call `PlaygroundForm` state handlers rather than mutating duplicate rail state.

## Task 2: Make Context Rail a First-Slice Control Surface

**Goal:** Add direct, accessible context/session controls to the left rail without replacing the composer toolbar.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Add `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

**Component API:**

Add controlled props to `PlaygroundContextRail`:

```ts
webSearch: boolean;
onToggleWebSearch: () => void;
temporaryChat: boolean;
onToggleTemporaryChat: (next: boolean) => void;
contextCounts: {
  files: number;
  knowledge: number;
  media: number;
  research: number;
};
onOpenSearchContext: () => void;
```

**Steps:**

- [ ] Write `PlaygroundContextRail.first-slice.test.tsx` first.
  - Assert the web-search control renders with `aria-pressed="false"` and calls `onToggleWebSearch`.
  - Assert the web-search control renders with `aria-pressed="true"` when `webSearch` is true.
  - Assert Search & Context calls `onOpenSearchContext`.
  - Assert temporary/saved session control calls `onToggleTemporaryChat(!temporaryChat)`.
  - Assert file, knowledge, media, and research counts render from `contextCounts`.
  - Assert the rail still communicates empty context without hiding the Search & Context action.
- [ ] Implement the rail markup using existing visual classes.
  - Use buttons/toggles, not static labels, for web search and temporary/saved state.
  - Keep compact text density matching current rails.
  - Preserve existing section labels and `data-testid="playground-context-rail"`.
- [ ] Wire `Playground.tsx` to pass current `webSearch`, `temporaryChat`, and context counts.
  - `onToggleWebSearch` dispatches `toggleWebSearchFromCockpit()`.
  - `onToggleTemporaryChat` dispatches `setTemporaryChatFromCockpit(next)`.
  - `onOpenSearchContext` dispatches `openSearchAndContext({ tab: "search" })`.
- [ ] Extend `Playground.cockpit-controls.test.tsx`.
  - Click the context-rail web-search button and assert the cockpit action event fires.
  - Click the temporary/saved session control and assert the cockpit action event includes the expected next value.
  - Confirm existing context summaries still render for active web/files/knowledge/media state.
- [ ] Extend `Playground.cockpit-a11y.test.tsx`.
  - Assert the new controls are reachable by role/name.
  - Assert the web-search button exposes pressed state.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

**Expected Outcome:** The left rail can toggle web search, open Search & Context, and switch temporary/saved session intent through the same owner state used by the composer.

## Task 3: Make Runtime Rail an Operational Inspector

**Goal:** Make the right rail clearer and more actionable for runtime/model/persona state in the first slice.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Add `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

**Component API:**

Add or normalize props:

```ts
selectedProvider: string | null | undefined;
selectedModel: string | null | undefined;
providerRouteLabel: string | null;
runtimeStatus: "ready" | "streaming" | "error" | "degraded";
runtimeStatusDetail?: string | null;
onOpenModelSettings: () => void;
onOpenCharacterSettings: () => void;
```

**Steps:**

- [ ] Write `PlaygroundRuntimeInspector.first-slice.test.tsx` first.
  - Assert provider and model render as separate fields when both are known.
  - Assert raw selected model still renders when provider is unknown.
  - Assert model settings calls `onOpenModelSettings`.
  - Assert character/persona settings calls `onOpenCharacterSettings`.
  - Assert streaming status is visually and semantically distinct from ready.
  - Assert degraded/error detail text renders when passed.
- [ ] Implement provider/model display.
  - Parse provider-qualified `provider:model` labels only for display if no explicit provider value exists.
  - Keep the original selected model value available for exact debugging copy.
  - Do not change provider selection or provider routing behavior in this task.
- [ ] Wire `Playground.tsx`.
  - Derive a display provider from the selected model/provider data already available in the component.
  - Pass `openModelSettings()` and `openActorSettings()` helpers.
  - Pass the existing streaming state and any existing degraded/error state already available in `/chat`.
- [ ] Extend `Playground.cockpit-controls.test.tsx`.
  - Assert model settings and character controls still dispatch through shared actions.
  - Assert provider/model summary appears in the runtime rail.
  - If the model settings entry path changes, assert it preserves provider-qualified identity for same-model-id cases.
- [ ] Extend `Playground.cockpit-a11y.test.tsx`.
  - Assert runtime rail controls have stable role/name labels.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

**Expected Outcome:** The right rail gives first-slice cockpit visibility into the actual model/provider/persona/runtime state, and its actions open the same settings surfaces as the composer.

## Task 4: Upgrade Status Strip for First-Slice State

**Goal:** Make the status strip report current chat operational state without becoming another command bar.

**Files:**

- Update `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Add or extend `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
- Update `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

**Component API:**

Extend status strip props:

```ts
selectedProvider?: string | null;
contextSummary?: string[];
temporaryChat?: boolean;
degradedChecks?: string[];
errorMessage?: string | null;
```

**Steps:**

- [ ] Write status-strip tests first.
  - Assert ready, streaming, degraded, and error states render distinctly.
  - Assert selected provider/model, context-active state, session persistence, and message count render together without replacing each other.
  - Assert long provider/model names do not remove accessible labels.
- [ ] Implement compact status tokens using existing status-strip visual style.
- [ ] Wire `Playground.tsx` to pass context summary, session state, and available degraded/error state.
- [ ] Avoid adding status-strip actions in this slice unless the action already exists and is fully wired elsewhere.

**Verification:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

**Expected Outcome:** The strip answers "what is active right now?" for runtime, model/provider, context, persistence, and degraded/error state.

## Task 5: Real-Server /chat Smoke Coverage

**Goal:** Prove the first-slice cockpit controls work against the actual running server and not mocked browser data.

**Files:**

- Update `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

**Rules:**

- Do not use `page.route`.
- Do not intercept `/api` responses.
- Do not inject synthetic provider/model/chat payloads.
- Do not test sidepanel/sidebar routes.
- Do not use Computer Use for this verification path.
- Use the real server configured by `TLDW_E2E_SERVER_URL`.

**Steps:**

- [ ] Add a source guard in the spec, or a nearby unit guard, that fails if `page.route(` appears in `chat-cockpit.real-server.spec.ts`.
- [ ] Extend the desktop real-server test.
  - Enter `/chat`.
  - Confirm degraded warnings are warning-only when the app admits `/chat`.
  - Click the cockpit web-search toggle.
  - Assert the rail and status strip reflect the changed web-search state.
  - Toggle focus mode, return to cockpit mode, and assert the state is still visible.
  - Send or attempt one chat message through the real server.
  - Accept either a visible assistant response or a recoverable real provider/server error. Do not treat fabricated success as passing.
- [ ] Extend the mobile real-server test.
  - Enter focus mode.
  - Reopen cockpit panels.
  - Expand the mobile context rail.
  - Toggle the same first-slice control and verify composer usability remains intact.
- [ ] If the real server has no usable configured model, assert the visible recoverable no-model/provider state and record that in the test name or assertion message.

**Verification:**

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_SERVER_URL=http://127.0.0.1:8000 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_WEB_URL=http://localhost:18014 TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014' bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line
```

**Expected Outcome:** The first-slice cockpit controls are proven in a real `/chat` browser session, and the test output clearly distinguishes real assistant success from recoverable real server/provider failure.

## Task 6: Regression Verification and Handoff

**Goal:** Finish the slice with focused tests, clean documentation, and explicit skips.

**Files:**

- Update `Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md` only if implementation reveals a scope correction.
- Update the related Backlog task with touched files, verification output, and skips.

**Steps:**

- [ ] Run the focused Vitest suite.

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

- [ ] Run the real-server Playwright smoke command from Task 5.
- [ ] Run repository diff checks.

```bash
git diff --check
git diff --cached --check
```

- [ ] Run frontend type/lint checks if they are practical in the current PR worktree. If repo-wide baseline noise blocks them, record the exact blocker and keep focused tests green.
- [ ] Bandit is not applicable if the implementation only touches TypeScript/Playwright/frontend docs. If Python files are touched, run Bandit on the touched Python paths from the activated `.venv`.
- [ ] Update the Backlog task final summary with:
  - What first-slice controls were implemented.
  - Which existing composer workflows remain unchanged.
  - Real-server evidence and whether the server produced an assistant response or a recoverable real error.
  - Any explicit follow-up items for later cockpit slices.

**Expected Outcome:** The first slice is reviewable as a narrow `/chat` cockpit control improvement with tests, real-server evidence, and no sidepanel/sidebar drift.

## Non-Goals for This Plan

- No sidepanel or extension sidebar parity work.
- No new backend capability contract for chat readiness unless it already exists and is cheap to consume.
- No full configured-model catalog redesign.
- No new MCP tool execution UX.
- No migration of all toolbar controls into rails.
- No visual redesign beyond the needed direct controls and status clarity.

## Risks and Mitigations

- **Risk:** Custom events become another hidden state channel.
  **Mitigation:** Keep the event bridge tiny, typed, tested, and only for `PlaygroundForm`-owned handlers that cannot be cleanly passed through `Playground` yet.

- **Risk:** The rail duplicates composer controls but drifts behavior.
  **Mitigation:** Rail controls dispatch to the existing composer owner handlers. Tests assert calls and visible shared state changes.

- **Risk:** Real-server chat send is flaky because providers may be unconfigured.
  **Mitigation:** The Playwright test accepts a visible, recoverable real provider/server error as valid evidence, but only if the request path used the real server.

- **Risk:** The first slice is mistaken for full cockpit completion.
  **Mitigation:** Keep the PR description and Backlog summary explicit that compare mode, image generation, voice, MCP execution, advanced presets, independent rail collapse, and full provider-health dashboards remain later slices.

## Review Checklist

- [ ] Plan/task scope says "first slice" and "main `/chat` only".
- [ ] No implementation task edits sidepanel/sidebar code.
- [ ] Every new rail control uses shared state or the centralized cockpit action bridge.
- [ ] Tests fail before implementation and pass after implementation.
- [ ] Real-server Playwright path uses the running server and no mocked API responses.
- [ ] Focus mode remains chat-first and composer controls remain reachable.
- [ ] Cockpit mode exposes direct first-slice controls without hiding the composer.
