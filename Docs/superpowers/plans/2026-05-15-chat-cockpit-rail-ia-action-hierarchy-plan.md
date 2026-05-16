# Main /chat Rail IA and Action Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the main `/chat` cockpit rails into predictable work surfaces so users can quickly find prompt, context, session, model, assistant, tools, and recovery controls.

**Architecture:** Keep `Playground` as the coordinator and keep rails presentational. Reuse the PR 1 composition preview and existing summary props; this slice changes grouping, labels, hierarchy, and tests, not the source of truth for chat state. Preserve focus mode, mobile rail tabs, shared dialogs, shared handlers, and provider:model identity.

**Tech Stack:** React 18, TypeScript, i18next, existing WebUI design-system tokens, Testing Library/Vitest, Playwright real-server checks.

---

## Scope Lock

In scope:

- Main WebUI `/chat` only.
- `apps/tldw-frontend/pages/chat/index.tsx` through `apps/packages/ui/src/routes/option-chat.tsx` and `Playground`.
- Left rail information architecture:
  - Composition preview
  - Context stack
  - Prompt management
  - Search/RAG sources
  - Files/media/research attachments
  - Session persistence
- Right rail information architecture:
  - Runtime state
  - Model route and settings
  - Assistant/persona/character
  - Tools/MCP
  - Recovery/run controls
- Copy and visible hierarchy for first-time comprehension and power-user scanning.
- Keyboard names and focus behavior around existing buttons and disclosures.

Out of scope:

- Browser-extension sidepanel/sidebar.
- Full model selector redesign.
- Backend API changes.
- MCP Hub lifecycle/policy work.
- New prompt library features.
- New character/persona management surfaces outside the existing selectors/settings paths.
- Replacing the composer or changing send behavior.

## Current Evidence

- `/chat` route renders `Playground`.
- `PlaygroundContextRail.tsx` already contains prompt, context status, context source inventory, web search, Search & Context, count-based clear actions, and session controls.
- `PlaygroundRuntimeInspector.tsx` already contains runtime status, model route, model/chat settings, scoped setting summaries, assistant/persona/character controls, MCP tool policy, stop/regenerate, and timeline count.
- `PlaygroundCompositionPreview.tsx` now provides the first-slice "Next message composition" card in the left rail.
- Focus mode hides rails through `PlaygroundCockpitShell`; PR 2 must not break that behavior.
- Real-server Playwright already has proof points for prompt, persona/character, model settings restore, MCP states, mobile tabs, focus mode, and actual conversation send.

## Design Decisions

- Keep rail group labels concrete: "Composition", "Context stack", "Prompt", "Search & sources", "Session", "Runtime", "Model & Chat", "Assistant", "Tools", "Run controls".
- Put durable configuration and inspection in rails. Keep turn-level draft composition in the composer.
- Prefer existing button names when tests and assistive technology already rely on them. If visible copy changes, keep `aria-label` compatible where practical.
- Use section order as hierarchy. Avoid nested cards and avoid turning the rail into a generic settings dump.
- Keep the PR 1 composition preview at the top of the left rail because it answers "what will happen next" before the lower controls explain "where to change it".

## File Structure

Modify:

- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
  - Responsibility: left-rail grouping, headings, summaries, and existing context/prompt/session controls.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
  - Responsibility: right-rail grouping, headings, summaries, and existing runtime/model/assistant/tools/run controls.
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
  - Responsibility: focused left-rail IA and action-hierarchy assertions.
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
  - Responsibility: focused right-rail IA and action-hierarchy assertions.
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
  - Responsibility: integrated rail behavior from `Playground`.
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
  - Responsibility: source-level guard that mobile/focus/cockpit wiring keeps rail surfaces present.
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Responsibility: real-server proof that reorganized rails remain usable without backend route mocking.
- `backlog/tasks/task-393 - Implement-main-chat-rail-information-architecture-and-action-hierarchy.md`
  - Responsibility: task tracking, plan link, verification notes, final summary.

Create only if duplication becomes noisy:

- `apps/packages/ui/src/components/Option/Playground/playground-rail-sections.ts`
  - Responsibility: shared rail class names or tiny presentational section wrappers. Do not move behavior into this file.

## Stage 1: Lock Left-Rail IA With Failing Tests

**Goal:** Prove the desired left-rail grouping before changing production UI.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`

- [ ] **Step 1: Add a test for the ordered left-rail regions**

Test that a populated rail exposes regions/headings in this order:

1. `Next message composition`
2. `Context stack`
3. `Prompt`
4. `Search & sources`
5. `Session`

Use Testing Library queries against visible headings/regions rather than snapshots.

- [ ] **Step 2: Add a test for first-time empty-state comprehension**

Render no prompt, no assistant context, no files, no knowledge, no media, no research, and no web search. Assert the rail still exposes:

- Composition summary
- Prompt selection control
- Search & Context action
- Web search toggle
- Session persistence control

- [ ] **Step 3: Add a test for existing control preservation**

With prompt, files, knowledge, media, research, and web search present, assert the existing clear/open/toggle callbacks still fire exactly once.

- [ ] **Step 4: Run the left-rail tests to verify RED**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --reporter=verbose
```

Expected: FAIL because the current rail does not expose the new IA labels/order.

## Stage 2: Implement Left-Rail Grouping

**Goal:** Reorganize the left rail without changing chat state.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/playground-rail-sections.ts`

- [ ] **Step 1: Introduce explicit left-rail group sections**

Keep `PlaygroundCompositionPreview` first when `compositionPreviewSummary` is supplied. Then render:

- `Context stack`: active source count and source inventory.
- `Prompt`: selected prompt state plus prompt selector and clear.
- `Search & sources`: Search & Context action, web search, files/knowledge/media/research counts and clear actions.
- `Session`: saved/temporary state, session status, history linkage.

- [ ] **Step 2: Preserve existing controls and accessible names**

Keep these existing interactive names working:

- `Select a prompt`
- `Clear prompt`
- `Open Search & Context`
- `Web search`
- `Clear files`
- `Clear knowledge`
- `Clear media scopes`
- `Clear research context`
- `Use temporary chat`
- `Save to history`

- [ ] **Step 3: Keep density restrained**

Use existing rail classes and compact text. Avoid adding nested card structures inside already framed rail sections.

- [ ] **Step 4: Run left-rail tests to verify GREEN**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --reporter=verbose
```

Expected: PASS.

## Stage 3: Lock Right-Rail IA With Failing Tests

**Goal:** Prove the desired right-rail grouping before changing production UI.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`

- [ ] **Step 1: Add a test for the ordered right-rail regions**

Test that a populated runtime rail exposes headings/regions in this order:

1. `Runtime`
2. `Model & Chat`
3. `Assistant`
4. `Tools`
5. `Run controls`

- [ ] **Step 2: Add a test for first-time empty-state comprehension**

Render no model, no assistant, unavailable tools, idle runtime, and no regenerable response. Assert the rail still makes model settings, assistant selection, tool policy, stop reason, and regenerate reason discoverable.

- [ ] **Step 3: Add a test for existing control preservation**

Assert model settings, assistant select, clear assistant, inspect/manage assistant, scene director, MCP settings, tool choice, stop, and regenerate callbacks still use shared props and disabled reasons.

- [ ] **Step 4: Run the right-rail tests to verify RED**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx --reporter=verbose
```

Expected: FAIL because the current rail does not expose the new IA labels/order consistently.

## Stage 4: Implement Right-Rail Grouping

**Goal:** Reorganize runtime controls into a clearer operational hierarchy.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/playground-rail-sections.ts`

- [ ] **Step 1: Keep runtime status first**

Preserve provider/model display and provider route summary in the `Runtime` section.

- [ ] **Step 2: Keep model configuration together**

Keep `Model & Chat` settings and scoped setting summaries together. Preserve provider:model route and setting scope visibility.

- [ ] **Step 3: Keep assistant controls together**

Group assistant/persona/character state, select/change, clear, inspect/manage, and character-only Scene Director behavior in `Assistant`.

- [ ] **Step 4: Keep tool policy together**

Group MCP summary, tool choice segmented buttons, and MCP settings in `Tools`. Keep unavailable/disabled/degraded copy distinct.

- [ ] **Step 5: Keep recovery/run controls together**

Move stop/regenerate/timeline recovery affordances into `Run controls` while preserving disabled reason descriptions and callback behavior.

- [ ] **Step 6: Run right-rail tests to verify GREEN**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx --reporter=verbose
```

Expected: PASS.

## Stage 5: Integrated Playground, Responsive, and Real-Server Proof

**Goal:** Prove the reorganized rails work through the real `/chat` page and existing state wiring.

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-393 - Implement-main-chat-rail-information-architecture-and-action-hierarchy.md`

- [ ] **Step 1: Add integrated `Playground` assertions**

Assert the rendered main cockpit exposes the new left/right rail groups while still reflecting selected prompt, assistant/persona/character, provider:model, scoped settings, and MCP state.

- [ ] **Step 2: Add focus-mode regression assertion**

Assert focus mode hides the reorganized rails and returning to cockpit restores the rail groups.

- [ ] **Step 3: Add real-server Playwright assertions**

In `chat-cockpit.real-server.spec.ts`, assert:

- Desktop cockpit contains the new rail groups.
- Prompt selection still appears under `Prompt` and in the composition preview.
- Character/persona selection still appears under `Assistant` and in the composition preview.
- MCP/tool state appears under `Tools`.
- Mobile context/runtime tabs expose the same group labels.
- No `page.route` backend mocking is introduced.

- [ ] **Step 4: Run focused unit/component verification**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 5: Run real-server Playwright verification**

Use the already-running real backend. Do not start a duplicate backend if port 8000 is already occupied.

```bash
cd apps/tldw-frontend
KEY=$(awk -F= '/^SINGLE_USER_API_KEY=/{print substr($0,index($0,"=")+1); exit}' /Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/Config_Files/.env)
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY="$KEY" bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Run design-system and whitespace verification**

```bash
cd apps/packages/ui
bun run verify:design-system-state
cd ../../..
git diff --check
```

Expected: PASS. Existing design-system baseline exceptions are acceptable if unchanged.

- [ ] **Step 7: Record verification and complete `TASK-393`**

Update the Backlog task with implementation notes, checked acceptance criteria, checked DoD items, and final summary.

## Definition of Done

- [ ] Main `/chat` left rail uses the PR 2 group hierarchy.
- [ ] Main `/chat` right rail uses the PR 2 group hierarchy.
- [ ] Existing controls and shared handlers still work.
- [ ] Keyboard-accessible names and focus behavior are preserved.
- [ ] Focus mode remains chat-first and hides rails.
- [ ] Mobile rail tabs expose the same organization without composer occlusion.
- [ ] Focused Vitest coverage passes.
- [ ] Real-server Playwright coverage passes against the running server.
- [ ] No sidepanel/sidebar files are touched.
- [ ] Bandit is run if Python files are touched; otherwise the skip is recorded.
