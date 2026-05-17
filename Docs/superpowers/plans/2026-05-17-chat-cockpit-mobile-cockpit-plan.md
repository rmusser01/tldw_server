# Main Chat Mobile Cockpit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete PR6 of the main `/chat` cockpit maturity roadmap by making mobile cockpit/focus behavior deliberate, accessible, and visually provable without adding bottom controls or touching extension sidepanel/sidebar code.

**Architecture:** Keep the existing `PlaygroundCockpitShell` as the owner of mobile cockpit layout. Preserve the current chat pipeline, rail visibility state, mobile panel state, and existing rail components; this slice only hardens mobile panel semantics, escape flow, and mobile proof coverage.

**Tech Stack:** React, TypeScript, Tailwind utility classes, Vitest + Testing Library, Playwright real-server E2E, Backlog.md.

---

## File Structure

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
  - Owns mobile panel tabs, tabpanel rendering, focus/focus-mode transition controls, and no-bottom layout guarantees.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`
  - Locks mobile tab/tabpanel semantics, panel switching, and draft-surface visibility.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
  - Locks keyboard activation, `aria-controls` validity, and focus-mode escape controls.
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Expands real-server mobile proof screenshots and interaction coverage.
- Modify: `backlog/tasks/task-416 - Implement-main-chat-mobile-cockpit-slice.md`
  - Records implementation notes, verification, and final status.

No extension sidepanel/sidebar files are in scope. Do not introduce bottom bars, bottom summaries, or composer-adjacent replacement controls.

---

### Task 1: Lock Mobile Tab Semantics Before Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

- [x] **Step 1: Add failing tabpanel existence coverage**

Add assertions to the existing controlled mobile panel test proving each visible tab's `aria-controls` id exists in the DOM even when that panel is inactive.

Expected test shape:

```tsx
const contextTab = within(mobilePanels).getByRole("tab", { name: "Context" })
const runtimeTab = within(mobilePanels).getByRole("tab", { name: "Runtime" })
expect(document.getElementById(contextTab.getAttribute("aria-controls") ?? "")).not.toBeNull()
expect(document.getElementById(runtimeTab.getAttribute("aria-controls") ?? "")).not.toBeNull()
```

- [x] **Step 2: Add failing hidden/inactive panel coverage**

In the same test, assert that the inactive runtime panel exists but is hidden from the accessibility tree while the active context panel remains visible.

- [x] **Step 3: Run red tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx --config vitest.config.ts
```

Expected: FAIL because inactive mobile tabs currently point at unmounted tabpanels.

---

### Task 2: Keep Mobile Tabpanels Mounted And Controlled

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`

- [x] **Step 1: Add stable mobile tab ids**

Inside `PlaygroundCockpitShell`, define stable ids for context/runtime tabs and panels. Keep names local to the shell.

- [x] **Step 2: Render both available mobile tabpanels**

For each visible rail, render its matching `<section role="tabpanel">` with:

- stable `id`
- `aria-labelledby` pointing at the tab
- `aria-describedby="playground-mobile-panel-summary"`
- `hidden={visibleMobilePanel !== "..."}`
- `aria-hidden={visibleMobilePanel !== "..."}` if the existing test pattern needs explicit state

Do not move the panel below the composer. Do not introduce bottom UI.

- [x] **Step 3: Preserve current layout constraints**

Keep the existing mobile rail container above the main chat surface, `lg:hidden`, and `max-h` scroll containment. If spacing changes are necessary, keep them to the mobile rail panel only.

- [x] **Step 4: Run green tests**

Run:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx --config vitest.config.ts
```

Expected: PASS.

---

### Task 3: Add A Mobile In-Panel Return-To-Chat Control

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

- [x] **Step 1: Add failing UX coverage**

Assert that the mobile panel area exposes a keyboard-reachable control using panel-local focus copy, and that activating it calls `onModeChange("focus")`.

- [x] **Step 2: Implement a small panel-local action row**

Inside the mobile rail container, add a compact top row with:

- panel state text or reuse the existing summary below
- a small button that calls `onModeChange("focus")`
- visible label `Focus`, accessible name `Return to focus chat` to distinguish the panel-local action from the header `Enter focus chat` action

Keep this control inside the mobile panel area, not in a bottom bar.

- [x] **Step 3: Preserve header control behavior**

The existing header mode toggle and rail visibility buttons must keep their current names, pressed state, and handlers.

- [x] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --config vitest.config.ts
```

Expected: PASS.

---

### Task 4: Expand Real-Server Mobile Proof

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [x] **Step 1: Extend the existing mobile test**

Expand `keeps mobile cockpit tabs and focus composer usable against the live server` to prove:

- context and runtime tabpanels both have valid `aria-controls` targets
- inactive panel target exists while hidden
- in-panel focus control returns to focus mode
- draft survives panel switching, dialogs, and return to focus mode
- no `playground-collapsed-composition-summary` or `composer-bottom-bar` appears

- [x] **Step 2: Add screenshot checkpoints**

Keep existing screenshots and add or rename as needed so final evidence covers:

- `chat-cockpit-mobile-context.png`
- `chat-cockpit-mobile-runtime.png`
- `chat-cockpit-mobile-active-draft.png`
- `chat-cockpit-mobile-focus.png`

- [x] **Step 3: Run real-server proof only when the real server is available**

Use the already-running server if present. Do not start or use a mocked server. Do not use `page.route`, `route.fulfill`, or synthetic backend payloads.

Command from `apps/tldw-frontend`:

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY="$TLDW_E2E_API_KEY" bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "keeps mobile cockpit tabs|sends a real mobile" --reporter=line
```

Expected: PASS against the real running server, or document the exact server/unavailable blocker.

---

### Task 5: Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-416 - Implement-main-chat-mobile-cockpit-slice.md`

- [x] **Step 1: Run focused Vitest**

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --config vitest.config.ts
```

- [x] **Step 2: Run broader adjacent cockpit coverage**

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --config vitest.config.ts
```

- [x] **Step 3: Run formatting/diff checks**

From repo root:

```bash
git diff --check
```

If locale keys changed, also run the existing locale mirror verification used by prior cockpit tasks.

- [x] **Step 4: Record Bandit applicability**

Bandit is skipped if this slice touches only TS/TSX/E2E/Markdown. If any Python file changes, run Bandit on the touched Python scope.

- [x] **Step 5: Update TASK-416**

Mark acceptance criteria and DoD complete only after verification. Include:

- focused Vitest results
- real-server Playwright result or exact blocker
- screenshot paths if captured
- no-bottom/no-sidepanel scope note

- [x] **Step 6: Commit**

```bash
git add Docs/superpowers/plans/2026-05-17-chat-cockpit-mobile-cockpit-plan.md \
  apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts \
  "backlog/tasks/task-416 - Implement-main-chat-mobile-cockpit-slice.md"
git commit -m "Polish mobile chat cockpit flow"
```
