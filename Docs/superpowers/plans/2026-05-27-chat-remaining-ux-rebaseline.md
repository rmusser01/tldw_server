# Chat Remaining UX Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining route-scoped `/chat` UX rebaseline work after cockpit rails were restored, without reintroducing the removed standalone `CharacterControlRail`.

**Architecture:** Preserve the existing `Playground` cockpit shell, context rail, runtime rail, sticky composer, and extension sidepanel contracts. Make small, test-first changes to the existing mobile cockpit, persistence feedback, and rail-copy surfaces, then rerun the `/chat` UX audit from the corrected rail-enabled page.

**Tech Stack:** React, TypeScript, Tailwind utility classes, Ant Design notifications, Vitest, Testing Library, Playwright, Backlog.md.

---

## Scope

In scope:

- `/chat` WebUI cockpit and focus modes.
- Direct extension sidepanel handoff into `/chat`.
- Mobile cockpit density, composer reachability, server-save feedback, error visibility, and remaining rail-label clarity.
- Focused regression coverage and refreshed UX evidence.

Out of scope for this plan:

- Broad app navigation redesign.
- New chat architecture.
- Reintroducing `CharacterControlRail` or any standalone character rail.
- Sidebar/history redesign and model selector contract redesign, except to record them as follow-up work if still observed during the re-audit.

## Backlog Ownership

- Active task: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`
- Related completed task with same numeric ID: `backlog/completed/task-521 - Fix-chat-UX-rebaseline-false-setup-and-handoff-affordances.md`
- Use the active task path when recording plan and verification notes because numeric IDs can collide between active and completed storage.

## File Map

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
  - Owns cockpit/focus mode, desktop rails, mobile rail tabs, mobile rail panel height, and shell-level accessibility labels.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
  - Existing shell accessibility and keyboard coverage. Extend it only for new mobile panel affordances or label changes.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
  - Existing cockpit shell integration coverage. Extend it for density-state and persistence-state contracts if needed.
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Existing real-server desktop/mobile cockpit coverage. Extend it for viewport overlap, toast/error visibility, and final screenshot evidence.
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx`
  - Owns the current "Chat now saved on server" success notification and inline server persistence hint state.
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx`
  - Existing persistence hook tests. Extend for notification suppression/shortening and inline hint behavior.
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
  - Renders the inline server persistence hint; keep changes small and responsive.
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
  - Remaining repeated context labels and empty assistant/context copy.
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
  - Remaining repeated runtime/MCP labels and provider/model state copy.
- Modify: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
  - Append the post-fix re-audit summary and mark which remaining findings were addressed or deferred.
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`
  - Record plan path, implementation notes, verification results, and final summary.

## Known Baseline Constraints

- `bunx tsc --noEmit --project tsconfig.json --pretty false` currently fails on an unrelated baseline issue in `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)`: `"comfortable"` is not assignable to `GalleryCardDensity`.
- Backend startup can generate untracked watchlist templates under `tldw_Server_API/Config_Files/templates/watchlists/`. Do not stage them unless this task intentionally changes watchlist templates.
- The previous rail rebaseline already removed the standalone `CharacterControlRail`; all guardrails must assert absence, not presence.

## Task 1: Lock Current Rail And Handoff Guardrails

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx`
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`

- [ ] **Step 1: Inspect existing guard expectations**

Run:

```bash
sed -n '1,220p' apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts
sed -n '1,220p' apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx
```

Expected:

- `Playground.cockpit-regression.guard.test.ts` asserts cockpit shell/context/runtime rails remain wired.
- It asserts `CharacterControlRail` is absent from the main `/chat` cockpit.
- `SidepanelHeaderSimple.fullscreen-route.test.tsx` asserts full-screen and dashboard handoffs target `/chat`.

- [ ] **Step 2: Write or tighten the guard first**

If the guard does not explicitly assert absence of the standalone rail, add:

```ts
expect(source).not.toContain("<CharacterControlRail")
expect(source).not.toContain("CharacterControlRail")
```

If the sidepanel test does not cover both full-screen and dashboard actions, add assertions that both open `/options.html#/chat`.

- [ ] **Step 3: Verify the guard fails for the removed rail if inverted**

Temporarily invert the `CharacterControlRail` absence assertion locally and run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx
```

Expected: the guard fails because the standalone rail is absent.

Restore the correct assertion immediately after confirming RED.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx
```

Expected: both focused guard files pass.

- [x] **Step 5: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx \
  "backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md"
git commit -m "test(chat): preserve restored rail contracts"
```

Skip commit if no file changes were needed; record the no-op verification in the Backlog task instead.

## Task 2: Reduce Mobile Cockpit Density Without Changing The Rail Model

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`

- [ ] **Step 1: Write failing mobile shell expectations**

Add a focused test to `Playground.cockpit-a11y.test.tsx` or `Playground.cockpit-shell.test.tsx` that renders `PlaygroundCockpitShell` in cockpit mode and asserts:

```ts
const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails")
expect(mobileRails).toHaveAttribute("data-mobile-panel", "context")

const contextPanel = screen.getByRole("tabpanel", { name: "Context" })
expect(contextPanel.className).toContain("max-h-[30vh]")
expect(contextPanel.className).not.toContain("max-h-[42vh]")

expect(
  screen.getByTestId("playground-cockpit-mobile-panel-summary"),
).toHaveClass("sr-only")
expect(
  screen.getByTestId("playground-cockpit-mobile-panel-summary"),
).toHaveTextContent("Context panel active. Composer draft remains available below.")
```

Expected RED: the test fails because current panels use `max-h-[42vh]` and the summary is visible.

- [ ] **Step 2: Run RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx
```

Expected: the new expectation fails for the current mobile panel class and visible summary.

- [ ] **Step 3: Implement minimal density changes**

In `PlaygroundCockpitShell.tsx`:

- Keep the mobile tablist and `role="tabpanel"` structure.
- Keep `data-testid="playground-cockpit-mobile-rails"` and `data-testid="playground-cockpit-mobile-panel-summary"`.
- Change the visible mobile summary paragraph to screen-reader-only copy:

```tsx
className="sr-only"
```

- Reduce each mobile rail panel from:

```tsx
className={`max-h-[42vh] overflow-y-auto rounded-md border border-border bg-surface p-2 ${
```

to:

```tsx
className={`max-h-[30vh] overflow-y-auto rounded-md border border-border bg-surface p-2 ${
```

Do not change desktop rail widths in this task.

- [ ] **Step 4: Extend real-server mobile overlap checks**

In `chat-cockpit.real-server.spec.ts`, extend the existing mobile cockpit test around the screenshots near the mobile context/runtime assertions. Add a helper if not already present:

```ts
const expectNoVerticalOverlap = async (
  first: Locator,
  second: Locator,
  label: string
) => {
  const firstBox = await first.boundingBox();
  const secondBox = await second.boundingBox();
  expect(firstBox, `${label}: first box`).not.toBeNull();
  expect(secondBox, `${label}: second box`).not.toBeNull();
  expect(firstBox!.y + firstBox!.height).toBeLessThanOrEqual(secondBox!.y + 1);
};
```

Use it after showing mobile cockpit panels:

```ts
await expectNoVerticalOverlap(
  mobileRails,
  page.getByTestId('chat-input'),
  'mobile cockpit rails should not overlap composer'
);
```

Also assert the visible panel height is bounded:

```ts
const panelBox = await contextPanelTarget.boundingBox();
expect(panelBox?.height ?? 0).toBeLessThanOrEqual(260);
```

Expected RED before implementation if the current `42vh` panel exceeds the new height target.

- [ ] **Step 5: Run GREEN**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx
```

Expected: focused shell/a11y tests pass.

Run real-server mobile cockpit only when the backend and WebUI are running:

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts \
  --grep "keeps mobile cockpit tabs and focus composer usable"
```

Expected: mobile cockpit test passes and captures updated mobile screenshots.

- [ ] **Step 6: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx \
  apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts \
  "backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md"
git commit -m "fix(chat): reduce mobile cockpit rail density"
```

## Task 3: Replace Blocking Server-Save Notification With Inline Feedback

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`

- [x] **Step 1: Write failing persistence hook test**

In `usePlaygroundPersistence.test.tsx`, add or update a test for first server save:

```ts
expect(notificationApi.success).not.toHaveBeenCalled()
expect(result.current.showServerPersistenceHint).toBe(true)
```

The test should still assert:

```ts
expect(setServerPersistenceHintSeen).toHaveBeenCalledWith(true)
```

Expected RED: current hook calls `notificationApi.success` with "Chat now saved on server".

- [x] **Step 2: Run RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx
```

Expected: the new notification suppression assertion fails.

- [x] **Step 3: Implement minimal feedback change**

In `usePlaygroundPersistence.tsx`, remove the `notificationApi.success({ message: "Chat now saved on server", ... })` call from the first-save path. Keep:

```ts
serverPersistenceHintSeenRef.current = true
setServerPersistenceHintSeen(true)
setShowServerPersistenceHint(true)
```

Do not remove error notifications in the catch path.

- [x] **Step 4: Keep inline hint concise on mobile**

If the inline hint in `ComposerToolbar.tsx` still creates mobile crowding, change only the copy length or responsive visibility:

```tsx
<p className="max-w-md text-xs text-text-muted sm:max-w-lg">
```

Avoid adding a new toast, modal, or second persistence banner.

- [x] **Step 5: Add real-server visibility assertion**

In the mobile send test in `chat-cockpit.real-server.spec.ts`, after `assertChatCompletionRenderedOrRecoverable(page, null)`, assert that the composer and any error/recovery card are not covered by an Ant Design notification:

```ts
await expect(page.locator('.ant-notification-notice').filter({
  hasText: 'Chat now saved on server',
})).toHaveCount(0);
await expect(page.getByTestId('chat-input')).toBeVisible();
```

- [ ] **Step 6: Run GREEN**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx
```

Then, when local backend/WebUI are running:

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts \
  --grep "sends a real mobile focus conversation"
```

Expected: focused persistence tests pass; mobile send flow shows no blocking server-save notification.

- [ ] **Step 7: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx \
  apps/packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx \
  apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts \
  "backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md"
git commit -m "fix(chat): keep server save feedback inline"
```

Only stage `ComposerToolbar.tsx` if it was changed.

## Task 4: Clean Remaining Rail Label Duplication

**Files:**

- Modify if needed: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`

- [x] **Step 1: Write failing accessible-name tests for the duplicated content only**

Use `Playground.cockpit-a11y.test.tsx` to add targeted assertions for the actual duplicated labels still observed after Tasks 2 and 3. Do not duplicate existing shell tests for:

- `Hide context rail`
- `Hide runtime rail`
- `Collapse context sidechannel`
- `Collapse runtime sidechannel`

Example assertions:

```ts
expect(screen.getAllByText("No assistant selected")).toHaveLength(1)
expect(screen.getAllByRole("heading", { name: "MCP tools" })).toHaveLength(1)
```

Expected RED only if those duplicates still exist. If live inspection shows the duplicates have already been resolved by other branch changes, skip this task and record the no-op.

- [x] **Step 2: Run RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

Expected: the new duplicate assertion fails if duplicates remain.

- [x] **Step 3: Implement copy/label cleanup**

Prefer these fixes:

- Keep one visible heading per section.
- Convert repeated explanatory text to shorter state copy.
- Use `aria-label` or `aria-describedby` only when it improves identification.
- Do not hide meaningful controls from assistive tech.

Avoid:

- Removing functional controls.
- Reusing the same label for multiple different controls.
- Moving rail content to a modal.

- [x] **Step 4: Run GREEN**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx
```

Expected: focused a11y tests pass.

- [x] **Step 5: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx \
  apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts \
  "backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md"
git commit -m "fix(chat): clarify rail state labels"
```

Only stage files that changed.

## Task 5: Re-Run Corrected `/chat` UX Evaluation

**Files:**

- Modify: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
- Add or update screenshots under: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`
- Modify: `backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md`

- [x] **Step 1: Start local backend and WebUI**

Use the established project startup commands for this worktree. Record exact ports in the review document.

- [x] **Step 2: Capture required evidence**

Capture:

- Desktop `/chat` cockpit.
- Desktop `/chat` focus.
- Mobile `/chat` focus.
- Mobile `/chat` cockpit context panel.
- Mobile `/chat` cockpit runtime panel.
- Mobile send success or recoverable error state.
- Extension sidepanel handoff into `/chat`.

- [x] **Step 3: Walk first-time journey**

Evaluate:

- Opening `/chat`.
- Understanding purpose and setup state.
- Model/provider readiness.
- First send.
- Loading/streaming/error/retry.
- Discovering history, context/RAG, persona/tools, and save/resume behavior.

- [x] **Step 4: Walk power-user journey**

Evaluate:

- Resume speed.
- Model/provider/settings switching.
- Persona/character and context workflows.
- Long-session controls.
- Failure/retry handling.
- Extension-to-WebUI continuity.

- [x] **Step 5: Update findings**

In `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`, include:

- Top 5 remaining UX risks.
- Evidence notes.
- First-time walkthrough.
- Power-user walkthrough.
- Severity-ranked table.
- Quick wins.
- Larger improvements.
- Ideal first-time and power-user `/chat` workflows.
- Open questions, assumptions, and non-goals.

Mark sidebar/history and model-selector data-contract work as follow-up tasks unless they remain direct P1 blockers after Tasks 2-4.

Task 5 evidence update:

- WebUI was audited at `http://127.0.0.1:18015` against the existing healthy backend on `http://127.0.0.1:8000`.
- Current `/chat` screenshots now include first-time unseeded, desktop cockpit, desktop focus, mobile focus, mobile cockpit context, mobile cockpit runtime, mobile send blocked state, and extension sidepanel debug route.
- The corrected page has context/runtime cockpit rails restored and the removed standalone `CharacterControlRail` remains absent from captured UI evidence.
- The current first-send path is blocked by inconsistent provider readiness: the page reports `No LLM provider configured` while runtime/model rails still show `tldw:gpt-4o` as ready/active.
- The directly connected sidepanel debug chat route horizontally overflows at 390 px (`documentElement.scrollWidth=420`, `body.scrollWidth=420`).
- Findings were refreshed in `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`; structured evidence was refreshed in `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`.

- [x] **Step 6: Verify no local servers remain running**

Run:

```bash
lsof -nP -iTCP:18015 -sTCP:LISTEN
lsof -nP -iTCP:18016 -sTCP:LISTEN
```

Expected: no listeners after shutdown.

Actual: after stopping the temporary WebUI process, final recheck showed no listeners on `18015` or `18016`. During cleanup, `18016` was briefly occupied by an unrelated Next dev server from `.worktrees/notes-list-reliability` (`node .../notes-list-reliability/apps/tldw-frontend/node_modules/.bin/next dev -p 18016`), so it was left untouched.

- [ ] **Step 7: Commit**

Run:

```bash
git add \
  Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline \
  "backlog/tasks/task-521 - Plan-remaining-chat-UX-rebaseline-slices.md"
git commit -m "docs(chat): refresh corrected chat UX audit"
```

## Final Verification

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx \
  src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx
```

Run from the repository root:

```bash
git diff --check
```

Run TypeScript and record the known baseline result:

```bash
cd apps/packages/ui
NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false
```

Expected current baseline caveat:

- May fail only on `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)` with `Type '"comfortable"' is not assignable to type 'GalleryCardDensity'.`

Current Task 5 verification:

- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx` passed: 5 files, 52 tests.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false` failed only on the known unrelated baseline at `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)`: `Type '"comfortable"' is not assignable to type 'GalleryCardDensity'.`
- Port cleanup: the audit-owned `18015` server was stopped, and final recheck showed no listeners on `18015` or `18016`. An intermediate unrelated `.worktrees/notes-list-reliability` Next dev server was observed on `18016` and left untouched.
- Bandit skipped: this slice touched Markdown, JSON, PNG screenshots, and frontend/Backlog metadata only; no Python files were changed.

Bandit:

- Skip for TS/TSX/docs-only changes and record the skip in Backlog.
- If Python files are touched unexpectedly, run Bandit on the touched Python scope before finalizing.

## Follow-Up Candidates

Create separate Backlog tasks if still observed after the corrected re-audit:

- Sidebar/history layout and search workflow when rails are open.
- Model selector/provider availability contract, especially `any_configured: false` with usable local models.
- Richer response failure copy for provider-specific no-response cases.
