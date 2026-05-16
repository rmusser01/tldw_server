# Chat Cockpit Mode and Session Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the main `/chat` cockpit clearly explain cockpit/focus mode, rail visibility, and conversation persistence/session state without changing chat send behavior.

**Architecture:** Keep `Playground` as the coordinator and keep `PlaygroundCockpitShell`, `PlaygroundContextRail`, and `PlaygroundStatusStrip` presentational. Extend existing summary props and copy rather than adding a new session store. Critical state that could be hidden by collapsed rails must also be visible in the status strip.

**Tech Stack:** React, TypeScript, i18next, existing WebUI design-system tokens, Testing Library/Vitest, Playwright real-server workflow.

---

## Scope Lock

In scope:

- Main WebUI `/chat` only.
- Cockpit/focus mode copy.
- Independent context/runtime rail visibility labels and persisted behavior.
- Conversation session labels for temporary, local unsaved, local history-linked, server-backed loading, loaded, and failed states.
- Status-strip propagation for important session/degraded state when rails are hidden.
- Focused Vitest plus real-server Playwright proof.

Out of scope:

- Browser-extension sidepanel/sidebar.
- Backend API changes.
- New saved-chat persistence behavior.
- New conversation recovery endpoints.
- Model selector redesign.
- Mobile cockpit redesign beyond preserving the existing mode/tabs behavior.

## Current Evidence

- `PlaygroundCockpitShell.tsx` already persists cockpit/focus mode and independent rail visibility through `Playground`.
- `PlaygroundContextRail.tsx` already renders session title, status, detail, error, history linkage, and temporary-chat toggle.
- `PlaygroundStatusStrip.tsx` already remains visible when rails are hidden, but it only receives `sessionLabel` and `temporaryChat`, so session details/errors can be hidden in collapsed-rail layouts.
- `buildCockpitSessionSummary` already distinguishes temporary, local, server-backed loading, loaded, and failed states.
- `chat-cockpit.real-server.spec.ts` already proves real-server cockpit/focus controls, rail visibility, mobile focus, model settings, prompt/persona/character, MCP, and conversation send.

## Task 1: Lock Session Summary and Status-Strip Critical State

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`

- [x] **Step 1: Add session-summary coverage**

Add table-style assertions for:

- temporary chat: label `Temporary chat`, detail `Not saved`, status label `Local only`
- local unsaved chat: label `Local chat`, detail `No saved history yet`, status `idle`
- local history-linked chat: detail `History linked`, status label `Local history`
- server-backed loading chat: title preserved and detail `Loading conversation`
- server-backed loaded chat: title preserved and detail includes topic/state/source when present
- server-backed failed chat: title, error detail, status label `Load failed`

- [x] **Step 2: Add failing status-strip coverage**

Assert `PlaygroundStatusStrip` can render session status/detail/error independently of the rails. Include a failed server-session case where the strip shows `Server chat`, `Load failed`, `Conversation no longer exists`, and a recovery/settings action remains available.

- [x] **Step 3: Run focused tests for RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx --reporter=verbose
```

Expected: summary tests may pass for existing behavior; status-strip critical-state test should fail because the strip does not yet accept session status/detail/error props.

## Task 2: Implement Session Status Propagation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`

- [x] **Step 1: Extend `PlaygroundStatusStrip` props**

Add optional props:

- `sessionTitle?: string | null`
- `sessionStatusLabel?: string | null`
- `sessionDetail?: string | null`
- `sessionError?: string | null`

Render a compact session-state pill when any of these values carries useful information beyond the session label. Error text must outrank detail text.

- [x] **Step 2: Pass session summary from `Playground`**

Pass `sessionSummary.title`, `sessionSummary.statusLabel`, `sessionSummary.detail`, and `sessionSummary.error` into `PlaygroundStatusStrip`.

- [x] **Step 3: Keep status strip compact**

Use existing pill styling. Avoid nested cards and avoid duplicating the full session rail.

- [x] **Step 4: Run focused tests for GREEN**

Run the same focused Vitest command from Task 1.

Expected: PASS.

## Task 3: Lock Mode and Rail Visibility Copy

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx`

- [x] **Step 1: Add shell copy coverage**

Assert the shell exposes a concise visible mode summary:

- cockpit with both rails visible: context and runtime rails are visible
- cockpit with one rail hidden: the hidden rail can be restored
- cockpit with both rails hidden: rails are hidden but status remains visible
- focus mode: rails are hidden and chat/composer stays active

- [x] **Step 2: Add collapsed-rail warning coverage**

Render the shell/status combination with both rails hidden and a failed/degraded status. Assert the status strip still exposes the critical warning text.

- [x] **Step 3: Run focused shell tests for RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx --reporter=verbose
```

Expected: FAIL until visible mode summary copy exists.

## Task 4: Implement Mode and Rail Visibility Copy

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`

- [x] **Step 1: Add mode summary helper copy**

Derive a short `modeSummary` from `mode`, `leftRailVisible`, and `rightRailVisible`.

Use copy in this family:

- `Context and runtime rails visible.`
- `Context rail hidden. Runtime rail visible.`
- `Runtime rail hidden. Context rail visible.`
- `Cockpit rails hidden. Status remains visible.`
- `Focus mode hides rails. Chat and composer remain active.`

- [x] **Step 2: Render the summary under the shell title**

Keep the header compact, truncate safely, and expose `data-testid="playground-cockpit-mode-summary"` for tests.

- [x] **Step 3: Preserve existing button names and persisted behavior**

Do not change `Hide context rail`, `Show context rail`, `Hide runtime rail`, `Show runtime rail`, `Enter focus chat`, or `Show cockpit panels` accessible names.

- [x] **Step 4: Run focused shell tests for GREEN**

Run the same focused Vitest command from Task 3.

Expected: PASS.

## Task 5: Real-Server Proof and Closeout

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `backlog/tasks/task-400 - Implement-main-chat-mode-and-session-clarity.md`

- [x] **Step 1: Expand real-server cockpit proof**

In the existing real-server `/chat` cockpit spec, assert:

- initial cockpit mode summary is visible
- hiding each rail updates the mode summary
- hiding both rails keeps the status strip visible
- focus mode summary is visible after entering focus
- degraded server warning text, when present, remains visible outside the rails

- [x] **Step 2: Run focused Vitest suite**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose
```

- [x] **Step 3: Run real-server Playwright**

Use the already-running real server at `http://127.0.0.1:8000` and the configured `.env` API key. Do not mock backend routes.

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium
```

- [x] **Step 4: Run static checks**

Run:

```bash
git diff --check
cd apps/packages/ui && bun run verify:design-system-state
```

- [x] **Step 5: Bandit decision**

No Python files should be touched. Record Bandit as skipped for no Python changes unless backend Python unexpectedly changes.

- [x] **Step 6: Update TASK-400 and commit**

Check acceptance criteria, record verification commands/results, add a final summary, and commit with:

```bash
git add ...
git commit -m "Clarify chat cockpit mode and session state"
```
