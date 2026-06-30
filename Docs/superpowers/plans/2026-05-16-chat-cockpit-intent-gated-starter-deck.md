# Chat Cockpit Intent-Gated Starter Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the main `/chat` starter deck only for a true blank state, then let the composer and transcript take priority once the user has chat intent.

**Architecture:** Add a narrow intent signal around the existing Playground coordinator state instead of creating a new mode system. The parent `/chat` surface should decide whether the starter deck is allowed; `PlaygroundEmpty` remains the reusable deck component and does not learn about global chat state.

**Tech Stack:** React, TypeScript, Vitest with Testing Library, Playwright real-server workflow for final browser proof if needed.

---

## Scope Guard

This slice is limited to the main WebUI `/chat` cockpit. Do not touch browser-extension sidepanel/sidebar routes, settings pages, character library pages, backend APIs, MCP Hub, or unrelated cockpit work.

The approved behavior is option 1, intent-gated collapse:

- Show the full starter deck only when there are no messages/history, no active draft text, and no active conversation.
- Hide the starter deck as soon as the composer has draft text.
- Keep it hidden for existing or loaded conversations.
- Allow it to reappear if the user clears the draft before any send.
- Do not add a bottom bar or composer-adjacent replacement summary.

## File Map

- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Derive a `showStarterDeck` or equivalent from existing messages/history/session state plus current composer draft state.
  - Render `PlaygroundEmpty` only when that signal is true.
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - If needed, expose the current composer draft text to the parent through a focused callback.
  - Keep existing composer controls and send behavior unchanged.
- Modify or create focused tests under `apps/packages/ui/src/components/Option/Playground/__tests__/`
  - Cover blank state, typed draft, existing conversation, and draft-cleared restoration.
- Update: `backlog/tasks/task-412 - Implement-main-chat-intent-gated-starter-deck.md`
  - Record plan, verification, and final state.

## Stage 1: Lock The Intent Contract

**Goal:** Add failing focused tests for the center starter deck visibility rules.
**Success Criteria:** Tests fail on current `origin/dev` because the starter deck remains visible after a draft appears.
**Tests:** `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --config vitest.config.ts`
**Status:** Complete

- [x] Add a focused Playground test that renders a true blank state and expects `playground-empty-mode-deck` to be visible.
- [x] Add a focused Playground test that simulates or injects composer draft text before send and expects `playground-empty-mode-deck` to be absent.
- [x] Add a focused Playground test with existing messages/history/server chat state and expects the starter deck to stay absent.
- [x] Add a focused Playground test that clears the draft before send and expects the starter deck to return.
- [x] Run the focused tests and confirm the new draft-intent assertion fails for the current behavior.

Red evidence: the focused cockpit-shell test failed on the draft, active conversation, and restored-draft assertions before the production wiring was added.

## Stage 2: Implement Intent-Gated Rendering

**Goal:** Hide the starter deck when the user has a draft or conversation intent without adding a new visible control.
**Success Criteria:** Existing `/chat` state paths continue to work and the starter deck only renders for true blank state.
**Tests:** Same focused Vitest command from Stage 1.
**Status:** Complete

- [x] Add a minimal parent-level draft intent signal in `Playground.tsx`.
- [x] Wire the composer draft signal from `PlaygroundForm.tsx` only if no existing parent-readable draft state exists.
- [x] Gate `PlaygroundEmpty` rendering on no messages/history, no active server/local conversation, and no draft text.
- [x] Keep the transcript, composer, rails, and status strip rendering unchanged.
- [x] Run the focused tests and confirm they pass.

## Stage 3: Regression And Real-Server Proof

**Goal:** Prove the slice does not regress cockpit rails, composer behavior, or real `/chat` operation.
**Success Criteria:** Focused unit coverage passes; real-server proof either captures the starter-deck transition or records why browser proof was not useful for this state-only slice.
**Tests:**

- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx --config vitest.config.ts`
- `git diff --check`
- Targeted ESLint/prettier if touched files require it.
- Real-server Playwright only if it materially helps validate the visible transition.

**Status:** Complete

- [x] Run focused Vitest.
- [x] Run `git diff --check`.
- [x] Run formatting/lint checks for touched frontend files where available.
- [x] Record Bandit skip because touched scope is frontend TypeScript and Backlog/plan Markdown only.
- [x] Update TASK-412 with verification evidence and final summary.

Verification evidence:

- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts --config vitest.config.ts` passed: 4 files, 22 tests.
- `bunx tsc --noEmit --project tsconfig.json --pretty false` still fails on the existing UI baseline; filtering `/tmp/chat-cockpit-intent-starter-tsc.log` found no touched-file errors for `Playground.tsx`, `PlaygroundChat.tsx`, `PlaygroundForm.tsx`, `Playground.cockpit-shell.test.tsx`, or `PlaygroundChat.server-load-state.test.tsx`.
- `git diff --check` passed.
- Real-server browser proof was not captured because no tldw_server2 frontend/backend listener was running on the expected local ports; `lsof` only showed unrelated local services including tldw_chatbook on `8837` and llama-server on `9099`.
- Bandit skipped: touched runtime code is frontend TypeScript and Markdown only.
