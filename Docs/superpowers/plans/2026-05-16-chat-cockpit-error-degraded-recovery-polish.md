# Chat Cockpit Error Degraded Recovery Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the main `/chat` cockpit status strip and adjacent runtime state clearly distinguish active work, warning-only degraded health, missing model recovery, and blocking errors.

**Architecture:** Keep this slice presentational and reuse the existing `Playground` readiness event, composition preview summary, and cockpit status strip. Do not introduce a second server health model or change chat submission behavior; degraded unrelated health remains warning-only.

**Tech Stack:** Next.js WebUI shared package, React, Vitest, Testing Library, Playwright real-server workflow.

---

## File Structure

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
  - Owns visible footer status priority, warning/action copy, and status action affordances.
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Passes existing server readiness and composition-preview state into the status strip without creating new runtime state.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
  - Focused red-green coverage for priority, warning-only degraded health, missing-model recovery, and context-loading copy.
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
  - Integration coverage that degraded readiness does not hide streaming state and remains visible as a non-blocking warning.
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Real-server proof for degraded warning copy when the running server reports degraded health.

## Stage 1: Status Priority And Copy

**Goal:** Make the status strip priority explicit: blocking error, active streaming, context loading, missing model, warning-only degraded, then ready.

**Success Criteria:** Users can tell whether chat is actively streaming, waiting on context preview, missing a model, degraded but usable, or blocked by an error from the status strip alone.

**Tests:** `PlaygroundStatusStrip.first-slice.test.tsx`

- [x] **Step 1: Write failing status-strip tests**
  - Add tests for streaming plus degraded health showing `Streaming` as the primary state while retaining degraded warning details.
  - Add tests for missing model showing `No model selected`, `Choose a model before sending.`, and the model settings action.
  - Add tests for context-loading copy showing `Loading context` without marking the page as degraded.

- [x] **Step 2: Run the focused status-strip tests and verify RED**
  - Run: `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
  - Expected: new assertions fail against the current priority/copy.

- [x] **Step 3: Implement minimal status-strip logic**
  - Add props only for already-known state, such as `compositionStatus`.
  - Compute a single display state inside `PlaygroundStatusStrip`.
  - Keep degraded checks visible as warning pills even when streaming or loading has primary status.
  - Preserve existing action names where tests and accessibility rely on them.

- [x] **Step 4: Run the focused status-strip tests and verify GREEN**
  - Run: `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
  - Expected: status-strip tests pass.

## Stage 2: Playground Wiring

**Goal:** Connect the status strip to existing `/chat` readiness and composition state without changing send behavior.

**Success Criteria:** `Playground` passes existing composition status into the strip, and degraded readiness remains non-blocking warning copy.

**Tests:** `Playground.cockpit-controls.test.tsx`

- [x] **Step 1: Add Playground integration guard**
  - Extend degraded readiness coverage so streaming plus degraded readiness keeps `Streaming` primary and still displays the degraded subsystem warning.

- [x] **Step 2: Run focused cockpit control test after status-strip red-green**
  - Run: `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
  - Expected: the integration guard passes with the fixed status priority.

- [x] **Step 3: Pass composition/readiness state through `Playground`**
  - Keep `compositionStatus` backed by the existing composition preview path.
  - Do not block chat submission for degraded server health.

- [x] **Step 4: Run focused cockpit tests and verify GREEN**
  - Run: `bun run test:run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`
  - Expected: cockpit control tests pass.

## Stage 3: Real Server Proof And Closeout

**Goal:** Prove the changed cockpit state works against the already-running real server and update task records.

**Success Criteria:** Focused Vitest coverage and real-server Playwright coverage pass or any environment blocker is recorded exactly.

**Tests:** focused Vitest files, `chat-cockpit.real-server.spec.ts`, touched-scope Bandit skip note for frontend-only TypeScript.

- [x] **Step 1: Update real-server Playwright degraded warning assertion**
  - When `/api/v1/health` reports `degraded`, assert the cockpit status strip includes warning-only copy.
  - Do not mock or route API responses.

- [x] **Step 2: Run focused unit/integration tests**
  - Run both touched Vitest files.

- [x] **Step 3: Run real-server browser proof**
  - Run the existing real-server Playwright workflow with API key loaded from `.env`.
  - Expected: no mocked data, real `/api/v1/health`, real `/chat` load, real conversation.

- [x] **Step 4: Update Backlog task and final notes**
  - Check completed acceptance criteria only after fresh verification.
  - Document Bandit as not applicable if only frontend TypeScript files changed.
