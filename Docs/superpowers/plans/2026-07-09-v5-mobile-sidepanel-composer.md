# V5 Mobile Sidepanel Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 360px sidepanel composer use the latest V5/Radial Command mobile direction with a usable full-width text input and visible document handling state.

**Architecture:** Keep desktop V5 unchanged. Add a compact-density V5 layout that separates metadata, text input, and actions into rows so controls cannot crush the textarea. Wire the sidepanel V5 path to compact density and V5-specific mobile controls instead of passing the legacy control stack as `facetsSlot`.

**Tech Stack:** React, Tailwind classes, existing composer slots, Playwright smoke tests, Vitest component tests.

---

### Task 1: Add V5 Compact Layout Guard

**Files:**
- Modify: `apps/packages/ui/src/components/Chat/composer/__tests__/ChatComposer.test.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/variants/RadialCommandV5.tsx`

- [x] **Step 1: Write the failing test**
  - Add a test that renders `ChatComposer variant="v5" density="compact"` with `textareaSlot`, `inlineSlot`, `sendSlot`, and facets.
  - Assert that the compact layout exposes `data-testid="v5-mobile-composer"`, `v5-mobile-text-row`, and `v5-mobile-action-row`, and that desktop `⌘K` text is absent.

- [x] **Step 2: Run the focused Vitest red check**
  - Run: `bunx vitest run src/components/Chat/composer/__tests__/ChatComposer.test.tsx`
  - Expected: FAIL because the compact V5 mobile test IDs/layout do not exist yet.

- [x] **Step 3: Implement compact V5 rows**
  - In `RadialCommandV5.tsx`, keep desktop rendering as-is.
  - When `density === "compact"`, render:
    - top metadata row from `facetsSlot` or default `FacetRow`
    - full-width textarea row
    - bottom action row containing inline actions and send
  - Change the default compact palette trigger from `⌘K` to a mobile command affordance.

- [x] **Step 4: Run the focused Vitest green check**
  - Run: `bunx vitest run src/components/Chat/composer/__tests__/ChatComposer.test.tsx`
  - Expected: PASS.

### Task 2: Wire Sidepanel V5 Mobile To Compact Layout

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Modify: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`

- [x] **Step 1: Update the smoke test red target**
  - Replace V1/V3/V5 mobile parity with V5-only sidepanel checks at 360px and `/chat` tablet guardrail at 768px.
  - Assert visible `data-variant="v5"`, `v5-mobile-composer`, `v5-mobile-action-row`, and usable chat input width.

- [x] **Step 2: Run the focused Playwright red check**
  - Run: `npx playwright test e2e/smoke/composer-mobile-viewport.spec.ts --reporter=line`
  - Expected: FAIL before sidepanel V5 compact wiring.

- [x] **Step 3: Wire V5 sidepanel props**
  - Pass `density="compact"` for the sidepanel V5 path.
  - Do not pass the legacy `composerControlAreaNode` as `facetsSlot`.
  - Build V5 facets for model, chat mode, web state, document count, and current document processing mode.
  - Build a compact inline action strip for image attach, voice/dictation, character, and commands using existing handlers.
  - Put Queue/Send state inside `sendSlot`.

- [x] **Step 4: Run verification**
  - Run focused Vitest.
  - Run focused Playwright mobile smoke.
  - Run `bunx tsc --noEmit --pretty false --project tsconfig.json`.
  - Capture a 360px V5 sidepanel screenshot for visual review.
