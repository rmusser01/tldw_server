# V5 Mobile Sidepanel Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove mobile sidepanel UX regressions in the latest V5 composer screen.

**Architecture:** Keep the sidepanel-specific route in charge of sidepanel chrome. Use existing composer slots and document upload handlers; do not add another upload system.

**Tech Stack:** Next.js, React, Tailwind, Vitest, Playwright.

---

### Task 1: Sidepanel Debug Shell

**Files:**
- Modify: `apps/tldw-frontend/pages/_app.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`

- [x] Add a smoke assertion that `/__debug__/sidepanel-chat` renders no global WebUI header/sidebar.
- [x] Hide the WebUI shell nav only for the sidepanel debug route.

### Task 2: Empty State And Composer Polish

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-chat.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/variants/RadialCommandV5.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`

- [x] Add smoke assertions for one sidepanel header, compact empty state, explicit document attach affordance, and readable V5 meta chips.
- [x] Remove redundant healthy connection status from the empty state.
- [x] Reduce nested composer framing in compact V5.
- [x] Use the existing context file upload handler for the V5 document attach action.

### Task 3: Verification

**Files:**
- Test: `apps/packages/ui/src/components/Chat/composer/__tests__/ChatComposer.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`

- [x] Run targeted Vitest.
- [x] Run TypeScript.
- [x] Run targeted Playwright.
- [x] Run `git diff --check`.
- [x] Record Bandit as skipped for this TS-only change.

**Verification Results:**
- `bunx vitest run src/components/Chat/composer/__tests__/ChatComposer.test.tsx src/components/Sidepanel/Chat/__tests__/empty.test.tsx --reporter=dot`: passed.
- `npx playwright test e2e/smoke/composer-mobile-viewport.spec.ts --reporter=line`: passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false --project tsconfig.json` in `apps/tldw-frontend`: passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false --project tsconfig.json` in `apps/packages/ui`: failed on existing unrelated Notes/background errors.
- `git diff --check`: passed.
- Bandit: skipped, no Python files touched.
