# Persona Visual Happy Path E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic Persona Visual fixtures and coverage proving default-pack and uploaded-pack setup can activate a sprite-frame visual that BuddyShell renders.

**Architecture:** Keep V1 scoped to the existing Persona Visual pack editor, service contract, and BuddyShell sprite-frame runtime. Reuse the backend starter-pack endpoints already added by the starter catalog and mock those same API shapes in E2E rather than adding a separate setup system. Import/upload remains review-gated: preview, commit to draft, explicit activation, then BuddyShell render.

**Tech Stack:** React, TypeScript, Vitest, Playwright, mocked Persona REST/WebSocket fixtures.

---

### Task 1: Starter-Pack Copy Service And Editor Affordance

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Write failing service and editor tests**

Add tests showing starter packs are listed from `/api/v1/persona/visual-starter-packs`, copied through `/copy`, and selected as a draft without activating.

- [x] **Step 2: Run tests to verify they fail**

Run: `bunx vitest run apps/packages/ui/src/services/__tests__/persona-visuals.test.ts apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 3: Implement minimal service/types/UI**

Add starter-pack response types, service helpers, and a compact "Default starter pack" panel inside `VisualPackEditor`.

- [x] **Step 4: Run tests to verify they pass**

Run the same focused Vitest command.

### Task 2: Deterministic Persona Visual E2E Fixtures

**Files:**
- Create: `apps/tldw-frontend/e2e/fixtures/persona-visual-packs.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Write failing E2E fixture consumers**

Update the Persona Live workflow to consume shared default and portable upload fixtures rather than inline ad hoc pack data.

- [x] **Step 2: Run the focused E2E to verify failure**

Run: `bunx playwright test e2e/workflows/persona-live.spec.ts --grep "visual pack" --reporter=line --workers=1`

- [x] **Step 3: Add deterministic fixture helpers**

Expose a sprite-frame starter summary, active/draft pack builders, and a `.tldw-persona-vpack` upload `File` helper with stable bytes, name, and MIME type.

- [x] **Step 4: Run the focused E2E to verify pass**

Run the same Playwright command.

### Task 3: Default And Upload Happy-Path E2E Coverage

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts`

- [x] **Step 1: Write failing happy-path tests**

Add tests for default starter copy -> activation -> BuddyShell render and upload preview -> commit -> activation -> BuddyShell render.

- [x] **Step 2: Run focused E2E and confirm failures**

Run: `bunx playwright test e2e/workflows/persona-live.spec.ts --grep "setup path" --reporter=line --workers=1`

- [x] **Step 3: Implement mocks and assertions**

Mock starter-pack list/copy, import preview/commit, activation, refreshed pack lists, and assert `persona-visual-frame` renders with the activated pack state.

- [x] **Step 4: Run focused validation**

Run focused Vitest and Playwright checks, plus `git diff --check`. Bandit is not applicable unless Python code changes.

## Validation

- `bun run test:run ../packages/ui/src/services/__tests__/persona-visuals.test.ts ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx ../packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx` passed: 55 tests.
- `bun -e "import { buildPortablePersonaVisualPackUpload } from './e2e/fixtures/persona-visual-packs.ts'; ..."` passed: repeated fixture builds produced identical bytes (`uploaded-visual-pack.tldw-persona-vpack`, 2055 bytes).
- `TLDW_WEB_URL=http://localhost:18099 TLDW_WEB_CMD='bun run dev -- -p 18099' bunx playwright test e2e/workflows/persona-live.spec.ts --grep "setup path" --reporter=line --workers=1` passed: 2 tests.
- `TLDW_WEB_URL=http://localhost:18100 TLDW_WEB_CMD='bun run dev -- -p 18100' bunx playwright test e2e/workflows/persona-live.spec.ts --reporter=line --workers=1` ran the full file; the 4 visual-pack tests passed and the live-backend WebSocket proof timed out waiting for Disconnect.
- `git diff --check` passed.
- Bandit not run because this slice changed TypeScript, JSON, Markdown, and Playwright fixture code only.
