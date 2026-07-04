# Skills UAT Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the `/skills` UAT quality-gates bundle with deterministic Playwright workflows, a manual QA checklist, and documented success metrics.

**Architecture:** Keep browser E2E at workflow level only. Extract the existing mocked Skills API setup into one focused fixture helper, then add only the missing power-user and trust-risk failure flows. Keep detailed state permutations in existing Vitest coverage.

**Tech Stack:** Playwright route mocking, existing frontend E2E fixtures, Vitest for touched component regressions only, Markdown QA documentation.

---

## Scope Guardrails

- Do not redesign `/skills`.
- Do not add telemetry.
- Do not add new dependencies.
- Do not duplicate existing `SkillsManager` and `SkillPreview` component-test permutations in Playwright.
- Use bulk delete confirmation, not bulk export, as the advanced workflow target because it covers the higher-risk path without actually confirming deletion.

## File Map

- Create: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
  - Owns scenario-level route mocks and request capture for Skills UAT.
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`
  - Keeps existing beginner and live-smoke tests, imports fixture helpers, adds power-user and failure workflows.
- Create: `Docs/Reviews/skills-page-uat.md`
  - Manual QA checklist and success metrics.
- Modify: `backlog/tasks/task-530.14 - Implement-Skills-UAT-and-quality-gates.md`
  - Link the implementation plan and record verification notes.

## Task 1: Extract Skills E2E Fixtures

**Status:** Complete.

**Files:**
- Create: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`

- [x] **Step 1: Move the existing route helpers**

Move these existing local pieces from `skills.spec.ts` to `skills-fixtures.ts`:

```ts
fulfillJson
seededSkillSummary
seededSkillResponse
mockSkillsBeginnerApi
forceSkillsConnectionState
```

Export only what tests use:

```ts
export {
  forceSkillsConnectionState,
  mockSkillsBeginnerApi,
  seededSkillSummary,
}
```

- [x] **Step 2: Import the helpers in the spec**

Replace local helper definitions with:

```ts
import {
  forceSkillsConnectionState,
  mockSkillsBeginnerApi,
} from "../../utils/skills-fixtures"
```

- [x] **Step 3: Run the existing mocked beginner checks**

Before running, extend the existing beginner test by clicking the post-seed `Copy invocation` action. Assert the visible success feedback. Assert clipboard text only if the current Playwright browser context already grants clipboard read reliably; otherwise leave exact clipboard contents to the manual checklist in Task 4.

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --grep "Skills beginner journey" --project=tier-5 --reporter=line
```

Expected: existing beginner tests still pass or fail only for a pre-existing environment setup issue.

- [x] **Step 4: Commit**

```bash
git add apps/tldw-frontend/e2e/utils/skills-fixtures.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
git commit -m "test: extract skills e2e fixtures"
```

## Task 2: Add Power-User Large-Library UAT

**Status:** Complete.

**Files:**
- Modify: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`

- [x] **Step 1: Add a failing Playwright test**

Add a mocked test under a deterministic describe block:

```ts
test("finds a skill outside page one and opens bulk delete confirmation", async ({ page, diagnostics }) => {
  const api = await mockPowerUserSkillsLibrary(page)
  await seedAuth(page, { serverUrl: TEST_CONFIG.serverUrl, allowOffline: true })

  await page.goto("/skills", { waitUntil: "domcontentloaded" })
  await forceSkillsConnectionState(page)

  await page.getByPlaceholder("Search skills...").fill("target research formatter")
  await expect(page.getByText("Target research formatter")).toBeVisible()
  expect(api.lastListUrl()?.searchParams.get("q")).toBe("target research formatter")

  await page.getByRole("button", { name: /filters/i }).click()
  await page.getByLabel(/fork/i).click()
  expect(api.lastListUrl()?.searchParams.toString()).toContain("mode")

  await page.getByRole("columnheader", { name: /name/i }).click()
  expect(api.lastListUrl()?.searchParams.toString()).toContain("sort")

  await page.getByRole("row", { name: /Target research formatter/i }).getByRole("checkbox").check()
  await page.getByRole("row", { name: /Batch cleanup helper/i }).getByRole("checkbox").check()
  await page.getByRole("button", { name: /Delete selected/i }).click()

  await expect(page.getByRole("dialog", { name: /delete/i })).toBeVisible()
  expect(api.deleteRequests).toHaveLength(0)
  await assertNoCriticalErrors(diagnostics)
})
```

Adjust labels only to match the actual rendered UI. Keep the assertion target the same: search request, filter/sort request, visible target row, delete confirmation opened, no delete submitted.

- [x] **Step 2: Add the fixture helper**

Add:

```ts
export async function mockPowerUserSkillsLibrary(page: Page) {
  const listUrls: URL[] = []
  const deleteRequests: unknown[] = []
  const skills = buildLargeSkillList()
  // route readiness, openapi, context, list, detail, delete
  return {
    deleteRequests,
    lastListUrl: () => listUrls.at(-1),
  }
}
```

Use simple in-memory filtering/sorting only for the fields asserted by the test. No fake backend framework.

- [x] **Step 3: Run the new test**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --grep "bulk delete confirmation" --project=tier-5 --reporter=line
```

Expected: pass.

- [x] **Step 4: Commit**

```bash
git add apps/tldw-frontend/e2e/utils/skills-fixtures.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
git commit -m "test: add skills power user uat"
```

## Task 3: Add Representative Failure UAT

**Status:** Complete.

**Files:**
- Modify: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`

- [x] **Step 1: Add failure helpers**

Add only the scenario helpers needed by the tests:

```ts
mockSkillsImportValidationFailure(page)
mockSkillsExecutionFailure(page)
mockSkillsStaleVersionConflict(page)
mockSkillsSlowList(page)
```

Each helper should route the smallest endpoint set needed for that UI state.

- [x] **Step 2: Add Playwright tests**

Add one test per trust-risk category:

- invalid import preview shows validation feedback and does not import
- execution failure shows an alert/retry affordance
- stale delete conflict says reload before retrying
- slow list shows loading state and blocks duplicate seed/list-trigger actions

Use existing labels and test IDs. Do not add UI code unless a scenario is impossible because the product state is genuinely missing.

Unsupported capability remains covered at component level in `SkillsWorkspace.test.tsx`. The browser E2E environment falls back to a bundled OpenAPI spec when capability discovery misses, which makes the unsupported gate non-deterministic in this mocked workflow.

- [x] **Step 3: Run the failure tests**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --grep "invalid import|execution failure|stale|slow" --project=tier-5 --reporter=line
```

Expected: pass.

- [x] **Step 4: Commit**

```bash
git add apps/tldw-frontend/e2e/utils/skills-fixtures.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
git commit -m "test: add skills failure uat"
```

## Task 4: Add Manual UAT Checklist And Metrics

**Status:** Complete.

**Files:**
- Create: `Docs/Reviews/skills-page-uat.md`
- Modify: `backlog/tasks/task-530.14 - Implement-Skills-UAT-and-quality-gates.md`

- [x] **Step 1: Create the checklist doc**

Include sections:

- setup prerequisites
- browser and auth assumptions
- beginner workflow
- advanced workflow
- accessibility checks
- responsive checks
- failure checks
- clipboard validation for `Copy invocation`
- success metrics

Use a compact table with columns:

```md
| ID | Scenario | Coverage | Pass criteria | Evidence |
| --- | --- | --- | --- | --- |
```

- [x] **Step 2: Add metrics without telemetry**

Document:

- task completion rate
- time to first successful skill use
- search/filter success
- configuration recovery success
- error categories
- user confidence rating

State explicitly that telemetry is not enabled by this task.

- [x] **Step 3: Link docs in Backlog**

Use Backlog MCP to add:

- `Docs/superpowers/plans/2026-07-04-skills-uat-quality-gates.md`
- `Docs/Reviews/skills-page-uat.md`

- [x] **Step 4: Commit**

```bash
git add Docs/Reviews/skills-page-uat.md "backlog/tasks/task-530.14 - Implement-Skills-UAT-and-quality-gates.md"
git commit -m "docs: add skills uat checklist"
```

## Task 5: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-530.14 - Implement-Skills-UAT-and-quality-gates.md`

- [ ] **Step 1: Run deterministic Skills Playwright UAT**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --project=chromium --reporter=line
```

Expected: mocked UAT tests pass. Live-server smoke tests may skip if the backend is unavailable.

- [ ] **Step 2: Run focused Vitest only if component files changed**

If only E2E/docs changed, skip this and record why.

If component files changed, run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx ../packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx
```

- [ ] **Step 3: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 4: Bandit decision**

If no Python files changed, record:

```text
Bandit skipped: frontend E2E and docs-only task.
```

If Python files changed, run scoped Bandit.

- [ ] **Step 5: Update Backlog final notes**

Record verification commands, known skips, and final summary.

- [ ] **Step 6: Commit closeout**

```bash
git add "backlog/tasks/task-530.14 - Implement-Skills-UAT-and-quality-gates.md"
git commit -m "docs: close skills uat task"
```
