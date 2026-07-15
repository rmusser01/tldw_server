# Skills Browser-Extension Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify the shared `/skills` experience through the packaged Chrome extension options shell with six deterministic, skip-free Playwright contracts and fix only defects reproduced by those contracts.

**Architecture:** Extend the existing built-extension launcher with one pre-navigation preparation hook and an initial options target. Use that hook to install page diagnostics, exact API routes, and a reload-persistent direct-request seam before the first `#/skills` navigation. Reuse and minimally extend the existing WebUI Skills fixtures; keep all new workflow orchestration in one extension Playwright spec.

**Tech Stack:** TypeScript, React, WXT browser extension, Playwright, Vitest, TanStack Query, Chrome MV3, existing tldw UI and E2E helpers.

**Approved spec:** `Docs/Design/2026-07-15-skills-extension-parity-design.md`

**Backlog task:** `TASK-12970`

---

## File Map

- Modify `apps/extension/tests/e2e/utils/extension-build.ts`: support a targeted initial options URL and pre-navigation page preparation.
- Modify `apps/extension/tests/e2e/utils/extension-build.test.ts`: prove preparation runs before navigation and defaults remain unchanged.
- Modify `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`: add exact extension bootstrap, model filtering, binary export, seed tracking, and list-recovery behavior.
- Create `apps/extension/tests/e2e/skills.parity.spec.ts`: own the six packaged-extension contracts and local diagnostics harness.
- Modify `apps/extension/entrypoints/options/index.html` and create `apps/packages/ui/src/public/theme-bootstrap.js`: replace the reproduced MV3-blocked inline options theme bootstrap with a synchronous same-origin script.
- Create `apps/extension/tests/unit/options-theme-bootstrap.test.ts`: prevent executable inline script regressions in the options entrypoint.
- Modify `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`: use the existing list-ready assertion in row-action tests that proved timing-dependent in the full owning suite.
- Modify `apps/extension/package.json`: add focused and strict one-worker Skills parity scripts with the deterministic mock origin.
- Update `backlog/tasks/task-12970 - Certify-Skills-browser-extension-parity-and-fix-shell-specific-regressions.md` only through Backlog MCP/CLI: track plan, verification, findings, and final status.
- Do not modify production UI files unless the built suite reproduces a product defect and a focused owning-boundary test fails first.

---

### Stage 1: Prepare the built extension before initial navigation

**Goal:** Let a caller install routes, diagnostics, and init scripts before the options page loads, while preserving every existing launcher caller.

**Success Criteria:** `launchWithBuiltExtension()` accepts `optionsTarget` and `prepareOptionsPage`; preparation is awaited before `goto()`; omitted options preserve the current `options.html` URL.

**Tests:** Focused Vitest coverage in `extension-build.test.ts` fails first and then passes.

**Status:** Complete

**Files:**
- Modify: `apps/extension/tests/e2e/utils/extension-build.test.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-build.ts`

- [ ] **Step 1: Add the failing launcher-order test**

  Extend the existing mocked launch test with an event log:

  ```ts
  const events: string[] = []
  page.goto.mockImplementation(async (url: string) => {
    events.push(`goto:${url}`)
  })
  const prepareOptionsPage = vi.fn(async ({ page: preparedPage }) => {
    expect(preparedPage).toBe(page)
    events.push("prepare")
  })

  await launchWithBuiltExtension({
    optionsTarget: "/skills",
    prepareOptionsPage
  })

  expect(events).toEqual([
    "prepare",
    `goto:chrome-extension://${"e".repeat(32)}/options.html#/skills`
  ])
  ```

  Keep or add a separate assertion that the no-option call still navigates to plain `options.html`.

- [ ] **Step 2: Run the focused test and confirm RED**

  Run from `apps/extension`:

  ```bash
  bunx vitest run tests/e2e/utils/extension-build.test.ts --reporter=dot
  ```

  Expected: FAIL because `LaunchOptions` does not accept the new fields and preparation is not invoked.

- [ ] **Step 3: Implement the smallest launcher change**

  In `extension-build.ts`:

  ```ts
  import {
    chromium,
    type BrowserContext,
    type Page
  } from "@playwright/test"

  type PrepareOptionsPage = (input: {
    context: BrowserContext
    page: Page
  }) => void | Promise<void>

  type LaunchOptions = {
    // existing fields
    optionsTarget?: string
    prepareOptionsPage?: PrepareOptionsPage
  }
  ```

  Rename the private `resolveSidepanelUrl()` helper to the platform-neutral `resolveExtensionPageUrl()` and use it for options and sidepanel targets. Change only the options-page launch sequence:

  ```ts
  const page = await context.newPage()
  await prepareOptionsPage?.({ context, page })
  await page.goto(resolveExtensionPageUrl(optionsUrl, optionsTarget))
  await waitForStorageSeed(page)
  ```

  Do not alter default storage, connection, browser, or service-worker behavior.

- [ ] **Step 4: Run focused launcher tests and confirm GREEN**

  ```bash
  bunx vitest run tests/e2e/utils/extension-build.test.ts tests/e2e/utils/extension-paths.test.ts --reporter=dot
  ```

  Expected: PASS with no changed existing expectations.

- [ ] **Step 5: Check diff hygiene and commit Stage 1**

  ```bash
  git diff --check
  git add apps/extension/tests/e2e/utils/extension-build.ts apps/extension/tests/e2e/utils/extension-build.test.ts
  git commit -m "test(extension): prepare targeted options launches"
  ```

---

### Stage 2: Establish bootstrap harness and beginner contract

**Goal:** Open the packaged extension directly at `#/skills`, complete production connection/capability bootstrap through deterministic routes, and verify the beginner journey through chat handoff.

**Success Criteria:** The first of six tests passes without force-connecting the store, skipping, leaking secrets, or leaving unexpected API/browser diagnostics.

**Tests:** One Playwright test named `completes bootstrap and the beginner journey` is written first and fails before fixture completion.

**Status:** Complete

**Files:**
- Create: `apps/extension/tests/e2e/skills.parity.spec.ts`
- Modify: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
- Modify: `apps/extension/entrypoints/options/index.html`
- Create: `apps/packages/ui/src/public/theme-bootstrap.js`
- Create: `apps/extension/tests/unit/options-theme-bootstrap.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Create the local extension harness and failing beginner test**

  Define only local helpers in `skills.parity.spec.ts`:

  ```ts
  const SKILLS_PARITY_SERVER_URL = "http://skills-parity.invalid"
  const SKILLS_PARITY_API_KEY = "skills-parity-test-key"
  const DEFAULT_VIEWPORT = { width: 1280, height: 900 }
  ```

  `installDirectRequestFallback(context)` must use `context.addInitScript()` before navigation. Patch `browser.runtime.sendMessage` and `chrome.runtime.sendMessage` for messages with `type === "tldw:request"` to throw the existing recognized transport error, `Could not establish connection. Receiving end does not exist.` Preserve all non-tldw messages and use `Object.defineProperty` only when direct assignment fails.

  `captureDiagnostics(page)` must collect `pageerror`, console messages whose type is `error`, `requestfailed` entries with bounded messages, and `unexpectedApiRequests`. Install a mock-origin fallback route before specific fixture routes; append any otherwise-unhandled `${SKILLS_PARITY_SERVER_URL}/api/**` request to `unexpectedApiRequests` and fulfill it with status 501. Each test must finish with explicit assertions equivalent to:

  ```ts
  expect(diagnostics.pageErrors).toEqual([])
  expect(diagnostics.consoleErrors).toEqual([])
  expect(diagnostics.requestFailures).toEqual([])
  expect(diagnostics.unexpectedApiRequests).toEqual([])
  ```

  Launch through:

  ```ts
  await launchWithBuiltExtension({
    seedConfig: {
      __tldw_first_run_complete: true,
      tldwConfig: {
        serverUrl: SKILLS_PARITY_SERVER_URL,
        authMode: "single-user",
        apiKey: SKILLS_PARITY_API_KEY
      }
    },
    optionsTarget: "/skills",
    prepareOptionsPage: async ({ context, page }) => {
      await page.setViewportSize(DEFAULT_VIEWPORT)
      diagnostics = captureDiagnostics(page)
      await installUnexpectedApiGuard(page, diagnostics)
      await installDirectRequestFallback(context)
      api = await mockSkillsBeginnerApi(page)
    }
  })
  ```

  Because Playwright routes are last-in-first-out, register the fallback guard before the specific fixture routes. Close every returned context in `finally`.

  The beginner test must assert:
  - `Skills` heading and first-use `Start with a reusable skill` state;
  - one seed request with `overwrite=false`, success confirmation, and `summarize` row;
  - details description and runtime disclosure;
  - Enter in Test run sends `{ dry_run: true }` and explicit Run test sends `{ dry_run: false }`;
  - Use in chat changes the hash to `#/chat` and leaves `/skill summarize` in `#textarea-message`;
  - no unexpected diagnostics or API requests.

- [ ] **Step 2: Run the beginner test and confirm RED**

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "bootstrap and the beginner" --workers=1 --reporter=line
  ```

  Expected: FAIL on missing `/api/v1/health/live`, `/api/v1/rag/health`, seed tracking, or exact chat bootstrap routes. Record the first owning failure before changing fixtures.

- [ ] **Step 3: Extend bootstrap and beginner fixture behavior minimally**

  In `mockSkillsCapabilityRoutes()`:

  ```ts
  await page.route(/\/api\/v1\/health\/live(?:\?.*)?$/, route =>
    fulfillJson(route, { status: "healthy" })
  )
  await page.route(/\/api\/v1\/rag\/health(?:\?.*)?$/, route =>
    fulfillJson(route, { status: "healthy" })
  )
  ```

Keep the existing OpenAPI fixture. In `mockSkillsBeginnerApi()`, capture each seed request URL and expose `seedRequests` with `executeRequests`. Add only exact chat bootstrap responses demonstrated by the RED run to a local `mockChatHandoffBootstrap(page)` helper in `skills.parity.spec.ts`; do not pollute the shared Skills fixture, add a permissive API catch-all, or create a general chat fixture framework.

The strict browser diagnostics reproduced the options entrypoint's inline theme script being rejected by MV3 CSP. Add a failing source regression test, move that exact bootstrap to `/theme-bootstrap.js`, rebuild, and require the packaged beginner test to pass with no CSP exception. Do not broaden this fix to the sidepanel. The full Manager suite also reproduced five row-action test timeouts; add the existing `1 skill` list-ready assertion before those role queries rather than adding sleeps or global timeouts.

- [ ] **Step 4: Run the beginner test and focused shared tests**

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "bootstrap and the beginner" --workers=1 --reporter=line

  bunx vitest run tests/unit/options-theme-bootstrap.test.ts --reporter=dot
  ```

  From `apps/packages/ui`:

  ```bash
  bunx vitest run \
    src/routes/__tests__/option-skills-shell.test.tsx \
    src/components/Option/Skills/__tests__/Manager.test.tsx \
    --reporter=dot
  ```

  Expected: PASS. The Playwright test reports one seed mutation, two execution requests, the chat composer handoff, and zero unexpected diagnostics.

- [ ] **Step 5: Commit Stage 2**

  ```bash
  git diff --check
  git add apps/extension/tests/e2e/skills.parity.spec.ts apps/tldw-frontend/e2e/utils/skills-fixtures.ts
  git commit -m "test(skills): cover extension beginner journey"
  ```

---

### Stage 3: Add power-user hash/export and Trash contracts

**Goal:** Verify dense library management, exact hash persistence/history, client-side bulk export, and durable Trash behavior in isolated extension contexts.

**Success Criteria:** Tests two and three pass with exact request/state assertions and no shared mutable fixture state.

**Tests:** Add each Playwright test before adding its missing fixture behavior.

**Status:** Complete

**Files:**
- Modify: `apps/extension/tests/e2e/skills.parity.spec.ts`
- Modify: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`

- [ ] **Step 1: Add the failing power-user test**

  Assert exactly two selected rows, two per-Skill export GETs as an order-independent set, one aggregate `.zip` download, and `2 selected` before and after download. While the unfiltered 30-row library still exposes pagination, set page size to 20; only then apply search, mode, tools, model, and sort filters. Assert the exact hash:

  ```text
  #/skills?q=target&mode=fork&tools=with-tools&model=gpt-4.1-mini&sort=name&order=desc&pageSize=20
  ```

  Assert the final list request has `q=target`, `context=fork`, `has_tools=true`, `model=gpt-4.1-mini`, `sort=name`, `order=desc`, `limit=20`, and `offset=0`. Reload and reassert. Change only tools to `without-tools`, then verify Back restores the exact `with-tools` state/row and Forward restores `without-tools`/no match.

- [ ] **Step 2: Run the power-user test and confirm RED**

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "power-user hash and export" --workers=1 --reporter=line
  ```

  Expected: FAIL because the fixture does not filter by model or serve binary exports.

- [ ] **Step 3: Add only the missing power-user fixture behavior**

  Update `filterLargeLibrarySkills()` to read `model` and reject nonmatching records. Add a route for `/api/v1/skills/{name}/export` that:
  - records decoded Skill names;
  - returns deterministic binary bytes;
  - sets `content-type: application/zip`;
  - sets `content-disposition: attachment; filename="{name}.zip"`.

  Expose `exportRequests` alongside `lastListUrl`. Keep the existing list and detail routes unchanged.

- [ ] **Step 4: Confirm the power-user test is GREEN**

  Run the Step 2 command again. Expected: PASS with one browser download, two export requests, exact query/hash restoration, and empty diagnostics.

- [ ] **Step 5: Add and run the Trash test**

  Add the third test with a fresh `mockSkillsTrashWorkflow()` context. Move `summarize` to Trash, assert `Undo delete summarize` is visible without clicking it, switch to Trash, restore, return to Library, and assert operations equal `['delete', 'restore']`.

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "Trash management" --workers=1 --reporter=line
  ```

  Expected: PASS without modifying the Trash fixture unless the extension exposes a real fixture gap.

- [ ] **Step 6: Commit Stage 3**

  ```bash
  git diff --check
  git add apps/extension/tests/e2e/skills.parity.spec.ts apps/tldw-frontend/e2e/utils/skills-fixtures.ts
  git commit -m "test(skills): cover extension power workflows"
  ```

---

### Stage 4: Add compact accessibility, recovery, and draft contracts

**Goal:** Complete tests four through six for compact layout, keyboard/focus behavior, real retry semantics, unreachable gating, and session draft recovery.

**Success Criteria:** All six tests pass independently with a fresh context and deterministic mutable fixture state.

**Tests:** Add and run each test separately before running the full file.

**Status:** In Progress

**Files:**
- Modify: `apps/extension/tests/e2e/skills.parity.spec.ts`
- Modify: `apps/tldw-frontend/e2e/utils/skills-fixtures.ts`
- Conditional only if a product defect is reproduced: the narrow owning shared UI file and its focused test.

- [ ] **Step 1: Add the compact keyboard/focus test**

  Launch with `{ width: 390, height: 844 }` before initial navigation and `mockSkillsBeginnerApi(page, { seeded: true })`. Assert roles/names, no document overflow initially and with each overlay open, 24-by-24 minimum target boxes for New Skill/details/test actions, keyboard activation, Escape closure, and focus return for details and Test run.

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "compact keyboard and focus" --workers=1 --reporter=line
  ```

  Expected: PASS. If it fails on product behavior, first add a focused failing shared component test before changing production code.

- [ ] **Step 2: Write the failing list-recovery test and fixture contract**

  Add `mockSkillsListRecovery(page)` with fresh closure state:

  ```ts
  let listRequests = 0
  // request 1 waits for releaseFirst(), then returns 503
  // request 2 is TanStack Query's automatic retry and returns 503
  // request 3 is the user's Try again request and returns seededSkillSummary
  ```

  Return `releaseFirst()` and `listRequestCount()`. The test must assert `Loading skills` before release, exactly two requests before the alert, no secret/path/raw-body leakage, `Try again`, a third successful request, then force only the documented unreachable state and assert recovery actions plus absence of New/Seed/Import controls.

- [ ] **Step 3: Run recovery RED, implement minimally, and confirm GREEN**

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "list retry and unreachable" --workers=1 --reporter=line
  ```

  Expected RED: missing recovery fixture. After implementation, expected GREEN with request count exactly three and no broad diagnostic exclusions.

- [ ] **Step 4: Add and run the session draft test**

  Use a new unseeded beginner fixture context. Enter a unique valid name and instructions, reload the same `options.html#/skills` tab, reopen New Skill, assert the recovery alert and both values, click `Discard recovered draft`, close/reopen, and assert blank base values with no recovery alert.

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --grep "session draft recovery" --workers=1 --reporter=line
  ```

  Expected: PASS, proving the context init script preserved deterministic routing across reload.

- [ ] **Step 5: Run all six tests together**

  ```bash
  TLDW_E2E_SERVER_URL=http://skills-parity.invalid \
    bunx playwright test tests/e2e/skills.parity.spec.ts \
    --workers=1 --reporter=line
  ```

  Expected: 6 passed, 0 failed, 0 skipped. Do not loosen assertions to mask a product defect. For any reproduced product defect, add one narrow failing unit/integration test at the owning boundary, apply the minimal fix, rerun that test, then rerun this six-test contract.

- [ ] **Step 6: Commit Stage 4**

  ```bash
  git diff --check
  git add apps/extension/tests/e2e/skills.parity.spec.ts apps/tldw-frontend/e2e/utils/skills-fixtures.ts
  # Add shared production/test files only if a defect was reproduced and fixed.
  git commit -m "test(skills): cover extension resilience and access"
  ```

---

### Stage 5: Add strict scripts and complete release verification

**Goal:** Provide stable local/strict entry points, verify all affected scopes, finalize Backlog evidence, and leave a reviewable branch.

**Success Criteria:** Strict JSON report contains six passes and zero skips; build, focused/shared tests, type checks, diff hygiene, and applicable security checks are recorded.

**Tests:** Package scripts plus full affected-scope verification.

**Status:** Not Started

**Files:**
- Modify: `apps/extension/package.json`
- Modify through Backlog MCP/CLI: `backlog/tasks/task-12970 - Certify-Skills-browser-extension-parity-and-fix-shell-specific-regressions.md`
- Remove after all stages are complete: `Docs/Plans/IMPLEMENTATION_PLAN_skills_extension_parity_TASK_12970.md`

- [ ] **Step 1: Add focused and strict package scripts**

  Add scripts parallel to existing parity gates:

  ```json
  {
    "test:e2e:skills-parity": "TLDW_E2E_SERVER_URL=http://skills-parity.invalid playwright test tests/e2e/skills.parity.spec.ts --reporter=line --workers=1",
    "test:e2e:skills-parity:strict": "rm -f .skills-parity-e2e-report.json && TLDW_E2E_SERVER_URL=http://skills-parity.invalid PLAYWRIGHT_JSON_OUTPUT_NAME=.skills-parity-e2e-report.json playwright test tests/e2e/skills.parity.spec.ts --reporter=json --workers=1 && node scripts/assert-playwright-no-skips.mjs .skills-parity-e2e-report.json && mkdir -p test-results && cp .skills-parity-e2e-report.json test-results/skills-parity-e2e-report.json"
  }
  ```

- [ ] **Step 2: Run the focused launcher and shared UI suites**

  From `apps/extension`:

  ```bash
  bunx vitest run \
    tests/e2e/utils/extension-build.test.ts \
    tests/unit/options-theme-bootstrap.test.ts \
    --reporter=dot
  ```

  From `apps/packages/ui`:

  ```bash
  bunx vitest run \
    src/routes/__tests__/option-skills-shell.test.tsx \
    src/components/Option/Skills/__tests__/skills-query-state.test.ts \
    src/components/Option/Skills/__tests__/Manager.test.tsx \
    src/components/Option/Skills/__tests__/SkillDetailsDrawer.test.tsx \
    src/components/Option/Skills/__tests__/SkillPreview.test.tsx \
    src/components/Option/Skills/__tests__/SkillDrawer.test.tsx \
    --reporter=dot
  ```

  Expected: all selected files pass.

- [ ] **Step 3: Run the existing WebUI Skills workflow because its fixture changed**

  From `apps/tldw-frontend`:

  ```bash
  bunx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts \
    --project=chromium --workers=1 --reporter=line
  ```

  Expected: all existing mocked Skills journeys pass with no new skips.

- [ ] **Step 4: Build and run the strict extension gate**

  From `apps/extension`:

  ```bash
  bun run build:chrome:prod
  bun run test:e2e:skills-parity:strict
  ```

  Expected: production build succeeds; JSON assertion reports six passed and zero skipped; copied report exists under `test-results/`.

- [ ] **Step 5: Run type and repository hygiene checks**

  From `apps/extension`:

  ```bash
  bunx tsc --noEmit -p tsconfig.json
  ```

  From repository root:

  ```bash
  git diff --check
  git status --short
  ```

  Expected: TypeScript passes or an unchanged repository baseline is documented precisely; diff check passes; only task-related files are modified. No Python paths are touched, so record Bandit as not applicable rather than running it.

- [ ] **Step 6: Review the complete diff before finalization**

  Confirm:
  - no unconditional skips, broad network regexes, arbitrary sleeps, or shared context state;
  - no new dependencies, Playwright config, mock-server framework, or duplicated launcher;
  - direct fallback scope is documented and the suite does not claim MV3 relay coverage;
  - no production file changed without a reproduced failing contract and owning test.

- [ ] **Step 7: Finalize TASK-12970 and remove this completed plan**

  Through Backlog MCP/CLI, record exact commands/results, touched files, any baseline warnings, PR URL when available, final summary, acceptance-criteria completion, and Definition of Done. After all stages are complete, remove this task-specific implementation plan and its Backlog documentation reference as required by repository guidance; retain the approved design and Backlog record.

- [ ] **Step 8: Commit final scripts, task evidence, and plan cleanup**

  ```bash
  git add apps/extension/package.json \
    'backlog/tasks/task-12970 - Certify-Skills-browser-extension-parity-and-fix-shell-specific-regressions.md'
  git add -u Docs/Plans/IMPLEMENTATION_PLAN_skills_extension_parity_TASK_12970.md
  git commit -m "test(skills): gate extension parity"
  ```

  Expected: clean worktree with reviewable, task-only commits.
