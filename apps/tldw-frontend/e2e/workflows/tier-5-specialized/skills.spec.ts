/**
 * Skills E2E Tests (Tier 5)
 *
 * Tests the /skills page:
 * - Page loads with connection gate or skills manager
 * - Skills list table or empty state renders
 * - Create button present when skills API available
 *
 * Run: npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { seedAuth, TEST_CONFIG } from "../../utils/helpers"
import {
  forceSkillsConnectionState,
  mockSkillsBeginnerApi,
  mockPowerUserSkillsLibrary,
  mockSkillsExecutionFailure,
  mockSkillsImportValidationFailure,
  mockSkillsSlowList,
  mockSkillsStaleVersionConflict,
  mockSkillsTrashWorkflow,
} from "../../utils/skills-fixtures"

test.describe("Skills beginner journey (mocked)", () => {
  test("fixture defaults a missing execute payload", async ({ page }) => {
    await mockSkillsBeginnerApi(page, { seeded: true })
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    const result = await page.evaluate(async () => {
      const response = await fetch("/api/v1/skills/summarize/execute", {
        method: "POST",
      })
      return {
        status: response.status,
        body: await response.json(),
      }
    })

    expect(result).toEqual({
      status: 200,
      body: expect.objectContaining({
        rendered_prompt: "Summarize this source: A long article about Skills UX",
        dry_run: false,
      }),
    })
  })

  test("seeds built-ins, opens test run, and persists after refresh", async ({
    page,
    diagnostics,
  }) => {
    await mockSkillsBeginnerApi(page)
    await page.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {
          writeText: async (value: string) => {
            const win = window as Window & { __skillsCopiedText?: string }
            win.__skillsCopiedText = value
          },
        },
      })
    })
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await expect(
      page.getByRole("heading", { name: "Start with a reusable skill" })
    ).toBeVisible()
    const emptyState = page.getByTestId("skills-empty-state")
    await expect(emptyState).toBeVisible()

    await emptyState.getByRole("button", { name: "Seed built-ins" }).click()

    const successActions = page.getByTestId("skills-success-actions")
    await expect(successActions).toContainText("Built-in skills seeded")
    await expect(successActions.getByRole("button", { name: "Test summarize" }))
      .toBeVisible()
    await expect(page.getByText("Summarize source material")).toBeVisible()

    await successActions.getByRole("button", { name: "Copy /skill summarize" }).click()
    await expect(page.getByText("Skill invocation copied")).toBeVisible()
    await expect
      .poll(() =>
        page.evaluate(() =>
          (window as Window & { __skillsCopiedText?: string }).__skillsCopiedText
        )
      )
      .toBe("/skill summarize")

    await successActions.getByRole("button", { name: "Test summarize" }).click()
    const previewDialog = page.getByRole("dialog", { name: "Test run" })
    await expect(previewDialog).toBeVisible()
    await previewDialog
      .getByPlaceholder("Enter test arguments...")
      .fill("A long article about Skills UX")
    await previewDialog.getByRole("button", { name: "Render prompt only" }).click()
    await expect(previewDialog.getByText("Rendered Prompt", { exact: true })).toBeVisible()
    await expect(
      previewDialog.locator("textarea").last()
    ).toHaveValue("Summarize this source: A long article about Skills UX")

    await page.reload({ waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)
    await expect(page.getByText("Summarize source material")).toBeVisible()
    await expect(page.getByTestId("skills-empty-state")).toHaveCount(0)

    await assertNoCriticalErrors(diagnostics)
  })

  test("supports keyboard test-run and create-cancel flow at extension width", async ({
    page,
    diagnostics,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    const api = await mockSkillsBeginnerApi(page, { seeded: true })
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    const searchInput = page.getByPlaceholder("Search skills...")
    await expect(searchInput).toBeVisible()
    await searchInput.focus()
    await page.keyboard.type("summarize")
    await expect(page.getByText("Summarize source material")).toBeVisible()

    const testRunButton = page.getByRole("button", { name: "Test run summarize" })
    await expect(testRunButton).toBeVisible()
    await testRunButton.focus()
    await testRunButton.press("Enter")

    const previewDialog = page.getByRole("dialog", { name: "Test run" })
    await expect(previewDialog).toBeVisible()
    const argsInput = previewDialog.getByPlaceholder("Enter test arguments...")
    await argsInput.focus()
    await page.keyboard.type("Keyboard-only workflow")
    await page.keyboard.press("Enter")
    await expect(previewDialog.getByText("Dry render", { exact: true })).toBeVisible()
    expect(api.executeRequests.at(-1)).toEqual({
      args: "Keyboard-only workflow",
      dry_run: true,
    })

    await previewDialog.getByRole("button", { name: "Run test" }).focus()
    await page.keyboard.press("Enter")
    await expect.poll(() => api.executeRequests.at(-1)).toEqual({
      args: "Keyboard-only workflow",
      dry_run: false,
    })
    await expect(previewDialog.getByText("Rendered Prompt", { exact: true })).toBeVisible()
    await expect(
      previewDialog.locator("textarea").last()
    ).toHaveValue("Summarize this source: Keyboard-only workflow")
    await page.keyboard.press("Escape")
    await expect(testRunButton).toBeFocused()

    const newSkillButton = page.getByRole("button", { name: "New Skill" })
    await expect(newSkillButton).toBeVisible()
    const newSkillBox = await newSkillButton.boundingBox()
    expect(newSkillBox).not.toBeNull()
    expect(newSkillBox!.x).toBeGreaterThanOrEqual(0)
    expect(newSkillBox!.x + newSkillBox!.width).toBeLessThanOrEqual(390)

    await newSkillButton.focus()
    await newSkillButton.press("Enter")
    const drawer = page.getByRole("dialog", { name: "New Skill" })
    await expect(drawer).toBeVisible()
    await drawer.getByRole("button", { name: "Cancel" }).focus()
    await page.keyboard.press("Enter")
    await expect(newSkillButton).toBeFocused()
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth
      )
    ).toBe(true)

    await assertNoCriticalErrors(diagnostics)
  })
})

test.describe("Skills power-user journey (mocked)", () => {
  test("restores committed filters with browser Back and Forward", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockPowerUserSkillsLibrary(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "Filters", exact: true }).click()
    await page.getByLabel("Skill mode filter").click()
    await page.getByTitle("Fork", { exact: true }).click()
    await expect(page).toHaveURL(/(?:\?|&)mode=fork(?:&|$)/)
    await expect.poll(() => api.lastListUrl()?.searchParams.get("context")).toBe("fork")
    await expect(page.getByText("target-research-formatter", { exact: true })).toBeVisible()

    await page.goBack()
    await expect(page).not.toHaveURL(/(?:\?|&)mode=fork(?:&|$)/)
    await expect(page.getByText("archive-helper-01", { exact: true })).toBeVisible()

    await page.goForward()
    await expect(page).toHaveURL(/(?:\?|&)mode=fork(?:&|$)/)
    await expect(page.getByText("target-research-formatter", { exact: true })).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("fixture accepts body-less skill deletes with trailing slashes", async ({ page }) => {
    const api = await mockPowerUserSkillsLibrary(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    const statuses = await page.evaluate(async () => {
      const paths = [
        "/api/v1/skills/target-research-formatter",
        "/api/v1/skills/target-research-formatter/",
      ]
      const results: number[] = []
      for (const path of paths) {
        const response = await fetch(path, { method: "DELETE" })
        results.push(response.status)
      }
      return results
    })

    expect(statuses).toEqual([200, 200])
    expect(api.deleteRequests).toEqual([
      { name: "target-research-formatter" },
      { name: "target-research-formatter" },
    ])
  })

  test("finds a skill outside page one and opens bulk delete confirmation", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockPowerUserSkillsLibrary(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    const searchInput = page.getByPlaceholder("Search skills...")
    await searchInput.pressSequentially("target-research-formatter")
    await expect
      .poll(() => api.lastListUrl()?.searchParams.get("q"))
      .toBe("target-research-formatter")
    await expect(page.getByText("Target research formatter")).toBeVisible()
    await expect(page.getByText("1 skill", { exact: true })).toBeVisible()
    await expect(
      page.getByRole("button", { name: "View target-research-formatter" })
    ).toBeVisible()

    await searchInput.clear()
    await expect(page.getByText("30 skills", { exact: true })).toBeVisible()
    await expect(page.getByRole("button", { name: "View archive-helper-02" })).toBeVisible()

    await page.getByRole("button", { name: "Filters", exact: true }).click()
    await page.getByLabel("Skill mode filter").click()
    await page.getByTitle("Fork", { exact: true }).click()
    await page.getByLabel("Skill tools filter").click()
    await page.getByTitle("Has tools", { exact: true }).click()
    await expect(page.getByText("Batch cleanup helper")).toBeVisible()
    await expect(page.getByText("Target research formatter")).toBeVisible()
    await expect
      .poll(() => api.lastListUrl()?.searchParams.get("context"))
      .toBe("fork")
    await expect
      .poll(() => api.lastListUrl()?.searchParams.get("has_tools"))
      .toBe("true")
    await page.getByRole("button", { name: /^Filters/ }).click()

    await page.getByRole("columnheader", { name: "Name" }).click()
    await expect
      .poll(() => api.lastListUrl()?.searchParams.get("sort"))
      .toBe("name")

    await page.getByLabel("Select target-research-formatter").check()
    await page.getByLabel("Select batch-cleanup-helper").check()
    await expect(page.getByText("2 selected")).toBeVisible()
    await page.getByRole("button", { name: "Delete selected" }).click()
    await expect(
      page.getByRole("dialog", { name: "Delete selected skills?" })
    ).toBeVisible()
    expect(api.deleteRequests).toHaveLength(0)

    await assertNoCriticalErrors(diagnostics)
  })
})

test.describe("Skills Trash journey (mocked)", () => {
  test("offers immediate undo after moving a skill to Trash", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockSkillsTrashWorkflow(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "More actions for summarize" }).click()
    await page.getByRole("menuitem", { name: "Delete" }).click()
    const confirmDialog = page.getByRole("dialog", { name: "Delete summarize?" })
    await confirmDialog.getByRole("button", { name: "Move to Trash" }).click()

    await expect(page.getByText("Skill moved to Trash")).toBeVisible()
    await page.getByRole("button", { name: "Undo delete summarize" }).click()
    await expect(page.getByText("Summarize source material")).toBeVisible()
    expect(api.operations).toEqual(["delete", "restore"])
    expect(api.state()).toBe("active")

    await assertNoCriticalErrors(diagnostics)
  })

  test("restores a skill from the durable Trash view", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockSkillsTrashWorkflow(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "More actions for summarize" }).click()
    await page.getByRole("menuitem", { name: "Delete" }).click()
    const deleteDialog = page.getByRole("dialog", { name: "Delete summarize?" })
    await deleteDialog.getByRole("button", { name: "Move to Trash" }).click()
    await expect(deleteDialog).toBeHidden()
    await page
      .getByRole("radiogroup", { name: "Skills view" })
      .getByText("Trash", { exact: true })
      .click()
    await expect(page.getByRole("radio", { name: "Trash" })).toBeChecked()

    await expect(page.getByText("1 in Trash", { exact: true })).toBeVisible()
    await page.getByRole("button", { name: "Restore summarize" }).click()
    await expect(page.getByRole("heading", { name: "Trash is empty" })).toBeVisible()
    await page
      .getByRole("radiogroup", { name: "Skills view" })
      .getByText("Library", { exact: true })
      .click()
    await expect(page.getByRole("radio", { name: "Library" })).toBeChecked()
    await expect(page.getByText("Summarize source material")).toBeVisible()
    expect(api.operations).toEqual(["delete", "restore"])
    expect(api.state()).toBe("active")

    await assertNoCriticalErrors(diagnostics)
  })

  test("permanently deletes only after explicit confirmation", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockSkillsTrashWorkflow(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "More actions for summarize" }).click()
    await page.getByRole("menuitem", { name: "Delete" }).click()
    const deleteDialog = page.getByRole("dialog", { name: "Delete summarize?" })
    await deleteDialog.getByRole("button", { name: "Move to Trash" }).click()
    await expect(deleteDialog).toBeHidden()
    await page
      .getByRole("radiogroup", { name: "Skills view" })
      .getByText("Trash", { exact: true })
      .click()
    await expect(page.getByRole("radio", { name: "Trash" })).toBeChecked()

    await page.getByRole("button", { name: "Permanently delete summarize" }).click()
    const confirmDialog = page.getByRole("dialog", {
      name: "Permanently delete summarize?",
    })
    await expect(confirmDialog).toContainText("This cannot be undone.")
    await confirmDialog.getByRole("button", { name: "Delete permanently" }).click()

    await expect(page.getByRole("heading", { name: "Trash is empty" })).toBeVisible()
    expect(api.operations).toEqual(["delete", "purge"])
    expect(api.state()).toBe("purged")

    await assertNoCriticalErrors(diagnostics)
  })
})

test.describe("Skills failure states (mocked)", () => {
  test("shows invalid import feedback without importing", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockSkillsImportValidationFailure(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "Import", exact: true }).click()
    await page.getByRole("menuitem", { name: "Import Text" }).click()
    const dialog = page.getByRole("dialog", { name: "Import Skill from Text" })
    await expect(dialog).toBeVisible()
    await dialog.getByLabel("SKILL.md Content").fill("not a valid skill")
    await dialog.getByRole("button", { name: "Review import" }).click()

    await expect(dialog.getByText("Fix these issues before importing.")).toBeVisible()
    await expect(dialog.getByText("Missing skill description")).toBeVisible()
    expect(api.importRequests).toHaveLength(0)

    await assertNoCriticalErrors(diagnostics)
  })

  test("shows execution failure details from a test run", async ({
    page,
    diagnostics,
  }) => {
    await mockSkillsExecutionFailure(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "Test run summarize" }).click()
    const previewDialog = page.getByRole("dialog", { name: "Test run" })
    await previewDialog.getByPlaceholder("Enter test arguments...").fill("failure test")
    await previewDialog.getByRole("button", { name: "Run test" }).click()

    await expect(previewDialog.getByRole("alert")).toContainText("Model unavailable")

    await assertNoCriticalErrors(diagnostics)
  })

  test("tells users to reload before retrying a stale delete", async ({
    page,
    diagnostics,
  }) => {
    await mockSkillsStaleVersionConflict(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await page.getByRole("button", { name: "More actions for summarize" }).click()
    await page.getByRole("menuitem", { name: "Delete" }).click()
    const confirmDialog = page.getByRole("dialog", { name: "Delete summarize?" })
    await expect(confirmDialog).toBeVisible()
    await confirmDialog.getByRole("button", { name: "Move to Trash" }).click()

    await expect(page.getByText("Skill changed elsewhere")).toBeVisible()
    await expect(page.getByText("Reload skills before deleting this version.")).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("announces slow list loading until the response resolves", async ({
    page,
    diagnostics,
  }) => {
    const api = await mockSkillsSlowList(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await expect.poll(() => api.listRequests()).toBeGreaterThan(0)
    await expect(
      page.locator('div[role="status"]').filter({ hasText: "Loading skills" })
    ).toHaveText("Loading skills")

    api.resolveList()
    await expect(page.getByText("Summarize source material")).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })
})

test.describe("Skills", () => {
  test.beforeEach(async ({ authedPage, serverInfo }) => {
    void authedPage
    skipIfServerUnavailable(serverInfo)
  })

  test("page loads with interactive elements", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const skillsTable = authedPage.locator("table")
    const createButtons = authedPage.getByRole("button", {
      name: /new skill|add skill|create/i,
    })

    await expect(
      unsupportedHeading.or(skillsTable).or(createButtons).first()
    ).toBeVisible({ timeout: 15_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("skills list fires API on load when available", async ({
    authedPage,
    diagnostics,
  }) => {
    let skillsRequestMade = false
    const handler = (req: import("@playwright/test").Request) => {
      if (req.url().includes("/api/v1/skills") && req.method() === "GET") {
        skillsRequestMade = true
      }
    }
    authedPage.on("request", handler)

    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const skillsTable = authedPage.locator("table")
    const createButtons = authedPage.getByRole("button", {
      name: /new skill|add skill|create/i,
    })

    await expect(
      unsupportedHeading.or(skillsTable).or(createButtons).first()
    ).toBeVisible({ timeout: 15_000 })

    const unsupportedVisible = await unsupportedHeading
      .isVisible()
      .catch(() => false)

    if (unsupportedVisible) {
      await expect(unsupportedHeading).toBeVisible()
      expect(skillsRequestMade).toBe(false)
    } else {
      await expect
        .poll(() => skillsRequestMade, { timeout: 15_000 })
        .toBe(true)
    }

    authedPage.removeListener("request", handler)

    await assertNoCriticalErrors(diagnostics)
  })

  test("page has buttons or table for skill management", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const unsupportedVisible = await unsupportedHeading
      .isVisible({ timeout: 5_000 })
      .catch(() => false)

    if (unsupportedVisible) {
      await expect(
        authedPage.getByText(/skills api/i)
      ).toBeVisible({ timeout: 10_000 })
    } else {
      const interactiveElements = authedPage.locator(
        "button, input, select, textarea, table"
      )
      await expect(interactiveElements.first()).toBeVisible({ timeout: 15_000 })
      expect(await interactiveElements.count()).toBeGreaterThan(0)
    }

    await assertNoCriticalErrors(diagnostics)
  })
})
