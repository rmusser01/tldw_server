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

test.describe("Skills", () => {
  test.beforeEach(async ({ authedPage, serverInfo }) => {
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
