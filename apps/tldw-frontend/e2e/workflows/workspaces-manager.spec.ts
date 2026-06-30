/**
 * Workspaces Manager real-backend smoke coverage.
 *
 * These tests intentionally exercise route handoffs against a configured
 * backend/WebUI pair. They skip when the backend preflight is unavailable.
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import { seedAuth } from "../utils/helpers"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

test.describe("Workspaces Manager", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    await page.setViewportSize(DESKTOP_VIEWPORT)
  })

  test("loads the canonical manager without workspace-playground compatibility", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)

    await authedPage.goto("/workspaces", { waitUntil: "domcontentloaded" })
    await expect(
      authedPage.getByRole("heading", { name: "Workspaces" }).first()
    ).toBeVisible({ timeout: 20_000 })
    await expect(authedPage.getByText(/server-backed research and project/i)).toBeVisible()

    await authedPage.goto("/workspace-playground", {
      waitUntil: "domcontentloaded"
    })
    await expect(
      authedPage.getByRole("heading", { name: "Workspaces" })
    ).toBeHidden()

    await assertNoCriticalErrors(diagnostics)
  })

  test("navigates from Research Workspace settings to the canonical manager", async ({
    authedPage,
    serverInfo,
    diagnostics
  }) => {
    skipIfServerUnavailable(serverInfo)

    await authedPage.goto("/research-workspace", {
      waitUntil: "domcontentloaded"
    })
    await expect(
      authedPage.getByTestId("workspace-header")
    ).toBeVisible({ timeout: 30_000 })

    await authedPage.getByRole("button", { name: "Workspace settings" }).click()
    await authedPage.getByText("Manage in Workspaces").click()

    await expect(authedPage).toHaveURL(/\/workspaces(?:$|[?#])/)
    await expect(
      authedPage.getByRole("heading", { name: "Workspaces" }).first()
    ).toBeVisible({ timeout: 20_000 })

    await assertNoCriticalErrors(diagnostics)
  })
})
