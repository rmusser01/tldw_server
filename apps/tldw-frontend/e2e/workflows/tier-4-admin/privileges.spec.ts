import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"

test.describe("Privileges Page", () => {
  test.beforeEach(async ({ serverInfo }) => {
    skipIfServerUnavailable(serverInfo)
  })

  test("privileges page loads and shows redirect panel", async ({ authedPage, diagnostics }) => {
    await authedPage.goto("/privileges", { waitUntil: "domcontentloaded" })

    // Privileges is a RouteRedirect to /settings — verify redirect panel or destination
    const redirectPanel = authedPage.getByTestId("route-redirect-panel")
    const panelVisible = await redirectPanel.isVisible().catch(() => false)

    if (panelVisible) {
      // Verify the redirect panel heading
      await expect(
        authedPage.getByRole("heading", { name: /privileges moved to settings/i })
      ).toBeVisible()

      // Verify redirect description
      await expect(
        authedPage.getByText(/role and permission controls/i)
      ).toBeVisible()
    } else {
      // If redirect happened already, we should be on /settings
      await expect(authedPage).toHaveURL(/\/settings/, { timeout: 10_000 })
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("privileges redirect targets /settings", async ({ authedPage, diagnostics }) => {
    await authedPage.goto("/privileges", { waitUntil: "domcontentloaded" })

    const redirectPanel = authedPage.getByTestId("route-redirect-panel")
    const panelVisible = await redirectPanel.isVisible().catch(() => false)

    if (panelVisible) {
      // Verify the destination path is shown as /settings
      await expect(authedPage.getByText("/settings")).toBeVisible()

      // "Open updated page" link should point to /settings
      const openUpdatedLink = authedPage.getByTestId("route-redirect-open-updated-page")
      await expect(openUpdatedLink).toBeVisible()
      await expect(openUpdatedLink).toHaveAttribute("href", "/settings")
    } else {
      await expect(authedPage).toHaveURL(/\/settings/, { timeout: 15_000 })
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("privileges redirect panel has fallback navigation links", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/privileges", { waitUntil: "domcontentloaded" })

    const redirectPanel = authedPage.getByTestId("route-redirect-panel")
    const panelVisible = await redirectPanel.isVisible().catch(() => false)

    if (panelVisible) {
      // "Open Home" fallback link
      const openHomeLink = authedPage.getByTestId("route-redirect-open-home")
      await expect(openHomeLink).toBeVisible()
      await expect(openHomeLink).toHaveAttribute("href", "/")

      // "Open Settings" fallback link
      const openSettingsLink = authedPage.getByTestId("route-redirect-open-settings")
      await expect(openSettingsLink).toBeVisible()
      await expect(openSettingsLink).toHaveAttribute("href", "/settings")
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("privileges page renders without critical errors", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/privileges", { waitUntil: "domcontentloaded" })

    // Verify the page rendered without uncaught errors regardless of redirect state
    const body = authedPage.locator("body")
    await expect(body).not.toBeEmpty()

    await assertNoCriticalErrors(diagnostics)
  })
})
