import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { seedAuth } from "../../utils/helpers"

test.describe("Admin Overview", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
  })

  test("renders module overview without redirecting to /admin/server", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/admin", { waitUntil: "domcontentloaded" })

    await expect(
      authedPage.getByRole("heading", { name: "Admin Operations" })
    ).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByTestId("route-redirect-panel")).toHaveCount(0)
    await expect(authedPage).toHaveURL(/\/admin$/)

    for (const [label, href] of [
      ["Server Admin", "/admin/server"],
      ["Workspace Integrations", "/admin/integrations"],
      ["Admin Sources", "/admin/sources"],
      ["Monitoring", "/admin/monitoring"],
    ] as const) {
      const moduleCard = authedPage.getByTestId(`admin-module-${href}`)
      await expect(
        moduleCard.getByRole("link", { name: label })
      ).toHaveAttribute("href", href)
    }

    await expect(authedPage.getByText("Route ready")).toHaveCount(4)

    await assertNoCriticalErrors(diagnostics)
  })
})
