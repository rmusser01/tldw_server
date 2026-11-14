/** Claims Review uses the shared decision workspace; native acceptance is run against a real server. */
import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../../utils/fixtures"

test.describe("Claims Review", () => {
  test.beforeEach(async ({ serverInfo }) => { skipIfServerUnavailable(serverInfo) })

  test("opens the Claims decision workflow without a content-draft redirect", async ({ authedPage, diagnostics }) => {
    await authedPage.goto("/claims-review", { waitUntil: "domcontentloaded" })
    await expect(authedPage.getByRole("heading", { name: "Claims Review", exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByTestId("route-redirect-panel")).toHaveCount(0)
    await expect(authedPage.getByLabel("Review status")).toBeVisible()
    await expect(authedPage.getByRole("button", { name: "Refresh queue" })).toBeVisible()
    await expect(authedPage.getByRole("button", { name: "Approve selected claims" })).toBeVisible()
    await expect(authedPage.getByRole("button", { name: "Reject selected claims" })).toBeVisible()
    await assertNoCriticalErrors(diagnostics)
  })
})
