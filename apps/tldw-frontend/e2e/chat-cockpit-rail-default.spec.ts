import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #5: the Context/Runtime cockpit rails are fully expanded on first
 * load, presenting a lot of dense panels before any interaction. Default them
 * collapsed for users without a saved preference; once a user restores a rail the
 * choice persists. Returning users who set a preference are unaffected.
 *
 * Auth and base URL come from the env-driven smoke config (seedAuth + Playwright baseURL).
 */
test("cockpit rails default collapsed for first-time users and persist once restored", async ({ page }) => {
  test.setTimeout(90_000)
  await page.setViewportSize({ width: 1440, height: 900 }) // lg width: restore affordance is shown
  await seedAuth(page)
  await page.goto("/chat", { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })

  // Default: both rails collapsed -> restore affordances visible, rail content absent.
  await expect(page.getByTestId("playground-cockpit-left-rail-restore")).toBeVisible()
  await expect(page.getByTestId("playground-cockpit-right-rail-restore")).toBeVisible()
  await expect(page.getByTestId("playground-cockpit-left-rail")).toHaveCount(0)
  await expect(page.getByTestId("playground-cockpit-right-rail")).toHaveCount(0)

  // Restoring a rail shows it and persists across reloads.
  await page.getByTestId("playground-cockpit-left-rail-restore").click()
  await expect(page.getByTestId("playground-cockpit-left-rail")).toBeVisible()

  await page.reload({ waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })
  await expect(page.getByTestId("playground-cockpit-left-rail")).toBeVisible()
})
