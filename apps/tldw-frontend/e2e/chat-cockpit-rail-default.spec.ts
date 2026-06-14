import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #5: the Context/Runtime cockpit rails are fully expanded on first
 * load, presenting a lot of dense panels before any interaction. Default them
 * collapsed for users without a saved preference; once a user restores a rail the
 * choice persists. Returning users who set a preference are unaffected.
 */
const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

test("cockpit rails default collapsed for first-time users and persist once restored", async ({ page }) => {
  test.setTimeout(90_000)
  await page.setViewportSize({ width: 1440, height: 900 }) // lg width: restore affordance is shown
  await seedAuth(page, { serverUrl: SERVER, apiKey: KEY })
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })

  // Default: both rails collapsed -> restore affordances visible, rail content absent.
  await expect(page.getByTestId("playground-cockpit-left-rail-restore")).toBeVisible()
  await expect(page.getByTestId("playground-cockpit-right-rail-restore")).toBeVisible()
  await expect(page.getByTestId("playground-cockpit-left-rail")).toHaveCount(0)

  // Restoring a rail shows it and persists across reloads.
  await page.getByTestId("playground-cockpit-left-rail-restore").click()
  await expect(page.getByTestId("playground-cockpit-left-rail")).toBeVisible()

  await page.reload({ waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })
  await expect(page.getByTestId("playground-cockpit-left-rail")).toBeVisible()
})
