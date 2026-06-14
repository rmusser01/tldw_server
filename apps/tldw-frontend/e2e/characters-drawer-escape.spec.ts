import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT (/characters): pressing Escape should close the "New character" drawer — the
 * page even documents "Esc Close modal". Root cause of the failure is app-wide: the
 * globally-mounted CommandPalette registers an Escape `useShortcut` with no `enabled`
 * gate, and useShortcut's capture-phase handler calls stopPropagation() on every
 * Escape, defeating antd Drawer/Modal's built-in Escape-to-close everywhere.
 *
 * Auth + base URL come from the env-driven smoke config (seedAuth + Playwright baseURL).
 */
test("Escape closes the New character drawer", async ({ page }) => {
  test.setTimeout(90_000)
  await seedAuth(page)
  await page.goto("/characters", { waitUntil: "domcontentloaded" })
  await page.getByTestId("characters-page").first().waitFor({ state: "visible", timeout: 30_000 })

  await page.getByTestId("characters-new-button").first().click()
  await expect(page.getByText(/Choose a template/i).first()).toBeVisible()

  await page.keyboard.press("Escape")
  await expect(page.getByText(/Choose a template/i).first()).toBeHidden()
})
