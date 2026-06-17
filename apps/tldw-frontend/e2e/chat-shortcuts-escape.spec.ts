import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #4: pressing Escape should dismiss the shortcuts help panel and
 * return focus to the trigger.
 *
 * Auth and base URL come from the env-driven smoke config (seedAuth + Playwright baseURL).
 */
test("Escape dismisses the shortcuts help panel and restores focus", async ({ page }) => {
  test.setTimeout(90_000)
  await seedAuth(page)
  await page.goto("/chat", { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })

  const trigger = page.getByTestId("playground-shortcuts-help-trigger").first()
  await trigger.click()
  await expect(page.getByTestId("playground-shortcuts-help-panel")).toBeVisible()

  await page.keyboard.press("Escape")
  await expect(page.getByTestId("playground-shortcuts-help-panel")).toBeHidden()

  // Focus is restored asynchronously (requestAnimationFrame); toBeFocused retries.
  await expect(page.getByTestId("playground-shortcuts-help-trigger")).toBeFocused()
})
