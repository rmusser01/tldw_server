import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #4: pressing Escape should dismiss the shortcuts help panel and
 * return focus to the trigger.
 */
const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

test("Escape dismisses the shortcuts help panel and restores focus", async ({ page }) => {
  test.setTimeout(90_000)
  await seedAuth(page, { serverUrl: SERVER, apiKey: KEY })
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })

  const trigger = page.getByTestId("playground-shortcuts-help-trigger").first()
  await trigger.click()
  await expect(page.getByTestId("playground-shortcuts-help-panel")).toBeVisible()

  await page.keyboard.press("Escape")
  await expect(page.getByTestId("playground-shortcuts-help-panel")).toBeHidden()

  // Focus should return to the trigger for keyboard users.
  const focusedTestId = await page.evaluate(() =>
    document.activeElement?.getAttribute("data-testid"),
  )
  expect(focusedTestId).toBe("playground-shortcuts-help-trigger")
})
