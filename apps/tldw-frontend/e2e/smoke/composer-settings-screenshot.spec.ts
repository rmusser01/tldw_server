import { test, type Page } from "@playwright/test"

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      /* ignore */
    }
  })
}

test("composer style settings — screenshot", async ({ page }) => {
  await bypassOnboarding(page)
  await page.setViewportSize({ width: 1440, height: 1200 })
  await page.goto("/settings/chat")
  await page.waitForSelector('[data-testid="composer-style-settings"]')
  const section = page.getByTestId("composer-style-settings")
  await section.screenshot({
    path: "test-results/composer-settings-picker.png",
  })
})
