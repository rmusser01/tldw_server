import { test, type Page } from "@playwright/test"

/**
 * Captures a full-page screenshot of the composer preview route.
 * Not an assertion — just produces an artifact for visual review.
 */
const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      /* ignore */
    }
  })
}

test("composer preview — full-page screenshot", async ({ page }, testInfo) => {
  await bypassOnboarding(page)
  await page.setViewportSize({ width: 1440, height: 2600 })
  await page.goto("/composer-variants-preview")
  // Let the client-side render settle
  await page.waitForSelector("[data-variant='v1']", { state: "visible" })
  await page.waitForSelector("[data-variant='v3']", { state: "visible" })
  await page.waitForSelector("[data-variant='v5']", { state: "visible" })
  await page.screenshot({
    path: testInfo.outputPath("composer-variants-preview.png"),
    fullPage: true,
  })
})
