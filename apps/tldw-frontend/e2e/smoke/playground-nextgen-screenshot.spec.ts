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

const captureVariant = async (
  page: Page,
  variant: "v1" | "v3" | "v5",
  outputPath: string
) => {
  await bypassOnboarding(page)
  await page.addInitScript((v: string) => {
    try {
      window.localStorage.setItem("tldw:composerVariant", v)
    } catch {
      /* ignore */
    }
  }, variant)
  await page.setViewportSize({ width: 1440, height: 1100 })
  await page.goto("/chat?nextgenComposer=1")
  await page.waitForSelector('[data-testid="nextgen-composer-wrapper"]', {
    timeout: 30_000,
  })
  // Dismiss the Next.js dev error overlay (backend isn't running, model
  // fetches throw — irrelevant to the composer visual).
  await page.evaluate(() => {
    document
      .querySelectorAll("nextjs-portal, [role='dialog']")
      .forEach((el) => {
        if (el instanceof HTMLElement) el.style.display = "none"
      })
  })
  const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
  await wrapper.screenshot({ path: outputPath })
}

test("playground · nextgen wrapper screenshot V1", async ({ page }) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v1", "test-results/playground-nextgen-v1.png")
})

test("playground · nextgen wrapper screenshot V3", async ({ page }) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v3", "test-results/playground-nextgen-v3.png")
})

test("playground · nextgen wrapper screenshot V5", async ({ page }) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v5", "test-results/playground-nextgen-v5.png")
})
