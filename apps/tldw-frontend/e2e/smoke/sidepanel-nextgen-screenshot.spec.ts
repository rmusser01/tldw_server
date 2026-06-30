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

const mockComposerProfile = async (page: Page) => {
  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    const method = route.request().method()
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          profile_version: "2026-04-20T00:00:00Z",
          preferences: {},
        }),
      })
      return
    }
    if (method === "PATCH") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ applied: [], skipped: [] }),
      })
      return
    }
    await route.continue()
  })
}

const captureVariant = async (
  page: Page,
  variant: "v1" | "v3" | "v5",
  outputPath: string
) => {
  await bypassOnboarding(page)
  await mockComposerProfile(page)
  await page.addInitScript((v: string) => {
    try {
      window.localStorage.setItem("tldw:composerVariant", v)
    } catch {
      /* ignore */
    }
  }, variant)
  // Sidepanel-ish viewport — narrower than desktop /chat
  await page.setViewportSize({ width: 480, height: 1000 })
  await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
  await page.waitForSelector('[data-testid="nextgen-composer-wrapper"]', {
    timeout: 30_000,
  })
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

test("sidepanel · nextgen wrapper screenshot V1", async ({ page }, testInfo) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v1", testInfo.outputPath("sidepanel-nextgen-v1.png"))
})

test("sidepanel · nextgen wrapper screenshot V3", async ({ page }, testInfo) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v3", testInfo.outputPath("sidepanel-nextgen-v3.png"))
})

test("sidepanel · nextgen wrapper screenshot V5", async ({ page }, testInfo) => {
  test.setTimeout(90_000)
  await captureVariant(page, "v5", testInfo.outputPath("sidepanel-nextgen-v5.png"))
})
