import { expect, test, type Page } from "@playwright/test"

/**
 * Smoke test for the Primer composer wire-up in the Sidepanel debug
 * route (`/__debug__/sidepanel-chat`). Mirrors the Playground spec.
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

test.describe("sidepanel · nextgen composer wire-up", () => {
  test("flag OFF: no nextgen composer rendered", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/__debug__/sidepanel-chat")
    // Wait long enough for the page to settle (no readiness gate here).
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    await expect(
      page.locator('[data-testid="nextgen-composer-wrapper"]')
    ).toHaveCount(0)
  })

  test("flag ON: nextgen composer mounts inside Sidepanel", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v1']")).toBeVisible()
  })

  test("flag ON: variant preference (v5) drives rendering in sidepanel", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("tldw:composerVariant", "v5")
      } catch {
        /* ignore */
      }
    })
    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()
  })

  test("nextgen composer textarea is rendered + accessible inside Sidepanel", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    // Real sidepanel textarea now lives inside the wrapper via textareaSlot.
    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
  })
})
