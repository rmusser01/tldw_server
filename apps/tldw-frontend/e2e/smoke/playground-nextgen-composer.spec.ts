import { expect, test, type Page } from "@playwright/test"

/**
 * Smoke test for the Primer composer wire-up in Playground (`/chat`).
 *
 *   - Without the flag, the new composer is NOT mounted (legacy path)
 *   - With `?nextgenComposer=1`, <ChatComposer> wraps the real
 *     `ComposerTextarea` + `ComposerToolbar` (legacy goes away)
 *   - The active variant's `data-variant` marker is present
 *   - Typing into the chat input still updates the form state
 *   - Variant preference from localStorage drives which variant renders
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

/**
 * `/chat` is gated behind a 15-second `ServerReadinessGate` health check.
 * Without a real backend, the gate eventually falls through on timeout
 * and renders Playground. Wait long enough to clear the gate before
 * asserting on Playground content.
 */
const waitForPlaygroundChrome = async (page: Page) => {
  // After the gate falls through, Playground mounts. The form area
  // (where our wrapper sits) appears soon after.
  await page.waitForSelector("body", { state: "attached" })
  await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
}

test.describe("playground · nextgen composer wire-up", () => {
  test("flag OFF: no nextgen composer rendered", async ({ page }) => {
    test.setTimeout(90_000)

    await bypassOnboarding(page)
    await page.goto("/chat")
    await waitForPlaygroundChrome(page)
    await expect(
      page.locator('[data-testid="nextgen-composer-wrapper"]')
    ).toHaveCount(0)
  })

  test("flag ON: nextgen composer mounts with chosen variant", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/chat?nextgenComposer=1")
    await waitForPlaygroundChrome(page)

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v1']")).toBeVisible()
  })

  test("flag ON: variant preference (v3) drives rendering", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("tldw:composerVariant", "v3")
      } catch {
        /* ignore */
      }
    })
    await page.goto("/chat?nextgenComposer=1")
    await waitForPlaygroundChrome(page)

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v3']")).toBeVisible()
  })

  test("typing in nextgen composer updates the textarea value", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/chat?nextgenComposer=1")
    await waitForPlaygroundChrome(page)

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })

    // Real ComposerTextarea now lives inside the wrapper via the textareaSlot
    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    await chatInput.fill("hello from nextgen")
    await expect(chatInput).toHaveValue("hello from nextgen")
  })
})
