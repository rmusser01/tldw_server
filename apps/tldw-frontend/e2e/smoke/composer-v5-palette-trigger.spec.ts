import { expect, test, type Page } from "@playwright/test"

/**
 * V5's pill shows a ⌘K button. Per the original plan, clicking it
 * should inject `/` into the textarea, opening the existing
 * ComposerTextarea slash menu rather than a separate palette UI.
 *
 * Asserts: after clicking the ⌘K button, the chat input value starts
 * with `/` (which is the trigger character for the slash menu).
 */

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
      window.localStorage.setItem("tldw:composerVariant", "v5")
    } catch {
      /* ignore */
    }
  })
}

test("composer · V5 ⌘K injects / into the textarea", async ({ page }) => {
  test.setTimeout(90_000)
  await bypassOnboarding(page)

  // Sidepanel surface — no ServerReadinessGate, less noise.
  await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
  await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

  const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
  await expect(wrapper).toBeVisible({ timeout: 30_000 })
  await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()

  const chatInput = wrapper.locator('[data-testid="chat-input"]')
  await expect(chatInput).toBeVisible()
  await expect(chatInput).toHaveValue("")

  // Locate by aria-label; use force-click because the Sidepanel form's
  // disabled-when-offline treatment can bubble up to the a11y tree
  // even though this button is not literally disabled.
  const paletteBtn = wrapper.locator('button[aria-label="Open command palette"]')
  await expect(paletteBtn).toBeVisible()
  await paletteBtn.click({ force: true })

  // A single `/` should now be in the textarea
  await expect(chatInput).toHaveValue("/")

  // Clicking again should NOT stack extra `/` — guard against
  // repeated-press spam doubling the input
  await paletteBtn.click({ force: true })
  await expect(chatInput).toHaveValue("/")
})
