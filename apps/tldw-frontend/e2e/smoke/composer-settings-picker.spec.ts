import { expect, test, type Page } from "@playwright/test"

/**
 * Smoke test for the Composer style picker on the Chat settings page.
 * Verifies the section mounts, radio semantics work, and the selection
 * persists to localStorage (the same key the `<ChatComposer>` dispatcher
 * reads).
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

test.describe("composer style settings picker", () => {
  test("picker renders and switches the stored preference", async ({ page }) => {
    await bypassOnboarding(page)
    await page.goto("/settings/chat")

    // Section mounts
    const section = page.getByTestId("composer-style-settings")
    await expect(section).toBeVisible()

    // All three radio cards are present
    const v1 = page.getByRole("radio", { name: /terminal stack/i })
    const v3 = page.getByRole("radio", { name: /split brief/i })
    const v5 = page.getByRole("radio", { name: /radial command/i })
    await expect(v1).toBeVisible()
    await expect(v3).toBeVisible()
    await expect(v5).toBeVisible()

    // V1 is the default
    await expect(v1).toHaveAttribute("aria-checked", "true")
    await expect(v3).toHaveAttribute("aria-checked", "false")
    await expect(v5).toHaveAttribute("aria-checked", "false")

    // Click V5 → selection flips + localStorage updates
    await v5.click()
    await expect(v5).toHaveAttribute("aria-checked", "true")
    await expect(v1).toHaveAttribute("aria-checked", "false")

    const stored = await page.evaluate(() =>
      window.localStorage.getItem("tldw:composerVariant")
    )
    expect(stored).toBe("v5")

    // Reload — selection persists
    await page.reload()
    await expect(
      page.getByRole("radio", { name: /radial command/i })
    ).toHaveAttribute("aria-checked", "true")
  })
})
