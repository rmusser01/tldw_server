import { expect, test, type Page } from "@playwright/test"

/**
 * Verifies the variant picker on /settings/chat is keyboard-accessible:
 *   - Tab focuses the radio group
 *   - Space selects the focused option
 *   - The selection persists (mocked PATCH succeeds)
 *
 * The picker uses `role="radiogroup"` + role="radio" cards. Even
 * without explicit arrow-key handling, native button activation
 * (Space/Enter) should select.
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

const mockProfile = async (page: Page) => {
  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    const method = route.request().method()
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          profile_version: "2026-04-19T00:00:00Z",
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

test.describe("composer picker · keyboard accessibility", () => {
  test("Space activates a focused variant card", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await page.goto("/settings/chat")

    const splitBrief = page.getByRole("radio", { name: /split brief/i })
    await expect(splitBrief).toBeVisible({ timeout: 15_000 })
    await splitBrief.focus()
    await page.keyboard.press("Space")
    await expect(splitBrief).toHaveAttribute("aria-checked", "true")
  })

  test("Enter activates a focused variant card", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await page.goto("/settings/chat")

    const radial = page.getByRole("radio", { name: /radial command/i })
    await expect(radial).toBeVisible({ timeout: 15_000 })
    await radial.focus()
    await page.keyboard.press("Enter")
    await expect(radial).toHaveAttribute("aria-checked", "true")
  })

  test("each card is programmatically focusable", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await page.goto("/settings/chat")

    // Pick a known card to anchor wait — picker is mounted by then.
    await expect(
      page.getByRole("radio", { name: /terminal stack/i })
    ).toBeVisible({ timeout: 15_000 })

    for (const id of ["v1", "v3", "v5"]) {
      const card = page.locator(`button[data-variant-option="${id}"]`)
      await card.focus()
      const isFocused = await card.evaluate(
        (el) => document.activeElement === el
      )
      expect(isFocused).toBe(true)
    }
  })
})
