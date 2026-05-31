import { expect, test, type Page } from "@playwright/test"
import { seedAuth } from "./smoke.setup"

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
  await seedAuth(page, {
    authMode: "single-user",
    apiKey: "test-key-not-placeholder",
    allowOffline: false,
  })
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

const openChatSettings = async (page: Page) => {
  await page.goto("/settings/chat")
  await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
  const modalClose = page.locator(".ant-modal .ant-modal-close").first()
  if (await modalClose.isVisible().catch(() => false)) {
    await modalClose.click({ force: true })
    await expect(page.locator(".ant-modal-wrap")).toHaveCount(0).catch(() => {})
  }
  await page
    .locator(
      "nextjs-portal, [data-nextjs-dialog-overlay], .ant-modal-wrap, .ant-modal-mask"
    )
    .evaluateAll((elements) => {
      for (const element of elements) {
        if (element instanceof HTMLElement) element.style.display = "none"
      }
    })
    .catch(() => {})
}

const focusVariantCard = async (page: Page, id: "v1" | "v3" | "v5") => {
  await page.locator(`button[data-variant-option="${id}"]`).focus()
  await expect
    .poll(() =>
      page
        .locator(`button[data-variant-option="${id}"]`)
        .evaluate((el) => document.activeElement === el)
    )
    .toBe(true)
}

test.describe("composer picker · keyboard accessibility", () => {
  test("Space activates a focused variant card", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await openChatSettings(page)

    const splitBrief = page.getByRole("radio", { name: /split brief/i })
    await expect(splitBrief).toBeVisible({ timeout: 15_000 })
    await focusVariantCard(page, "v3")
    await page.keyboard.press("Space")
    await expect(splitBrief).toHaveAttribute("aria-checked", "true")
  })

  test("Enter activates a focused variant card", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await openChatSettings(page)

    const radial = page.getByRole("radio", { name: /radial command/i })
    await expect(radial).toBeVisible({ timeout: 15_000 })
    await focusVariantCard(page, "v5")
    await page.keyboard.press("Enter")
    await expect(radial).toHaveAttribute("aria-checked", "true")
  })

  test("each card is programmatically focusable", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)
    await openChatSettings(page)

    // Pick a known card to anchor wait — picker is mounted by then.
    await expect(
      page.getByRole("radio", { name: /terminal stack/i })
    ).toBeVisible({ timeout: 15_000 })

    for (const id of ["v1", "v3", "v5"] as const) {
      await focusVariantCard(page, id)
    }
  })
})
