import { expect, test, type Page } from "@playwright/test"

/**
 * End-to-end coverage for the user flow:
 *   1. Open `/settings/chat`
 *   2. Click a variant card under "Composer style"
 *   3. Open `/chat?nextgenComposer=1`
 *   4. The picked variant renders
 *
 * This is the path a real user takes to switch composer styles. It
 * exercises the picker UI, the `useComposerVariantPreference` hook
 * write path, the localStorage persistence layer, and the dispatcher
 * read on /chat — all in one shot.
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

test.describe("composer · picker → /chat round-trip", () => {
  test("clicking V3 in Settings renders V3 on /chat", async ({ page }) => {
    test.setTimeout(120_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    await page.goto("/settings/chat")
    const card = page.getByRole("radio", { name: /split brief/i })
    await expect(card).toBeVisible({ timeout: 15_000 })
    await card.click()
    await expect(card).toHaveAttribute("aria-checked", "true")

    await page.goto("/chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v3']")).toBeVisible()
  })

  test("selected card shows the ✓ Active badge", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    await page.goto("/settings/chat")
    const v3Card = page.getByRole("radio", { name: /split brief/i })
    await expect(v3Card).toBeVisible({ timeout: 15_000 })
    await v3Card.click()

    // The selected card replaces its tag with the Active badge.
    const activeBadge = page.getByTestId("composer-variant-active-badge")
    await expect(activeBadge).toBeVisible()
    await expect(activeBadge).toHaveText(/Active/i)

    // Switching variants moves the badge to the new selection.
    const v5Card = page.getByRole("radio", { name: /radial command/i })
    await v5Card.click()
    await expect(activeBadge).toBeVisible()
    // Only one badge at a time
    await expect(
      page.getByTestId("composer-variant-active-badge")
    ).toHaveCount(1)
  })

  test("clicking V5 in Settings renders V5 on the Sidepanel", async ({
    page,
  }) => {
    test.setTimeout(120_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    await page.goto("/settings/chat")
    const card = page.getByRole("radio", { name: /radial command/i })
    await expect(card).toBeVisible({ timeout: 15_000 })
    await card.click()
    await expect(card).toHaveAttribute("aria-checked", "true")

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()
  })
})
