import { expect, test, type Page } from "@playwright/test"

/**
 * Verifies the "Enable new composer" toggle on /settings/chat lets
 * users opt into the nextgen composer without the URL flag.
 *
 * Flow:
 *   - Without the flag, /chat renders the legacy composer
 *   - Check the Settings toggle
 *   - /chat (no flag) now renders the nextgen composer wrapper
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

test.describe("composer · Settings enable-toggle", () => {
  test("toggle off → /chat has no nextgen wrapper; toggle on → wrapper mounts without URL flag", async ({
    page,
  }) => {
    test.setTimeout(120_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    // 1. Baseline: flag off, no storage toggle → legacy composer
    await page.goto("/chat")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    await expect(
      page.locator('[data-testid="nextgen-composer-wrapper"]')
    ).toHaveCount(0)

    // 2. Flip the toggle on from Settings
    await page.goto("/settings/chat")
    const toggle = page.getByTestId("composer-enabled-toggle")
    await expect(toggle).toBeVisible({ timeout: 15_000 })
    await expect(toggle).not.toBeChecked()
    await toggle.check()
    await expect(toggle).toBeChecked()

    // 3. Revisit /chat — no URL flag — the wrapper should now mount
    await page.goto("/chat")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v1']")).toBeVisible()

    // 4. Turn it back off — subsequent /chat loads should revert
    await page.goto("/settings/chat")
    const toggleAgain = page.getByTestId("composer-enabled-toggle")
    await expect(toggleAgain).toBeChecked()
    await toggleAgain.uncheck()
    await expect(toggleAgain).not.toBeChecked()

    await page.goto("/chat")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    await expect(
      page.locator('[data-testid="nextgen-composer-wrapper"]')
    ).toHaveCount(0)
  })

  test("full user journey: enable + pick V3 + /chat shows V3 composer", async ({
    page,
  }) => {
    test.setTimeout(120_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    // 1. User opens Settings
    await page.goto("/settings/chat")

    // 2. Checks "Enable new composer"
    const toggle = page.getByTestId("composer-enabled-toggle")
    await expect(toggle).toBeVisible({ timeout: 15_000 })
    await toggle.check()
    await expect(toggle).toBeChecked()

    // 3. Picks V3
    const v3Card = page.getByRole("radio", { name: /split brief/i })
    await v3Card.click()
    await expect(v3Card).toHaveAttribute("aria-checked", "true")
    await expect(
      page.getByTestId("composer-variant-active-badge")
    ).toBeVisible()

    // 4. Opens /chat (no URL flag)
    await page.goto("/chat")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    // 5. Sees V3 composer with real chat input
    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v3']")).toBeVisible()
    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
  })

  test("toggle on → /__debug__/sidepanel-chat also mounts nextgen without URL flag", async ({
    page,
  }) => {
    test.setTimeout(120_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    // 1. Flip toggle from Settings
    await page.goto("/settings/chat")
    const toggle = page.getByTestId("composer-enabled-toggle")
    await expect(toggle).toBeVisible({ timeout: 15_000 })
    await toggle.check()
    await expect(toggle).toBeChecked()

    // 2. Sidepanel without URL flag should render the nextgen wrapper
    await page.goto("/__debug__/sidepanel-chat")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v1']")).toBeVisible()
  })

  test("URL flag still works independently of the toggle", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await mockProfile(page)

    // Ensure the localStorage toggle is off.
    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("tldw:nextgenComposerEnabled", "0")
      } catch {
        /* ignore */
      }
    })

    // URL flag alone should enable the composer even if toggle is off.
    await page.goto("/chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
  })
})
