import { expect, test, type Page, type Route } from "@playwright/test"
import { seedAuth } from "./smoke.setup"

/**
 * Smoke test for the Primer composer redesign preview route.
 * Verifies:
 *   - Route renders without console errors
 *   - All three variants mount under the expected `[data-variant]` attrs
 *   - The <ChatComposer> dispatcher switches variants on button click
 *   - Key primitives are present in the DOM (source chip, brief, facets, palette)
 *
 * Run: npm run e2e:pw -- e2e/smoke/composer-variants-preview.spec.ts
 */

/**
 * Bypass the "Build Your Assistant" onboarding modal by pre-setting the
 * dismiss flag the app checks. Must run BEFORE page.goto() — addInitScript
 * applies on every navigation in the context.
 */
const bypassOnboarding = async (page: Page) => {
  await seedAuth(page)
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      // ignore
    }
  })
}

const waitForPreviewHarness = async (page: Page) => {
  await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
  await expect(page.getByRole("button", { name: "V1" })).toBeVisible({
    timeout: 30_000,
  })
}

const notificationHeaders = {
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,OPTIONS",
  "access-control-allow-origin": "*"
}

const fulfillNotificationOptions = async (route: Route) => {
  if (route.request().method() !== "OPTIONS") return false
  await route.fulfill({ status: 204, headers: notificationHeaders })
  return true
}

const stubNotifications = async (page: Page) => {
  await page.route(/\/api\/v1\/notifications\/stream(?:\?.*)?$/, async (route) => {
    if (await fulfillNotificationOptions(route)) return
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: {
        ...notificationHeaders,
        "cache-control": "no-cache"
      },
      body: ""
    })
  })

  await page.route(/\/api\/v1\/notifications\/unread-count(?:\?.*)?$/, async (route) => {
    if (await fulfillNotificationOptions(route)) return
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: notificationHeaders,
      body: JSON.stringify({ unread_count: 0 })
    })
  })

  await page.route(/\/api\/v1\/notifications(?:\?.*)?$/, async (route) => {
    if (await fulfillNotificationOptions(route)) return
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: notificationHeaders,
      body: JSON.stringify({
        items: [],
        total: 0,
        limit: 1,
        offset: 0,
        has_more: false
      })
    })
  })
}

test.describe("composer variants preview", () => {
  test("renders all three variants + live dispatcher", async ({ page }) => {
    test.setTimeout(90_000)
    const errors: string[] = []
    page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`))
    page.on("console", (msg) => {
      if (msg.type() !== "error") return
      const text = msg.text()
      // Filter out backend resource failures — the app's shell fires
      // API calls (notifications, connection status, etc.) that aren't
      // part of the preview route under test.
      if (/Failed to load resource/i.test(text)) return
      if (/status of 5\d{2}/.test(text)) return
      if (/status of 4\d{2}/.test(text)) return
      errors.push(`console: ${text}`)
    })

    await bypassOnboarding(page)
    await stubNotifications(page)
    await page.goto("/composer-variants-preview")
    await waitForPreviewHarness(page)

    // Each variant mounted at least once
    await expect(page.locator("[data-variant='v1']").first()).toBeVisible({
      timeout: 30_000,
    })
    await expect(page.locator("[data-variant='v3']").first()).toBeVisible({
      timeout: 30_000,
    })
    await expect(page.locator("[data-variant='v5']").first()).toBeVisible({
      timeout: 30_000,
    })

    // V1 source chip is in the DOM (14 + irb-archive label)
    await expect(page.getByText("irb-archive").first()).toBeVisible()

    // V3 Brief section header
    await expect(page.getByText("Brief").first()).toBeVisible()

    // V5 facet row rendered (role=group with label "Composer facets")
    await expect(
      page.getByRole("group", { name: /composer facets/i }).first()
    ).toBeVisible()

    // V5 inline slash palette is open in the demo state
    await expect(
      page.getByRole("listbox", { name: /composer slash commands/i })
    ).toBeVisible()

    // Live dispatcher section has variant-picker buttons
    const v1Btn = page.getByRole("button", { name: "V1" })
    const v3Btn = page.getByRole("button", { name: "V3" })
    const v5Btn = page.getByRole("button", { name: "V5" })
    await expect(v1Btn).toBeVisible()
    await expect(v3Btn).toBeVisible()
    await expect(v5Btn).toBeVisible()

    // Click V3 — dispatcher should re-render with v3 variant attribute
    await v3Btn.click()
    const liveRegion = page.locator("section").filter({
      has: page.getByText(/previewVariant=v3/),
    })
    await expect(liveRegion).toBeVisible()

    // Click V5 — dispatcher should render with v5 variant
    await v5Btn.click()
    const liveRegionV5 = page.locator("section").filter({
      has: page.getByText(/previewVariant=v5/),
    })
    await expect(liveRegionV5).toBeVisible()

    // No runtime errors
    expect(errors, `Console/runtime errors: ${errors.join(" | ")}`).toEqual([])
  })

  test("preview variant resets on reload instead of mutating real preferences", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await stubNotifications(page)
    await page.goto("/composer-variants-preview")
    await waitForPreviewHarness(page)
    await page.getByRole("button", { name: "V5" }).click()

    const v5Btn = page.getByRole("button", { name: "V5" })
    await expect(v5Btn).toHaveAttribute("aria-pressed", "true")

    // Reload — the preview returns to its local default rather than
    // persisting into the real composer preference store.
    await page.reload()
    await waitForPreviewHarness(page)

    const v1Btn = page.getByRole("button", { name: "V1" })
    await expect(v1Btn).toHaveAttribute("aria-pressed", "true")
    await expect(page.getByText(/previewVariant=v1/)).toBeVisible()
  })
})
