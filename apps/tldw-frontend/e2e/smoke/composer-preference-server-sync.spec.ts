import { expect, test, type Page } from "@playwright/test"

/**
 * End-to-end verification of the Phase-4 server preference sync.
 * Confirms the picker now round-trips through `/api/v1/users/me/profile`
 * via Playwright route mocking — no real backend required.
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

test.describe("composer preference · server sync", () => {
  test("hydrates from server on mount + PATCHes on change", async ({ page }) => {
    test.setTimeout(60_000)
    await bypassOnboarding(page)

    let patchedPayload: any = null

    // Mock the profile endpoints on whatever absolute API URL the app uses.
    await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
      const method = route.request().method()
      if (method === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            profile_version: "2026-04-19T00:00:00Z",
            catalog_version: "1.0.0",
            preferences: {
              "preferences.ui.composer_variant": "v5",
            },
          }),
        })
        return
      }
      if (method === "PATCH") {
        patchedPayload = route.request().postDataJSON()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            profile_version: "2026-04-19T00:00:01Z",
            applied: [{ key: "preferences.ui.composer_variant", value: "v3" }],
            skipped: [],
          }),
        })
        return
      }
      await route.continue()
    })

    await page.goto("/settings/chat")

    // Server says v5 — the picker should reflect it once hydration completes.
    const v5 = page.getByRole("radio", { name: /radial command/i })
    await expect(v5).toHaveAttribute("aria-checked", "true", {
      timeout: 10_000,
    })

    // Click V3 — picker updates immediately, PATCH fires.
    await page.getByRole("radio", { name: /split brief/i }).click()
    await expect(
      page.getByRole("radio", { name: /split brief/i })
    ).toHaveAttribute("aria-checked", "true")

    // Wait briefly for the fire-and-forget PATCH to land in our mock.
    await expect.poll(() => patchedPayload, { timeout: 10_000 }).toBeTruthy()

    expect(patchedPayload).toEqual({
      updates: [
        { key: "preferences.ui.composer_variant", value: "v3" },
      ],
    })
  })
})
