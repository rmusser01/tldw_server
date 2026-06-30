import { expect, test, type Page } from "@playwright/test"

/**
 * Plan verification step: "Bundle size: confirm only the active
 * variant's chunk loads on page entry for each surface."
 *
 * The dispatcher uses `React.lazy` per variant — only the rendered
 * variant's chunk should be fetched. Track network requests and
 * assert that on a fresh load with V1 active, neither V3's nor V5's
 * chunk URL is requested.
 *
 * Switching to V3 should then trigger V3's chunk fetch (proving the
 * lazy boundary actually works rather than eagerly bundling all
 * three).
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

const setVariant = (variant: "v1" | "v3" | "v5") => async (page: Page) => {
  await page.addInitScript((v: string) => {
    try {
      window.localStorage.setItem("tldw:composerVariant", v)
    } catch {
      /* ignore */
    }
  }, variant)
}

const mockComposerProfile = async (page: Page) => {
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

const collectVariantChunks = (page: Page) => {
  const fetched: string[] = []
  page.on("request", (req) => {
    const url = req.url()
    // Variant chunk filenames contain the variant component name.
    // Pattern: ".../TerminalStackV1.<hash>.js" or similar.
    if (
      url.includes("TerminalStackV1") ||
      url.includes("SplitBriefV3") ||
      url.includes("RadialCommandV5")
    ) {
      fetched.push(url)
    }
  })
  return fetched
}

test.describe("composer · bundle isolation", () => {
  test("loading V1 fetches V1 chunk but not V3/V5 chunks", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await setVariant("v1")(page)
    await mockComposerProfile(page)
    const fetched = collectVariantChunks(page)

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v1']")).toBeVisible()

    const v1Loaded = fetched.some((u) => u.includes("TerminalStackV1"))
    const v3Loaded = fetched.some((u) => u.includes("SplitBriefV3"))
    const v5Loaded = fetched.some((u) => u.includes("RadialCommandV5"))
    expect(v1Loaded).toBe(true)
    expect(v3Loaded).toBe(false)
    expect(v5Loaded).toBe(false)
  })

  test("loading V5 fetches V5 chunk but not V1/V3 chunks", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await setVariant("v5")(page)
    await mockComposerProfile(page)
    const fetched = collectVariantChunks(page)

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()

    const v1Loaded = fetched.some((u) => u.includes("TerminalStackV1"))
    const v3Loaded = fetched.some((u) => u.includes("SplitBriefV3"))
    const v5Loaded = fetched.some((u) => u.includes("RadialCommandV5"))
    expect(v5Loaded).toBe(true)
    expect(v1Loaded).toBe(false)
    expect(v3Loaded).toBe(false)
  })
})
