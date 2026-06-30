import { expect, test, type Page } from "@playwright/test"

/**
 * Verifies V3's left brief panel shows the surface-specific fields
 * we wire from PlaygroundForm / Sidepanel form. Without this guard,
 * regressions where the brief defaults back to `briefSections=[]`
 * would silently leave V3 looking empty.
 */

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
      window.localStorage.setItem("tldw:composerVariant", "v3")
    } catch {
      /* ignore */
    }
  })
}

test.describe("composer · V3 brief data", () => {
  test("playground V3 brief shows mdl/chr/src/web fields", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v3']")).toBeVisible()

    // Desktop brief panel — role="group" aria-label="Brief"
    const brief = wrapper.getByRole("group", { name: "Brief" })
    await expect(brief).toBeVisible({ timeout: 10_000 })

    // Section header + each field key chip
    await expect(brief.getByText("Brief", { exact: true }).first()).toBeVisible()
    await expect(brief.getByText("mdl", { exact: true })).toBeVisible()
    await expect(brief.getByText("chr", { exact: true })).toBeVisible()
    await expect(brief.getByText("src", { exact: true })).toBeVisible()
    await expect(brief.getByText("web", { exact: true })).toBeVisible()
  })

  test("playground V3 brief: clicking 'web' toggles web search state", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    // Suppress the Next.js dev runtime error overlay (no backend = LLM
    // model metadata fetch returns 500). It blocks pointer events.
    await page.evaluate(() => {
      document.querySelectorAll("nextjs-portal, [role='dialog']").forEach(
        (el) => {
          if (el instanceof HTMLElement) el.style.display = "none"
        }
      )
    })

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })

    const brief = wrapper.getByRole("group", { name: "Brief" })
    await expect(brief).toBeVisible({ timeout: 10_000 })

    // The web field is an interactive button (BriefField sets role=button
    // + aria-pressed when onClick is provided). Locate it via its child
    // `web` key glyph rather than the aria-label, since the label flips
    // on toggle ("Turn web search on" → "off").
    const webField = brief
      .locator("button", { has: page.locator("text=web") })
      .first()
    await expect(webField).toBeVisible()
    await expect(webField).toHaveText(/weboff/)
    await webField.click()
    await expect(webField).toHaveText(/webon/)
  })

  test("sidepanel V3 brief renders fields (compact density hides keys)", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v3']")).toBeVisible()

    // The brief panel is present in compact mode too (collapsed to a chip
    // strip). Existence of the role="group" Brief is enough — exact field
    // keys aren't rendered when hideKey is set.
    const brief = wrapper.getByRole("group", { name: "Brief" })
    await expect(brief).toBeVisible({ timeout: 10_000 })
    // Field values DO render in compact mode — assert the chat mode is
    // present (the mode value we wire is always rendered: normal/rag/vision).
    await expect(brief.getByText(/normal|rag|vision/)).toBeVisible()
  })
})
