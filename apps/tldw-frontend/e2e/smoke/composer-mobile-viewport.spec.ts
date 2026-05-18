import { expect, test, type Page } from "@playwright/test"

/**
 * Mobile-viewport smoke for {V1, V3, V5} at narrow widths. The plan
 * calls for "resize each surface to ~360px; verify all three variants
 * degrade cleanly (especially V3's brief panel collapsing to a chip
 * strip above the textarea)."
 *
 * The Playground page itself isn't designed for sub-tablet widths
 * (the legacy composer also overflows there), so we test the
 * Sidepanel surface at 360px (it IS designed for narrow widths) and
 * the Playground at 768px (tablet — its breakpoint where the layout
 * collapses gracefully).
 *
 * We don't snapshot pixels. Instead we assert structural invariants:
 *   - the variant root [data-variant=vN] exists
 *   - the chat input is present and visible
 *   - the composer WRAPPER itself does not exceed its container
 *     (rules out variant-level layout regressions, even if the rest
 *     of the page has its own overflow story)
 */

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    const authConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
    }

    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      /* ignore */
    }
    try {
      window.localStorage.setItem("__tldw_first_run_complete", "true")
    } catch {
      /* ignore */
    }
    try {
      window.localStorage.setItem("__tldw_allow_offline", "true")
    } catch {
      /* ignore */
    }
    try {
      window.localStorage.setItem("__tldw_test_bypass", "true")
    } catch {
      /* ignore */
    }
    try {
      window.localStorage.setItem("tldwConfig", JSON.stringify(authConfig))
      window.localStorage.setItem("apiKey", authConfig.apiKey)
      window.localStorage.setItem("authMode", authConfig.authMode)
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

const SIDEPANEL_MATRIX: Array<{ variant: "v1" | "v3" | "v5"; width: number }> = [
  { variant: "v1", width: 360 },
  { variant: "v3", width: 360 },
  { variant: "v5", width: 360 },
]

const PLAYGROUND_MATRIX: Array<{ variant: "v1" | "v3" | "v5"; width: number }> = [
  { variant: "v1", width: 768 },
  { variant: "v3", width: 768 },
  { variant: "v5", width: 768 },
]

test.describe("composer · mobile viewport", () => {
  for (const { variant, width } of SIDEPANEL_MATRIX) {
    test(`sidepanel: ${variant} at ${width}px fits without composer overflow`, async ({
      page,
    }) => {
      test.setTimeout(90_000)
      await bypassOnboarding(page)
      await setVariant(variant)(page)
      await page.setViewportSize({ width, height: 800 })

      await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
      await page
        .waitForLoadState("networkidle", { timeout: 30_000 })
        .catch(() => {})

      const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
      await expect(wrapper).toBeVisible({ timeout: 30_000 })
      await expect(
        wrapper.locator(`[data-variant='${variant}']`)
      ).toBeVisible()

      const chatInput = wrapper.locator('[data-testid="chat-input"]')
      await expect(chatInput).toBeVisible()

      // The composer wrapper should not exceed its parent container.
      const overflow = await wrapper.evaluate((el) => {
        const parent = el.parentElement
        if (!parent) return false
        return el.scrollWidth > parent.clientWidth
      })
      expect(overflow).toBe(false)
    })
  }

  for (const { variant, width } of PLAYGROUND_MATRIX) {
    test(`/chat: ${variant} at ${width}px fits without composer overflow`, async ({
      page,
    }) => {
      test.setTimeout(90_000)
      await bypassOnboarding(page)
      await setVariant(variant)(page)
      await page.setViewportSize({ width, height: 800 })

      await page.goto("/chat?nextgenComposer=1")
      await page
        .waitForLoadState("networkidle", { timeout: 30_000 })
        .catch(() => {})

      // Hide dev runtime overlay (no backend)
      await page.evaluate(() => {
        document
          .querySelectorAll("nextjs-portal, [role='dialog']")
          .forEach((el) => {
            if (el instanceof HTMLElement) el.style.display = "none"
          })
      })

      const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
      await expect(wrapper).toBeVisible({ timeout: 30_000 })
      await expect(
        wrapper.locator(`[data-variant='${variant}']`)
      ).toBeVisible()

      const chatInput = wrapper.locator('[data-testid="chat-input"]')
      await expect(chatInput).toBeVisible()

      // The composer wrapper should not exceed its parent container.
      const overflow = await wrapper.evaluate((el) => {
        const parent = el.parentElement
        if (!parent) return false
        return el.scrollWidth > parent.clientWidth
      })
      expect(overflow).toBe(false)
    })
  }
})
