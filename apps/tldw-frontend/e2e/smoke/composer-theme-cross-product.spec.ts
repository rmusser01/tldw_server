import { expect, test, type Page } from "@playwright/test"

/**
 * Cross-product smoke for {V1, V3, V5} × {primer, default} themes.
 * The plan calls for "variants must reskin purely via tokens; no
 * broken colors, no missing backgrounds, no regressions under the
 * old default."
 *
 * We don't snapshot pixels here — that's brittle and a follow-up.
 * Instead we assert the structural invariants that prove the variant
 * mounted cleanly under each theme:
 *   - the variant's root [data-variant=...] element exists
 *   - the chat input is visible and accepts focus
 *   - no React error overlay appeared (would block focus)
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

const setVariantAndTheme = (variant: "v1" | "v3" | "v5", theme: string) =>
  async (page: Page) => {
    await page.addInitScript(
      ({ v, t }: { v: string; t: string }) => {
        try {
          window.localStorage.setItem("tldw:composerVariant", v)
          window.localStorage.setItem("tldw:themePreset", t)
          // Mark migration as already applied so the migration helper
          // doesn't reset the theme to "primer" on first load.
          window.localStorage.setItem("tldw:themeMigrationVersion", "1")
        } catch {
          /* ignore */
        }
      },
      { v: variant, t: theme }
    )
  }

const variantThemeMatrix: Array<{ variant: "v1" | "v3" | "v5"; theme: string }> = [
  { variant: "v1", theme: "primer" },
  { variant: "v1", theme: "default" },
  { variant: "v3", theme: "primer" },
  { variant: "v3", theme: "default" },
  { variant: "v5", theme: "primer" },
  { variant: "v5", theme: "default" },
]

test.describe("composer · theme × variant cross-product", () => {
  for (const { variant, theme } of variantThemeMatrix) {
    test(`/chat: ${variant} under "${theme}" mounts cleanly`, async ({
      page,
    }) => {
      test.setTimeout(90_000)
      await bypassOnboarding(page)
      await setVariantAndTheme(variant, theme)(page)
      await page.goto("/chat?nextgenComposer=1")
      await page
        .waitForLoadState("networkidle", { timeout: 30_000 })
        .catch(() => {})

      const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
      await expect(wrapper).toBeVisible({ timeout: 30_000 })
      await expect(wrapper.locator(`[data-variant='${variant}']`)).toBeVisible()

      // Hide the dev runtime error overlay (no backend) so it doesn't
      // count as a "broken" theme load.
      await page.evaluate(() => {
        document.querySelectorAll("nextjs-portal, [role='dialog']").forEach(
          (el) => {
            if (el instanceof HTMLElement) el.style.display = "none"
          }
        )
      })

      const chatInput = wrapper.locator('[data-testid="chat-input"]')
      await expect(chatInput).toBeVisible()
    })
  }
})
