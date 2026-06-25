/**
 * Audiobook Studio compatibility E2E Tests (Tier 2)
 *
 * Tests that the legacy /audiobook-studio path routes into Audio Studio
 * Narration compatibility.
 *
 * Run: bunx playwright test e2e/workflows/tier-2-features/audiobook-studio.spec.ts
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AudiobookStudioPage } from "../../utils/page-objects/AudiobookStudioPage"
import { seedAuth } from "../../utils/helpers"

test.describe("Audiobook Studio compatibility", () => {
  let studio: AudiobookStudioPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    studio = new AudiobookStudioPage(page)
  })

  test("routes the legacy path into Audio Studio Narration", async ({
    authedPage,
    diagnostics,
  }) => {
    studio = new AudiobookStudioPage(authedPage)
    await studio.goto()
    await studio.assertPageReady()

    await expect(authedPage).toHaveURL(/\/audio-studio\?workflow=narration/)
    await expect(studio.heading).toBeVisible()
    await expect(studio.narrationTab).toHaveAttribute("aria-selected", "true")
    await expect(studio.contentTab).toBeVisible()
    await expect(studio.chaptersTab).toBeVisible()
    await expect(studio.voiceTab).toBeVisible()
    await expect(studio.outputTab).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })
})
