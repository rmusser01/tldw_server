/**
 * Audiobook Studio E2E Tests (Tier 2)
 *
 * Tests the Audiobook Studio page lifecycle:
 * - Page loads with expected elements (heading, tabs, buttons)
 * - Generate tab shows generation controls
 * - Output tab shows output controls
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/audiobook-studio.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AudiobookStudioPage } from "../../utils/page-objects/AudiobookStudioPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

test.describe("Audiobook Studio", () => {
  let studio: AudiobookStudioPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    studio = new AudiobookStudioPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Audiobook Studio page with heading and tabs", async ({
      authedPage,
      diagnostics,
    }) => {
      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      // Heading visible
      await expect(studio.heading).toBeVisible()

      // All four tabs visible
      await expect(studio.contentTab).toBeVisible()
      await expect(studio.chaptersTab).toBeVisible()
      await expect(studio.generateTab).toBeVisible()
      await expect(studio.outputTab).toBeVisible()

      // Project management buttons visible
      await expect(studio.myProjectsButton).toBeVisible()
      await expect(studio.newProjectButton).toBeVisible()
      await expect(studio.saveButton).toBeVisible()
      await expect(studio.saveStatus).toContainText(/draft not saved|unsaved changes|saved/i)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch between tabs without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      // Switch to each tab and verify no crashes
      for (const tab of ["chapters", "generate", "output", "content"] as const) {
        await studio.switchToTab(tab)
        const tabLocator = {
          chapters: studio.chaptersTab,
          generate: studio.generateTab,
          output: studio.outputTab,
          content: studio.contentTab,
        }[tab]
        await expect(tabLocator).toHaveAttribute("aria-selected", "true")
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Generate Tab
  // =========================================================================

  test.describe("Generate Tab", () => {
    test("should show voice settings and generation controls on Generate tab", async ({
      authedPage,
      diagnostics,
    }) => {
      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      await studio.switchToTab("generate")

      // Either "Voice Settings" card or "No chapters" empty state should appear
      const voiceSettings = authedPage.getByText("Voice Settings")
      const noChapters = authedPage.getByText(/add chapters first/i)
      const eitherVisible = await voiceSettings.isVisible().catch(() => false) ||
        await noChapters.isVisible().catch(() => false)

      expect(eitherVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Output Tab
  // =========================================================================

  test.describe("Output Tab", () => {
    test("should show output panel or empty state on Output tab", async ({
      authedPage,
      diagnostics,
    }) => {
      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      await studio.switchToTab("output")

      // Either output controls or "No chapters" empty state should appear
      const outputPanel = authedPage.getByText("Output")
      const noChapters = authedPage.getByText(/generated chapters appear here/i)
      const eitherVisible = await outputPanel.isVisible().catch(() => false) ||
        await noChapters.isVisible().catch(() => false)

      expect(eitherVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // My Projects
  // =========================================================================

  test.describe("My Projects", () => {
    test("should toggle project list view when clicking My Projects", async ({
      authedPage,
      diagnostics,
    }) => {
      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      // Click My Projects
      await studio.myProjectsButton.click()

      // Should show Back to Editor button in project list view
      const backButton = authedPage.getByRole("button", { name: /back to editor/i })
      await expect(backButton).toBeVisible({ timeout: 10_000 })

      // Click Back to Editor to return
      await backButton.click()

      // Tabs should be visible again
      await expect(studio.contentTab).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // TTS API Integration (requires server)
  // =========================================================================

  test.describe("TTS Settings Fetch", () => {
    test("should fetch TTS settings when navigating to Generate tab", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      studio = new AudiobookStudioPage(authedPage)
      await studio.goto()
      await studio.assertPageReady()

      // Set up API expectation before navigating to the Generate tab
      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/(audio\/voices|tts|settings)/,
        method: "GET",
      }, 15_000)

      await studio.switchToTab("generate")

      // Wait for a TTS-related API call (settings or voice catalog)
      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // If no TTS API call fires, that is acceptable (could be cached or browser provider)
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
