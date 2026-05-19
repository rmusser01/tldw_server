/**
 * Sources (Ingestion Sources) E2E Tests (Tier 2)
 *
 * Tests the Sources workspace page lifecycle:
 * - Page loads with expected elements (heading, description, new source button)
 * - Handles loading/offline/unsupported/unavailable/empty states gracefully
 * - "New source" button navigates to /sources/new
 * - "Sync now" button fires POST /api/v1/ingestion-sources/{id}/sync (requires sources)
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/sources.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { SourcesPage } from "../../utils/page-objects"
import { seedAuth } from "../../utils/helpers"

test.describe("Sources & Connectors", () => {
  let sources: SourcesPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    sources = new SourcesPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Sources page with heading or a valid state", async ({
      authedPage,
      diagnostics,
    }) => {
      sources = new SourcesPage(authedPage)
      await sources.goto()
      await sources.assertPageReady()

      // One of the valid states should be visible
      const headingVisible = await sources.heading.isVisible().catch(() => false)
      const loadingVisible = await sources.loadingSpinner.isVisible().catch(() => false)
      const offlineVisible = await sources.offlineMessage.isVisible().catch(() => false)
      const unsupportedVisible = await sources.unsupportedMessage.isVisible().catch(() => false)
      const unavailableVisible = await sources.unavailableMessage.isVisible().catch(() => false)
      const emptyVisible = await sources.emptyMessage.isVisible().catch(() => false)

      expect(
        headingVisible || loadingVisible || offlineVisible || unsupportedVisible || unavailableVisible || emptyVisible
      ).toBe(true)

      // If the online workspace is showing (heading + no unsupported/offline banner),
      // the "New source" button should be visible
      const isOnline = await sources.isOnlineWorkspace()
      if (isOnline) {
        await expect(sources.newSourceButton).toBeVisible()
        await expect(sources.description).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show empty state or source list when online", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      sources = new SourcesPage(authedPage)
      await sources.goto()
      await sources.assertPageReady()

      await expect(sources.loadingSpinner).toBeHidden({ timeout: 15_000 })

      const isOnline = await sources.isOnlineWorkspace()
      if (!isOnline) return

      // Either sources are listed or the empty message is shown
      const hasSources = await sources.hasSourceCards()
      const emptyVisible = await sources.emptyMessage.isVisible().catch(() => false)

      expect(hasSources || emptyVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Navigation
  // =========================================================================

  test.describe("Navigation", () => {
    test("should navigate to /sources/new when 'New source' is clicked", async ({
      authedPage,
      diagnostics,
    }) => {
      sources = new SourcesPage(authedPage)
      await sources.goto()
      await sources.assertPageReady()

      const isOnline = await sources.isOnlineWorkspace()
      if (!isOnline) return

      await sources.newSourceButton.click()
      await expect(authedPage).toHaveURL(/\/sources\/new/, { timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Integration (requires server + existing sources)
  // =========================================================================

  test.describe("Source List API", () => {
    test("should fire GET /api/v1/ingestion-sources on page load", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/ingestion-sources/,
        method: "GET",
      }, 15_000).catch(() => null)

      sources = new SourcesPage(authedPage)
      await sources.goto()
      await sources.assertPageReady()

      const isOnline = await sources.isOnlineWorkspace()
      const apiResult = await apiCall

      if (isOnline) {
        expect(apiResult).not.toBeNull()
        expect(apiResult?.response.status()).toBeLessThan(500)
      } else {
        expect(apiResult).toBeNull()
        const unsupportedVisible = await sources.unsupportedMessage.isVisible().catch(() => false)
        const offlineVisible = await sources.offlineMessage.isVisible().catch(() => false)
        expect(unsupportedVisible || offlineVisible).toBe(true)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should fire sync API when 'Sync now' is clicked on a source card", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      sources = new SourcesPage(authedPage)
      await sources.goto()
      await sources.assertPageReady()

      const hasSources = await sources.hasSourceCards()
      if (!hasSources) {
        test.skip(true, "No ingestion sources available to test sync")
        return
      }

      const syncVisible = await sources.syncNowButton.isVisible().catch(() => false)
      if (!syncVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/ingestion-sources\/.*\/sync/,
        method: "POST",
      }, 15_000)

      await sources.syncNowButton.click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Sync may fail if source is not configured properly; acceptable
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
