/**
 * Sources (Ingestion Sources) E2E Tests (Tier 2)
 *
 * Tests the Sources workspace page lifecycle:
 * - Page loads with expected elements (heading, description, new source button)
 * - Handles offline/unsupported/unavailable/empty states gracefully
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
import {
  fetchWithApiKey,
  seedAuth,
  TEST_CONFIG,
} from "../../utils/helpers"

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
      const offlineVisible = await sources.offlineMessage.isVisible().catch(() => false)
      const unsupportedVisible = await sources.unsupportedMessage.isVisible().catch(() => false)
      const unavailableVisible = await sources.unavailableMessage.isVisible().catch(() => false)
      const emptyVisible = await sources.emptyMessage.isVisible().catch(() => false)

      expect(
        headingVisible || offlineVisible || unsupportedVisible || unavailableVisible || emptyVisible
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
      await expect(sources.loadingSpinner).toBeHidden({ timeout: 15_000 })

      const apiResult = await apiCall
      expect(apiResult).not.toBeNull()
      expect(apiResult?.response.status()).toBeLessThan(500)
      expect(await sources.isOnlineWorkspace()).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    if (
      process.env.TLDW_E2E_INGESTION_SOURCE_ROOT ||
      process.env.TLDW_LIVE_TIER_UAT === "1"
    ) {
      test("should fire sync API when 'Sync now' is clicked on a source card", async ({
        authedPage,
        serverInfo,
        diagnostics,
      }) => {
        skipIfServerUnavailable(serverInfo)

        const sourceRoot = process.env.TLDW_E2E_INGESTION_SOURCE_ROOT
        expect(sourceRoot, "live-tier runner must expose an allowed source root").toBeTruthy()

        const label = `Live Tier source ${Date.now()}`
        const createResponse = await fetchWithApiKey(
          `${TEST_CONFIG.serverUrl}/api/v1/ingestion-sources/`,
          TEST_CONFIG.apiKey,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              source_type: "local_directory",
              sink_type: "notes",
              policy: "canonical",
              enabled: true,
              config: { path: sourceRoot, label },
            }),
          }
        )
        expect(createResponse.status).toBe(201)

        sources = new SourcesPage(authedPage)
        await sources.goto()
        await sources.assertPageReady()
        const sourceCard = authedPage.locator(".ant-card").filter({ hasText: label })
        await expect(sourceCard).toBeVisible({ timeout: 15_000 })
        const syncButton = sourceCard.getByRole("button", { name: /^Sync now$/i })
        await expect(syncButton).toBeVisible()

        const apiCall = expectApiCall(authedPage, {
          url: /\/api\/v1\/ingestion-sources\/.*\/sync/,
          method: "POST",
        }, 15_000)

        await syncButton.click()
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)

        await assertNoCriticalErrors(diagnostics)
      })
    }
  })
})
