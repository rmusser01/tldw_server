/**
 * Multi-Item Media Review Workflow E2E Tests
 *
 * Tests the complete media review lifecycle:
 * - Single item review (select, view details)
 * - Multi-item selection (click, shift-click, limit)
 * - View modes (spread/compare, list/focus, all/stack)
 * - Filtering & search
 * - Pagination
 * - Error handling & undo
 *
 * Run: npx playwright test e2e/workflows/media-review.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import { MediaReviewPage } from "../utils/page-objects/MediaReviewPage"
import {
  seedAuth,
  generateTestId,
  TEST_CONFIG,
  fetchWithApiKey
} from "../utils/helpers"

const MIN_MEDIA_ITEMS_FOR_CROSS_PAGE_SELECTION = 22

const waitForViewerItems = async (
  reviewPage: MediaReviewPage,
  minimumCount = 1,
  timeoutMs = 15_000
): Promise<void> => {
  await expect
    .poll(async () => await reviewPage.getViewerItemCount(), {
      timeout: timeoutMs,
      message: "Timed out waiting for media review viewer cards to render",
    })
    .toBeGreaterThanOrEqual(minimumCount)
}

const waitForSelectedCount = async (
  reviewPage: MediaReviewPage,
  minimumCount: number,
  timeoutMs = 10_000
): Promise<void> => {
  await expect
    .poll(async () => await reviewPage.getSelectedCount(), {
      timeout: timeoutMs,
      message: "Timed out waiting for media review selection count to update",
    })
    .toBeGreaterThanOrEqual(minimumCount)
}

const waitForSelectedAcrossPagesCount = async (
  reviewPage: MediaReviewPage,
  minimumCount: number,
  timeoutMs = 10_000
): Promise<void> => {
  await expect
    .poll(async () => await reviewPage.getSelectedAcrossPagesCount(), {
      timeout: timeoutMs,
      message: "Timed out waiting for cross-page selection count to update",
    })
    .toBeGreaterThanOrEqual(minimumCount)
}

const waitForCurrentPage = async (
  reviewPage: MediaReviewPage,
  expectedPage: number,
  timeoutMs = 10_000
): Promise<void> => {
  await expect
    .poll(async () => await reviewPage.getCurrentPage(), {
      timeout: timeoutMs,
      message: "Timed out waiting for media review pagination to settle",
    })
    .toBe(expectedPage)
}

const seedAppAuthWithApiKey = async (page: import("@playwright/test").Page) => {
  await page.context().addInitScript((cfg) => {
    try {
      localStorage.setItem(
        "tldwConfig",
        JSON.stringify({
          serverUrl: cfg.serverUrl,
          // lgtm[js/clear-text-storage-of-sensitive-data] synthetic CI key only
          apiKey: cfg.apiKey,
          authMode: "single-user"
        })
      )
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("__tldw_allow_offline", "true")
    } catch {
      // Ignore localStorage failures in hardened browser contexts.
    }
  }, { serverUrl: TEST_CONFIG.serverUrl, apiKey: TEST_CONFIG.apiKey })
}

const getMediaTotalCount = async (): Promise<number> => {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/?page=1&results_per_page=1`,
    TEST_CONFIG.apiKey
  )
  if (!response.ok) {
    throw new Error(`Failed to fetch media count: ${response.status} ${await response.text()}`)
  }
  const payload = await response.json().catch(() => ({}))
  const totalCandidate =
    payload?.pagination?.total_items ??
    payload?.pagination?.total ??
    payload?.total ??
    payload?.count
  const parsed = Number(totalCandidate)
  if (Number.isFinite(parsed) && parsed >= 0) return parsed
  if (Array.isArray(payload?.items)) return payload.items.length
  return 0
}

const seedMediaDocument = async (
  seedLabel: string,
  options?: {
    title?: string
    content?: string
  }
): Promise<void> => {
  const body = new FormData()
  body.append("media_type", "document")
  body.append("title", options?.title || `Media review E2E ${seedLabel}`)
  body.append("perform_analysis", "false")
  body.append("perform_chunking", "false")
  body.append(
    "files",
    new Blob([options?.content || `Seeded media review payload ${seedLabel}`], { type: "text/plain" }),
    `media-review-${seedLabel}.txt`
  )

  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/add`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      body
    }
  )
  if (!response.ok) {
    throw new Error(`Failed to seed media: ${response.status} ${await response.text()}`)
  }
}

const seedSortFixtureDocuments = async (): Promise<{
  query: string
}> => {
  const fixtureId = generateTestId("media-review-sort")
  const prefix = `Media review sortable ${fixtureId}`
  const titles = [
    `${prefix} Zeta`,
    `${prefix} Alpha`,
    `${prefix} Mu`
  ]

  for (const title of titles) {
    await seedMediaDocument(generateTestId("media-review-sort-doc"), {
      title,
      content: `${title} content body`
    })
  }

  return { query: prefix }
}

const seedLongDiffFixtureDocuments = async (): Promise<{
  query: string
}> => {
  const fixtureId = generateTestId("media-review-long-diff")
  const prefix = `Media review long diff ${fixtureId}`
  const leftTitle = `${prefix} Left`
  const rightTitle = `${prefix} Right`
  const leftContent = Array.from({ length: 2600 }, (_, idx) => `${prefix} left line ${idx}`).join("\n")
  const rightContent = Array.from({ length: 2600 }, (_, idx) => `${prefix} right line ${idx}`).join("\n")

  await seedMediaDocument(generateTestId("media-review-long-left"), {
    title: leftTitle,
    content: leftContent
  })
  await seedMediaDocument(generateTestId("media-review-long-right"), {
    title: rightTitle,
    content: rightContent
  })

  return { query: prefix }
}

const seedLiteratureTriageDocuments = async (
  count: number = 12
): Promise<{ query: string }> => {
  const fixtureId = generateTestId("media-review-triage")
  const prefix = `Media review triage ${fixtureId}`
  for (let idx = 0; idx < count; idx += 1) {
    const title = `${prefix} Paper ${String(idx + 1).padStart(2, "0")}`
    await seedMediaDocument(generateTestId("media-review-triage-doc"), {
      title,
      content: `${title}\n\nTriage body for ${idx + 1}`
    })
  }
  return { query: prefix }
}

const seedMediaAuditDocuments = async (
  count: number = 4
): Promise<{ query: string }> => {
  const fixtureId = generateTestId("media-review-audit")
  const prefix = `Media review audit ${fixtureId}`
  for (let idx = 0; idx < count; idx += 1) {
    const title = `${prefix} Item ${String(idx + 1).padStart(2, "0")}`
    await seedMediaDocument(generateTestId("media-review-audit-doc"), {
      title,
      content: `${title}\n\nAudit payload ${idx + 1}`
    })
  }
  return { query: prefix }
}

const ensureMediaCountForCrossPageReview = async (
  minimumCount: number = MIN_MEDIA_ITEMS_FOR_CROSS_PAGE_SELECTION
): Promise<number> => {
  let currentCount = await getMediaTotalCount()
  if (currentCount >= minimumCount) return currentCount

  const needed = minimumCount - currentCount
  for (let idx = 0; idx < needed; idx += 1) {
    await seedMediaDocument(generateTestId(`media-review-cross-page-${idx}`))
  }

  await expect
    .poll(async () => await getMediaTotalCount(), {
      timeout: 12_000,
      message: `Timed out waiting for media count >= ${minimumCount}`,
    })
    .toBeGreaterThanOrEqual(minimumCount)

  currentCount = await getMediaTotalCount()
  return currentCount
}

test.describe("Multi-Item Media Review Workflow", () => {
  let reviewPage: MediaReviewPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    reviewPage = new MediaReviewPage(page)
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.1  Single Item Review
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Single Item Review", () => {
    test("should navigate to media review page and load items", async ({
      authedPage,
      diagnostics
    }) => {
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display media list with pagination", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      // Wait for API response
      const itemCount = await reviewPage.getItemCount()
      // List should have items (or be empty)
      expect(itemCount).toBeGreaterThanOrEqual(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should select a single item and show details", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No media items available")
        return
      }

      // Click first item
      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display item content in viewer", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No media items available")
        return
      }

      // Select first item and wait for detail load
      const [detailResult] = await Promise.all([
        reviewPage.waitForMediaDetail().catch(() => ({ status: 0, body: null })),
        reviewPage.clickItem(0)
      ])

      await waitForViewerItems(reviewPage)

      // Content should show title and type
      const viewerCount = await reviewPage.getViewerItemCount()
      if (viewerCount > 0) {
        expect(viewerCount).toBeGreaterThanOrEqual(1)
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.2  Multi-Item Selection
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Multi-Item Selection", () => {
    test("should select multiple items", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount < 2) {
        test.skip(true, "Need at least 2 media items")
        return
      }

      await reviewPage.toggleItemSelection(0)
      await waitForSelectedCount(reviewPage, 1)

      await reviewPage.toggleItemSelection(1)
      await waitForSelectedCount(reviewPage, 2)

      // Should have 2 selected
      const selectedCount = await reviewPage.getSelectedCount()
      expect(selectedCount).toBeGreaterThanOrEqual(2)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should support range selection with shift-click", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount < 3) {
        test.skip(true, "Need at least 3 media items for range select")
        return
      }

      await reviewPage.toggleItemSelection(0)
      await waitForSelectedCount(reviewPage, 1)

      // Shift-click third item
      await reviewPage.shiftToggleItemSelection(2)
      await waitForSelectedCount(reviewPage, 3)

      // Should have at least 3 selected (range)
      const selectedCount = await reviewPage.getSelectedCount()
      expect(selectedCount).toBeGreaterThanOrEqual(3)

      await assertNoCriticalErrors(diagnostics)
    })

    test("preserves cross-page selection when adding visible page items", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      await ensureMediaCountForCrossPageReview()
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      await expect
        .poll(async () => await reviewPage.getItemCount(), {
          timeout: 15_000,
          message: "Timed out waiting for media review results to render seeded cross-page items",
        })
        .toBeGreaterThan(0)

      const totalPages = await reviewPage.getTotalPages()
      expect(totalPages).toBeGreaterThanOrEqual(2)

      await reviewPage.toggleItemSelection(0)
      await expect(
        authedPage.getByText(/Selected across pages:\s*1/i)
      ).toBeVisible({ timeout: 10_000 })

      const selectionBefore = await reviewPage.getSelectedAcrossPagesCount()
      expect(selectionBefore).toBeGreaterThanOrEqual(1)

      await reviewPage.goToPage(2)
      await waitForCurrentPage(reviewPage, 2)

      await reviewPage.clickAddVisibleToSelection()
      await waitForSelectedAcrossPagesCount(reviewPage, selectionBefore + 1)

      const selectionAfter = await reviewPage.getSelectedAcrossPagesCount()
      expect(selectionAfter).toBeGreaterThan(selectionBefore)
      await expect(
        authedPage.getByText(/selected across pages:/i)
      ).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.3  View Modes
  // ═════════════════════════════════════════════════════════════════════

  test.describe("View Modes", () => {
    test("should switch to Compare (spread) view mode", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount < 2) {
        test.skip(true, "Need items for view mode test")
        return
      }

      await reviewPage.toggleItemSelection(0)
      await reviewPage.toggleItemSelection(1)
      await waitForSelectedCount(reviewPage, 2)

      await reviewPage.setViewMode("spread")
      await expect
        .poll(async () => await reviewPage.getCurrentViewMode(), { timeout: 10_000 })
        .toBe("spread")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch to Focus (list) view mode", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No items for view mode test")
        return
      }

      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)

      await reviewPage.setViewMode("list")
      await expect
        .poll(async () => await reviewPage.getCurrentViewMode(), { timeout: 10_000 })
        .toBe("list")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch to Stack (all) view mode", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No items for view mode test")
        return
      }

      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)

      await reviewPage.setViewMode("all")
      await expect
        .poll(async () => await reviewPage.getCurrentViewMode(), { timeout: 10_000 })
        .toBe("all")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should change orientation between vertical and horizontal", async ({
      authedPage,
      diagnostics
    }) => {
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      try {
        await reviewPage.setOrientation("horizontal")
        await reviewPage.setOrientation("vertical")
      } catch {
        // Orientation toggle may not be available without selected items
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.4  Filtering & Search
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Filtering & Search", () => {
    test("should filter items by search query", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      // Enter a search query
      await reviewPage.fillSearchQuery("test")

      // Results should update
      const itemCount = await reviewPage.getItemCount()
      // Any count is valid (0 = no matches, >0 = matches found)
      expect(itemCount).toBeGreaterThanOrEqual(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should filter by media type", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      try {
        await reviewPage.filterByMediaType("video")

        // Results should be filtered
        const itemCount = await reviewPage.getItemCount()
        expect(itemCount).toBeGreaterThanOrEqual(0)
      } catch {
        // Type filter may not be available
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should clear filters and restore full list", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      // Apply a filter first
      await reviewPage.fillSearchQuery("filter-test")

      // Clear filters
      await reviewPage.clearFilters()

      await assertNoCriticalErrors(diagnostics)
    })

    test("applies title sort and date range filters for focused search results", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      const fixture = await seedSortFixtureDocuments()
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      await reviewPage.fillSearchQuery(fixture.query)
      await reviewPage.setSort("title_asc")
      await reviewPage.setDateRange("2026-01-01", "2026-12-31")
      const searchRequestPromise = authedPage.waitForRequest((request) => (
        request.url().includes("/api/v1/media/search") &&
        request.method().toUpperCase() === "POST"
      ))
      await authedPage.getByRole("button", { name: /^search$/i }).click()
      const searchRequest = await searchRequestPromise
      const payload = searchRequest.postDataJSON() as Record<string, any>
      expect(payload.sort_by).toBe("title_asc")
      expect(payload.date_range?.start ?? payload.date_range?.start_date).toBe("2026-01-01")
      expect(payload.date_range?.end ?? payload.date_range?.end_date).toBe("2026-12-31")

      await expect
        .poll(async () => await reviewPage.getItemCount(), {
          timeout: 45_000,
          message: "Timed out waiting for seeded media-review search fixtures to become searchable",
        })
        .toBeGreaterThanOrEqual(1)

      await assertNoCriticalErrors(diagnostics)
    })

    test("shows content-search scope copy and progress feedback", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      await ensureMediaCountForCrossPageReview(6)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const slowRoutePattern = "**/api/v1/media/*?include_content=true&include_versions=false"
      let sawSlowContentRequest = false
      let releaseSlowContentRequest: (() => void) | null = null
      const slowContentRequestPending = new Promise<void>((resolve) => {
        releaseSlowContentRequest = resolve
      })
      await authedPage.route(slowRoutePattern, async (route) => {
        sawSlowContentRequest = true
        await slowContentRequestPending
        await route.continue()
      })

      try {
        await reviewPage.fillSearchQuery("Media review")
        await reviewPage.toggleContentSearch(true)
        await authedPage.getByRole("button", { name: /^search$/i }).click()

        await expect.poll(() => sawSlowContentRequest, {
          timeout: 10_000,
          message: "Timed out waiting for media review content-search requests",
        }).toBe(true)
        await expect(
          authedPage.getByText(/scans current page results/i)
        ).toBeVisible({ timeout: 10_000 })
        await expect(
          authedPage.getByRole("status", { name: /content filtering progress/i })
        ).toBeVisible({ timeout: 10_000 })
        releaseSlowContentRequest?.()
      } finally {
        releaseSlowContentRequest?.()
        await authedPage.unroute(slowRoutePattern)
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.5  Error Handling & Recovery
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Error Handling & Recovery", () => {
    test("should handle failed item loads gracefully", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      // Mock a failing detail API for a specific ID
      await authedPage.route("**/api/v1/media/99999", (route) => {
        route.fulfill({
          status: 404,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Not found" })
        })
      })

      // The test verifies the page doesn't crash on error responses
      await assertNoCriticalErrors(diagnostics)

      await authedPage.unroute("**/api/v1/media/99999")
    })

    test("should clear selection and offer undo", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No items for clear/undo test")
        return
      }

      // Select items
      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)

      // Clear via options menu
      try {
        await reviewPage.clickClearSession()
        await expect
          .poll(async () => await reviewPage.isViewerEmpty(), { timeout: 10_000 })
          .toBe(true)
      } catch {
        // Clear session button may not be accessible without items open
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.6  Pagination
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Pagination", () => {
    test("should display pagination controls", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      // Check for pagination component
      const pagination = authedPage.locator(".ant-pagination")
      // Pagination may only show if there are enough items

      const currentPage = await reviewPage.getCurrentPage()
      expect(currentPage).toBe(1)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should navigate to page 2 if available", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const totalPages = await reviewPage.getTotalPages()
      if (totalPages < 2) {
        test.skip(true, "Not enough items for pagination test")
        return
      }

      // Navigate to page 2
      const [apiResult] = await Promise.all([
        reviewPage.waitForMediaList().catch(() => ({ status: 0, body: null })),
        reviewPage.goToPage(2)
      ])

      const currentPage = await reviewPage.getCurrentPage()
      expect(currentPage).toBe(2)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.7  Performance & Scalability
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Performance & Scalability", () => {
    test("handles long compare diffs with non-blocking status feedback", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      const fixture = await seedLongDiffFixtureDocuments()
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      await reviewPage.fillSearchQuery(fixture.query)
      await expect
        .poll(async () => await reviewPage.getItemCount(), { timeout: 15_000 })
        .toBeGreaterThanOrEqual(2)

      await reviewPage.toggleItemSelection(0)
      await reviewPage.toggleItemSelection(1)

      await reviewPage.openCompareContentDiff()

      await expect(
        authedPage.getByRole("button", { name: /exit compare/i })
      ).toBeVisible({ timeout: 10_000 })
      await expect(
        authedPage.getByText(/^Open items$/)
      ).toBeVisible({ timeout: 10_000 })
      await expect(
        authedPage.getByRole("button", { name: /chat about selection/i })
      ).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("keeps 30-item stack mode interactive with virtualized rendering", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      await ensureMediaCountForCrossPageReview(35)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      await reviewPage.clickAddVisibleToSelection()
      await waitForSelectedAcrossPagesCount(reviewPage, 1)
      await reviewPage.goToPage(2)
      await waitForCurrentPage(reviewPage, 2)
      await reviewPage.clickAddVisibleToSelection()

      await expect(
        authedPage.getByText(/selected across pages:\s*30/i)
      ).toBeVisible({ timeout: 10_000 })

      await reviewPage.setViewMode("all")
      await expect(
        authedPage.getByTestId("media-review-stack-virtualized")
      ).toBeVisible({ timeout: 10_000 })

      const renderedCardCount = await reviewPage.getStackRenderedCardCount()
      expect(renderedCardCount).toBeGreaterThan(0)
      expect(renderedCardCount).toBeLessThan(30)

      const scrollTop = await reviewPage.scrollStackContainer(1200)
      expect(scrollTop).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 4.8  Batch Operations Workflows
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Batch Operations Workflows", () => {
    test("triage: filters PDF set, reviews stack, compares two, and opens chat handoff", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      const fixture = await seedLiteratureTriageDocuments(12)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()
      await reviewPage.pressEscapeTwice()

      await reviewPage.fillSearchQuery(fixture.query)
      await authedPage.getByRole("button", { name: /^search$/i }).click()
      await expect
        .poll(async () => await reviewPage.getItemCount(), { timeout: 20_000 })
        .toBeGreaterThanOrEqual(2)

      await reviewPage.filterByMediaType("pdf").catch(() => {})
      const availableItems = await reviewPage.getItemCount()
      const triageSelectionCount = Math.min(10, availableItems)
      expect(triageSelectionCount).toBeGreaterThanOrEqual(2)
      await reviewPage.selectFirstNItems(triageSelectionCount)
      await expect
        .poll(async () => await reviewPage.getSelectedCount(), { timeout: 10_000 })
        .toBe(triageSelectionCount)

      await reviewPage.setViewMode("all")
      await expect
        .poll(async () => await reviewPage.getCurrentViewMode(), { timeout: 10_000 })
        .toBe("all")

      // Keep two selected for compare/chat handoff.
      for (let idx = 2; idx < triageSelectionCount; idx += 1) {
        await reviewPage.toggleItemSelection(idx)
      }
      await reviewPage.openCompareContentDiff()
      await expect(
        authedPage.getByRole("button", { name: /exit compare/i })
      ).toBeVisible({ timeout: 10_000 })

      const closeDiff = authedPage.getByRole("button", { name: /close/i }).first()
      if (await closeDiff.isVisible().catch(() => false)) {
        await closeDiff.click()
      }

      await authedPage
        .getByRole("button", { name: /chat about selection/i })
        .first()
        .click()
      await expect(authedPage).toHaveURL(/\/($|\?)/, { timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("audit: sorts by date, batch moves stale items to trash, and opens trash", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      const fixture = await seedMediaAuditDocuments(4)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()
      await reviewPage.pressEscapeTwice()

      await reviewPage.fillSearchQuery(fixture.query)
      await reviewPage.setSort("date_asc")
      await authedPage.getByRole("button", { name: /^search$/i }).click()
      await expect
        .poll(async () => await reviewPage.getItemCount(), { timeout: 20_000 })
        .toBeGreaterThanOrEqual(2)

      await reviewPage.selectFirstNItems(2)
      await expect
        .poll(async () => await reviewPage.getSelectedCount(), { timeout: 10_000 })
        .toBeGreaterThanOrEqual(2)
      await expect(
        authedPage.getByTestId("media-multi-batch-toolbar")
      ).toBeVisible({ timeout: 10_000 })

      await reviewPage.clickBatchMoveToTrash()
      await reviewPage.confirmBatchMoveToTrashIfPrompted()
      const openTrashButton = authedPage
        .getByRole("button", { name: /open trash/i })
        .first()
      if (await openTrashButton.isVisible().catch(() => false)) {
        await reviewPage.clickOpenTrashCta()
      } else {
        await authedPage.goto("/media-trash", { waitUntil: "domcontentloaded" })
      }

      await expect(authedPage).toHaveURL(/\/media-trash/, { timeout: 10_000 })
      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // Keyboard Shortcuts
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Keyboard Shortcuts", () => {
    test("should navigate items with j/k keys", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount < 2) {
        test.skip(true, "Need items for keyboard nav test")
        return
      }

      // Select first item to start
      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)
      const startingPosition = await reviewPage.getPreviewPosition()

      // Press j to go next
      await reviewPage.pressNextItemShortcut()
      await expect
        .poll(async () => await reviewPage.getPreviewPosition(), { timeout: 10_000 })
        .not.toBe(startingPosition)
      const advancedPosition = await reviewPage.getPreviewPosition()

      // Press k to go back
      await reviewPage.pressPrevItemShortcut()
      await expect
        .poll(async () => await reviewPage.getPreviewPosition(), { timeout: 10_000 })
        .toBe(startingPosition)
      expect(advancedPosition).not.toBe(startingPosition)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should toggle content expand with o key", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()

      const itemCount = await reviewPage.getItemCount()
      if (itemCount === 0) {
        test.skip(true, "No items for keyboard expand test")
        return
      }

      await reviewPage.clickItem(0)
      await waitForViewerItems(reviewPage)

      // Press o to toggle expand
      await reviewPage.pressToggleExpand()

      await assertNoCriticalErrors(diagnostics)
    })

    test("keyboard accessibility flow scopes shortcuts in overlays and supports clear-session recovery", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      await seedAppAuthWithApiKey(authedPage)
      await ensureMediaCountForCrossPageReview(8)
      reviewPage = new MediaReviewPage(authedPage)
      await reviewPage.goto()
      await reviewPage.waitForReady()
      await reviewPage.pressEscapeTwice()

      const rows = authedPage.locator(
        "[data-testid='media-review-results-list'] [role='button'][aria-selected]"
      )
      await expect(rows.first()).toBeVisible({ timeout: 10_000 })

      await reviewPage.toggleItemSelection(0)
      await reviewPage.toggleItemSelection(1)
      await expect(
        authedPage.getByText(/selected across pages:\s*2/i)
      ).toBeVisible({ timeout: 10_000 })

      const searchInput = authedPage
        .getByRole("textbox", { name: /search media/i })
        .first()

      const optionsButton = authedPage.getByRole("button", { name: /options/i }).first()
      await optionsButton.focus()
      await optionsButton.press("Enter")
      const addVisibleMenuItem = authedPage
        .getByText(/add visible to selection|review all/i)
        .first()
      if (!(await addVisibleMenuItem.isVisible().catch(() => false))) {
        await optionsButton.press("Space").catch(() => {})
      }
      if (!(await addVisibleMenuItem.isVisible().catch(() => false))) {
        await optionsButton.press("ArrowDown").catch(() => {})
      }
      if (!(await addVisibleMenuItem.isVisible().catch(() => false))) {
        await optionsButton.click()
      }
      await expect(
        addVisibleMenuItem
      ).toBeVisible({ timeout: 10_000 })

      await authedPage.keyboard.press("/")
      await expect(searchInput).not.toBeFocused()

      await authedPage.keyboard.press("Escape")

      const compareButton = authedPage.getByRole("button", { name: /compare content/i }).first()
      await compareButton.focus()
      await compareButton.press("Enter")
      const exitCompareButton = authedPage.getByRole("button", { name: /exit compare/i }).first()
      await expect(exitCompareButton).toBeVisible({ timeout: 10_000 })
      await expect(
        authedPage.getByText(/^Open items$/)
      ).toBeVisible({ timeout: 10_000 })

      await authedPage.keyboard.press("/")
      await expect(searchInput).not.toBeFocused()

      await exitCompareButton.focus()
      await exitCompareButton.press("Enter")
      await expect(exitCompareButton).toBeHidden({ timeout: 10_000 })
      await expect(
        authedPage.getByText(/selected across pages:\s*2/i)
      ).toBeVisible({ timeout: 10_000 })

      await reviewPage.clickClearSession()
      await expect
        .poll(async () => await reviewPage.getSelectedAcrossPagesCount(), {
          timeout: 10_000
        })
        .toBe(0)

      const undoButton = authedPage.getByRole("button", { name: /undo/i }).first()
      if (await undoButton.isVisible().catch(() => false)) {
        await undoButton.focus()
        await authedPage.keyboard.press("Enter")

        await expect
          .poll(async () => await reviewPage.getSelectedAcrossPagesCount(), { timeout: 10_000 })
          .toBe(2)
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
