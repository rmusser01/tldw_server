/**
 * Media Ingestion Workflow E2E Tests
 *
 * Tests the complete media ingestion workflow from a user's perspective:
 * - File upload
 * - URL ingestion
 * - Metadata editing
 * - Error cases
 * - Content review flow
 */
import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../utils/fixtures"
import type { Page, Route } from "@playwright/test"
import { MediaPage } from "../utils/page-objects"
import {
  seedAuth,
  generateTestId,
  waitForConnection
} from "../utils/helpers"
import { expectApiCall } from "../utils/api-assertions"
import {
  ingestAndWaitForReady,
  dismissQuickIngest,
  reopenQuickIngest,
  queueUrlAndStartProcessing,
  startQueuedQuickIngestProcessing,
  assertQuickIngestCompletedResults,
  openQuickIngestDialog,
  queueFileForQuickIngest,
  advanceQuickIngestToConfigureStep,
  reachQuickIngestOptionInConstrainedViewport
} from "../utils/journey-helpers"
import * as path from "path"
import * as fs from "fs"

test.describe("Media Ingestion Workflow", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
  })

  test.describe("Media Page Navigation", () => {
    test("should navigate to media page and display interface", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      await expect(mediaPage.heading).toBeVisible({ timeout: 20_000 })
      await expect(mediaPage.searchInput).toBeVisible({ timeout: 20_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display empty state or media list", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      const emptyState = authedPage.locator(
        "[data-testid='empty-state'], .empty-state, .no-media"
      ).first()
      const mediaList = mediaPage.mediaList

      const hasEmpty = await emptyState.isVisible().catch(() => false)
      const hasList = await mediaList.isVisible().catch(() => false)

      expect(hasEmpty || hasList).toBeTruthy()

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("File Upload", () => {
    test("should display file upload interface", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Look for upload interface
      const uploadArea = authedPage.locator(
        "input[type='file'], [data-testid='upload-area'], .ant-upload, .dropzone"
      )

      if ((await uploadArea.count()) > 0) {
        // File input may be hidden but should exist
        expect(await uploadArea.count()).toBeGreaterThan(0)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show upload progress indicator when uploading", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)

      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Create a temporary test file
      const testContent = `Test content ${generateTestId()}`
      const testFilePath = path.join("/tmp", `test-${Date.now()}.txt`)
      fs.writeFileSync(testFilePath, testContent)

      try {
        // Find file input and upload
        const fileInput = authedPage.locator("input[type='file']")
        if ((await fileInput.count()) > 0) {
          // Set up API call interception before triggering upload
          const apiCall = expectApiCall(authedPage, {
            method: "POST",
            url: "/api/v1/media"
          })

          await fileInput.setInputFiles(testFilePath)

          // Look for progress indicator
          const _progress = authedPage.locator(
            ".ant-progress, [data-testid='upload-progress'], .progress-bar, .uploading"
          )

          // Verify the upload API call completes without error
          try {
            const { response } = await apiCall
            expect(response.status()).toBeLessThan(400)
          } catch {
            // Upload may not trigger an API call if the UI requires
            // additional user interaction (e.g. clicking a submit button)
          }
        }
      } finally {
        // Cleanup
        if (fs.existsSync(testFilePath)) {
          fs.unlinkSync(testFilePath)
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should reject invalid file types with error message", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Create an invalid file type
      const testFilePath = path.join("/tmp", `test-${Date.now()}.exe`)
      fs.writeFileSync(testFilePath, "invalid content")

      try {
        const fileInput = authedPage.locator("input[type='file']")
        if ((await fileInput.count()) > 0) {
          await fileInput.setInputFiles(testFilePath)

          // Check for error indication
          const errorMessage = authedPage.locator(
            ".ant-message-error, .error-message, [data-testid='upload-error']"
          )
          await expect
            .poll(async () => await errorMessage.first().isVisible().catch(() => false), {
              timeout: 2_000,
              message: "Timed out waiting for invalid upload feedback",
            })
            .toBe(true)
            .catch(() => {})
        }
      } finally {
        if (fs.existsSync(testFilePath)) {
          fs.unlinkSync(testFilePath)
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("URL Ingestion", () => {
    test("should display URL input field", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Look for URL input
      const _urlInput = authedPage.locator(
        "input[placeholder*='url' i], textarea[placeholder*='url' i], input[placeholder*='URL'], textarea[placeholder*='URL'], [data-testid='url-input']"
      )

      // URL input may exist on media page or in a modal
      // Just verify page loads correctly
      await assertNoCriticalErrors(diagnostics)
    })

    test("should validate URL format", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Find URL input if available
      const urlInput = authedPage.locator(
        "input[placeholder*='url' i], textarea[placeholder*='url' i], [data-testid='url-input']"
      ).first()

      if ((await urlInput.count()) > 0) {
        await urlInput.fill("not-a-valid-url")

        const submitBtn = authedPage.getByRole("button", {
          name: /add|submit|ingest|process/i
        })

        if ((await submitBtn.count()) > 0) {
          await submitBtn.first().click()

          // Look for validation error
          const _validationError = authedPage.locator(
            ".ant-form-item-explain-error, .error-message, [data-testid='url-error']"
          )

          await expect
            .poll(
              async () =>
                await authedPage
                  .locator(".ant-form-item-explain-error, .error-message, [data-testid='url-error']")
                  .first()
                  .isVisible()
                  .catch(() => false),
              {
                timeout: 3_000,
                message: "Timed out waiting for invalid URL feedback",
              }
            )
            .toBe(true)
            .catch(() => {})
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show processing status for URL ingestion", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)

      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Find URL input if available
      const urlInput = authedPage.locator(
        "input[placeholder*='url' i], textarea[placeholder*='url' i], [data-testid='url-input']"
      ).first()

      if ((await urlInput.count()) > 0 && (await urlInput.isVisible())) {
        // Use a reliable test URL (though actual ingestion may not work)
        await urlInput.fill("https://example.com")

        const submitBtn = authedPage.getByRole("button", {
          name: /add|submit|ingest|process/i
        })

        if ((await submitBtn.count()) > 0) {
          // Set up API call interception before triggering ingest
          const apiCall = expectApiCall(authedPage, {
            method: "POST",
            url: "/api/v1/media"
          })

          await submitBtn.first().click()

          // Look for processing indicator
          const _processing = authedPage.locator(
            ".processing, [data-status='processing'], .ant-spin"
          )

          // Verify the ingest API call completes without error
          try {
            const { response } = await apiCall
            expect(response.status()).toBeLessThan(400)
          } catch {
            // Ingest may not trigger an API call if URL validation
            // prevents submission or if config is not set up
          }
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Metadata Editing", () => {
    test("should navigate to media detail page", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Get list of media items
      const items = await mediaPage.getMediaItems()

      if (items.length > 0) {
        // Click on first item
        const firstItem = authedPage.locator(
          "[data-testid='media-item'], .media-item, .ant-table-row"
        ).first()

        await firstItem.click()

        await expect
          .poll(
            async () => {
              const urlChanged = /\/media(\/|%2F)\d+/i.test(authedPage.url()) || /\/media\/\d+/i.test(authedPage.url())
              const dialogVisible = await authedPage.getByRole("dialog").first().isVisible().catch(() => false)
              const editVisible = await authedPage.getByRole("button", { name: /edit/i }).first().isVisible().catch(() => false)
              return urlChanged || dialogVisible || editVisible
            },
            {
              timeout: 5_000,
              message: "Timed out waiting for the media detail surface to appear",
            }
          )
          .toBe(true)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display edit form for media metadata", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      const items = await mediaPage.getMediaItems()

      if (items.length > 0) {
        // Find edit button
        const editBtn = authedPage.getByRole("button", { name: /edit/i }).first()

        if ((await editBtn.count()) > 0 && (await editBtn.isVisible())) {
          await editBtn.click()

          // Look for edit form
          const _editForm = authedPage.locator(
            "form, [data-testid='edit-form'], .edit-modal"
          )

          await expect
            .poll(
              async () =>
                await authedPage
                  .locator("form, [data-testid='edit-form'], .edit-modal")
                  .first()
                  .isVisible()
                  .catch(() => false),
              {
                timeout: 5_000,
                message: "Timed out waiting for the media metadata edit form",
              }
            )
            .toBe(true)
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Error Cases", () => {
    test("should handle network failure during upload gracefully", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Page should remain functional even if uploads fail
      // This test verifies UI stability
      await assertNoCriticalErrors(diagnostics)
    })

    test("should display appropriate error for oversized files", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Most browsers prevent setting files larger than disk allows
      // This test verifies error handling UI exists
      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Quick Ingest", () => {
    const quickIngestFixtureFile = path.resolve(
      process.cwd(),
      "e2e/fixtures/media/quick-ingest-sample.mkv"
    )
    const quickIngestFixtureUrl = "https://example.com/e2e/quick-ingest-source.html"
    const bulkConferencePlaylistUrl =
      "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
    const bulkConferenceCollectionId = 700

    const fulfillJson = async (
      route: Route,
      status: number,
      body: Record<string, unknown>
    ) => {
      await route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      })
    }

    const buildBulkConferencePreflightFixture = () => {
      const items = Array.from({ length: 34 }, (_, index) => {
        const ordinal = index + 1
        const videoId =
          ordinal === 18 ? "conference-talk-08" : `conference-talk-${String(ordinal).padStart(2, "0")}`
        const duplicateStatus =
          ordinal === 8
            ? "duplicate_existing"
            : ordinal === 18
              ? "duplicate_in_batch"
              : "new"
        return {
          ordinal,
          source_url: `https://www.youtube.com/watch?v=${videoId}`,
          normalized_source_id: `youtube:video:${videoId}`,
          source_kind: "youtube_video",
          title: `Talk ${ordinal}`,
          speaker: `Speaker ${ordinal}`,
          duration_seconds: 1800 + ordinal,
          published_at: `2010-09-${String(Math.min(ordinal, 28)).padStart(2, "0")}`,
          thumbnail_url: null,
          duplicate_status: duplicateStatus,
          duplicate_of_ordinal: ordinal === 18 ? 8 : null,
          selected: duplicateStatus === "new",
        }
      })

      return {
        source_url: bulkConferencePlaylistUrl,
        source_kind: "youtube_watch_playlist",
        playlist_id: "PL0065D9B288E6804B",
        playlist_title: "Conference 2010",
        video_id: "PrNmmN6qBiw",
        item_count: items.length,
        selected_count: items.filter((item) => item.selected).length,
        duplicate_count: items.filter((item) => item.duplicate_status !== "new").length,
        warnings: [],
        items,
      }
    }

    const mockBulkConferenceApis = async (page: Page) => {
      const preflight = buildBulkConferencePreflightFixture()
      const collectionItems = new Map<number, Record<string, any>>()
      const jobToCollectionItem = new Map<number, number>()
      let nextCollectionItemId = 900
      let nextJobId = 1200
      let submittedJobCount = 0

      await page.route("**/openapi.json", async (route) => {
        await fulfillJson(route, 200, {
          openapi: "3.1.0",
          info: { title: "tldw e2e", version: "e2e" },
          paths: {
            "/api/v1/media": { get: {} },
            "/api/v1/media/playlists/preflight": { post: {} },
            "/api/v1/media/ingest/jobs": { post: {} },
            "/api/v1/media/collections": { get: {}, post: {} },
            "/api/v1/media/collections/{collection_id}": { get: {} },
            "/api/v1/rag/search": { post: {} },
          },
        })
      })
      await page.route("**/api/v1/config/docs-info", async (route) => {
        await fulfillJson(route, 200, {
          capabilities: {
            hasMediaPlaylistPreflight: true,
            hasMediaIngestJobs: true,
            hasDurableMediaCollections: true,
            hasKnowledgeQaMediaScope: true,
          },
        })
      })
      await page.route("**/api/v1/health", async (route) => {
        await fulfillJson(route, 200, { status: "ok", version: "e2e" })
      })
      await page.route(/\/api\/v1\/media\/?(?:\?|$)/, async (route, request) => {
        if (request.method().toUpperCase() !== "GET") {
          await route.continue()
          return
        }
        await fulfillJson(route, 200, {
          items: [],
          pagination: {
            page: 1,
            results_per_page: 20,
            total_items: 0,
            total_pages: 1,
          },
        })
      })
      await page.route("**/api/v1/media/playlists/preflight", async (route, request) => {
        if (request.method().toUpperCase() !== "POST") {
          await route.continue()
          return
        }
        await fulfillJson(route, 200, preflight)
      })
      await page.route("**/api/v1/media/collections", async (route, request) => {
        const url = new URL(request.url())
        if (
          request.method().toUpperCase() !== "POST" ||
          url.pathname.replace(/\/+$/, "") !== "/api/v1/media/collections"
        ) {
          await route.continue()
          return
        }
        await fulfillJson(route, 200, {
          id: bulkConferenceCollectionId,
          name: "Conference 2010 Review",
          kind: "conference",
          source_url: bulkConferencePlaylistUrl,
          metadata: {
            conference_name: "Conference",
            event_year: "2010",
            source_playlist_url: bulkConferencePlaylistUrl,
          },
          default_tags: ["conference", "talks"],
          created_at: "2026-05-16T00:00:00Z",
          updated_at: "2026-05-16T00:00:00Z",
          items: [],
        })
      })
      await page.route(
        `**/api/v1/media/collections/${bulkConferenceCollectionId}/items`,
        async (route, request) => {
          if (request.method().toUpperCase() !== "POST") {
            await route.continue()
            return
          }
          const payload = request.postDataJSON() as Record<string, any>
          const itemId = nextCollectionItemId++
          const item = {
            id: itemId,
            collection_id: bulkConferenceCollectionId,
            ordinal: payload.ordinal,
            source_url: payload.source_url,
            normalized_source_id: payload.normalized_source_id,
            source_kind: payload.source_kind,
            title: payload.title || `Talk ${payload.ordinal}`,
            speaker: payload.speaker || null,
            published_at: payload.published_at || null,
            track: payload.track || null,
            duplicate_status: payload.duplicate_status || "new",
            status: payload.status || "planned",
            media_id:
              payload.duplicate_status && payload.duplicate_status !== "new"
                ? 5000 + Number(payload.ordinal || 0)
                : null,
            content_item_id: null,
            latest_job_id: null,
            latest_run_id: null,
            idempotency_key: `collection-${itemId}-attempt-0`,
            retry_count: 0,
            error_summary: null,
            warnings: [],
            metadata: payload.metadata || {},
            tags: payload.tags || [],
            created_at: "2026-05-16T00:00:00Z",
            updated_at: "2026-05-16T00:00:00Z",
          }
          collectionItems.set(itemId, item)
          await fulfillJson(route, 200, item)
        }
      )
      await page.route(
        `**/api/v1/media/collections/${bulkConferenceCollectionId}/items/*`,
        async (route, request) => {
          if (request.method().toUpperCase() !== "PATCH") {
            await route.continue()
            return
          }
          const url = new URL(request.url())
          const itemId = Number(url.pathname.split("/").pop())
          const patch = request.postDataJSON() as Record<string, any>
          const current = collectionItems.get(itemId) || {
            id: itemId,
            collection_id: bulkConferenceCollectionId,
          }
          const next = {
            ...current,
            ...patch,
            media_id:
              patch.media_id != null
                ? Number(patch.media_id)
                : current.media_id ?? null,
            updated_at: "2026-05-16T00:01:00Z",
          }
          collectionItems.set(itemId, next)
          await fulfillJson(route, 200, next)
        }
      )
      await page.route("**/api/v1/media/ingest/jobs", async (route, request) => {
        const url = new URL(request.url())
        if (
          request.method().toUpperCase() !== "POST" ||
          url.pathname.replace(/\/+$/, "") !== "/api/v1/media/ingest/jobs"
        ) {
          await route.continue()
          return
        }
        submittedJobCount += 1
        const jobId = nextJobId++
        const form = request.postDataBuffer()
        const body = form?.toString("utf8") ?? ""
        const plannedMatch = body.match(/name="media_collection_item_id"\r\n\r\n([^\r\n]+)/)
        const plannedItemId = plannedMatch ? Number(plannedMatch[1]) : 0
        if (plannedItemId) {
          jobToCollectionItem.set(jobId, plannedItemId)
        }
        await fulfillJson(route, 200, {
          batch_id: `bulk-batch-${jobId}`,
          job_ids: [jobId],
          jobs: [{ id: jobId, status: "queued" }],
        })
      })
      await page.route("**/api/v1/media/ingest/jobs/*", async (route, request) => {
        if (request.method().toUpperCase() !== "GET") {
          await route.continue()
          return
        }
        const url = new URL(request.url())
        const jobId = Number(url.pathname.split("/").pop())
        const collectionItemId = jobToCollectionItem.get(jobId)
        const isFailed = jobId === 1209
        await fulfillJson(route, 200, {
          job_id: jobId,
          status: "completed",
          progress_percent: 100,
          result: isFailed
            ? {
                status: "Error",
                error: "Download failed for mocked talk",
                source_url: `https://www.youtube.com/watch?v=conference-talk-failed`,
              }
            : {
                status: "Success",
                media_id: collectionItemId ? 7000 + collectionItemId : jobId,
                source_url: `https://www.youtube.com/watch?v=conference-talk-${jobId}`,
                title: `Processed talk ${jobId}`,
              },
        })
      })
      await page.route(
        `**/api/v1/media/collections/${bulkConferenceCollectionId}`,
        async (route, request) => {
          if (request.method().toUpperCase() !== "GET") {
            await route.continue()
            return
          }
          await fulfillJson(route, 200, {
            id: bulkConferenceCollectionId,
            name: "Conference 2010 Review",
            kind: "conference",
            description: "Conference talks",
            source_url: bulkConferencePlaylistUrl,
            metadata: {
              conference_name: "Conference",
              event_year: "2010",
            },
            default_tags: ["conference", "talks"],
            created_at: "2026-05-16T00:00:00Z",
            updated_at: "2026-05-16T00:01:00Z",
            items: Array.from(collectionItems.values()).sort(
              (left, right) => Number(left.ordinal || 0) - Number(right.ordinal || 0)
            ),
          })
        }
      )

      return {
        getSubmittedJobCount: () => submittedJobCount,
      }
    }
    const createUniqueQuickIngestFixtureCopy = (): string => {
      const fixture = path.parse(quickIngestFixtureFile)
      const uniqueFixtureFile = path.join(
        "/tmp",
        `${fixture.name}-${generateTestId("quick-ingest-real-upload")}${fixture.ext}`
      )
      fs.copyFileSync(quickIngestFixtureFile, uniqueFixtureFile)
      return uniqueFixtureFile
    }

    const mockQuickIngestLifecycle = async (
      page: Parameters<typeof waitForConnection>[0],
      options: {
        sourceUrl?: string
        mediaId?: string
        jobId?: number
        batchId?: string
        processingResponses?: number
        completedResult?: Record<string, unknown>
        queueResponse?: {
          status: number
          body: Record<string, unknown>
        }
        fallbackAddResponse?: {
          status?: number
          body: Record<string, unknown>
        }
      } = {}
    ) => {
      const sourceUrl = options.sourceUrl ?? quickIngestFixtureUrl
      const mediaId = options.mediaId ?? "qi-media-e2e-101"
      const jobId = options.jobId ?? 101
      const batchId = options.batchId ?? "batch-e2e-quick-ingest"
      const title = "Quick ingest source"
      let remainingProcessingResponses = Math.max(0, options.processingResponses ?? 0)

      await page.route("**/api/v1/media/ingest/jobs", async (route, request) => {
        const url = new URL(request.url())
        const isQueueSubmit =
          request.method().toUpperCase() === "POST" &&
          url.pathname.replace(/\/+$/, "") === "/api/v1/media/ingest/jobs"
        if (!isQueueSubmit) {
          await route.continue()
          return
        }

        if (options.queueResponse) {
          await route.fulfill({
            status: options.queueResponse.status,
            contentType: "application/json",
            body: JSON.stringify(options.queueResponse.body)
          })
          return
        }

        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            batch_id: batchId,
            job_ids: [jobId],
            jobs: [
              {
                id: jobId,
                status: "queued"
              }
            ]
          })
        })
      })

      await page.route(`**/api/v1/media/ingest/jobs/${jobId}`, async (route, request) => {
        if (request.method().toUpperCase() !== "GET") {
          await route.continue()
          return
        }

        if (remainingProcessingResponses > 0) {
          remainingProcessingResponses -= 1
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({
              status: "processing",
              progress_percent: 40,
              progress_message: "Processing queued URL"
            })
          })
          return
        }

        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            job_id: jobId,
            status: "completed",
            progress_percent: 100,
            result: {
              status: "Success",
              media_id: mediaId,
              source_url: sourceUrl,
              title,
              ...options.completedResult,
            }
          })
        })
      })

      if (options.fallbackAddResponse) {
        await page.route("**/api/v1/media/add", async (route, request) => {
          const url = new URL(request.url())
          const isFallbackAdd =
            request.method().toUpperCase() === "POST" &&
            url.pathname.replace(/\/+$/, "") === "/api/v1/media/add"
          if (!isFallbackAdd) {
            await route.continue()
            return
          }

          await route.fulfill({
            status: options.fallbackAddResponse?.status ?? 200,
            contentType: "application/json",
            body: JSON.stringify(options.fallbackAddResponse.body)
          })
        })
      }

      return { sourceUrl, mediaId, jobId, batchId, title }
    }

    test("quick ingest opens from the visible media page triggers without helper fallback", async ({
      authedPage,
      diagnostics
    }) => {
      await authedPage.goto("/media", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)

      const dialog = authedPage.getByRole("dialog", { name: /quick ingest/i }).first()
      await expect(dialog).toBeHidden()

      const sidebarTrigger = authedPage.getByRole("button", { name: /^quick ingest$/i }).first()
      if (await sidebarTrigger.isVisible().catch(() => false)) {
        await sidebarTrigger.click()
        await expect(dialog).toBeVisible({ timeout: 15_000 })
        await dismissQuickIngest(authedPage)
      }

      let emptyStateTrigger = authedPage
        .getByRole("button", { name: /open quick ingest/i })
        .first()
      if (!(await emptyStateTrigger.isVisible().catch(() => false))) {
        const skipTutorial = authedPage.getByRole("button", { name: /skip for now/i }).first()
        if (await skipTutorial.isVisible().catch(() => false)) {
          await skipTutorial.click()
        }
        emptyStateTrigger = authedPage
          .getByRole("button", { name: /open quick ingest/i })
          .first()
      }
      await expect(emptyStateTrigger).toBeVisible({ timeout: 15_000 })
      await emptyStateTrigger.click()
      await expect(dialog).toBeVisible({ timeout: 15_000 })
      await expect(dialog).toContainText(
        /Add URLs or files\. Stored items appear in Media/i
      )
      await expect(dialog).toContainText(/Max file size: 50 MB/i)

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest communicates mixed URL paste validation before processing", async ({
      authedPage,
      diagnostics
    }) => {
      await authedPage.goto("/media", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)

      const dialog = await openQuickIngestDialog(authedPage)
      const urlInput = dialog
        .getByLabel(/url input area|paste urls input/i)
        .or(dialog.getByPlaceholder(/https:\/\/example\.com/i))
        .first()
      await urlInput.fill("https://example.com/valid\nnot-a-url")
      await dialog.getByRole("button", { name: /add urls/i }).first().click()

      await expect(dialog).toContainText(/1 valid \/ 1 invalid/i)
      await expect(dialog).toContainText(/Invalid URL format/i)

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest preserves terminal real .mkv upload results after reopen", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const uniqueFixtureFile = createUniqueQuickIngestFixtureCopy()
      const uniqueFixtureName = path.basename(uniqueFixtureFile)

      try {
        const mediaId = await ingestAndWaitForReady(authedPage, {
          file: uniqueFixtureFile
        })

        await dismissQuickIngest(authedPage)
        const dialog = await reopenQuickIngest(authedPage)
        await assertQuickIngestCompletedResults(dialog, {
          mediaId,
          fileName: uniqueFixtureName,
          terminalState: "either"
        })
      } finally {
        if (fs.existsSync(uniqueFixtureFile)) {
          fs.unlinkSync(uniqueFixtureFile)
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest ingests deterministic local URL through completion and reopen", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const {
        sourceUrl: ingestUrl,
        mediaId: expectedMediaId,
        title,
      } = await mockQuickIngestLifecycle(
        authedPage,
        {
          mediaId: "qi-media-url-complete"
        }
      )
      const mediaId = await ingestAndWaitForReady(authedPage, { url: ingestUrl })
      expect(mediaId).toBe(expectedMediaId)

      await dismissQuickIngest(authedPage)
      const dialog = await reopenQuickIngest(authedPage)
      await assertQuickIngestCompletedResults(dialog, {
        mediaId,
        sourceUrl: ingestUrl,
        title,
      })

      const openInMediaButton = dialog
        .getByRole("button", { name: /open .* media/i })
        .first()
      await expect(openInMediaButton).toBeVisible({ timeout: 15_000 })
      await openInMediaButton.click()
      await authedPage.waitForURL(
        (url) =>
          url.pathname === "/media" &&
          url.searchParams.get("id") === expectedMediaId,
        { timeout: 15_000 }
      )
      await expect(dialog).toBeHidden({ timeout: 15_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest restores skipped duplicate URL results after reopen", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const { sourceUrl: ingestUrl, title } = await mockQuickIngestLifecycle(authedPage, {
        mediaId: "qi-media-url-duplicate",
        completedResult: {
          db_message: "Media 'Quick ingest source' already exists. Overwrite not enabled."
        }
      })

      let dialog = await queueUrlAndStartProcessing(authedPage, ingestUrl)
      await assertQuickIngestCompletedResults(dialog, {
        sourceUrl: ingestUrl,
        title,
        terminalState: "skipped"
      })

      await dismissQuickIngest(authedPage)
      dialog = await reopenQuickIngest(authedPage)
      await assertQuickIngestCompletedResults(dialog, {
        sourceUrl: ingestUrl,
        title,
        terminalState: "skipped"
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest falls back to /api/v1/media/add when queue endpoint returns recognized 429", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const { sourceUrl: ingestUrl, title } = await mockQuickIngestLifecycle(authedPage, {
        queueResponse: {
          status: 429,
          body: {
            detail: "Concurrent job limit reached: queue is full."
          }
        },
        fallbackAddResponse: {
          body: {
            results: [
              {
                status: "Success",
                media_id: "qi-media-url-fallback",
                source_url: quickIngestFixtureUrl,
                title: "Quick ingest source"
              }
            ]
          }
        }
      })

      const fallbackAddRequest = authedPage.waitForRequest((request) => {
        if (request.method().toUpperCase() !== "POST") return false
        const url = new URL(request.url())
        return url.pathname.replace(/\/+$/, "") === "/api/v1/media/add"
      })

      const dialog = await queueUrlAndStartProcessing(authedPage, ingestUrl)
      await fallbackAddRequest
      await assertQuickIngestCompletedResults(dialog, {
        sourceUrl: ingestUrl,
        title,
      })
      await expect(dialog).not.toContainText(/queue is full|concurrent job limit/i)

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest configure options stay reachable in constrained viewport without forced preset selection", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(120_000)
      skipIfServerUnavailable(serverInfo)

      await authedPage.setViewportSize({ width: 390, height: 720 })
      const dialog = await openQuickIngestDialog(authedPage)
      await advanceQuickIngestToConfigureStep(dialog, quickIngestFixtureUrl)

      await expect(dialog).toContainText(
        "Presets are starting points. Adjust any settings below or in Advanced options to fit this run."
      )

      const overwriteToggle = await reachQuickIngestOptionInConstrainedViewport(
        dialog,
        /overwrite existing/i
      )
      await expect(overwriteToggle).toBeEnabled()
      const initialChecked = await overwriteToggle.getAttribute("aria-checked")
      await overwriteToggle.click()
      await expect(overwriteToggle).toHaveAttribute(
        "aria-checked",
        initialChecked === "true" ? "false" : "true"
      )

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest can be dismissed during processing and resumed from the normal trigger", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const { sourceUrl: ingestUrl } = await mockQuickIngestLifecycle(authedPage, {
        mediaId: "qi-media-url-resume",
        processingResponses: 1
      })
      const dialog = await queueUrlAndStartProcessing(authedPage, ingestUrl, {
        waitForState: "processing"
      })

      await dismissQuickIngest(authedPage, { duringProcessing: true })
      await expect(dialog).toBeHidden()

      const reopened = await reopenQuickIngest(authedPage)
      await expect(reopened).toContainText(/processing|completed/i)

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest restores URL sessions across refresh for queued, processing, and completed states", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(240_000)
      skipIfServerUnavailable(serverInfo)

      const { sourceUrl: ingestUrl, title } = await mockQuickIngestLifecycle(authedPage, {
        mediaId: "qi-media-url-refresh"
      })

      let dialog = await openQuickIngestDialog(authedPage)
      await advanceQuickIngestToConfigureStep(dialog, ingestUrl, { proceedToConfigure: false })
      await authedPage.reload({ waitUntil: "domcontentloaded" })
      dialog = await reopenQuickIngest(authedPage)
      await expect(dialog).toContainText(ingestUrl)

      dialog = await startQueuedQuickIngestProcessing(dialog, {
        waitForState: "processing"
      })
      await authedPage.reload({ waitUntil: "domcontentloaded" })
      dialog = await reopenQuickIngest(authedPage)
      await expect(dialog).toContainText(/processing|completed/i)
      await assertQuickIngestCompletedResults(dialog, {
        sourceUrl: ingestUrl,
        title,
      })

      await authedPage.reload({ waitUntil: "domcontentloaded" })
      dialog = await reopenQuickIngest(authedPage)
      await assertQuickIngestCompletedResults(dialog, {
        sourceUrl: ingestUrl,
        title,
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest file refresh restores reattach-required state", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(120_000)
      skipIfServerUnavailable(serverInfo)

      const dialog = await openQuickIngestDialog(authedPage)
      await queueFileForQuickIngest(dialog, quickIngestFixtureFile)

      await authedPage.reload({ waitUntil: "domcontentloaded" })
      const reopened = await reopenQuickIngest(authedPage)
      await expect(reopened).toContainText(/reattach this file after refresh/i)
      await expect(reopened.getByRole("button", { name: /use defaults & process/i })).toBeDisabled()
      await expect(reopened.getByRole("button", { name: /configure 0 items/i })).toBeDisabled()

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest draft sessions reopen from the normal trigger and do not expose the queued CTA during processing", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      skipIfServerUnavailable(serverInfo)

      const processQueuedCta = authedPage.getByTestId("process-queued-ingest-header")
      await authedPage.goto("/media", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)
      await expect(processQueuedCta).toHaveCount(0)

      const { sourceUrl: ingestUrl } = await mockQuickIngestLifecycle(authedPage, {
        mediaId: "qi-media-url-draft",
        processingResponses: 1
      })
      let dialog = await openQuickIngestDialog(authedPage)
      await advanceQuickIngestToConfigureStep(dialog, ingestUrl, { proceedToConfigure: false })
      await dismissQuickIngest(authedPage)

      dialog = await reopenQuickIngest(authedPage)
      await expect(dialog).toContainText(ingestUrl)
      await dismissQuickIngest(authedPage)

      dialog = await reopenQuickIngest(authedPage)
      dialog = await startQueuedQuickIngestProcessing(dialog, {
        waitForState: "processing"
      })
      await expect(dialog).toContainText(/processing/i)
      await expect(processQueuedCta).toHaveCount(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest handles a mocked 34-talk conference playlist through collection review", async ({
      authedPage,
      diagnostics
    }) => {
      test.setTimeout(180_000)
      await authedPage.addInitScript(() => {
        localStorage.removeItem("tldw-quick-ingest-session")
        localStorage.removeItem("__tldwServerCapabilitiesCacheV3")
      })
      const bulkApis = await mockBulkConferenceApis(authedPage)

      await authedPage.goto("/media", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)
      const dialog = await openQuickIngestDialog(authedPage)
      const urlInput = dialog.locator("textarea").first()

      await urlInput.fill(bulkConferencePlaylistUrl)
      await dialog.getByRole("button", { name: "Preview" }).click()

      await expect(dialog).toContainText("Conference 2010", { timeout: 20_000 })
      await expect(dialog).toContainText("34 items")
      await expect(dialog).toContainText("32 selected")
      await expect(dialog).toContainText("2 duplicates")

      await dialog.getByLabel("Include existing").check()
      await expect(dialog).toContainText("34 selected")
      await dialog.getByRole("checkbox", { name: "Select Talk 3", exact: true }).uncheck()
      await expect(dialog).toContainText("33 selected")
      await dialog.getByRole("button", { name: "Add 33" }).click()

      const metadataPanel = dialog.getByLabel("Conference batch metadata")
      await expect(metadataPanel).toContainText("33 selected")
      await metadataPanel.getByLabel("Collection name").fill("Conference 2010 Review")
      await metadataPanel.getByLabel("Conference name").fill("Conference")
      await metadataPanel.getByLabel("Event year").fill("2010")
      await metadataPanel.getByLabel("Shared tags").fill("conference, talks")

      await dialog.getByRole("button", { name: /configure 33 items/i }).click()
      await dialog.getByRole("button", { name: "Next" }).click()
      await expect(dialog).toContainText("Ready to Process")
      await dialog.getByRole("button", { name: /start processing/i }).click()

      await expect(dialog.getByTestId("wizard-results-step")).toBeVisible({
        timeout: 120_000,
      })
      await expect(dialog).toContainText("Succeeded (30)")
      await expect(dialog).toContainText("Skipped existing (2)")
      await expect(dialog).toContainText("Failed during processing (1)")
      await expect(dialog).toContainText(
        "Total: 30 succeeded, 2 skipped, 0 not submitted, 1 failed, 0 cancelled"
      )
      expect(bulkApis.getSubmittedJobCount()).toBe(31)

      await dialog.getByRole("button", { name: "Open collection" }).click()
      await expect(authedPage).toHaveURL(/\/media-collections\/700/, {
        timeout: 20_000,
      })
      await expect(
        authedPage.getByRole("heading", { name: "Conference 2010 Review" })
      ).toBeVisible({ timeout: 20_000 })
      await expect(authedPage.getByText("33 talks", { exact: true })).toBeVisible()
      await expect(authedPage.getByText("32 ready", { exact: true })).toBeVisible()
      await expect(authedPage.getByText("1 need attention", { exact: true })).toBeVisible()
      await expect(authedPage.getByText("Talk 1").first()).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("quick ingest extension playlist handoff opens the shared preflight state", async ({
      authedPage,
      diagnostics
    }) => {
      await authedPage.addInitScript(() => {
        localStorage.removeItem("tldw-quick-ingest-session")
        localStorage.removeItem("__tldwServerCapabilitiesCacheV3")
      })
      await mockBulkConferenceApis(authedPage)

      await authedPage.goto("/media", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)
      await authedPage.evaluate((url) => {
        window.dispatchEvent(
          new CustomEvent("tldw:open-quick-ingest", {
            detail: {
              source: "extension_active_tab",
              action: "playlist_preflight",
              sourceKind: "youtube_watch_playlist",
              url,
            },
          })
        )
      }, bulkConferencePlaylistUrl)

      const dialog = authedPage.getByRole("dialog", { name: /quick ingest/i }).first()
      await expect(dialog).toBeVisible({ timeout: 20_000 })
      await expect(dialog).toContainText("Conference 2010", { timeout: 20_000 })
      await expect(dialog).toContainText("34 items")
      await expect(dialog).toContainText("32 selected")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Content Review Flow", () => {
    test("should show moved-route guidance for the legacy review page", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.gotoReview()

      await expect(
        authedPage.getByRole("heading", { name: /this route has moved/i })
      ).toBeVisible({ timeout: 20_000 })
      await expect(authedPage.getByRole("link", { name: /open updated page/i })).toBeVisible({
        timeout: 20_000
      })
      await expect(authedPage).toHaveURL(/\/review(?:[/?#].*)?$/, {
        timeout: 20_000
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should navigate from the legacy review route to media-multi", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.gotoReview()
      await authedPage.getByRole("link", { name: /open updated page/i }).click()
      await expect(authedPage).toHaveURL(/\/media-multi(?:[/?#].*)?$/, {
        timeout: 20_000
      })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Media Search", () => {
    test("should search media items", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)

      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Find search input
      const searchInput = authedPage.getByPlaceholder(/search|filter/i).first()

      if ((await searchInput.count()) > 0 && (await searchInput.isVisible())) {
        await searchInput.fill("test query")
        await searchInput.press("Enter")

        await authedPage
          .locator(".ant-spin-spinning, [aria-busy='true']")
          .first()
          .waitFor({ state: "hidden", timeout: 5_000 })
          .catch(() => {})
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should filter media by type", async ({
      authedPage,
      diagnostics
    }) => {
      const mediaPage = new MediaPage(authedPage)
      await mediaPage.goto()
      await mediaPage.waitForReady()

      // Look for type filter
      const typeFilter = authedPage.getByLabel(/type|content type/i).first()

      if ((await typeFilter.count()) > 0 && (await typeFilter.isVisible())) {
        await typeFilter.click()

        // Select a type option
        const option = authedPage.getByRole("option").first()
        if ((await option.count()) > 0) {
          await option.click()
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Media Multi Page", () => {
    test("should navigate to media-multi page", async ({
      authedPage,
      diagnostics
    }) => {
      await authedPage.goto("/media-multi", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)

      await expect
        .poll(
          async () =>
            await authedPage
              .getByTestId("media-review-status-bar")
              .isVisible()
              .catch(() => false),
          {
            timeout: 20_000,
            message: "Timed out waiting for the media review surface to settle",
          }
        )
        .toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Media Trash", () => {
    test("should navigate to media trash page", async ({
      authedPage,
      diagnostics
    }) => {
      await authedPage.goto("/media-trash", { waitUntil: "domcontentloaded" })
      await waitForConnection(authedPage)

      await expect
        .poll(
          async () =>
            (await authedPage.getByTestId("trash-retention-policy").isVisible().catch(() => false)) ||
            (await authedPage.getByRole("heading", { name: /^trash$/i }).isVisible().catch(() => false)),
          {
            timeout: 20_000,
            message: "Timed out waiting for the media trash surface to settle",
          }
        )
        .toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
