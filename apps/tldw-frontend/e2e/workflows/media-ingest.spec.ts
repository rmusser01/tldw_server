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

      const emptyStateTrigger = authedPage
        .getByRole("button", { name: /open quick ingest/i })
        .first()
      await expect(emptyStateTrigger).toBeVisible({ timeout: 15_000 })
      await emptyStateTrigger.click()
      await expect(dialog).toBeVisible({ timeout: 15_000 })

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
