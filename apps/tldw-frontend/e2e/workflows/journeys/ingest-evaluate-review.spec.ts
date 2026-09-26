/**
 * Journey: Ingest -> Evaluate -> Review
 *
 * End-to-end workflow that ingests content, runs an evaluation on it,
 * then navigates to content review to inspect results.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from "../../utils/fixtures"
import { EvaluationsPage, ContentReviewPage } from "../../utils/page-objects"
import { ingestAndWaitForReady } from "../../utils/journey-helpers"

test.describe("Ingest -> Evaluate -> Review journey", () => {
  test("closes terminal ingest before the next evaluation click", async ({
    authedPage: page,
    serverInfo,
  }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    skipIfNoModels(serverInfo)

    await ingestAndWaitForReady(page, {
      url: "https://en.wikipedia.org/wiki/Unit_testing",
    })

    const quickIngestDialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
    await testInfo.attach("quick-ingest-modal-state.json", {
      body: JSON.stringify({
        count: await quickIngestDialog.count(),
        visible: await quickIngestDialog.isVisible().catch(() => false),
      }),
      contentType: "application/json",
    })
    await expect(quickIngestDialog).toBeHidden()

    const evalsPage = new EvaluationsPage(page)
    await evalsPage.goto()
    await evalsPage.assertPageReady()
    await evalsPage.switchTab("synthetic-review")
  })

  test("ingest content, run evaluation, check content review", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)
    skipIfNoModels(serverInfo)

    const testUrl = "https://en.wikipedia.org/wiki/Unit_testing"
    let mediaId: string

    await test.step("Ingest content via URL", async () => {
      mediaId = await ingestAndWaitForReady(page, { url: testUrl })
      expect(mediaId).toBeTruthy()
    })

    await test.step("Navigate to evaluations and run an evaluation", async () => {
      const evalsPage = new EvaluationsPage(page)
      await evalsPage.goto()
      await evalsPage.assertPageReady()

      // Attempt to run an evaluation
      // The evaluations page may need configuration; this step verifies the
      // page loads and the evaluation workflow is accessible
      const evalTabVisible = await evalsPage.evaluationsTab.isVisible().catch(() => false)
      expect(evalTabVisible).toBe(true)

      // Try to trigger an evaluation run
      await evalsPage.runEvaluation()
    })

    await test.step("Check content review page", async () => {
      const reviewPage = new ContentReviewPage(page)
      await reviewPage.goto()
      await reviewPage.assertPageReady()

      // Content review may show drafts from the ingestion or be empty
      // depending on the ingestion pipeline configuration
      const headingVisible = await reviewPage.heading.isVisible().catch(() => false)
      const emptyVisible = await reviewPage.emptyState.isVisible().catch(() => false)

      // Either the heading or empty state should be visible
      expect(headingVisible || emptyVisible).toBe(true)
    })
  })
})
