/**
 * Journey: Ingest -> Evaluate -> Review
 *
 * End-to-end workflow that ingests content, runs an evaluation on it,
 * then navigates to content review to inspect results.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from "../../utils/fixtures"
import { EvaluationsPage, ContentReviewPage } from "../../utils/page-objects"
import { dismissQuickIngest, ingestAndWaitForReady } from "../../utils/journey-helpers"

test.describe("Ingest -> Evaluate -> Review journey", () => {
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
      expect(mediaId).toMatch(/^[1-9]\d*$/)
      await dismissQuickIngest(page)
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
