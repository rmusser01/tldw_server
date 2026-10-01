/**
 * Seeded engineering journey: successful ingestion and a real two-row exact-match run.
 * C-03 batch/export/foreign-owner variants and B-09 draft edit/commit remain separate.
 * Visiting Content Review does not qualify its draft workflow. No inference is needed
 * for precomputed exact-match output; no recipe-worker failure can count as a pass.
 */
import { randomUUID } from "node:crypto"
import { writeFile } from "node:fs/promises"
import { test, expect } from "../../utils/fixtures"
import { EvaluationsPage, ContentReviewPage } from "../../utils/page-objects"
import { dismissQuickIngest, ingestAndWaitForReady } from "../../utils/journey-helpers"
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from "../../utils/helpers"

const SAMPLES = [
  { input: { output: "ORBIT-742" }, expected: { output: "ORBIT-742" } },
  { input: { output: "ORBIT-999" }, expected: { output: "ORBIT-742" } },
]

test.describe("Ingest -> Evaluate -> Review journey", () => {
  test("ingests content and retains a completed two-example evaluation across reload", async ({
    authedPage: page, serverInfo,
  }, testInfo) => {
    test.setTimeout(240_000)
    expect(serverInfo.available, "The real application backend is required").toBe(true)
    await page.unrouteAll({ behavior: "wait" })
    const name = `c03_${randomUUID().replaceAll("-", "")}`
    const evidence: Record<string, unknown> = { name, auth: "seeded", mode: "deterministic-precomputed" }
    const apiGet = async (path: string) => {
      const response = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}${path}`)
      expect(response.ok, `Canonical GET ${path}: HTTP ${response.status}`).toBe(true)
      return response.json()
    }
    try {
      // A literal uploaded document keeps this evaluation journey independent of
      // external-site access; URL-source acceptance is tracked separately.
      const sourceText = `${name}: Exact-match evaluation compares each output against its expected answer.`
      const sourceFile = testInfo.outputPath(`${name}.txt`)
      await writeFile(sourceFile, sourceText, "utf8")
      const mediaId = await ingestAndWaitForReady(page, { file: sourceFile })
      const source = await apiGet(`/api/v1/media/${mediaId}`)
      expect(String(source.media_id)).toBe(mediaId)
      expect(source.content.text.trim()).toBe(sourceText)
      evidence.mediaId = mediaId
      await dismissQuickIngest(page)

      const evaluations = new EvaluationsPage(page)
      await evaluations.goto()
      await evaluations.assertPageReady()
      await evaluations.switchTab("evaluations")
      await page.getByTestId("evaluations-create-button").click()
      const wizard = page.getByRole("dialog", { name: "New evaluation", exact: true })
      await wizard.getByLabel("Name", { exact: true }).fill(name)
      await wizard.getByLabel("Description", { exact: true }).fill("F-EVAL v1: one exact match and one mismatch")
      await wizard.getByLabel("Evaluation type", { exact: true }).click()
      // Ant Select exposes only adjacent virtual options by role; select the visible option.
      await page.getByTitle("exact_match", { exact: true }).click()
      await wizard.getByRole("button", { name: "Next", exact: true }).click()
      const caseSensitive = wizard.getByRole("switch", { name: "Case sensitive", exact: true })
      await expect(caseSensitive).toHaveAttribute("aria-checked", "false")
      await caseSensitive.click()
      await expect(caseSensitive).toHaveAttribute("aria-checked", "true")
      await wizard.getByRole("button", { name: "Next", exact: true }).click()
      await wizard.getByRole("checkbox", { name: "Attach inline dataset instead of referencing dataset_id" }).check()
      const datasetEditor = wizard.locator("textarea")
      await expect(datasetEditor).toHaveCount(1)
      await datasetEditor.fill(JSON.stringify(SAMPLES))
      const created = page.waitForResponse(response =>
        response.request().method() === "POST" && new URL(response.url()).pathname.replace(/\/$/, "") === "/api/v1/evaluations"
      )
      const [creation] = await Promise.all([created, wizard.getByRole("button", { name: "Create", exact: true }).click()])
      expect(creation.ok()).toBe(true)
      expect(creation.request().postDataJSON()).toMatchObject({
        name, eval_type: "exact_match", dataset: SAMPLES,
        eval_spec: { metrics: ["exact_match"], case_sensitive: true },
      })
      const evaluation = await creation.json()
      expect(evaluation).toMatchObject({ name, eval_type: "exact_match", eval_spec: { case_sensitive: true } })
      expect(evaluation.id).toMatch(/^eval_/)
      expect(evaluation.dataset_id).toMatch(/^dataset_/)
      await expect(wizard).toBeHidden()
      const datasetPath = `/api/v1/evaluations/datasets/${evaluation.dataset_id}?include_samples=true`
      const dataset = await apiGet(datasetPath)
      expect(dataset).toMatchObject({ id: evaluation.dataset_id, sample_count: 2, samples: SAMPLES })
      evidence.evaluation = evaluation
      evidence.dataset = dataset

      await evaluations.switchTab("runs")
      await page.getByLabel("Target model", { exact: true }).fill("precomputed")
      const started = page.waitForResponse(response =>
        response.request().method() === "POST" && new URL(response.url()).pathname === `/api/v1/evaluations/${evaluation.id}/runs`
      )
      const [submission] = await Promise.all([
        started, page.locator('[data-eval-tour="start-run"]').click(),
      ])
      expect(submission.ok()).toBe(true)
      expect(submission.request().postDataJSON().target_model).toBe("precomputed")
      const runId = (await submission.json()).id
      expect(runId).toMatch(/^run_/)
      const runPath = `/api/v1/evaluations/runs/${runId}`
      await expect.poll(async () => (await apiGet(runPath)).status, { timeout: 60_000 }).toBe("completed")
      const run = await apiGet(runPath)
      expect(run).toMatchObject({
        id: runId, eval_id: evaluation.id, status: "completed", target_model: "precomputed",
        progress: { completed_samples: 2, total_samples: 2 },
        results: {
          aggregate: { total_samples: 2, failed_samples: 0, pass_rate: 0.5, mean_score: 0.5, min_score: 0, max_score: 1 },
          by_metric: { exact_match: { mean: 0.5, min: 0, max: 1 } },
          failed_samples: [],
          sample_results: [
            { sample_id: "sample_000000", scores: { exact_match: 1 }, passed: true, avg_score: 1 },
            { sample_id: "sample_000001", scores: { exact_match: 0 }, passed: false, avg_score: 0 },
          ],
        },
      })
      evidence.run = run
      const visibleResults = page.locator("pre").filter({ hasText: '"sample_results"' })
      await expect.poll(async () => JSON.parse(await visibleResults.innerText())).toEqual(run.results)
      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await evaluations.switchTab("evaluations")
      await page.getByTestId("evaluations-list-card").getByText(name, { exact: true }).click()
      await evaluations.switchTab("runs")
      await page.getByText(`Run ${runId}`, { exact: true }).first().click()
      await expect.poll(async () => JSON.parse(await visibleResults.innerText())).toEqual(run.results)
      expect(await apiGet(runPath)).toEqual(run)
      expect(await apiGet(datasetPath)).toEqual(dataset)
      expect(await apiGet(`/api/v1/evaluations/${evaluation.id}`)).toEqual(evaluation)

      // Preserve this navigation check without claiming B-09 draft/commit acceptance.
      const review = new ContentReviewPage(page)
      await review.goto()
      await review.assertPageReady()
      expect(String((await apiGet(`/api/v1/media/${mediaId}`)).media_id)).toBe(mediaId)
    } finally {
      await testInfo.attach("c03-evaluation-evidence.json", {
        body: JSON.stringify(evidence, null, 2), contentType: "application/json",
      })
    }
  })
})
