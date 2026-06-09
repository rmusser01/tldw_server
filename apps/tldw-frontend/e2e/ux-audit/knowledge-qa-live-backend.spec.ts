import { type Page } from "@playwright/test"

import {
  test,
  expect,
  assertNoCriticalErrors,
  type ServerInfo,
} from "../utils/fixtures"
import { seedAuth, TEST_CONFIG, dismissConnectionModals } from "../utils/helpers"
import { KnowledgeQAPage } from "../utils/page-objects/KnowledgeQAPage"
import {
  getRequiredKnowledgeQaLiveSourceId,
  loadKnowledgeQaLiveManifest,
  type KnowledgeQaLiveManifest,
} from "../fixtures/knowledge-qa-live"

const expectLiveBackendAvailable = (serverInfo: ServerInfo): void => {
  expect(
    serverInfo.available,
    "Launch the backend and seed fixtures before running this release gate. " +
      "Run Helper_Scripts/seed_knowledge_qa_uat.py and set TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST."
  ).toBe(true)
}

const seedKnowledgeAuth = async (page: Page): Promise<void> => {
  await seedAuth(page, {
    serverUrl: TEST_CONFIG.serverUrl,
    allowOffline: false,
  })
}

test.describe("Knowledge QA live backend UAT gates", () => {
  let manifest: KnowledgeQaLiveManifest

  test.beforeEach(() => {
    manifest = loadKnowledgeQaLiveManifest()
  })

  test("fixture manifest precondition includes seeded source IDs", async () => {
    expect(manifest.schemaVersion).toBe(1)
    expect(getRequiredKnowledgeQaLiveSourceId(manifest, "cited_media")).toBeTruthy()
    expect(getRequiredKnowledgeQaLiveSourceId(manifest, "distractor_media")).toBeTruthy()
    expect(getRequiredKnowledgeQaLiveSourceId(manifest, "scoped_note")).toBeTruthy()
  })

  test("backend unavailable recovery is visible and search is blocked", async ({ page }) => {
    await seedAuth(page, {
      serverUrl: "http://127.0.0.1:1",
      allowOffline: false,
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })

    const recoveryRegion = page.getByRole("region", {
      name: /Can't reach your tldw server right now/i,
    })
    await expect(recoveryRegion).toBeVisible({ timeout: 30_000 })
    await expect(recoveryRegion).toContainText("http://127.0.0.1:1")
    await expect(recoveryRegion.getByRole("button", { name: /Retry connection/i }))
      .toBeVisible()
  })

  test("known cited answer renders with citations and inspectable evidence", async ({
    authedPage,
    serverInfo,
    diagnostics,
  }) => {
    expectLiveBackendAvailable(serverInfo)
    await seedKnowledgeAuth(authedPage)
    const qaPage = new KnowledgeQAPage(authedPage)

    await qaPage.goto()
    await qaPage.waitForReady()
    await qaPage.setWebFallback(false)

    const [ragResult] = await Promise.all([
      qaPage.waitForRagSearch(),
      qaPage.search(manifest.queries.cited),
    ])

    expect(ragResult.status).toBe(200)
    expect(ragResult.requestBody?.query).toBe(manifest.queries.cited)
    expect(ragResult.requestBody?.sources).toEqual(
      expect.arrayContaining(["media_db", "notes"])
    )
    expect(ragResult.requestBody?.enable_web_fallback).toBe(false)

    await qaPage.waitForResults(90_000)
    const answer = await qaPage.getAnswerText()
    expect(answer).toContain(manifest.expected.citedAnswerPhrase)
    expect(answer).toMatch(/\[\d+\]/)

    await expect(qaPage.getEvidencePanel()).toBeVisible({ timeout: 15_000 })
    await expect(
      qaPage.getEvidencePanel().getByText(manifest.sources.cited_media.title).first()
    ).toBeVisible({ timeout: 15_000 })
    await expect(
      qaPage.getEvidencePanel().getByText(manifest.expected.citedAnswerPhrase).first()
    ).toBeVisible({ timeout: 15_000 })

    await qaPage.openEvidenceDetails()
    await expect(qaPage.getEvidencePanel().getByText(/Web fallback disabled/i).first())
      .toBeVisible({ timeout: 15_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("known no-results query shows recovery without inventing an answer", async ({
    authedPage,
    serverInfo,
    diagnostics,
  }) => {
    expectLiveBackendAvailable(serverInfo)
    await seedKnowledgeAuth(authedPage)
    const qaPage = new KnowledgeQAPage(authedPage)

    await qaPage.goto()
    await qaPage.waitForReady()
    await qaPage.setWebFallback(false)

    const [ragResult] = await Promise.all([
      qaPage.waitForRagSearch(),
      qaPage.search(manifest.queries.noMatch),
    ])

    expect(ragResult.status).toBe(200)
    expect(ragResult.requestBody?.query).toBe(manifest.queries.noMatch)
    expect(ragResult.requestBody?.enable_web_fallback).toBe(false)

    await qaPage.waitForResults(90_000)
    await expect.poll(() => qaPage.hasNoResults(), { timeout: 30_000 }).toBe(true)
    const answer = await qaPage.getAnswerText()
    expect(answer).not.toContain(manifest.expected.citedAnswerPhrase)
    expect(answer).not.toContain(manifest.expected.scopedIncludedPhrase)

    await assertNoCriticalErrors(diagnostics)
  })

  test("scoped exact-note search excludes seeded distractor sources", async ({
    authedPage,
    serverInfo,
    diagnostics,
  }) => {
    expectLiveBackendAvailable(serverInfo)
    await seedKnowledgeAuth(authedPage)
    const qaPage = new KnowledgeQAPage(authedPage)
    const noteId = String(getRequiredKnowledgeQaLiveSourceId(manifest, "scoped_note"))

    await qaPage.goto()
    await qaPage.waitForReady()
    await qaPage.setWebFallback(false)
    await qaPage.selectSpecificSource("notes", manifest.sources.scoped_note.title)

    const [ragResult] = await Promise.all([
      qaPage.waitForRagSearch(),
      qaPage.search(manifest.queries.scopedIncluded),
    ])

    expect(ragResult.status).toBe(200)
    expect(ragResult.requestBody?.query).toBe(manifest.queries.scopedIncluded)
    expect(ragResult.requestBody?.sources).toEqual(expect.arrayContaining(["notes"]))
    expect(ragResult.requestBody?.include_note_ids).toEqual(expect.arrayContaining([noteId]))
    expect(ragResult.requestBody?.enable_web_fallback).toBe(false)

    await qaPage.waitForResults(90_000)
    const answer = await qaPage.getAnswerText()
    const evidenceText = await qaPage.getEvidencePanel().textContent()
    expect(`${answer}\n${evidenceText ?? ""}`).toContain(
      manifest.expected.scopedIncludedPhrase
    )
    expect(`${answer}\n${evidenceText ?? ""}`).not.toContain(
      manifest.expected.scopedExcludedPhrase
    )

    await assertNoCriticalErrors(diagnostics)
  })

  test("exported markdown preserves trust and evidence labels", async ({
    authedPage,
    serverInfo,
    diagnostics,
  }) => {
    expectLiveBackendAvailable(serverInfo)
    await seedKnowledgeAuth(authedPage)
    const qaPage = new KnowledgeQAPage(authedPage)

    await qaPage.goto()
    await qaPage.waitForReady()
    await qaPage.setWebFallback(false)

    await Promise.all([
      qaPage.waitForRagSearch(),
      qaPage.search(manifest.queries.cited),
    ])
    await qaPage.waitForResults(90_000)

    await dismissConnectionModals(authedPage)
    await qaPage.openExportDialog()
    const exportedMarkdown = await qaPage.copyExportMarkdown()

    expect(exportedMarkdown).toContain("## Trust and Evidence")
    expect(exportedMarkdown).toMatch(/^Trust:/m)
    expect(exportedMarkdown).toMatch(/^Evidence origin: local library$/im)
    expect(exportedMarkdown).toContain(manifest.sources.cited_media.title)
    expect(exportedMarkdown).toContain(manifest.expected.citedAnswerPhrase)

    await assertNoCriticalErrors(diagnostics)
  })
})
