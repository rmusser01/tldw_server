import { expect, test, type Page } from "@playwright/test"

import {
  assertRealServerHealth,
  getRequiredKnowledgeQaLiveSourceId,
  launchWithBuiltExtensionForLiveUat,
  loadKnowledgeQaLiveManifest,
  requireRealServerConfigStrict,
  type KnowledgeQaLiveManifest,
} from "./utils/real-server"

const createKnowledgeSeedConfig = (serverUrl: string, apiKey: string) => ({
  __tldw_first_run_complete: true,
  quickIngestInspectorIntroDismissed: true,
  quickIngestOnboardingDismissed: true,
  tldw_skip_landing_hub: true,
  "tldw:workflow:landing-config": {
    showOnFirstRun: true,
    dismissedAt: Date.now(),
    completedWorkflows: [],
  },
  tldwConfig: {
    serverUrl,
    authMode: "single-user",
    apiKey,
  },
})

const watchRuntimeFailures = (page: Page) => {
  const pageErrors: string[] = []
  const failedRequests: string[] = []

  page.on("pageerror", (error) => {
    pageErrors.push(error.message)
  })
  page.on("requestfailed", (request) => {
    const url = request.url()
    if (!url.startsWith("chrome-extension://")) return
    failedRequests.push(`${url} ${request.failure()?.errorText || ""}`.trim())
  })

  return { pageErrors, failedRequests }
}

const openKnowledge = async (
  manifest: KnowledgeQaLiveManifest
): Promise<{
  context: Awaited<ReturnType<typeof launchWithBuiltExtensionForLiveUat>>["context"]
  page: Page
  failures: ReturnType<typeof watchRuntimeFailures>
}> => {
  const { serverUrl, apiKey } = requireRealServerConfigStrict()
  const launchResult = await launchWithBuiltExtensionForLiveUat({
    seedConfig: createKnowledgeSeedConfig(serverUrl, apiKey),
    allowOffline: false,
    launchTimeoutMs: 60_000,
  })
  const failures = watchRuntimeFailures(launchResult.page)

  await launchResult.page.goto(`${launchResult.optionsUrl}#/knowledge`, {
    waitUntil: "domcontentloaded",
  })
  await expect(launchResult.page).toHaveURL(/options\.html#\/knowledge$/)
  await expect(launchResult.page.locator("#root")).not.toBeEmpty()
  await expect(
    launchResult.page.getByRole("heading", { name: /Ask Your Library/i })
  ).toBeVisible({ timeout: 45_000 })
  await expect(
    launchResult.page.getByLabel(/Search your knowledge base/i)
  ).toBeVisible({ timeout: 30_000 })
  expect(manifest.schemaVersion).toBe(1)

  return {
    context: launchResult.context,
    page: launchResult.page,
    failures,
  }
}

const setWebFallback = async (page: Page, enabled: boolean): Promise<void> => {
  const toggle = page.getByRole("button", { name: /Web fallback is currently/i }).first()
  await expect(toggle).toBeVisible({ timeout: 15_000 })
  const pressed = (await toggle.getAttribute("aria-pressed")) === "true"
  if (pressed !== enabled) {
    await toggle.click()
    await expect(toggle).toHaveAttribute("aria-pressed", enabled ? "true" : "false")
  }
}

const searchKnowledge = async (page: Page, query: string): Promise<void> => {
  const input = page.getByLabel(/Search your knowledge base/i)
  await expect(input).toBeVisible({ timeout: 20_000 })
  await input.fill(query)
  await input.press("Enter")
}

const waitForKnowledgeResult = async (page: Page): Promise<void> => {
  await expect
    .poll(
      async () => {
        const stopVisible = await page
          .getByRole("button", { name: /^Stop$/i })
          .isVisible()
          .catch(() => false)
        const hasAnswer = await page
          .getByTestId("knowledge-answer-content")
          .isVisible()
          .catch(() => false)
        const hasNoResults = await page
          .getByRole("button", { name: /Broaden scope|Enable web|Show nearest matches/i })
          .first()
          .isVisible()
          .catch(() => false)
        const hasSourceOnlyState = await page
          .getByText(/Found \d+ relevant source/i)
          .first()
          .isVisible()
          .catch(() => false)

        return !stopVisible && (hasAnswer || hasNoResults || hasSourceOnlyState)
      },
      { timeout: 90_000 }
    )
    .toBe(true)
}

const selectSpecificNote = async (page: Page, title: string): Promise<void> => {
  const scopeButton = page
    .getByRole("button", { name: /Open source scope and saved profiles/i })
    .first()
  await expect(scopeButton).toBeVisible({ timeout: 20_000 })
  await scopeButton.click()

  const scopeDialog = page.getByRole("dialog", { name: "Source scope and profiles" })
  await expect(scopeDialog).toBeVisible({ timeout: 15_000 })
  await scopeDialog.getByRole("button", { name: /^Specific:/i }).click()

  const specificDialog = page.getByRole("dialog", { name: "Specific source selector" })
  await expect(specificDialog).toBeVisible({ timeout: 15_000 })
  await specificDialog.getByRole("button", { name: /^Notes\b/i }).click()
  await specificDialog.getByPlaceholder(/Filter notes by title/i).fill(title)
  await specificDialog
    .getByRole("checkbox", { name: new RegExp(title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i") })
    .check({ timeout: 20_000 })

  await page.keyboard.press("Escape").catch(() => {})
  const closeScope = scopeDialog.getByRole("button", { name: "Close source scope" })
  if (await closeScope.isVisible().catch(() => false)) {
    await closeScope.click()
    await expect(scopeDialog).toBeHidden({ timeout: 10_000 })
  }
}

test.describe.configure({ mode: "serial" })

test.describe("Knowledge QA extension live backend UAT gates", () => {
  let manifest: KnowledgeQaLiveManifest

  test.beforeAll(async () => {
    manifest = loadKnowledgeQaLiveManifest()
    const { serverUrl, apiKey } = requireRealServerConfigStrict()
    await assertRealServerHealth(serverUrl, apiKey)
  })

  test("launches packaged extension Knowledge QA route with live config", async () => {
    const { context, page, failures } = await openKnowledge(manifest)

    try {
      await expect(page.getByText(/Queries stay on your tldw server/i)).toBeVisible()
      expect(failures.pageErrors).toEqual([])
      expect(failures.failedRequests).toEqual([])
    } finally {
      await context.close()
    }
  })

  test("runs cited Knowledge QA search with evidence in the extension", async () => {
    const { context, page, failures } = await openKnowledge(manifest)

    try {
      await setWebFallback(page, false)
      await searchKnowledge(page, manifest.queries.cited)
      await waitForKnowledgeResult(page)

      const answer = (await page.getByTestId("knowledge-answer-content").textContent()) ?? ""
      expect(answer).toContain(manifest.expected.citedAnswerPhrase)
      expect(answer).toMatch(/\[\d+\]/)
      await expect(page.getByRole("complementary", { name: /Evidence panel/i }))
        .toContainText(manifest.sources.cited_media.title)
      expect(failures.pageErrors).toEqual([])
      expect(failures.failedRequests).toEqual([])
    } finally {
      await context.close()
    }
  })

  test("shows no-results recovery in the extension", async () => {
    const { context, page, failures } = await openKnowledge(manifest)

    try {
      await setWebFallback(page, false)
      await searchKnowledge(page, manifest.queries.noMatch)
      await waitForKnowledgeResult(page)

      await expect(
        page.getByRole("button", { name: /Broaden scope|Enable web|Show nearest matches/i })
          .first()
      ).toBeVisible()
      expect(failures.pageErrors).toEqual([])
      expect(failures.failedRequests).toEqual([])
    } finally {
      await context.close()
    }
  })

  test("scopes extension search to selected note and excludes distractor", async () => {
    const noteId = String(getRequiredKnowledgeQaLiveSourceId(manifest, "scoped_note"))
    const { context, page, failures } = await openKnowledge(manifest)

    try {
      await setWebFallback(page, false)
      await selectSpecificNote(page, manifest.sources.scoped_note.title)
      await searchKnowledge(page, manifest.queries.scopedIncluded)
      await waitForKnowledgeResult(page)

      const answer = (await page.getByTestId("knowledge-answer-content").textContent()) ?? ""
      const evidence = (await page.getByRole("complementary", { name: /Evidence panel/i }).textContent()) ?? ""
      expect(`${answer}\n${evidence}`).toContain(manifest.expected.scopedIncludedPhrase)
      expect(`${answer}\n${evidence}`).not.toContain(manifest.expected.scopedExcludedPhrase)
      await expect(page.getByText(new RegExp(`ID: ${noteId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`)).first())
        .toBeVisible()
      expect(failures.pageErrors).toEqual([])
      expect(failures.failedRequests).toEqual([])
    } finally {
      await context.close()
    }
  })
})
