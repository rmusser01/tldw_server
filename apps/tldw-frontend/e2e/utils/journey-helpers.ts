/**
 * Reusable helpers for cross-feature journey specs.
 * Each helper performs a common setup action and returns an identifier.
 */
import { type Locator, type Page, expect } from "@playwright/test"
import { expectApiCall } from "./api-assertions"
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from "./helpers"
import { NotesPage } from "./page-objects"

const QUICK_INGEST_JOB_STATUS_PATH = "/api/v1/media/ingest/jobs/"
const QUICK_INGEST_JOB_SUBMIT_MATCHER =
  /\/api\/v1\/media\/ingest\/jobs(?:[/?]|$)/i
const QUICK_INGEST_SUBMIT_MATCHER =
  /\/api\/v1\/media\/(?:ingest\/jobs|process-web-scraping|add)(?:[/?]|$)/i

const extractMediaId = (payload: unknown): string | undefined => {
  const body = payload as Record<string, any> | null
  const candidates = [
    body?.result?.media_id,
    body?.media_id,
    body?.data?.media_id,
    body?.job?.result?.media_id,
    ...(Array.isArray(body?.media_ids) ? body.media_ids : []),
    ...(Array.isArray(body?.result?.media_ids) ? body.result.media_ids : []),
    ...(Array.isArray(body?.data?.media_ids) ? body.data.media_ids : []),
  ]

  for (const candidate of candidates) {
    const value = String(candidate ?? "").trim()
    if (/^[1-9]\d*$/.test(value)) return value
  }

  return undefined
}

/** Require a saved Media ID, never a submission job or batch identifier. */
export const requireIngestMediaId = (
  payload: unknown,
  completedMediaId?: string
): string => {
  const mediaId = completedMediaId === undefined
    ? extractMediaId(payload)
    : extractMediaId({ media_id: completedMediaId })
  if (!mediaId) throw new Error("Ingestion did not return a canonical Media ID")
  return mediaId
}

const extractIngestJobIds = (payload: unknown): number[] => {
  const body = payload as Record<string, any> | null
  const rawIds = [
    body?.job_id,
    body?.id,
    ...(Array.isArray(body?.job_ids) ? body.job_ids : []),
    ...(Array.isArray(body?.jobs) ? body.jobs.map((job: any) => job?.id) : []),
  ]

  return rawIds
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value > 0)
}

const dismissAssistantSetupBlockingModal = async (
  page: Page,
  timeoutMs = 5_000
): Promise<void> => {
  const setupHeading = page.getByText(/build your assistant/i).first()
  const overlay = page.getByTestId("assistant-setup-overlay").first()
  const overlayVisible =
    (await overlay.isVisible({ timeout: 1_000 }).catch(() => false)) ||
    (await setupHeading.isVisible({ timeout: 1_000 }).catch(() => false))
  if (!overlayVisible) return

  const skipControls = [
    page.getByRole("button", { name: /skip for now/i }).first(),
    page.getByText(/^skip for now$/i).first(),
  ]

  for (const control of skipControls) {
    if (!(await control.isVisible({ timeout: 1_000 }).catch(() => false))) {
      continue
    }
    await control.click()
    await expect
      .poll(
        async () =>
          !(await overlay.isVisible().catch(() => false)) &&
          !(await setupHeading.isVisible().catch(() => false)),
        {
          timeout: timeoutMs,
          message: "Timed out dismissing the assistant setup modal",
        }
    )
      .toBe(true)
    return
  }

  throw new Error(
    "Assistant setup modal is visible, but no dismiss control matching 'Skip for now' was found."
  )
}

const clickQuickIngestTrigger = async (page: Page): Promise<void> => {
  const dialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
  const triggers = [
    page.getByTestId("open-quick-ingest").first(),
    page.getByRole("button", { name: /^quick ingest$/i }).first(),
    page.getByRole("button", { name: /open quick ingest/i }).first(),
  ]

  for (const trigger of triggers) {
    if (await trigger.isVisible({ timeout: 3_000 }).catch(() => false)) {
      try {
        await trigger.click({ timeout: 2_000 })
      } catch {
        if (await dialog.isVisible({ timeout: 1_000 }).catch(() => false)) {
          return
        }
        continue
      }
      if (await dialog.isVisible({ timeout: 2_000 }).catch(() => false)) {
        return
      }
    }
  }

  await page.evaluate(() => {
    ;(
      window as Window & {
        __tldwPendingQuickIngestOpen?: {
          mode: "normal"
          at: number
        }
      }
    ).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
    }
    window.dispatchEvent(new CustomEvent("tldw:open-quick-ingest"))
  })
}

const requestQuickIngestDialogFromPage = async (page: Page): Promise<void> => {
  await page.evaluate(() => {
    ;(
      window as Window & {
        __tldwPendingQuickIngestOpen?: {
          mode: "normal"
          at: number
        }
      }
    ).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
    }
    window.dispatchEvent(new CustomEvent("tldw:open-quick-ingest"))
  })
}

const waitForQuickIngestProcessingUi = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const minimizeButton = dialog
    .getByRole("button", { name: /minimize to background/i })
    .first()
  await expect(minimizeButton).toBeVisible({ timeout: timeoutMs })
}

const waitForQuickIngestCompletionUi = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const resultsStep = dialog.getByTestId("wizard-results-step")
  const completedRegion = dialog.getByRole("region", { name: /completed items/i }).first()
  const errorRegion = dialog.getByRole("region", { name: /error items/i }).first()

  await expect
    .poll(
      async () => {
        const resultsVisible = await resultsStep.isVisible().catch(() => false)
        const regionVisible = await completedRegion.isVisible().catch(() => false)
        const errorVisible = await errorRegion.isVisible().catch(() => false)
        return resultsVisible || regionVisible || errorVisible
      },
      { timeout: timeoutMs, message: "Timed out waiting for quick ingest completion UI" }
    )
    .toBe(true)
}

type QuickIngestCompletionExpectation = {
  mediaId?: string
  sourceUrl?: string
  fileName?: string
  title?: string
  terminalState?: "completed" | "skipped" | "failed" | "either"
}

type QuickIngestProcessingTarget = "processing" | "completed"

type QueueUrlAndStartProcessingOptions = {
  waitForState?: QuickIngestProcessingTarget
  performAnalysis?: boolean
  performChunking?: boolean
  requireJobSubmit?: boolean
  timeoutMs?: number
}

type DismissQuickIngestOptions = {
  duringProcessing?: boolean
  timeoutMs?: number
}

const clickQuickIngestCloseControl = async (
  page: Page,
  dialog: Locator
): Promise<void> => {
  const doneButton = dialog
    .getByRole("button", { name: /close the ingest wizard|done/i })
    .first()
  if (await doneButton.isVisible({ timeout: 1_000 }).catch(() => false)) {
    await doneButton.click()
    return
  }

  const closeButton = dialog.locator(".ant-modal-close").first()
  if (await closeButton.isVisible({ timeout: 1_000 }).catch(() => false)) {
    await closeButton.click()
    return
  }

  await page.keyboard.press("Escape")
}

const selectQuickIngestSource = async (
  dialog: Locator, source: "Files" | "Links", timeoutMs: number
): Promise<void> => {
  if (await dialog.getByTestId("wizard-results-step").isVisible()) {
    await dialog.getByText("More actions", { exact: true }).click()
    await dialog.getByRole("button", { name: "Add more sources", exact: true }).click()
  }
  await dialog.getByRole("button", { name: source, exact: true }).click({ timeout: timeoutMs })
}

const quickIngestOptionName = (option: "analysis" | "chunking"): RegExp =>
  new RegExp(`^Ingestion options\\s*[–-]\\s*${option}$`, "i")

const setQuickIngestProcessOptionEnabled = async (
  dialog: Locator,
  option: "analysis" | "chunking",
  enabled: boolean,
  timeoutMs: number
): Promise<void> => {
  const analysisToggle = dialog
    .getByRole("switch", { name: quickIngestOptionName(option) })
    .first()

  await expect(analysisToggle).toBeVisible({ timeout: timeoutMs })
  const current = (await analysisToggle.getAttribute("aria-checked")) === "true"
  if (current === enabled) return

  await analysisToggle.click()
  await expect(analysisToggle).toHaveAttribute(
    "aria-checked",
    enabled ? "true" : "false",
    { timeout: timeoutMs }
  )
}

const startQueuedSources = async (
  dialog: Locator,
  timeoutMs: number,
  options: Pick<QueueUrlAndStartProcessingOptions, "performAnalysis" | "performChunking"> = {}
): Promise<void> => {
  const settings = dialog.getByRole("button", { name: "Change settings", exact: true })
  if (await settings.isVisible()) await settings.click()
  await setQuickIngestProcessOptionEnabled(dialog, "analysis", options.performAnalysis === true, timeoutMs)
  await setQuickIngestProcessOptionEnabled(dialog, "chunking", options.performChunking === true, timeoutMs)
  const action = dialog.getByRole("button", { name: /^save [1-9]\d* sources?$/i })
  await expect(action).toBeEnabled({ timeout: timeoutMs })
  await action.click()
}

/**
 * Open the quick ingest dialog from any page surface that exposes the trigger.
 */
export async function openQuickIngestDialog(
  page: Page,
  timeoutMs = 30_000
): Promise<Locator> {
  const dialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
  await dismissAssistantSetupBlockingModal(page)
  if (await dialog.isVisible({ timeout: Math.min(timeoutMs, 5_000) }).catch(() => false)) {
    return dialog
  }

  await waitForConnection(page, Math.min(timeoutMs, 20_000))
  await dismissAssistantSetupBlockingModal(page)
  if (await dialog.isVisible({ timeout: Math.min(timeoutMs, 5_000) }).catch(() => false)) {
    return dialog
  }

  const quickIngestBtn = page.getByTestId("open-quick-ingest")
  if (!(await quickIngestBtn.isVisible({ timeout: 1_000 }).catch(() => false))) {
    await page.goto("/media", { waitUntil: "domcontentloaded" })
    await waitForConnection(page)
    await dismissAssistantSetupBlockingModal(page)
    if (await dialog.isVisible({ timeout: 3_000 }).catch(() => false)) {
      return dialog
    }
  }

  await clickQuickIngestTrigger(page)
  if (await dialog.isVisible({ timeout: Math.min(timeoutMs, 5_000) }).catch(() => false)) {
    return dialog
  }

  await requestQuickIngestDialogFromPage(page)
  await expect(dialog).toBeVisible({ timeout: timeoutMs })
  return dialog
}

/**
 * Close the quick ingest dialog.
 * When closing during processing we minimize the session instead of cancelling it.
 */
export async function dismissQuickIngest(
  page: Page,
  options: DismissQuickIngestOptions = {}
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? (options.duringProcessing ? 30_000 : 20_000)
  const dialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
  if (!(await dialog.isVisible({ timeout: 1_000 }).catch(() => false))) {
    return
  }

  await clickQuickIngestCloseControl(page, dialog)

  const minimizeButton = page
    .getByRole("button", { name: /minimize to background/i })
    .first()
  if (await minimizeButton.isVisible({ timeout: 2_000 }).catch(() => false)) {
    await minimizeButton.click()
  }

  await expect(dialog).toBeHidden({
    timeout: timeoutMs,
  })
}

/**
 * Reopen the quick ingest dialog after dismissal or refresh.
 */
export async function reopenQuickIngest(
  page: Page,
  timeoutMs = 30_000
): Promise<Locator> {
  return openQuickIngestDialog(page, timeoutMs)
}

/**
 * Queue a URL and submit it from the hosted Quick Ingest dialog.
 */
export async function queueUrlAndStartProcessing(
  page: Page,
  url: string,
  options: QueueUrlAndStartProcessingOptions = {}
): Promise<Locator> {
  const timeoutMs = options.timeoutMs ?? 120_000
  const dialog = await openQuickIngestDialog(page, timeoutMs)
  const submitRequest = expectApiCall(
    page,
    {
      method: "POST",
      url: options.requireJobSubmit
        ? QUICK_INGEST_JOB_SUBMIT_MATCHER
        : QUICK_INGEST_SUBMIT_MATCHER,
    },
    timeoutMs
  )

  await queueUrlForQuickIngest(dialog, url, timeoutMs)
  await startQueuedSources(dialog, timeoutMs, options)

  await submitRequest

  if (options.waitForState === "processing") {
    await waitForQuickIngestProcessingUi(dialog, timeoutMs)
    return dialog
  }

  await waitForQuickIngestCompletionUi(dialog, timeoutMs)
  return dialog
}

/**
 * Start processing from an already queued quick-ingest dialog without re-adding inputs.
 */
export async function startQueuedQuickIngestProcessing(
  dialog: Locator,
  options: QueueUrlAndStartProcessingOptions = {}
): Promise<Locator> {
  const timeoutMs = options.timeoutMs ?? 120_000
  const submitRequest = expectApiCall(
    dialog.page(),
    {
      method: "POST",
      url: options.requireJobSubmit
        ? QUICK_INGEST_JOB_SUBMIT_MATCHER
        : QUICK_INGEST_SUBMIT_MATCHER,
    },
    timeoutMs
  )

  await startQueuedSources(dialog, timeoutMs, options)
  await submitRequest

  if (options.waitForState === "processing") {
    await waitForQuickIngestProcessingUi(dialog, timeoutMs)
    return dialog
  }

  await waitForQuickIngestCompletionUi(dialog, timeoutMs)
  return dialog
}

/**
 * Assert the quick ingest results view has reached a completed state.
 */
export async function assertQuickIngestCompletedResults(
  dialog: Locator,
  expectation: QuickIngestCompletionExpectation = {},
  timeoutMs = 60_000
): Promise<void> {
  await waitForQuickIngestCompletionUi(dialog, timeoutMs)
  await expect(dialog.getByTestId("wizard-results-step")).toBeVisible({
    timeout: timeoutMs,
  })
  const completedRegion = dialog.getByRole("region", { name: /completed items/i }).first()
  const skippedRegion = dialog.getByRole("region", { name: /skipped items/i }).first()
  const errorRegion = dialog.getByRole("region", { name: /error items/i }).first()
  const terminalState = expectation.terminalState ?? "completed"
  let sawSkippedRegion = false

  if (terminalState === "completed") {
    await expect(completedRegion).toBeVisible({
      timeout: timeoutMs,
    })
  } else if (terminalState === "skipped") {
    await expect(skippedRegion).toBeVisible({
      timeout: timeoutMs,
    })
    sawSkippedRegion = true
  } else if (terminalState === "failed") {
    await expect(errorRegion).toBeVisible({
      timeout: timeoutMs,
    })
  } else {
    await expect
      .poll(
        async () =>
          (await completedRegion.isVisible().catch(() => false)) ||
          (await skippedRegion.isVisible().catch(() => false)) ||
          (await errorRegion.isVisible().catch(() => false)),
        {
          timeout: timeoutMs,
          message: "Timed out waiting for quick ingest terminal results to show completed, skipped, or failed items",
        }
      )
      .toBe(true)
    sawSkippedRegion = await skippedRegion.isVisible().catch(() => false)
  }

  const summary = dialog.getByTestId("wizard-results-step").getByRole("status").first()
  await expect(summary).toContainText(/\d+ (?:saved to Media|extracted|skipped|failed|cancelled)/i, { timeout: timeoutMs })
  if (sawSkippedRegion) {
    await expect(skippedRegion).toContainText(/already|duplicate|overwrite/i, { timeout: timeoutMs })
  }

  const candidateTexts = [
    expectation.title,
    expectation.fileName,
    expectation.mediaId,
    expectation.sourceUrl,
    expectation.sourceUrl
      ? decodeURIComponent(new URL(expectation.sourceUrl).pathname.split("/").filter(Boolean).pop() || "")
      : undefined,
  ].filter((candidate): candidate is string => Boolean(String(candidate || "").trim()))

  if (candidateTexts.length > 0) {
    await expect
      .poll(
        async () => {
          const dialogText = ((await dialog.textContent().catch(() => "")) || "").toLowerCase()
          return candidateTexts.some((candidate) =>
            dialogText.includes(candidate.toLowerCase())
          )
        },
        {
          timeout: timeoutMs,
          message: `Timed out waiting for quick ingest results to contain one of: ${candidateTexts.join(", ")}`,
        }
      )
      .toBe(true)
  }
}

/**
 * Queue a local file in the quick ingest dialog.
 */
export async function queueFileForQuickIngest(
  dialog: Locator,
  filePath: string,
  timeoutMs = 30_000
): Promise<void> {
  await selectQuickIngestSource(dialog, "Files", timeoutMs)
  await dialog.getByTestId("qi-file-input").setInputFiles(filePath)
  const fileName = filePath.split(/[\\/]/).pop() || filePath
  await expect(dialog.getByRole("region", { name: "Queued sources" })).toContainText(fileName, { timeout: timeoutMs })
}

/** Select Links and add the source to the same queue as file uploads. */
export async function queueUrlForQuickIngest(dialog: Locator, url: string, timeoutMs = 60_000): Promise<void> {
  await selectQuickIngestSource(dialog, "Links", timeoutMs)
  await dialog.getByRole("textbox", { name: /url input area|paste urls input/i }).fill(url)
  await dialog.getByRole("button", { name: /add urls to queue/i }).click()
  await expect(dialog.getByRole("region", { name: "Queued sources" })).toContainText(url, { timeout: timeoutMs })
}

export const waitForCompletedIngestJob = async (
  page: Page,
  jobIds: number[],
  timeoutMs: number
): Promise<string | undefined> => {
  if (jobIds.length === 0) return undefined

  const successStatuses = ["completed", "succeeded", "success"]
  const terminalStatuses = [...successStatuses, "failed", "error", "cancelled", "canceled"]
  const readStatus = (payload: { status?: unknown; job?: { status?: unknown }; result?: { status?: unknown } } | null): string =>
    String(payload?.status ?? payload?.job?.status ?? payload?.result?.status ?? "").toLowerCase()
  const response = await page.waitForResponse(
    async (candidate) => {
      if (candidate.request().method().toUpperCase() !== "GET") return false
      const pathname = new URL(candidate.url()).pathname.replace(/\/+$/, "")
      if (!jobIds.some(jobId => pathname === `${QUICK_INGEST_JOB_STATUS_PATH}${jobId}`)) {
        return false
      }
      if (!candidate.ok()) return true
      return terminalStatuses.includes(readStatus(await candidate.json()))
    },
    { timeout: timeoutMs }
  )
  expect(response.ok(), `Ingest job status returned HTTP ${response.status()}`).toBe(true)
  const payload = await response.json()
  expect(successStatuses, `Ingest job ended with status ${readStatus(payload)}`).toContain(readStatus(payload))
  return requireIngestMediaId(payload)
}

/**
 * Ingest content via the media page and wait until processing completes.
 * Returns the canonical media_id only after successful ingestion.
 */
export async function ingestAndWaitForReady(
  page: Page,
  input: { url: string } | { file: string },
  timeoutMs = 120_000
): Promise<string> {
  await page.goto("/media", { waitUntil: "domcontentloaded" })
  await waitForConnection(page)

  const quickIngestDialog = await openQuickIngestDialog(page, timeoutMs)

  const submitRequest = expectApiCall(
    page,
    {
      method: "POST",
      url: QUICK_INGEST_SUBMIT_MATCHER,
    },
    timeoutMs
  )

  if ("url" in input) {
    await queueUrlForQuickIngest(quickIngestDialog, input.url, timeoutMs)
  } else {
    await queueFileForQuickIngest(quickIngestDialog, input.file, timeoutMs)
  }
  await startQueuedSources(quickIngestDialog, timeoutMs)

  const { response } = await submitRequest
  expect(response.ok(), `Ingestion submission returned HTTP ${response.status()}`).toBe(true)
  const body = await response.json()
  const isJobSubmission = QUICK_INGEST_JOB_SUBMIT_MATCHER.test(response.url())
  const ingestJobIds = isJobSubmission ? extractIngestJobIds(body) : []
  if (isJobSubmission) {
    expect(ingestJobIds.length, "Ingestion accepted without a job ID").toBeGreaterThan(0)
  } else {
    requireIngestMediaId(body)
  }
  const [completedMediaId] = await Promise.all([
    waitForCompletedIngestJob(page, ingestJobIds, timeoutMs),
    waitForQuickIngestCompletionUi(quickIngestDialog, timeoutMs),
  ])
  return requireIngestMediaId(body, completedMediaId)
}

/**
 * Create a note via the notes page. Returns the note title for later lookup.
 * Uses the same selectors as NotesPage page object for reliability.
 */
export async function createNote(
  page: Page,
  opts: { title: string; content: string }
): Promise<string> {
  const notesPage = new NotesPage(page)
  await notesPage.goto()
  await notesPage.assertPageReady()
  await notesPage.createNote({
    title: opts.title,
    content: opts.content,
  })
  return opts.title
}

/** Capture an SSE fetch in the page before Chromium discards its network body. */
export async function captureStreamedResponseBody(page: Page, pathname: string): Promise<() => Promise<string>> {
  await page.evaluate((targetPath) => {
    const originalFetch = window.fetch.bind(window)
    const captured = window as typeof window & { __uatStreamBody?: Promise<string> }
    window.fetch = async (...args) => {
      const response = await originalFetch(...args)
      if (response.url && new URL(response.url).pathname === targetPath) {
        captured.__uatStreamBody = response.clone().text()
      }
      return response
    }
  }, pathname)
  return () => page.evaluate(async () => {
    const body = (window as typeof window & { __uatStreamBody?: Promise<string> }).__uatStreamBody
    if (!body) throw new Error("No streamed response body was captured")
    return body
  })
}

/**
 * Wait for streaming response to complete in chat.
 */
export async function waitForStreamComplete(
  page: Page,
  timeoutMs = 60_000
): Promise<void> {
  const assistantMessages = page.locator("article[aria-label*='Assistant message']")

  await expect
    .poll(
      async () => {
        const assistantCount = await assistantMessages.count()
        if (assistantCount === 0) return false

        const latestAssistant = assistantMessages.last()
        const isGenerating = await latestAssistant
          .getByText(/Generating response/i)
          .isVisible()
          .catch(() => false)
        const hasStopStreaming = await latestAssistant
          .getByRole("button", { name: /Stop streaming response|Stop Streaming|Stop/i })
          .isVisible()
          .catch(() => false)
        const text = ((await latestAssistant.textContent().catch(() => "")) || "")
          .replace(/▋/g, "")
          .trim()

        return Boolean(text) && !isGenerating && !hasStopStreaming
      },
      {
        timeout: timeoutMs,
        message: "Timed out waiting for a completed streamed assistant response",
      }
    )
    .toBe(true)
}

/**
 * Verify server is available and return basic info.
 * Useful at the start of journey tests.
 */
export async function checkServerHealth(): Promise<{
  available: boolean
  version?: string
}> {
  try {
    const res = await fetchWithApiKey(
      `${TEST_CONFIG.serverUrl}/api/v1/health`,
      TEST_CONFIG.apiKey
    )
    if (res.ok) {
      const data = await res.json()
      return { available: true, version: data.version }
    }
    return { available: false }
  } catch {
    return { available: false }
  }
}
