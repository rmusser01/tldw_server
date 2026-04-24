/**
 * Reusable helpers for cross-feature journey specs.
 * Each helper performs a common setup action and returns an identifier.
 */
import { type Locator, type Page, expect } from "@playwright/test"
import { expectApiCall } from "./api-assertions"
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from "./helpers"
import { NotesPage } from "./page-objects"

const QUICK_INGEST_JOB_STATUS_PATH = "/api/v1/media/ingest/jobs/"
const QUICK_INGEST_SUBMIT_MATCHER =
  /\/api\/v1\/media\/(?:ingest\/jobs|process-web-scraping|add)(?:[/?]|$)/i

const extractMediaId = (payload: unknown): string | undefined => {
  const body = payload as Record<string, any> | null
  const candidate =
    body?.result?.media_id ??
    body?.media_id ??
    body?.data?.media_id ??
    body?.job?.result?.media_id

  return typeof candidate === "string" && candidate.trim() ? candidate.trim() : undefined
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

const waitForQuickIngestDialog = async (page: Page): Promise<Locator> => {
  const dialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
  await expect(dialog).toBeVisible({ timeout: 30_000 })
  return dialog
}

const findQuickIngestUrlInput = (dialog: Locator): Locator =>
  dialog.locator("textarea").first()

const findQuickIngestFileInput = (dialog: Locator): Locator =>
  dialog.locator('[data-testid="qi-file-input"], input[type="file"]').first()

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

const waitForQuickIngestQueueAdvance = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  await expect
    .poll(
      async () => {
        const useDefaultsVisible = await dialog
          .getByRole("button", { name: /use defaults/i })
          .isVisible()
          .catch(() => false)
        const configureVisible = await dialog
          .getByRole("button", { name: /configure \d+ items/i })
          .isVisible()
          .catch(() => false)
        const runVisible = await dialog
          .getByTestId("quick-ingest-run")
          .isVisible()
          .catch(() => false)
        return useDefaultsVisible || configureVisible || runVisible
      },
      {
        timeout: Math.min(timeoutMs, 20_000),
        message: "Timed out waiting for quick ingest to queue the submitted input",
      }
    )
    .toBe(true)
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
  const completionSummary = dialog.getByTestId("quick-ingest-complete")
  const completedRegion = dialog.getByRole("region", { name: /completed items/i }).first()
  const errorRegion = dialog.getByRole("region", { name: /error items/i }).first()

  await expect
    .poll(
      async () => {
        const resultsVisible = await resultsStep.isVisible().catch(() => false)
        const summaryVisible = await completionSummary.isVisible().catch(() => false)
        const regionVisible = await completedRegion.isVisible().catch(() => false)
        const errorVisible = await errorRegion.isVisible().catch(() => false)
        return resultsVisible || summaryVisible || regionVisible || errorVisible
      },
      { timeout: timeoutMs, message: "Timed out waiting for quick ingest completion UI" }
    )
    .toBe(true)
}

const waitForQuickIngestConfigureUi = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const ingestionOptionsHeading = dialog.getByText(/ingestion options/i).first()
  const nextButton = dialog.getByRole("button", { name: /^next$/i }).first()
  const backButton = dialog.getByRole("button", { name: /^back$/i }).first()

  await expect
    .poll(
      async () => {
        const headingVisible = await ingestionOptionsHeading.isVisible().catch(() => false)
        if (headingVisible) return true

        const nextVisible = await nextButton.isVisible().catch(() => false)
        const backVisible = await backButton.isVisible().catch(() => false)
        return nextVisible && backVisible
      },
      {
        timeout: timeoutMs,
        message: "Timed out waiting for the quick ingest configure step",
      }
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
  timeoutMs?: number
}

type AdvanceQuickIngestToConfigureStepOptions = {
  proceedToConfigure?: boolean
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

const ensureQuickIngestAddStep = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const urlInput = findQuickIngestUrlInput(dialog)
  if (await urlInput.isVisible({ timeout: 1_000 }).catch(() => false)) {
    return
  }

  const ingestMoreBtn = dialog.getByRole("button", { name: /ingest more/i }).first()
  if (await ingestMoreBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await ingestMoreBtn.click()
    await expect(urlInput).toBeVisible({ timeout: timeoutMs })
    return
  }
}

const advanceQuickIngestToReviewStep = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const nextBtn = dialog.getByRole("button", { name: /^next$/i }).first()
  await expect(nextBtn).toBeVisible({ timeout: timeoutMs })
  await nextBtn.click()
  await expect(dialog.getByRole("button", { name: /start processing/i }).first()).toBeVisible({
    timeout: timeoutMs,
  })
}

const startQuickIngestProcessing = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const startProcessingBtn = dialog.getByRole("button", { name: /start processing/i }).first()
  await expect(startProcessingBtn).toBeVisible({ timeout: timeoutMs })
  await startProcessingBtn.click()
}

const startQueuedQuickIngestFromCurrentStep = async (
  dialog: Locator,
  timeoutMs: number
): Promise<void> => {
  const useDefaultsBtn = dialog.getByRole("button", { name: /use defaults/i }).first()
  if (await useDefaultsBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await useDefaultsBtn.click()
    return
  }

  const configureBtn = dialog
    .getByRole("button", { name: /configure \d+ items/i })
    .first()
  if (await configureBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await configureBtn.click()
    await waitForQuickIngestConfigureUi(dialog, timeoutMs)
    await advanceQuickIngestToReviewStep(dialog, timeoutMs)
    await startQuickIngestProcessing(dialog, timeoutMs)
    return
  }

  const runBtn = dialog.getByTestId("quick-ingest-run").first()
  if (await runBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await runBtn.click()
    return
  }

  const startProcessingBtn = dialog
    .getByRole("button", { name: /start processing|run quick ingest/i })
    .first()
  await expect(startProcessingBtn).toBeVisible({ timeout: timeoutMs })
  await startProcessingBtn.click()
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
 * Add a URL, advance through the quick ingest steps, and start processing.
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
      url: QUICK_INGEST_SUBMIT_MATCHER,
    },
    timeoutMs
  )

  await advanceQuickIngestToConfigureStep(dialog, url, { timeoutMs })
  await advanceQuickIngestToReviewStep(dialog, timeoutMs)
  await startQuickIngestProcessing(dialog, timeoutMs)

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

  await startQueuedQuickIngestFromCurrentStep(dialog, timeoutMs)

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

  await expect(
    dialog.getByText(/Total:\s*\d+\s+succeeded(?:,\s*\d+\s+skipped)?,\s*\d+\s+failed/i)
  ).toBeVisible({
    timeout: timeoutMs,
  })

  if (sawSkippedRegion) {
    await expect(
      dialog.getByText(/already exists|overwrite/i).first()
    ).toBeVisible({
      timeout: timeoutMs,
    })
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
  await ensureQuickIngestAddStep(dialog, timeoutMs)
  const fileInput = findQuickIngestFileInput(dialog)
  await fileInput.setInputFiles(filePath)
  await waitForQuickIngestQueueAdvance(dialog, timeoutMs)

  const fileName = filePath.split(/[\\/]/).pop() || filePath
  await expect(dialog.getByText(fileName, { exact: false })).toBeVisible({
    timeout: timeoutMs,
  })
}

/**
 * Advance a queued URL into the configure step.
 */
export async function advanceQuickIngestToConfigureStep(
  dialog: Locator,
  url: string,
  options: AdvanceQuickIngestToConfigureStepOptions = {}
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? 60_000
  await ensureQuickIngestAddStep(dialog, timeoutMs)
  const urlInput = findQuickIngestUrlInput(dialog)
  await urlInput.fill(url)

  const addUrlsBtn = dialog.getByRole("button", { name: /add url/i }).first()
  if (await addUrlsBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await addUrlsBtn.click()
  } else {
    await dialog.getByRole("button", { name: /^add$/i }).first().click()
  }

  await waitForQuickIngestQueueAdvance(dialog, timeoutMs)

  if (options.proceedToConfigure === false) {
    return
  }

  const configureBtn = dialog
    .getByRole("button", { name: /configure \d+ items/i })
    .first()
  await configureBtn.click()
  await waitForQuickIngestConfigureUi(dialog, timeoutMs)
}

/**
 * Reach a configurable quick ingest option in a constrained viewport.
 */
export async function reachQuickIngestOptionInConstrainedViewport(
  dialog: Locator,
  optionLabel: RegExp,
  timeoutMs = 30_000
): Promise<Locator> {
  const option = dialog.getByLabel(optionLabel).first()
  await expect(option).toBeAttached({ timeout: timeoutMs })
  await expect
    .poll(
      async () => {
        const visible = await option.isVisible().catch(() => false)
        if (visible) return true
        await option
          .evaluate((element) => {
            if (!(element instanceof HTMLElement)) return
            element.scrollIntoView({ block: "center", inline: "nearest" })
            let current: HTMLElement | null = element.parentElement
            while (current) {
              const style = window.getComputedStyle(current)
              const scrollable =
                /(auto|scroll)/.test(style.overflowY) &&
                current.scrollHeight > current.clientHeight
              if (scrollable) {
                current.scrollTop = Math.max(0, element.offsetTop - current.clientHeight / 2)
                break
              }
              current = current.parentElement
            }
          })
          .catch(() => {})
        return option.isVisible().catch(() => false)
      },
      {
        timeout: timeoutMs,
        message: `Timed out reaching quick ingest option ${String(optionLabel)}`,
      }
    )
    .toBe(true)
  return option
}

const waitForCompletedIngestJob = async (
  page: Page,
  jobIds: number[],
  timeoutMs: number
): Promise<string | undefined> => {
  if (jobIds.length === 0) return undefined

  const response = await page
    .waitForResponse(
      async (candidate) => {
        if (candidate.request().method().toUpperCase() !== "GET") return false
        if (
          !jobIds.some((jobId) =>
            candidate.url().includes(`${QUICK_INGEST_JOB_STATUS_PATH}${jobId}`)
          )
        ) {
          return false
        }

        const payload = await candidate.json().catch(() => null)
        if (!payload) return false

        const status = String(
          payload?.status ?? payload?.job?.status ?? payload?.result?.status ?? ""
        ).toLowerCase()

        return (
          status === "completed" ||
          status === "succeeded" ||
          status === "success" ||
          Boolean(extractMediaId(payload))
        )
      },
      { timeout: timeoutMs }
    )
    .catch(() => null)

  if (!response) return undefined

  const payload = await response.json().catch(() => null)
  return extractMediaId(payload)
}

/**
 * Ingest content via the media page and wait until processing completes.
 * Returns the media_id from the completed ingest job when available.
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
    await advanceQuickIngestToConfigureStep(quickIngestDialog, input.url)

    const useDefaultsBtn = quickIngestDialog
      .getByRole("button", { name: /use defaults/i })
      .first()
    if (await useDefaultsBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await useDefaultsBtn.click()
    } else {
      const nextBtn = quickIngestDialog.getByRole("button", { name: /^next$/i }).first()
      if (await nextBtn.isVisible({ timeout: 5_000 }).catch(() => false)) {
        await nextBtn.click()
        await expect(quickIngestDialog.getByText(/ready to process/i)).toBeVisible({
          timeout: 20_000,
        })
      }

      const runBtn = quickIngestDialog.getByTestId("quick-ingest-run").first()
      if (await runBtn.isVisible({ timeout: 5_000 }).catch(() => false)) {
        await runBtn.click()
      } else {
        const startProcessingBtn = quickIngestDialog
          .getByRole("button", { name: /start processing|run quick ingest/i })
          .first()
        await startProcessingBtn.click()
      }
    }
  } else {
    await queueFileForQuickIngest(quickIngestDialog, input.file, timeoutMs)

    const useDefaultsBtn = quickIngestDialog
      .getByRole("button", { name: /use defaults/i })
      .first()
    if (await useDefaultsBtn.isVisible({ timeout: 3_000 }).catch(() => false)) {
      await useDefaultsBtn.click()
    } else {
      const configureBtn = quickIngestDialog
        .getByRole("button", { name: /configure \d+ items/i })
        .first()
      if (await configureBtn.isVisible({ timeout: 5_000 }).catch(() => false)) {
        await configureBtn.click()
        await waitForQuickIngestConfigureUi(quickIngestDialog, timeoutMs)
      }

      const nextBtn = quickIngestDialog.getByRole("button", { name: /^next$/i }).first()
      if (await nextBtn.isVisible({ timeout: 5_000 }).catch(() => false)) {
        await nextBtn.click()
        await expect(quickIngestDialog.getByText(/ready to process/i)).toBeVisible({
          timeout: timeoutMs,
        })
      }

      const startProcessingBtn = quickIngestDialog
        .getByRole("button", { name: /start processing/i })
        .first()
      await startProcessingBtn.click()
    }
  }

  const { response } = await submitRequest
  const body = await response.json().catch(() => ({}))
  const ingestJobIds = extractIngestJobIds(body)
  const completedMediaIdPromise = waitForCompletedIngestJob(page, ingestJobIds, timeoutMs)

  await waitForQuickIngestCompletionUi(quickIngestDialog, timeoutMs)

  return (
    (await completedMediaIdPromise) ??
    extractMediaId(body) ??
    String(body.id ?? ingestJobIds[0] ?? body.batch_id ?? "unknown")
  )
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
