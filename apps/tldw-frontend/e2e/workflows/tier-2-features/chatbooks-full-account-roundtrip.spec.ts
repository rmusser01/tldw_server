import { expect, test, type Page } from "@playwright/test"
import { existsSync, mkdirSync } from "node:fs"
import path from "node:path"

import { seedAuth } from "../../smoke/smoke.setup"

type UatPhase = "export" | "import"
type JobResponse = {
  status?: unknown
  error_message?: unknown
  progress_percentage?: unknown
  metadata?: {
    post_write_verification?: unknown
    imported_items?: Record<string, unknown>
  }
}

const phase = process.env.TLDW_CHATBOOK_UAT_PHASE as UatPhase | undefined
const archivePath = process.env.TLDW_CHATBOOK_UAT_ARCHIVE_PATH
const accessToken = process.env.TLDW_CHATBOOK_UAT_ACCESS_TOKEN
const apiUrl = process.env.TLDW_CHATBOOK_UAT_API_URL?.replace(/\/$/, "")
const externalHarnessEnabled =
  (phase === "export" || phase === "import") &&
  Boolean(archivePath && accessToken && apiUrl)
const liveTierEnabled = process.env.TLDW_LIVE_TIER_UAT === "1"
const liveApiUrl = (
  process.env.TLDW_E2E_SERVER_URL || process.env.TLDW_SERVER_URL || ""
).replace(/\/$/, "")
const liveApiKey = process.env.TLDW_E2E_API_KEY || process.env.TLDW_API_KEY || ""
const liveArchivePath = path.join(
  process.cwd(),
  "test-results/live-tier-uat",
  process.env.TLDW_LIVE_TIER_UAT_RUN_ID || "local",
  "chatbooks-full-account.chatbook",
)

const authHeaders = (token?: string, apiKey?: string): Record<string, string> =>
  token ? { Authorization: `Bearer ${token}` } : { "X-API-KEY": apiKey || "" }

async function openChatbooks(
  page: Page,
  auth: { apiUrl: string; apiKey?: string; accessToken?: string },
): Promise<void> {
  await seedAuth(
    page,
    auth.accessToken
      ? {
          serverUrl: auth.apiUrl,
          authMode: "multi-user",
          apiKey: "",
          accessToken: auth.accessToken,
          allowOffline: false,
        }
      : {
          serverUrl: auth.apiUrl,
          authMode: "single-user",
          apiKey: auth.apiKey,
          accessToken: "",
          allowOffline: false,
        },
  )
  await page.goto("/chatbooks", { waitUntil: "domcontentloaded" })
  await expect(
    page.getByRole("heading", { name: /Chatbooks Backup & Import/i })
  ).toBeVisible({ timeout: 30_000 })
}

async function waitForJob(
  kind: "export" | "import",
  jobId: string,
  auth: { apiUrl: string; apiKey?: string; accessToken?: string },
): Promise<JobResponse> {
  const deadline = Date.now() + 180_000
  let lastStatus = "unknown"
  while (Date.now() < deadline) {
    const response = await fetch(
      `${auth.apiUrl}/api/v1/chatbooks/${kind}/jobs/${encodeURIComponent(jobId)}`,
      { headers: authHeaders(auth.accessToken, auth.apiKey) },
    )
    if (!response.ok) {
      throw new Error(
        `${kind} job status failed: ${response.status} ${await response.text()}`,
      )
    }
    const job = (await response.json()) as JobResponse
    lastStatus = String(job.status || "unknown")
    if (lastStatus === "completed") return job
    if (["failed", "cancelled", "expired"].includes(lastStatus)) {
      throw new Error(
        `${kind} job ${jobId} ended ${lastStatus}: ${String(job.error_message || "")}`,
      )
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`${kind} job ${jobId} did not complete; last status ${lastStatus}`)
}

async function exportArchive(
  page: Page,
  auth: { apiUrl: string; apiKey?: string; accessToken?: string },
  outputPath: string,
  exportName: string,
): Promise<void> {
  await openChatbooks(page, auth)
  await page
    .getByRole("textbox", { name: /^Name$/i })
    .fill(exportName)
  await page
    .getByPlaceholder(/Description/i)
    .fill("Browser-created full-account archive for roundtrip UAT")
  await expect(page.getByText(/Backup all scope/i)).toBeVisible()

  const exportRequestPromise = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      /\/api\/v1\/chatbooks\/export(?:$|\?)/.test(request.url()),
  )
  await page.getByRole("button", { name: /^Backup all$/i }).click()
  const exportRequest = await exportRequestPromise
  const payload = exportRequest.postDataJSON() as Record<string, unknown>
  expect(payload).not.toHaveProperty("content_selections")
  expect(payload).toMatchObject({
    format_version: "1.1.0",
    include_media: true,
    include_embeddings: true,
    include_generated_content: true,
    media_quality: "original",
  })
  const exportResponse = await exportRequest.response()
  expect(exportResponse).not.toBeNull()
  expect(exportResponse!.ok()).toBe(true)
  const exportResult = (await exportResponse!.json()) as Record<string, unknown>
  const jobId = String(exportResult.job_id || "")
  expect(jobId).not.toBe("")
  const completedJob = await waitForJob("export", jobId, auth)
  expect(completedJob.progress_percentage).toBe(100)
  expect(completedJob.metadata?.post_write_verification).toBe(true)

  await page.getByRole("tab", { name: /Jobs/i }).click()
  const exportRow = page
    .locator(".ant-table-row")
    .filter({ hasText: exportName })
    .first()
  await expect(exportRow).toContainText(/Completed/i, { timeout: 30_000 })
  const downloadButton = exportRow.getByRole("button", { name: /Download/i })
  await expect(downloadButton).toBeVisible()

  mkdirSync(path.dirname(outputPath), { recursive: true })
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    downloadButton.click(),
  ])
  await download.saveAs(outputPath)
  expect(existsSync(outputPath)).toBe(true)
}

async function importArchive(
  page: Page,
  auth: { apiUrl: string; apiKey?: string; accessToken?: string },
  inputPath: string,
  requirePopulatedMedia: boolean,
): Promise<void> {
  expect(existsSync(inputPath)).toBe(true)
  await openChatbooks(page, auth)
  await page.getByRole("tab", { name: /Import/i }).click()

  const previewResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      /\/api\/v1\/chatbooks\/preview(?:$|\?)/.test(response.url()),
  )
  await page.locator('input[type="file"]').first().setInputFiles(inputPath)
  const previewResponse = await previewResponsePromise
  expect(previewResponse.ok()).toBe(true)
  const restoreSummary = page.getByRole("region", {
    name: /What will be restored/i,
  })
  await expect(restoreSummary).toBeVisible({ timeout: 30_000 })
  await expect(restoreSummary.getByText(/Account profile/i)).toBeVisible()
  await expect(restoreSummary.getByText(/Account settings/i)).toBeVisible()
  if (requirePopulatedMedia) {
    await expect(restoreSummary.getByText(/Stored media artifacts/i)).toBeVisible()
  }
  await expect(restoreSummary.getByText(/^Verified$/i)).toBeVisible()

  const importRequestPromise = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      /\/api\/v1\/chatbooks\/import(?:$|\?)/.test(request.url()),
  )
  await page.getByRole("button", { name: /^Import chatbook$/i }).click()
  const importRequest = await importRequestPromise
  const importResponse = await importRequest.response()
  expect(importResponse).not.toBeNull()
  expect(importResponse!.ok()).toBe(true)
  const importResult = (await importResponse!.json()) as Record<string, unknown>
  const jobId = String(importResult.job_id || "")
  expect(jobId).not.toBe("")
  const completedJob = await waitForJob("import", jobId, auth)
  expect(completedJob.progress_percentage).toBe(100)
  if (requirePopulatedMedia) {
    expect(Number(completedJob.metadata?.imported_items?.media || 0)).toBeGreaterThan(0)
    expect(
      Number(completedJob.metadata?.imported_items?.embedding || 0),
    ).toBeGreaterThan(0)
  }

  await page.getByRole("tab", { name: /Jobs/i }).click()
  const importRow = page
    .locator(".ant-table-row")
    .filter({ hasText: path.basename(inputPath) })
    .first()
  await expect(importRow).toContainText(/Completed/i, { timeout: 30_000 })
}

if (liveTierEnabled) {
  test.describe.serial("Chatbooks full-account live-tier round trip", () => {
    const liveAuth = { apiUrl: liveApiUrl, apiKey: liveApiKey }

    test("exports the live account through Backup all", async ({ page }) => {
      test.setTimeout(360_000)
      await exportArchive(
        page,
        liveAuth,
        liveArchivePath,
        `Live Tier full account backup ${Date.now()}`,
      )
    })

    test("imports the exact live-tier browser archive", async ({ page }) => {
      test.setTimeout(360_000)
      await importArchive(page, liveAuth, liveArchivePath, false)
    })
  })
} else if (externalHarnessEnabled) {
  test.describe("Chatbooks full-account browser round trip", () => {
    const externalAuth = { apiUrl: apiUrl!, accessToken }

    test("exports the source account through Backup all", async ({ page }) => {
      test.skip(phase !== "export", "Import phase invocation")
      test.setTimeout(360_000)
      await exportArchive(
        page,
        externalAuth,
        archivePath!,
        "Browser UAT full account backup",
      )
    })

    test("imports the exact browser archive into the clean destination", async ({
      page,
    }) => {
      test.skip(phase !== "import", "Export phase invocation")
      test.setTimeout(360_000)
      await importArchive(page, externalAuth, archivePath!, true)
    })
  })
}
