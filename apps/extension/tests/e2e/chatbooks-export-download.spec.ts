import { expect, test, type Page } from "@playwright/test"
import { existsSync, mkdirSync } from "node:fs"
import path from "node:path"

import { forceConnected, waitForConnectionStore } from "./utils/connection"
import { launchWithExtension } from "./utils/extension"
import { grantHostPermission } from "./utils/permissions"

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
const serverUrl = process.env.TLDW_CHATBOOK_UAT_API_URL?.replace(/\/$/, "")
const enabled =
  (phase === "export" || phase === "import") &&
  Boolean(archivePath && accessToken && serverUrl)

const authHeaders = () => ({ Authorization: `Bearer ${accessToken}` })

async function waitForJob(
  kind: "export" | "import",
  jobId: string,
): Promise<JobResponse> {
  const deadline = Date.now() + 180_000
  let lastStatus = "unknown"
  while (Date.now() < deadline) {
    const response = await fetch(
      `${serverUrl}/api/v1/chatbooks/${kind}/jobs/${encodeURIComponent(jobId)}`,
      { headers: authHeaders() },
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

async function openChatbooks(page: Page, optionsUrl: string): Promise<void> {
  await page.goto(`${optionsUrl}#/chatbooks`, { waitUntil: "domcontentloaded" })
  await waitForConnectionStore(page, "chatbooks-open")
  await forceConnected(page, { serverUrl: serverUrl! }, "chatbooks-connected")
  await expect(
    page.getByRole("heading", { name: /Chatbooks Backup & Import/i }),
  ).toBeVisible({ timeout: 30_000 })
}

test.describe("Chatbooks packaged-extension full-account round trip", () => {
  test.skip(!enabled, "Run through chatbooks_full_account_browser_uat.py")

  test("exports or imports the configured full-account UAT phase", async () => {
    test.setTimeout(240_000)
    const health = await fetch(`${serverUrl}/api/v1/chatbooks/health`, {
      headers: authHeaders(),
    })
    expect(health.ok).toBe(true)

    const extPath = path.resolve("build/chrome-mv3")
    const { context, page, extensionId, optionsUrl } = await launchWithExtension(
      extPath,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: false,
          authMode: "multi-user",
          accessToken,
          tldwConfig: {
            serverUrl,
            authMode: "multi-user",
            accessToken,
          },
        },
      },
    )

    try {
      const origin = new URL(serverUrl!).origin + "/*"
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        throw new Error(`Extension host permission was not granted for ${origin}`)
      }
      await openChatbooks(page, optionsUrl)

      if (phase === "export") {
        const exportName = "Extension UAT full account backup"
        await page.getByPlaceholder(/^Name$/i).fill(exportName)
        await page
          .getByPlaceholder(/Description/i)
          .fill("Packaged-extension full-account archive for clean-destination UAT")
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
        const completedJob = await waitForJob("export", jobId)
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
        mkdirSync(path.dirname(archivePath!), { recursive: true })
        const [download] = await Promise.all([
          page.waitForEvent("download", { timeout: 30_000 }),
          downloadButton.click(),
        ])
        await download.saveAs(archivePath!)
        expect(existsSync(archivePath!)).toBe(true)
        return
      }

      expect(existsSync(archivePath!)).toBe(true)
      await page.getByRole("tab", { name: /Import/i }).click()
      const previewResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          /\/api\/v1\/chatbooks\/preview(?:$|\?)/.test(response.url()),
      )
      await page.locator('input[type="file"]').first().setInputFiles(archivePath!)
      const previewResponse = await previewResponsePromise
      expect(previewResponse.ok()).toBe(true)
      await expect(
        page.getByRole("heading", { name: /What will be restored/i }),
      ).toBeVisible({ timeout: 30_000 })
      await expect(page.getByText(/Account profile/i).first()).toBeVisible()
      await expect(page.getByText(/Account settings/i).first()).toBeVisible()
      await expect(page.getByText(/Media stored artifacts/i).first()).toBeVisible()
      await expect(page.getByText(/^Verified$/i).first()).toBeVisible()

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
      const completedJob = await waitForJob("import", jobId)
      expect(completedJob.progress_percentage).toBe(100)
      expect(Number(completedJob.metadata?.imported_items?.media || 0)).toBeGreaterThan(0)
      expect(
        Number(completedJob.metadata?.imported_items?.embedding || 0),
      ).toBeGreaterThan(0)

      await page.getByRole("tab", { name: /Jobs/i }).click()
      const importRow = page
        .locator(".ant-table-row")
        .filter({ hasText: path.basename(archivePath!) })
        .first()
      await expect(importRow).toContainText(/Completed/i, { timeout: 30_000 })
    } finally {
      await context.close()
    }
  })
})
