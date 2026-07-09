import { test, expect } from "@playwright/test"
import path from "path"
import { grantHostPermission } from "./utils/permissions"
import { forceConnected, waitForConnectionStore } from "./utils/connection"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"

test.describe("Chatbooks export download", () => {
  test("exports a chatbook and downloads the zip", async () => {
    test.setTimeout(180_000)
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = serverUrl.replace(/\/$/, "")
    const headers = {
      "Content-Type": "application/json",
      "x-api-key": apiKey
    }

    const healthRes = await fetch(
      `${normalizedServerUrl}/api/v1/chatbooks/health`,
      { headers }
    ).catch(() => null)
    if (!healthRes || !healthRes.ok) {
      test.skip(true, "Chatbooks API not available on the configured server.")
      return
    }
    const healthPayload = await healthRes.json().catch(() => null)
    if (healthPayload?.available === false) {
      test.skip(true, "Chatbooks API disabled on the configured server.")
      return
    }

    const promptName = `E2E Chatbook Prompt ${Date.now()}`
    let promptId: string | number | null = null

    try {
      const createRes = await fetch(`${normalizedServerUrl}/api/v1/prompts`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          name: promptName,
          system_prompt: "You are an export prompt for chatbooks.",
          user_prompt: "Generate a short answer.",
          keywords: ["e2e", "chatbook"]
        })
      })
      if (!createRes.ok) {
        const body = await createRes.text().catch(() => "")
        throw new Error(
          `Prompt create failed: ${createRes.status} ${createRes.statusText} ${body}`
        )
      }
      const created = await createRes.json().catch(() => null)
      promptId = created?.id ?? created?.uuid ?? created?.name ?? null

      const extPath = path.resolve("build/chrome-mv3")
      const { context, page, extensionId, optionsUrl } = await launchWithExtensionOrSkip(
        test,
        extPath,
        {
          seedConfig: {
            __tldw_first_run_complete: true,
            __tldw_allow_offline: true,
            tldwConfig: {
              serverUrl: normalizedServerUrl,
              authMode: "single-user",
              apiKey
            }
          }
        }
      )

      try {
        const origin = new URL(normalizedServerUrl).origin + "/*"
        const granted = await grantHostPermission(context, extensionId, origin)
        if (!granted) {
          test.skip(
            true,
            "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
          )
          return
        }

        await page.goto(`${optionsUrl}#/chatbooks`, {
          waitUntil: "domcontentloaded"
        })
        await waitForConnectionStore(page, "chatbooks-open")
        await forceConnected(page, { serverUrl: normalizedServerUrl }, "chatbooks-connected")

        await expect(
          page.getByRole("heading", { name: /Chatbooks Backup & Import/i })
        ).toBeVisible({ timeout: 15000 })

        const unavailableAlert = page.getByText(
          /Chatbooks is not available on this server/i
        )
        if (await unavailableAlert.isVisible().catch(() => false)) {
          test.skip(true, "Chatbooks API not available on this server.")
          return
        }

        const exportName = `E2E Chatbook ${Date.now()}`
        await page.getByPlaceholder(/^Name$/i).fill(exportName)
        await page
          .getByPlaceholder(/Description/i)
          .fill("E2E chatbook export")

        await expect(page.getByText(/Backup all scope/i)).toBeVisible({
          timeout: 15000
        })
        const exportRequestPromise = page.waitForRequest(
          (request) =>
            request.method() === "POST" &&
            /\/api\/v1\/chatbooks\/export(?:$|\?)/.test(request.url()),
          { timeout: 15000 }
        )

        await page.getByRole("button", { name: /^Backup all$/i }).click()

        const exportRequest = await exportRequestPromise
        const exportPayload = exportRequest.postDataJSON() as Record<string, unknown>
        expect(exportPayload).not.toHaveProperty("content_selections")
        const exportResponse = await exportRequest.response()
        expect(exportResponse).not.toBeNull()
        expect(exportResponse!.status()).toBeLessThan(500)

        const errorNotice = page
          .getByText(
            /Select at least one item to export|Name and description are required|Export failed/i
          )
          .first()
        const errorVisible = await errorNotice
          .waitFor({ state: "visible", timeout: 5000 })
          .then(() => true)
          .catch(() => false)
        if (errorVisible) {
          const errorText = await errorNotice.textContent()
          throw new Error(
            `Chatbook export failed: ${errorText?.trim() || "unknown error"}`
          )
        }

        await page
          .getByText(/Export job created|Export complete/i)
          .first()
          .waitFor({ state: "visible", timeout: 30000 })
          .catch(() => {})

        const jobsTab = page.getByRole("tab", { name: /Jobs/i })
        await jobsTab.click()
        const jobsPanelId = await jobsTab.getAttribute("aria-controls")
        const jobsPanel = jobsPanelId ? page.locator(`#${jobsPanelId}`) : page

        const exportCard = jobsPanel
          .locator(".ant-card")
          .filter({ hasText: /Export jobs/i })
          .first()
        await expect(exportCard).toBeVisible({ timeout: 15000 })

        const exportRow = exportCard
          .locator(".ant-table-row")
          .filter({ hasText: exportName })
          .first()
        await expect(exportRow).toBeVisible({ timeout: 90000 })

        const downloadButton = exportRow.getByRole("button", {
          name: /Download/i
        })
        await expect(downloadButton).toBeVisible({ timeout: 120000 })

        const [download] = await Promise.all([
          page.waitForEvent("download", { timeout: 30000 }),
          downloadButton.click()
        ])

        expect(download.suggestedFilename()).toMatch(/\.zip$/)
      } finally {
        await context.close()
      }
    } finally {
      if (promptId != null) {
        await fetch(
          `${normalizedServerUrl}/api/v1/prompts/${encodeURIComponent(
            String(promptId)
          )}`,
          {
            method: "DELETE",
            headers: { "x-api-key": apiKey }
          }
        ).catch(() => {})
      }
    }
  })
})
