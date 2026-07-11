import http from "node:http"
import type { AddressInfo } from "node:net"

import { expect, test } from "@playwright/test"
import { launchWithBuiltExtension } from "./utils/extension-build"
import { waitForConnectionStore } from "./utils/connection"
import { setQuickIngestSwitch } from "./utils/quick-ingest-options"

const API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

test.describe("Quick ingest cancel flow", () => {
  let server: http.Server
  let serverUrl: string
  const unexpectedRequests: string[] = []

  test.beforeAll(async () => {
    server = http.createServer((request, response) => {
      const method = (request.method || "GET").toUpperCase()
      const url = new URL(request.url || "/", "http://127.0.0.1")

      if (method === "OPTIONS") {
        response.writeHead(204, {
          "access-control-allow-origin": "*",
          "access-control-allow-headers": "content-type, x-api-key, authorization"
        })
        response.end()
        return
      }

      let body: unknown = {}
      if (url.pathname === "/openapi.json") {
        body = {
          openapi: "3.1.0",
          info: { title: "tldw mock", version: "e2e" },
          paths: {
            "/api/v1/health": {},
            "/api/v1/media": {},
            "/api/v1/media/process-web-scraping": {}
          }
        }
      } else if (url.pathname === "/api/v1/health") {
        body = { status: "ok" }
      } else if (url.pathname === "/api/v1/health/live") {
        body = { status: "alive" }
      } else if (url.pathname === "/api/v1/rag/health") {
        body = { status: "healthy" }
      } else if (url.pathname === "/api/v1/setup/first-run/state") {
        body = { status: "completed" }
      } else if (url.pathname === "/api/v1/users/storage") {
        body = {
          user_id: 1,
          storage_used_mb: 0,
          storage_quota_mb: 5120,
          available_mb: 5120,
          usage_percentage: 0
        }
      } else if (url.pathname === "/api/v1/media") {
        body = {
          items: [],
          pagination: {
            page: 1,
            results_per_page: 20,
            total_items: 0,
            total_pages: 0
          }
        }
      } else if (url.pathname === "/api/v1/ingestion-sources/capabilities") {
        body = { can_create_local_directory: false }
      } else if (
        url.pathname !== "/api/v1/config/docs-info" &&
        url.pathname !== "/api/v1/users/me/profile"
      ) {
        unexpectedRequests.push(`${method} ${url.pathname}`)
      }

      response.writeHead(200, {
        "content-type": "application/json",
        "access-control-allow-origin": "*"
      })
      response.end(JSON.stringify(body))
    })
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
    const address = server.address() as AddressInfo
    serverUrl = `http://127.0.0.1:${address.port}`
  })

  test.afterAll(async () => {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()))
    })
  })

  test("quick ingest cancel all reaches terminal wizard results", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl,
        authMode: "single-user",
        apiKey: API_KEY
      }
    })
    const browserErrors: string[] = []
    page.on("pageerror", (error) => {
      browserErrors.push(error.stack || error.message)
    })
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text())
    })
    let releaseDirectResponse: (() => void) | null = null
    await page.route("**/api/v1/media/process-web-scraping", async (route) => {
      await new Promise<void>((resolve) => {
        releaseDirectResponse = resolve
      })
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "persist-ok", media_ids: [4242] })
      })
    })

    try {
      await page.goto(optionsUrl + "#/media", { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "quick-ingest-cancel")
      await page.waitForFunction(
        () => {
          const state = (window as any).__tldw_useConnectionStore?.getState?.().state
          return state?.isConnected === true && state?.phase === "connected"
        },
        null,
        { timeout: 20_000 }
      )

      const openQuickIngestButton = page
        .getByRole("button", { name: /quick ingest/i })
        .first()
      await expect(openQuickIngestButton).toBeVisible()
      await openQuickIngestButton.click()

      const dialog = page.getByRole("dialog", { name: /quick ingest/i }).first()
      await expect(dialog).toBeVisible()

      const urlInput = dialog
        .getByLabel(/Paste URLs input/i)
        .or(dialog.getByLabel(/URL input area/i))
        .or(dialog.getByPlaceholder(/https:\/\/example\.com/i))
        .first()
      await expect(urlInput).toBeEnabled({ timeout: 20000 })
      await urlInput.fill("https://example.com/cancel-me")
      await dialog.getByRole("button", { name: /add urls/i }).first().click()

      const configureButton = dialog
        .getByRole("button", { name: /configure \d+ items/i })
        .first()
      await expect(configureButton).toBeVisible()
      await configureButton.click()

      await setQuickIngestSwitch(dialog, "analysis", false)
      await setQuickIngestSwitch(dialog, "chunking", false)

      const nextButton = dialog.getByRole("button", { name: /^next$/i }).first()
      await expect(nextButton).toBeVisible()
      await nextButton.click()

      const startButton = dialog.getByRole("button", { name: /start processing/i }).first()
      await expect(startButton).toBeVisible()
      const directRequestStarted = page.waitForRequest(
        "**/api/v1/media/process-web-scraping",
        { timeout: 10_000 }
      )
      await startButton.click()

      const cancelButton = dialog.getByRole("button", { name: /cancel all/i }).first()
      await expect(cancelButton).toBeVisible({ timeout: 10000 })
      await directRequestStarted

      // A global tutorial notification can overlap the modal briefly. The
      // regression is about the cancellation/completion race once this visible
      // control is invoked, so bypass unrelated notification hit-testing.
      await cancelButton.click({ force: true })

      const resultsStep = dialog.getByTestId("wizard-results-step")
      const errorBoundary = page.getByTestId("error-boundary")
      const terminalView = await Promise.race([
        resultsStep.waitFor({ state: "visible", timeout: 10000 }).then(() => "results"),
        errorBoundary.waitFor({ state: "visible", timeout: 10000 }).then(() => "error")
      ])
      if (terminalView === "error") {
        await errorBoundary.getByText("View error details", { exact: true }).click()
        const details = (await errorBoundary.locator("pre").textContent()) || ""
        throw new Error(
          `Options error boundary rendered after Cancel All.\n${details}\n${browserErrors.join("\n")}`
        )
      }

      await expect(resultsStep).toBeVisible()
      await expect(dialog.getByRole("region", { name: /cancelled items/i })).toBeVisible()

      const lateResponse = page.waitForResponse((response) =>
        response.url().includes("/api/v1/media/process-web-scraping")
      )
      if (!releaseDirectResponse) {
        throw new Error("Deferred direct response was not registered.")
      }
      releaseDirectResponse()
      await lateResponse
      await page.evaluate(
        () =>
          new Promise<void>((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
          )
      )
      await expect(dialog.getByTestId("wizard-results-step")).toBeVisible()
      await expect(dialog.getByRole("region", { name: /cancelled items/i })).toBeVisible()
      await expect(
        dialog.getByRole("region", { name: /completed items/i })
      ).toHaveCount(0)
      expect(browserErrors).toEqual([])
      expect(unexpectedRequests).toEqual([])
    } finally {
      releaseDirectResponse?.()
      await context.close()
    }
  })
})
