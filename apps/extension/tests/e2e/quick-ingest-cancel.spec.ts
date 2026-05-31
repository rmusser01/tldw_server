import { expect, test } from "@playwright/test"
import { launchWithBuiltExtension } from "./utils/extension-build"
import { forceConnected, waitForConnectionStore } from "./utils/connection"

const API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

test.describe("Quick ingest cancel flow", () => {
  test("quick ingest cancel all reaches terminal wizard results", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: API_KEY
      },
      allowOffline: true
    })

    try {
      await page.goto(optionsUrl + "#/media", { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "quick-ingest-cancel")
      await forceConnected(page, {}, "quick-ingest-cancel")

      const patched = await page.evaluate(() => {
        try {
          const runtime =
            (globalThis as any)?.browser?.runtime ||
            (globalThis as any)?.chrome?.runtime
          const onMessage = runtime?.onMessage
          const originalSendMessage =
            typeof runtime?.sendMessage === "function"
              ? runtime.sendMessage.bind(runtime)
              : null
          if (!runtime || !onMessage || !originalSendMessage) {
            return false
          }

          const originalAddListener =
            typeof onMessage.addListener === "function"
              ? onMessage.addListener.bind(onMessage)
              : null
          const originalRemoveListener =
            typeof onMessage.removeListener === "function"
              ? onMessage.removeListener.bind(onMessage)
              : null
          const listeners = new Set<(message: any, sender?: any, sendResponse?: any) => void>()

          const emit = (message: any) => {
            for (const listener of [...listeners]) {
              try {
                listener(message, {}, () => undefined)
              } catch {
                // best-effort test emitter
              }
            }
          }

          onMessage.addListener = (listener: any) => {
            listeners.add(listener)
          }
          onMessage.removeListener = (listener: any) => {
            listeners.delete(listener)
          }

          runtime.sendMessage = async (message: any) => {
            if (message?.type === "tldw:quick-ingest/start") {
              return { ok: true, sessionId: "qi-e2e-cancel-session" }
            }
            if (message?.type === "tldw:quick-ingest/cancel") {
              emit({
                type: "tldw:quick-ingest/cancelled",
                payload: {
                  sessionId: "qi-e2e-cancel-session",
                  reason: "Cancelled by user."
                }
              })
              setTimeout(() => {
                emit({
                  type: "tldw:quick-ingest/completed",
                  payload: {
                    sessionId: "qi-e2e-cancel-session",
                    results: [
                      {
                        id: "qi-e2e-item-1",
                        status: "ok",
                        type: "html",
                        url: "https://example.com/cancel-me"
                      }
                    ]
                  }
                })
              }, 500)
              return { ok: true }
            }
            return originalSendMessage(message)
          }

          ;(window as any).__restoreQuickIngestCancelPatch = () => {
            runtime.sendMessage = originalSendMessage
            if (originalAddListener) {
              onMessage.addListener = originalAddListener
            }
            if (originalRemoveListener) {
              onMessage.removeListener = originalRemoveListener
            }
            listeners.clear()
          }
          return true
        } catch {
          return false
        }
      })

      if (!patched) {
        test.skip(true, "Unable to patch runtime messaging in extension page context.")
        return
      }

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

      const useDefaultsButton = dialog
        .getByRole("button", { name: /use defaults & process/i })
        .first()
      await expect(useDefaultsButton).toBeVisible()
      await useDefaultsButton.click()

      const cancelButton = dialog.getByRole("button", { name: /cancel all/i }).first()
      await expect(cancelButton).toBeVisible({ timeout: 10000 })

      await cancelButton.click()

      await expect(dialog.getByTestId("wizard-results-step")).toBeVisible({
        timeout: 10000
      })
      await expect(dialog.getByRole("region", { name: /error items/i })).toBeVisible()

      await page.waitForTimeout(1200)
      await expect(dialog.getByTestId("wizard-results-step")).toBeVisible()
    } finally {
      try {
        await page.evaluate(() => {
          try {
            const restore = (window as any).__restoreQuickIngestCancelPatch
            if (typeof restore === "function") {
              restore()
            }
            delete (window as any).__restoreQuickIngestCancelPatch
          } catch {
            // ignore best-effort cleanup failures
          }
        })
      } catch {
        // ignore cleanup failures if page already closed
      }
      await context.close()
    }
  })
})
