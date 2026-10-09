import { browser } from "wxt/browser"
import { defineContentScript } from "wxt/utils/define-content-script"

/**
 * Minimal every-page stub for the copilot popup.
 *
 * WXT builds content scripts as single-file IIFE bundles, so any bundler-
 * analyzable import of the shared popup module would be inlined and the
 * heavy popup code (chat client + DOM) would ship on every page. Instead
 * this stub:
 *
 *   1. marks readiness immediately (E2E contract: the dataset flag must be
 *      present before any popup message is sent), and
 *   2. on the first `tldw:popup:open` message, dynamically imports the
 *      web-accessible `copilot-popup-main.js` chunk via a runtime URL the
 *      bundler cannot inline, then delegates to the registered handler.
 *
 * Top-frame only: the background addresses the popup via
 * `tabs.sendMessage(tabId, ..., { frameId })` from the `contextual-popup-pa`
 * context-menu handler, and frame 0 (top frame) is where the popup is
 * expected to render.
 */

const COPILOT_POPUP_CHUNK_RESOURCE = "copilot-popup-main.js"
const COPILOT_POPUP_HANDLE_KEY = "__tldwCopilotPopupHandle"

type CopilotPopupPayload = Record<string, unknown>

let chunkLoad: Promise<void> | null = null

const ensurePopupChunkLoaded = (): Promise<void> => {
  if (!chunkLoad) {
    chunkLoad = (async () => {
      const runtime = browser?.runtime as
        | { getURL?: (path: string) => string }
        | undefined
      const getURL = runtime?.getURL?.bind(runtime)
      if (!getURL) {
        throw new Error("Copilot popup chunk requires an extension runtime.")
      }
      const chunkUrl = getURL(COPILOT_POPUP_CHUNK_RESOURCE)
      // Runtime URL: intentionally NOT bundler-analyzable so the heavy
      // module becomes a real lazily-imported chunk.
      await import(/* @vite-ignore */ chunkUrl)
    })().catch((error) => {
      chunkLoad = null
      throw error
    })
  }
  return chunkLoad
}

const handlePopupMessage = async (payload: CopilotPopupPayload) => {
  await ensurePopupChunkLoaded()
  const handle = (globalThis as Record<string, unknown>)[
    COPILOT_POPUP_HANDLE_KEY
  ]
  if (typeof handle !== "function") {
    throw new Error("Copilot popup chunk did not register its handler.")
  }
  await (handle as (payload?: CopilotPopupPayload) => Promise<void>)(payload)
}

export default defineContentScript({
  matches: ["http://*/*", "https://*/*"],
  main() {
    try {
      ;(window as unknown as Record<string, unknown>).__tldwCopilotPopupReady =
        true
      document.documentElement.dataset.tldwCopilotPopupReady = "true"
    } catch {
      // ignore readiness flag failures
    }

    browser.runtime.onMessage.addListener((message: any) => {
      if (message?.type !== "tldw:popup:open") return
      void handlePopupMessage(message.payload || {}).catch(() => {
        // Loading or popup failure is surfaced by the popup itself; the
        // message acknowledgement below still satisfies the sender.
      })
      return Promise.resolve({ ok: true })
    })
  }
})
