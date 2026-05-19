import { browser } from "wxt/browser"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import { extractClipPageTextFromDocument } from "@/services/web-clipper/content-extract"

const CAPTURE_REQUEST_MESSAGE_TYPE = "capture-web-clipper"

type WebClipperCaptureRequest = {
  type?: string
  requestedType?: "bookmark" | "article" | "full_page" | "selection" | "screenshot"
  selectionText?: string
}

export default defineContentScript({
  main() {
    const listener = (message: WebClipperCaptureRequest) => {
      if (message?.type !== CAPTURE_REQUEST_MESSAGE_TYPE) return undefined
      const requestedType = message.requestedType || "article"
      const pageExtraction = extractClipPageTextFromDocument(document)
      const explicitSelectionText = String(message.selectionText || "").trim()
      return Promise.resolve(buildClipDraft({
        requestedType,
        pageUrl: window.location.href,
        pageTitle: document.title || window.location.hostname,
        extracted: {
          ...pageExtraction,
          selectionText:
            explicitSelectionText || pageExtraction.selectionText || undefined
        }
      }))
    }

    try {
      browser.runtime.onMessage.addListener(listener as never)
    } catch {
      // The content script is best-effort during local testing.
    }
  },
  allFrames: false,
  matches: ["http://*/*", "https://*/*"]
})
