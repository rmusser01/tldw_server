import { browser } from "wxt/browser"

// BFCache retains the document, so navigation must retire its unfinished reads.
let documentLifetime = typeof window !== "undefined" && /^https?:$/.test(window.location?.protocol ?? "")
  ? new AbortController()
  : undefined
if (documentLifetime) {
  window.addEventListener("pagehide", () => {
    documentLifetime?.abort(new DOMException("The page request was aborted", "AbortError"))
  })
  window.addEventListener("pageshow", () => {
    if (documentLifetime?.signal.aborted) documentLifetime = new AbortController()
  })
}

export const captureDocumentGetSignal = (method?: string): AbortSignal | undefined =>
  !browser?.runtime?.id && String(method || "GET").toUpperCase() === "GET" ? documentLifetime?.signal : undefined
