import { tldwClient } from "@/services/tldw/TldwApiClient"
import { validateStandaloneHtmlSource } from "./standalone-html-source"

type DraftAttachmentClient = (
  presentationId: string,
  source: string,
  options?: { abortSignal?: AbortSignal }
) => Promise<Uint8Array>

const exactBytes = (left: Uint8Array, right: Uint8Array): boolean => {
  if (left.byteLength !== right.byteLength) return false
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false
  }
  return true
}

export class StandaloneHtmlDownloadManager {
  private readonly downloadDraft: DraftAttachmentClient
  private activeUrl: string | null = null
  private revokeTimer: ReturnType<typeof setTimeout> | null = null
  private requestController: AbortController | null = null
  private disposed = false

  constructor(options: { downloadDraft?: DraftAttachmentClient } = {}) {
    this.downloadDraft =
      options.downloadDraft ??
      ((presentationId, source, requestOptions) =>
        tldwClient.downloadStandaloneHtmlDraft(presentationId, source, requestOptions))
    if (typeof window !== "undefined") window.addEventListener("pagehide", this.handlePagehide)
  }

  private abortRequest = () => {
    const controller = this.requestController
    this.requestController = null
    try {
      controller?.abort()
    } catch {
      // Cleanup remains source-free and continues through platform failures.
    }
  }

  private handlePagehide = () => {
    this.abortRequest()
    this.revokeActiveUrl()
  }

  private revokeActiveUrl = () => {
    if (this.revokeTimer !== null) {
      const timer = this.revokeTimer
      this.revokeTimer = null
      try {
        clearTimeout(timer)
      } catch {
        // The URL is still detached and revoked below.
      }
    }
    if (this.activeUrl !== null) {
      const url = this.activeUrl
      this.activeUrl = null
      try {
        URL.revokeObjectURL(url)
      } catch {
        // Ref ownership is cleared even when browser revocation is unavailable.
      }
    }
  }

  private scheduleRevoke = (url: string) => {
    try {
      this.revokeTimer = setTimeout(() => {
        this.revokeTimer = null
        if (this.activeUrl === url) {
          this.activeUrl = null
          try {
            URL.revokeObjectURL(url)
          } catch {
            // The scheduled cleanup cannot reintroduce the detached URL reference.
          }
        }
      }, 0)
    } catch {
      this.revokeActiveUrl()
    }
  }

  async download(input: { presentationId: string; source: string }): Promise<void> {
    if (this.disposed) throw new Error("Download manager is disposed")
    const accepted = await validateStandaloneHtmlSource(input.source)
    if (!accepted.ok) throw accepted
    if (this.disposed) throw new DOMException("Aborted", "AbortError")

    this.abortRequest()
    const controller = new AbortController()
    this.requestController = controller
    let bytes: Uint8Array
    try {
      bytes = await this.downloadDraft(input.presentationId, accepted.source, {
        abortSignal: controller.signal
      })
    } finally {
      if (this.requestController === controller) this.requestController = null
    }
    if (this.disposed || controller.signal.aborted) throw new DOMException("Aborted", "AbortError")
    if (!exactBytes(bytes, accepted.bytes)) {
      throw new Error("Downloaded draft could not be verified")
    }

    this.revokeActiveUrl()
    const blob = new Blob([bytes.slice().buffer], { type: "application/octet-stream" })
    const objectUrl = URL.createObjectURL(blob)
    this.activeUrl = objectUrl
    let anchor: HTMLAnchorElement | null = null
    try {
      anchor = document.createElement("a")
      anchor.dataset.standaloneHtmlDownload = ""
      anchor.download = "presentation.html"
      anchor.href = objectUrl
      document.body.appendChild(anchor)
      anchor.click()
    } catch (error) {
      this.revokeActiveUrl()
      throw error
    } finally {
      try {
        anchor?.remove()
      } catch {
        try {
          if (anchor?.parentNode) anchor.parentNode.removeChild(anchor)
        } catch {
          // URL cleanup remains independent when DOM removal is unavailable.
        }
      }
    }
    this.scheduleRevoke(objectUrl)
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.abortRequest()
    this.revokeActiveUrl()
    if (typeof window !== "undefined") {
      try {
        window.removeEventListener("pagehide", this.handlePagehide)
      } catch {
        // All sensitive refs have already been cleared.
      }
    }
  }
}
