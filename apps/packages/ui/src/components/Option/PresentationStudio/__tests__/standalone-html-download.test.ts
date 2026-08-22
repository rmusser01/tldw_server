import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const loadDownload = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-download"].join("/"))

const SOURCE = "<!doctype html><title>Download me 😀</title>"
const BYTES = new TextEncoder().encode(SOURCE)
const OBJECT_URL = "blob:application-owned-standalone-html-download"

describe("standalone HTML attachment handoff", () => {
  const createObjectUrlDescriptor = Object.getOwnPropertyDescriptor(URL, "createObjectURL")
  const revokeObjectUrlDescriptor = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL")
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let clickedAnchors: HTMLAnchorElement[]
  let click: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.useFakeTimers()
    clickedAnchors = []
    createObjectURL = vi.fn(() => OBJECT_URL)
    revokeObjectURL = vi.fn()
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: createObjectURL
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: revokeObjectURL
    })
    click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      clickedAnchors.push(this)
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    if (createObjectUrlDescriptor) {
      Object.defineProperty(URL, "createObjectURL", createObjectUrlDescriptor)
    } else {
      delete (URL as any).createObjectURL
    }
    if (revokeObjectUrlDescriptor) {
      Object.defineProperty(URL, "revokeObjectURL", revokeObjectUrlDescriptor)
    } else {
      delete (URL as any).revokeObjectURL
    }
    document.querySelectorAll("a[data-standalone-html-download]").forEach((node) => node.remove())
  })

  it("creates one octet-stream URL only after authenticated validation and exposes it only to a fixed temporary anchor", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    let resolveAttachment: ((value: Uint8Array) => void) | null = null
    const downloadDraft = vi.fn(
      () =>
        new Promise<Uint8Array>((resolve) => {
          resolveAttachment = resolve
        })
    )
    const manager = new StandaloneHtmlDownloadManager({ downloadDraft })
    const open = vi.spyOn(window, "open")
    const pushState = vi.spyOn(history, "pushState")
    const replaceState = vi.spyOn(history, "replaceState")
    const localWrite = vi.spyOn(localStorage, "setItem")
    const sessionWrite = vi.spyOn(sessionStorage, "setItem")
    const log = vi.spyOn(console, "log")
    const WorkerConstructor = vi.fn()
    Object.defineProperty(globalThis, "Worker", {
      configurable: true,
      writable: true,
      value: WorkerConstructor
    })

    const resultPromise = manager.download({ presentationId: "html-1", source: SOURCE })
    await vi.waitFor(() =>
      expect(downloadDraft).toHaveBeenCalledWith(
        "html-1",
        SOURCE,
        expect.objectContaining({ abortSignal: expect.any(AbortSignal) })
      )
    )
    expect(createObjectURL).not.toHaveBeenCalled()

    resolveAttachment?.(BYTES)
    await expect(resultPromise).resolves.toBeUndefined()

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    const blob = createObjectURL.mock.calls[0][0] as Blob
    expect(blob.type).toBe("application/octet-stream")
    expect(blob.size).toBe(BYTES.byteLength)
    expect(click).toHaveBeenCalledTimes(1)
    expect(clickedAnchors).toHaveLength(1)
    expect(clickedAnchors[0]).toEqual(
      expect.objectContaining({
        download: "presentation.html",
        target: ""
      })
    )
    expect(clickedAnchors[0].href).toBe(OBJECT_URL)
    expect(clickedAnchors[0].isConnected).toBe(false)
    expect(document.querySelector("a[data-standalone-html-download]")).toBeNull()
    expect(revokeObjectURL).not.toHaveBeenCalled()

    expect(open).not.toHaveBeenCalled()
    expect(pushState).not.toHaveBeenCalled()
    expect(replaceState).not.toHaveBeenCalled()
    expect(localWrite).not.toHaveBeenCalled()
    expect(sessionWrite).not.toHaveBeenCalled()
    expect(log).not.toHaveBeenCalledWith(expect.stringContaining(OBJECT_URL))
    expect(WorkerConstructor).not.toHaveBeenCalled()
    for (const selector of ["iframe", "img", "script", "object", "embed"]) {
      expect(document.querySelector(`${selector}[src=\"${OBJECT_URL}\"]`)).toBeNull()
      expect(document.querySelector(`${selector}[data=\"${OBJECT_URL}\"]`)).toBeNull()
    }

    await vi.advanceTimersByTimeAsync(1_000)
    expect(revokeObjectURL).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).toHaveBeenCalledWith(OBJECT_URL)
    manager.dispose()
  })

  it("rejects invalid scalar source before encoding, client dispatch, Blob, or DOM work", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const downloadDraft = vi.fn()
    const manager = new StandaloneHtmlDownloadManager({ downloadDraft })
    const RealEncoder = globalThis.TextEncoder
    const encoder = vi.fn(() => {
      throw new Error("invalid source must not be encoded")
    })
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: encoder
    })

    try {
      await expect(
        manager.download({ presentationId: "html-1", source: "private\ud800source" })
      ).rejects.toMatchObject({ code: "invalid_unicode_scalar" })
      expect(encoder).not.toHaveBeenCalled()
      expect(downloadDraft).not.toHaveBeenCalled()
      expect(createObjectURL).not.toHaveBeenCalled()
      expect(click).not.toHaveBeenCalled()
    } finally {
      Object.defineProperty(globalThis, "TextEncoder", {
        configurable: true,
        writable: true,
        value: RealEncoder
      })
      manager.dispose()
    }
  })

  it("does not dispatch source if disposal occurs while exact validation is pending", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const digest = await crypto.subtle.digest("SHA-256", BYTES)
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => {
        resolveDigest = resolve
      })
    )
    const downloadDraft = vi.fn().mockResolvedValue(BYTES)
    const manager = new StandaloneHtmlDownloadManager({ downloadDraft })

    const pending = manager.download({ presentationId: "html-1", source: SOURCE })
    await vi.waitFor(() => expect(crypto.subtle.digest).toHaveBeenCalledTimes(1))
    manager.dispose()
    resolveDigest?.(digest)

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(downloadDraft).not.toHaveBeenCalled()
    expect(createObjectURL).not.toHaveBeenCalled()
  })

  it("creates no URL when the authenticated attachment client rejects response headers", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const failure = new Error("Invalid standalone HTML attachment response")
    const manager = new StandaloneHtmlDownloadManager({
      downloadDraft: vi.fn().mockRejectedValue(failure)
    })

    await expect(
      manager.download({ presentationId: "html-1", source: SOURCE })
    ).rejects.toBe(failure)
    expect(createObjectURL).not.toHaveBeenCalled()
    expect(click).not.toHaveBeenCalled()
    manager.dispose()
  })

  it("creates no URL when authenticated bytes do not exactly match the accepted draft", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const manager = new StandaloneHtmlDownloadManager({
      downloadDraft: vi.fn().mockResolvedValue(new TextEncoder().encode("different source"))
    })

    await expect(
      manager.download({ presentationId: "html-1", source: SOURCE })
    ).rejects.toThrow("Downloaded draft could not be verified")
    expect(createObjectURL).not.toHaveBeenCalled()
    expect(click).not.toHaveBeenCalled()
    manager.dispose()
  })

  it("removes the anchor in finally and revokes the URL even when the synthetic click fails", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const manager = new StandaloneHtmlDownloadManager({
      downloadDraft: vi.fn().mockResolvedValue(BYTES)
    })
    click.mockImplementationOnce(function (this: HTMLAnchorElement) {
      clickedAnchors.push(this)
      throw new Error("synthetic click blocked")
    })

    await expect(
      manager.download({ presentationId: "html-1", source: SOURCE })
    ).rejects.toThrow("synthetic click blocked")
    expect(clickedAnchors[0].isConnected).toBe(false)
    expect(document.querySelector("a[data-standalone-html-download]")).toBeNull()
    await vi.runOnlyPendingTimersAsync()
    expect(revokeObjectURL).toHaveBeenCalledWith(OBJECT_URL)
    manager.dispose()
  })

  it("revokes synchronously if temporary anchor setup fails after URL creation", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    const manager = new StandaloneHtmlDownloadManager({
      downloadDraft: vi.fn().mockResolvedValue(BYTES)
    })
    const append = vi.spyOn(document.body, "appendChild").mockImplementationOnce(() => {
      throw new Error("anchor setup blocked")
    })

    await expect(
      manager.download({ presentationId: "html-1", source: SOURCE })
    ).rejects.toThrow("anchor setup blocked")
    expect(append).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).toHaveBeenCalledWith(OBJECT_URL)
    expect(document.querySelector("a[data-standalone-html-download]")).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
    manager.dispose()
  })

  it("keeps at most one live URL and revokes it synchronously on pagehide or dispose", async () => {
    const { StandaloneHtmlDownloadManager } = await loadDownload()
    createObjectURL.mockReturnValueOnce("blob:first").mockReturnValueOnce("blob:second")
    const manager = new StandaloneHtmlDownloadManager({
      downloadDraft: vi.fn().mockResolvedValue(BYTES)
    })

    await manager.download({ presentationId: "html-1", source: SOURCE })
    await manager.download({ presentationId: "html-1", source: SOURCE })
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:first")
    expect(revokeObjectURL).not.toHaveBeenCalledWith("blob:second")

    window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:second")
    const callsAfterPagehide = revokeObjectURL.mock.calls.length
    manager.dispose()
    await vi.runAllTimersAsync()
    expect(revokeObjectURL).toHaveBeenCalledTimes(callsAfterPagehide)
  })
})
