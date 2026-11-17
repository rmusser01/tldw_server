import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const runtime = vi.hoisted(() => ({ id: "" }))
vi.mock("wxt/browser", () => ({ browser: { runtime } }))

beforeEach(() => {
  runtime.id = ""
  vi.resetModules()
})
afterEach(() => vi.unstubAllGlobals())

describe("document GET scope initialization", () => {
  it.each([undefined, {}])(
    "imports without a browser location (%j)",
    async (value) => {
      vi.stubGlobal("window", value)
      const { captureDocumentGetSignal } = await import("../document-get-scope")
      expect(captureDocumentGetSignal()).toBeUndefined()
    }
  )

  it("does not attach document lifetime to an extension page", async () => {
    const addEventListener = vi.fn()
    runtime.id = "owned-extension"
    vi.stubGlobal("window", {
      location: { protocol: "chrome-extension:" },
      addEventListener
    })
    const { captureDocumentGetSignal } = await import("../document-get-scope")
    expect(captureDocumentGetSignal()).toBeUndefined()
    expect(addEventListener).not.toHaveBeenCalled()
  })

  it("retires HTTP GETs and renews them on BFCache restoration without scoping writes or runtime requests", async () => {
    const listeners = new Map<string, () => void>()
    vi.stubGlobal("window", {
      location: { protocol: "https:" },
      addEventListener: (name: string, listener: () => void) =>
        listeners.set(name, listener)
    })
    const { captureDocumentGetSignal } = await import("../document-get-scope")
    const initial = captureDocumentGetSignal()
    expect(initial?.aborted).toBe(false)
    expect(captureDocumentGetSignal("POST")).toBeUndefined()
    listeners.get("pagehide")!()
    expect(initial?.aborted).toBe(true)
    listeners.get("pageshow")!()
    const restored = captureDocumentGetSignal("get")
    expect(restored).not.toBe(initial)
    expect(restored?.aborted).toBe(false)
    runtime.id = "owned-extension"
    expect(captureDocumentGetSignal()).toBeUndefined()
  })
})
