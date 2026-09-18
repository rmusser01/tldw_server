import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { transferFlashcardsSource } from "@/services/tldw/flashcards-generate-transfer"
import { FLASHCARDS_GENERATE_HANDOFF_PREFIX } from "@/services/tldw/flashcards-generate-handoff"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  createTab: vi.fn(),
  removeTab: vi.fn(),
  extension: false
}))
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: mocks.initialize,
  ensureConfigForRequest: async () => JSON.parse(window.localStorage.getItem("tldwConfig") || "null")
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1, is_active: true }) } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => mocks.extension }))
vi.mock("wxt/browser", () => ({ browser: {
  runtime: { getURL: (path: string) => `chrome-extension://test/${path}` },
  tabs: { create: mocks.createTab, remove: mocks.removeTab }
} }))

const intent = { text: "Private source", sourceType: "note" as const, sourceId: "note-1" }
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const transferKeys = () => Array.from(
  { length: window.localStorage.length }, (_, index) => window.localStorage.key(index)
).filter(key => key?.startsWith(FLASHCARDS_GENERATE_HANDOFF_PREFIX))

describe("private Flashcards producer platform boundary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.extension = false
    mocks.initialize.mockResolvedValue(undefined)
    window.localStorage.clear()
    window.localStorage.setItem("tldwConfig", JSON.stringify({
      serverUrl: "https://server.test", authMode: "single-user", apiKey: "synthetic-key"
    }))
    let tail = Promise.resolve()
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
      request: (_name: string, work: () => unknown) => {
        const next = tail.then(work)
        tail = next.then(() => undefined, () => undefined)
        return next
      }
    } } }))
  })
  afterEach(() => vi.unstubAllGlobals())

  it("waits for authority and storage before same-tab navigation", async () => {
    const ready = deferred<void>()
    mocks.initialize.mockReturnValueOnce(ready.promise)
    const acquire = vi.fn(() => intent)
    const navigate = vi.fn(() => expect(transferKeys()).toHaveLength(1))
    const pending = transferFlashcardsSource(acquire, { navigate })
    expect(acquire).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
    ready.resolve()
    await pending
    expect(navigate.mock.calls[0][0]).not.toContain("Private")
  })

  it("captures the authority before delayed source acquisition and rejects A to B to A", async () => {
    const source = deferred<typeof intent>()
    const acquire = vi.fn(() => source.promise)
    const navigate = vi.fn()
    const pending = transferFlashcardsSource(acquire, { navigate })
    await vi.waitFor(() => expect(acquire).toHaveBeenCalled())
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    source.resolve(intent)
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(navigate).not.toHaveBeenCalled()
    expect(transferKeys()).toHaveLength(0)
  })

  it("rejects a cross-tab A to B to A change while initial authority resolution is delayed", async () => {
    const ready = deferred<void>()
    mocks.initialize.mockReturnValueOnce(ready.promise)
    const acquire = vi.fn(() => intent)
    const navigate = vi.fn()
    const pending = transferFlashcardsSource(acquire, { navigate })
    const original = window.localStorage.getItem("tldwConfig")!
    const other = JSON.stringify({ serverUrl: "https://other.test", authMode: "single-user", apiKey: "other-key" })
    window.localStorage.setItem("tldwConfig", other)
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: original, newValue: other }))
    window.localStorage.setItem("tldwConfig", original)
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: other, newValue: original }))
    ready.resolve()
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(acquire).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })

  it("reserves a real WebUI target during the click and navigates only after storage", async () => {
    const source = deferred<typeof intent>()
    const popup = { closed: false, opener: {}, location: { replace: vi.fn() }, close: vi.fn() }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const pending = transferFlashcardsSource(() => source.promise, { newTab: true })
    expect(window.open).toHaveBeenCalledWith("about:blank", "_blank")
    expect(popup.location.replace).not.toHaveBeenCalled()
    source.resolve(intent)
    await pending
    expect(popup.opener).toBeNull()
    expect(popup.location.replace).toHaveBeenCalledWith(expect.stringContaining("generate_handoff="))
    expect(popup.close).not.toHaveBeenCalled()
  })

  it("reports a blocked popup before acquiring or discarding the source", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    const acquire = vi.fn(() => intent)
    await expect(transferFlashcardsSource(acquire, { newTab: true })).rejects.toThrow(/popup|tab/i)
    expect(acquire).not.toHaveBeenCalled()
    expect(transferKeys()).toHaveLength(0)
  })

  it("closes a reserved target and removes its transfer when navigation fails", async () => {
    const popup = { closed: false, opener: {}, location: { replace: vi.fn(() => { throw new Error("Closed") }) }, close: vi.fn() }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    await expect(transferFlashcardsSource(() => intent, { newTab: true })).rejects.toThrow()
    expect(popup.close).toHaveBeenCalled()
    expect(transferKeys()).toHaveLength(0)
  })

  it("removes a same-tab transfer when asynchronous navigation rejects", async () => {
    const navigate = vi.fn(async () => { throw new Error("Navigation failed") })
    await expect(transferFlashcardsSource(() => intent, { navigate })).rejects.toThrow("Navigation failed")
    expect(transferKeys()).toHaveLength(0)
  })

  it("closes a newly opened extension tab when the source was cancelled while opening", async () => {
    mocks.extension = true
    const opened = deferred<{ id: number }>()
    mocks.createTab.mockReturnValue(opened.promise)
    const controller = new AbortController()
    const pending = transferFlashcardsSource(() => intent, { newTab: true }, controller.signal)
    await vi.waitFor(() => expect(mocks.createTab).toHaveBeenCalled())
    controller.abort()
    opened.resolve({ id: 4 })
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.removeTab).toHaveBeenCalledWith(4)
    expect(transferKeys()).toHaveLength(0)
  })

  it("requires an actual extension tab result rather than a resolving shim", async () => {
    mocks.extension = true
    mocks.createTab.mockResolvedValue(undefined)
    await expect(transferFlashcardsSource(() => intent, { newTab: true })).rejects.toThrow(/tab/i)
    expect(transferKeys()).toHaveLength(0)
  })

  it("opens extension options with only the opaque token", async () => {
    mocks.extension = true
    mocks.createTab.mockResolvedValue({ id: 4 })
    await transferFlashcardsSource(() => intent, { newTab: true })
    expect(mocks.createTab).toHaveBeenCalledWith({ url: expect.stringMatching(/^chrome-extension:\/\/test\/options.html#\/flashcards\?tab=importExport&generate_handoff=/) })
    expect(mocks.removeTab).not.toHaveBeenCalled()
  })
})
