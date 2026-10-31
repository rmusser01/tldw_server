import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Storage as PlasmoStorage } from "@plasmohq/storage"
import { createLocalRegistryBucket } from "../local-bucket"

vi.mock("@plasmohq/storage", () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
const prefix = "registry:creation-claim-test:"
const bucket = () => createLocalRegistryBucket<string>({ prefix })
const locks = () => {
  const tails = new Map<string, Promise<unknown>>()
  vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
    request: (key: string, operation: () => unknown) => {
      const next = (tails.get(key) ?? Promise.resolve()).then(operation)
      tails.set(key, next.catch(() => undefined))
      return next
    }
  } } }))
}

describe("existing registry durable creation reservation boundary", () => {
  beforeEach(() => { localStorage.clear(); sessionStorage.clear(); locks() })
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it("allows exactly one same-key absence claim across independent bucket instances", async () => {
    const a = bucket(), b = bucket()
    const results = await Promise.all([a.set("intent", "A", 1, null), b.set("intent", "B", 2, null)])
    expect(results.filter(Boolean)).toHaveLength(1)
    expect((await bucket().get("intent", { failClosed: true }))?.value).toBe(results[0] ? "A" : "B")
  })

  it("blocks a cold same-intent claim and retains an old nonexpiring unresolved record", async () => {
    expect(await bucket().set("intent", "UNCONFIRMED", Date.now() - 366 * 86400000, null)).toBe(true)
    expect(await bucket().set("intent", "REPLAY", Date.now(), null)).toBe(false)
    expect((await bucket().get("intent", { failClosed: true }))?.value).toBe("UNCONFIRMED")
  })

  it("returns actual durable absence without a session pin", async () => {
    expect(await bucket().get("never-created", { failClosed: true })).toBeNull()
    expect(sessionStorage.length).toBe(0)
  })

  it.each(
    [false, true].flatMap((backendRemains) =>
      [false, true].map((present) => ({ backendRemains, present }))
    )
  )(
    "certifies held strict read presence=$present only while the backend remains=$backendRemains",
    async ({ backendRemains, present }) => {
      const storage = new PlasmoStorage({ area: "local" })
      const record = { value: "ORIGINAL UNCONFIRMED", updatedAt: 1 }
      if (present) await storage.set(prefix + "intent", record)
      const before = localStorage.getItem(prefix + "intent")
      const get = PlasmoStorage.prototype.get
      let persistent = true
      vi.spyOn(
        PlasmoStorage.prototype,
        "hasPersistentBackend",
        "get"
      ).mockImplementation(() => persistent)
      let release!: () => void
      let entered!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      const started = new Promise<void>((resolve) => {
        entered = resolve
      })
      vi.spyOn(PlasmoStorage.prototype, "get").mockImplementationOnce(
        async function (this: PlasmoStorage, key: string) {
          const result = await get.call(this, key)
          entered()
          await held
          return result
        }
      )
      const write = vi.spyOn(PlasmoStorage.prototype, "set")
      const remove = vi.spyOn(PlasmoStorage.prototype, "remove")
      const pending = bucket().get("intent", { failClosed: true })
      await started
      persistent = backendRemains
      release()
      if (backendRemains) expect(await pending).toEqual(present ? record : null)
      else
        await expect(pending).rejects.toThrow("registry_durable_read_unavailable")
      expect(write).not.toHaveBeenCalled()
      expect(remove).not.toHaveBeenCalled()
      expect(localStorage.getItem(prefix + "intent")).toBe(before)
      expect(sessionStorage.length).toBe(0)
    }
  )

  it("does not interpret failed durable reads as absence while legacy reads retain null behavior", async () => {
    vi.spyOn(PlasmoStorage.prototype, "get").mockRejectedValue(new Error("storage blocked"))
    await expect(bucket().get("intent", { failClosed: true })).rejects.toThrow("storage blocked")
    expect(await bucket().get("ordinary")).toBeNull()
  })

  it("refuses corrupt durable records without deleting uncertainty", async () => {
    const storage = new PlasmoStorage({ area: "local" })
    await storage.set(prefix + "intent", { value: "UNCONFIRMED", updatedAt: "invalid" })
    const before = localStorage.getItem(prefix + "intent")
    await expect(bucket().get("intent", { failClosed: true })).rejects.toThrow()
    expect(localStorage.getItem(prefix + "intent")).toBe(before)
    expect(await bucket().set("intent", "REPLAY", Date.now(), null)).toBe(false)
  })

  it("refuses stored literal null without deleting it while ordinary reads remain unchanged", async () => {
    const storage = new PlasmoStorage({ area: "local" })
    await storage.set(prefix + "intent", null)
    const before = localStorage.getItem(prefix + "intent")
    await expect(bucket().get("intent", { failClosed: true })).rejects.toThrow()
    expect(localStorage.getItem(prefix + "intent")).toBe(before)
    expect(await bucket().get("intent")).toBeNull()
  })

  it("does not replace stored literal null with an absence claim", async () => {
    const storage = new PlasmoStorage({ area: "local" })
    await storage.set(prefix + "intent", null)
    expect(await bucket().set("intent", "REPLAY", Date.now(), null)).toBe(false)
    expect(await storage.get(prefix + "intent")).toBeNull()
  })

  it("refuses expired strict records without deleting them or enabling replay", async () => {
    const timed = createLocalRegistryBucket<string>({ prefix, ttlMs: 10 })
    await bucket().set("intent", "UNCONFIRMED", Date.now() - 1000)
    const before = localStorage.getItem(prefix + "intent")
    await expect(timed.get("intent", { failClosed: true })).rejects.toThrow()
    expect(localStorage.getItem(prefix + "intent")).toBe(before)
    expect(await timed.set("intent", "REPLAY", Date.now(), null)).toBe(false)
  })

  it("refuses volatile persistence before strict read or conditional claim writes", async () => {
    vi.spyOn(PlasmoStorage.prototype, "hasPersistentBackend", "get").mockReturnValue(false)
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    await expect(bucket().get("intent", { failClosed: true })).rejects.toThrow()
    expect(await bucket().set("intent", "DISPATCH", Date.now(), null)).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })

  it("refuses an uncertified storage instance even when backend flags are missing", async () => {
    vi.spyOn(PlasmoStorage.prototype, "hasPersistentBackend", "get").mockReturnValue(undefined as unknown as boolean)
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    await expect(bucket().get("intent", { failClosed: true })).rejects.toThrow()
    expect(await bucket().set("intent", "DISPATCH", Date.now(), null)).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })

  it("checks the captured lease after waiting for the record lock", async () => {
    let release!: () => void
    let entered!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const started = new Promise<void>(resolve => { entered = resolve })
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
      request: async (_key: string, operation: () => unknown) => { entered(); await held; return operation() }
    } } }))
    let current = true
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    const claim = bucket().set("intent", "DISPATCH", Date.now(), null, () => current)
    await started
    current = false
    release()
    expect(await claim).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })

  it("checks the captured lease after an awaited comparison read", async () => {
    let release!: () => void
    let entered!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const started = new Promise<void>(resolve => { entered = resolve })
    vi.spyOn(PlasmoStorage.prototype, "get").mockImplementationOnce(async () => { entered(); await held; return undefined })
    let current = true
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    const claim = bucket().set("intent", "DISPATCH", Date.now(), null, () => current)
    await started
    current = false
    release()
    expect(await claim).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })

  it("refuses lockless absence claims before any write", async () => {
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: undefined } }))
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    expect(await bucket().set("intent", "DISPATCH", Date.now(), null)).toBe(false)
    expect(write).not.toHaveBeenCalled()
    expect(await bucket().set("ordinary", "UNCHANGED")).toBe(true)
  })

  it("does not let a tab pin hide newer durable uncertainty from a strict read", async () => {
    const pinned = createLocalRegistryBucket<string>({ prefix, tabScoped: true })
    await pinned.get("intent")
    await bucket().set("intent", "NEWER UNCONFIRMED")
    await expect(pinned.get("intent", { failClosed: true })).rejects.toThrow()
    expect((await bucket().get("intent", { failClosed: true }))?.value).toBe("NEWER UNCONFIRMED")
  })

  it("refuses an unavailable locked comparison before writing a claim", async () => {
    vi.spyOn(PlasmoStorage.prototype, "get").mockRejectedValue(new Error("comparison unavailable"))
    const write = vi.spyOn(PlasmoStorage.prototype, "set")
    expect(await bucket().set("intent", "DISPATCH", Date.now(), null)).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })
})
