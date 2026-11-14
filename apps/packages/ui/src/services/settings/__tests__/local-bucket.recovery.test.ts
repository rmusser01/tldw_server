import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { Storage as PlasmoStorage } from "@plasmohq/storage"

import { createLocalRegistryBucket } from "../local-bucket"

vi.mock(
  "@plasmohq/storage",
  () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
const prefix = "registry:draft:recovery-test:"
const area = (): Storage => {
  const values = new Map<string, string>()
  return {
    get length() {
      return values.size
    },
    key: (index) => [...values.keys()][index] ?? null,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value)
    },
    removeItem: (key) => {
      values.delete(key)
    },
    clear: () => values.clear()
  }
}

describe("nonmutating registry recovery enumeration", () => {
  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("lists readable current records without pinning unread candidates or deleting stale records", async () => {
    const bucket = createLocalRegistryBucket<string>({
      prefix,
      ttlMs: 1000,
      tabScoped: true
    })
    const writer = createLocalRegistryBucket<string>({ prefix })
    await writer.set("fresh", "KEEP")
    await writer.set("expired", "OLD", Date.now() - 2000)
    const before = localStorage.getItem(prefix + "expired")
    expect(await bucket.list()).toMatchObject({
      unavailable: false,
      records: { fresh: { value: "KEEP" } }
    })
    expect((await bucket.list()).records.expired).toBeUndefined()
    expect(sessionStorage.getItem(prefix + "fresh")).toBeNull()
    expect(localStorage.getItem(prefix + "expired")).toBe(before)
  })

  it("uses only this physical session's pin over the shared last-record recovery", async () => {
    const first = area(),
      second = area()
    vi.stubGlobal("sessionStorage", first)
    const a = createLocalRegistryBucket<string>({ prefix, tabScoped: true })
    await a.set("draft", "A")
    vi.stubGlobal("sessionStorage", second)
    const b = createLocalRegistryBucket<string>({ prefix, tabScoped: true })
    await b.set("draft", "B")
    expect((await a.list()).records.draft.value).toBe("A")
    expect((await b.list()).records.draft.value).toBe("B")
  })

  it("does not resurrect another tab's later text through an explicit empty tombstone", async () => {
    const a = createLocalRegistryBucket<string>({ prefix, tabScoped: true })
    const b = createLocalRegistryBucket<string>({ prefix })
    await a.set("draft", "A")
    await b.set("draft", "B")
    await a.remove("draft")
    expect((await a.list()).records.draft).toBeUndefined()
    expect((await b.get("draft"))?.value).toBe("B")
  })

  it("retains this mounted bucket's fallback pins during quota and durable enumeration failure", async () => {
    const tab = area()
    vi.stubGlobal("sessionStorage", tab)
    const bucket = createLocalRegistryBucket<string>({
      prefix,
      tabScoped: true
    })
    vi.spyOn(tab, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    await bucket.set("draft", "PINNED ONLY")
    vi.spyOn(PlasmoStorage.prototype, "getAll").mockRejectedValue(
      new Error("offline storage")
    )
    expect(await bucket.list()).toMatchObject({
      unavailable: true,
      records: { draft: { value: "PINNED ONLY" } }
    })
  })

  it("overlays a same-tab edit made while durable enumeration is pending", async () => {
    const bucket = createLocalRegistryBucket<string>({
      prefix,
      tabScoped: true
    })
    await bucket.set("draft", "OLD")
    const getAll = PlasmoStorage.prototype.getAll
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    let entered!: () => void
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    vi.spyOn(PlasmoStorage.prototype, "getAll").mockImplementation(
      async function (this: PlasmoStorage) {
        const old = await getAll.call(this)
        entered()
        await held
        return old
      }
    )
    const listing = bucket.list()
    await started
    await bucket.set("draft", "NEW")
    release()
    expect((await listing).records.draft.value).toBe("NEW")
  })

  it("retires a same-tab marker without replacing another tab's newer durable record", async () => {
    const a = createLocalRegistryBucket<{ text: string; marker?: boolean }>({
      prefix,
      tabScoped: true
    })
    const b = createLocalRegistryBucket<{ text: string; marker?: boolean }>({
      prefix
    })
    await a.set("draft", { text: "A", marker: true })
    const previous = (await a.get("draft"))!
    await b.set("draft", { text: "B", marker: true })
    expect(
      await a.set("draft", { text: "A" }, previous.updatedAt, previous)
    ).toBe(false)
    expect((await a.list()).records.draft.value).toEqual({ text: "A" })
    expect((await b.get("draft"))?.value).toEqual({ text: "B", marker: true })
  })

  it("reports unavailable storage without presenting a successful empty result", async () => {
    const bucket = createLocalRegistryBucket<string>({ prefix })
    vi.spyOn(PlasmoStorage.prototype, "getAll").mockRejectedValue(
      new Error("blocked")
    )
    expect(await bucket.list()).toEqual({ records: {}, unavailable: true })
  })
  it("reads native serialized enumeration alongside decoded objects without pinning malformed or stale values", async () => {
    const now = Date.now()
    vi.spyOn(PlasmoStorage.prototype, "getAll").mockResolvedValue({
      [prefix + "native"]: JSON.stringify({ value: "NATIVE", updatedAt: now }),
      [prefix + "decoded"]: { value: "DECODED", updatedAt: now },
      [prefix + "expired"]: JSON.stringify({
        value: "EXPIRED",
        updatedAt: now - 2000
      }),
      [prefix + "null"]: "null",
      [prefix + "malformed"]: "{",
      [prefix + "invalid-date"]: JSON.stringify({
        value: "INVALID DATE",
        updatedAt: 1e100
      })
    })
    const bucket = createLocalRegistryBucket<string>({
      prefix,
      ttlMs: 1000,
      tabScoped: true
    })
    expect(await bucket.list()).toEqual({
      unavailable: false,
      records: {
        native: { value: "NATIVE", updatedAt: now },
        decoded: { value: "DECODED", updatedAt: now }
      }
    })
    expect(sessionStorage.length).toBe(0)
  })
  it("reports a volatile fallback as unavailable while keeping its reachable draft", async () => {
    const bucket = createLocalRegistryBucket<string>({
      prefix,
      tabScoped: true
    })
    await bucket.set("draft", "PINNED")
    vi.spyOn(
      PlasmoStorage.prototype,
      "hasPersistentBackend",
      "get"
    ).mockReturnValue(false)
    expect(await bucket.list()).toMatchObject({
      unavailable: true,
      records: { draft: { value: "PINNED" } }
    })
  })
  it("skips lockless durable retirement so a later writer cannot be replaced after an old comparison read", async () => {
    const a = createLocalRegistryBucket<{ text: string; marker?: boolean }>({
      prefix,
      tabScoped: true
    })
    const b = createLocalRegistryBucket<{ text: string; marker?: boolean }>({
      prefix
    })
    await a.set("draft", { text: "A", marker: true })
    const previous = (await a.get("draft"))!
    const get = PlasmoStorage.prototype.get
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    let readHeld = false
    const comparison = vi
      .spyOn(PlasmoStorage.prototype, "get")
      .mockImplementation(async function (this: PlasmoStorage, key: string) {
        const hold = !readHeld
        readHeld = true
        const old = await get.call(this, key)
        if (hold) await held
        return old
      })
    const retire = a.set("draft", { text: "A" }, previous.updatedAt, previous)
    comparison.mockRestore()
    await b.set("draft", { text: "B", marker: true })
    release()
    expect(await retire).toBe(false)
    expect((await a.list()).records.draft.value).toEqual({ text: "A" })
    expect((await b.get("draft"))?.value).toEqual({ text: "B", marker: true })
  })

  it("refuses a native no-op write when extension storage is absent", async () => {
    // Native Plasmo has no WebUI backend certification getter.
    vi.spyOn(PlasmoStorage.prototype, "hasPersistentBackend", "get").mockReturnValue(undefined as unknown as boolean)
    Object.defineProperty(PlasmoStorage.prototype, "hasExtensionApi", {
      configurable: true,
      value: false
    })
    try {
      vi.spyOn(PlasmoStorage.prototype, "set").mockResolvedValue(undefined)
      const bucket = createLocalRegistryBucket<string>({
        prefix,
        tabScoped: true
      })
      expect(await bucket.set("draft", "PINNED ONLY")).toBe(false)
      expect(await bucket.list()).toMatchObject({
        unavailable: true,
        records: { draft: { value: "PINNED ONLY" } }
      })
    } finally {
      Reflect.deleteProperty(PlasmoStorage.prototype, "hasExtensionApi")
    }
  })
})
