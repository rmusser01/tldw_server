import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Storage as PlasmoStorage } from "@plasmohq/storage"
import { createLocalRegistryBucket } from "../local-bucket"

vi.mock("@plasmohq/storage", () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>(done => { resolve = done })
  return { promise, resolve }
}
const prefix = "registry:draft:locking-test:"

describe("registry durable recovery across concurrent tabs", () => {
  beforeEach(() => {
    localStorage.clear(); sessionStorage.clear()
    const tails = new Map<string, Promise<unknown>>()
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
      request: (key: string, operation: () => unknown) => {
        const next = (tails.get(key) ?? Promise.resolve()).then(operation)
        tails.set(key, next.catch(() => undefined))
        return next
      }
    } } }))
  })
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it("keeps a newer write queued while another tab conditionally clears its own record", async () => {
    const a = createLocalRegistryBucket<string>({ prefix, tabScoped: true })
    const b = createLocalRegistryBucket<string>({ prefix })
    await a.set("draft", "OLD")
    const entered = deferred(), release = deferred()
    const get = PlasmoStorage.prototype.get
    let blocked = false
    vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function<T>(this: PlasmoStorage, key: string): Promise<T | undefined> {
      const value = await get.call(this, key) as T | undefined
      if (key === prefix + "draft" && !blocked) {
        blocked = true; entered.resolve(); await release.promise
      }
      return value
    })
    const clearing = a.remove("draft")
    await entered.promise
    const saving = b.set("draft", "NEWER")
    // Give an unlocked write enough microtasks to commit before releasing deletion.
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    release.resolve()
    await Promise.all([clearing, saving])
    expect((await b.get("draft"))?.value).toBe("NEWER")
  })

  it("rechecks default-bucket cleanup candidates after another tab refreshes them", async () => {
    const a = createLocalRegistryBucket<string>({ prefix, ttlMs: 1000 })
    const b = createLocalRegistryBucket<string>({ prefix, ttlMs: 1000, tabScoped: true })
    await a.set("draft", "EXPIRED", Date.now() - 2000)
    const entered = deferred(), release = deferred()
    const getAll = PlasmoStorage.prototype.getAll
    vi.spyOn(PlasmoStorage.prototype, "getAll").mockImplementation(async function(this: PlasmoStorage) {
      const entries = await getAll.call(this)
      entered.resolve(); await release.promise
      return entries
    })
    const cleanup = a.cleanup()
    await entered.promise
    await b.set("draft", "FRESH")
    release.resolve()
    expect(await cleanup).toBe(0)
    expect((await a.get("draft"))?.value).toBe("FRESH")
  })

  it("does not delete a newer record after reading an expired durable snapshot", async () => {
    const a = createLocalRegistryBucket<string>({ prefix, ttlMs: 1000 })
    const b = createLocalRegistryBucket<string>({ prefix, ttlMs: 1000 })
    await a.set("draft", "EXPIRED", Date.now() - 2000)
    const entered = deferred(), release = deferred()
    const get = PlasmoStorage.prototype.get
    let blocked = false
    vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function<T>(this: PlasmoStorage, key: string): Promise<T | undefined> {
      const value = await get.call(this, key) as T | undefined
      if (key === prefix + "draft" && !blocked) {
        blocked = true; entered.resolve(); await release.promise
      }
      return value
    })
    const staleRead = a.get("draft")
    await entered.promise
    await b.set("draft", "FRESH")
    release.resolve()
    expect(await staleRead).toBeNull()
    expect((await b.get("draft"))?.value).toBe("FRESH")
  })
})
