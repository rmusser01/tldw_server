import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { browser } from "@web/extension/shims/wxt-browser"

const { storage } = browser

describe("wxt-browser storage shim", () => {
  beforeEach(async () => {
    localStorage.clear()
    // session is memory-only; clear it explicitly between tests.
    await storage.session.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("clears only the target area, not the whole origin (H9)", async () => {
    await storage.local.set({ "tldw-api-host": "http://localhost:8000" })
    await storage.sync.set({ theme: "dark" })

    await storage.sync.clear()

    // sync area is empty, local area untouched
    const syncAll = await storage.sync.get(null)
    const localAfter = await storage.local.get("tldw-api-host")
    expect(syncAll).toEqual({})
    expect(localAfter["tldw-api-host"]).toBe("http://localhost:8000")
    // the raw local key must still physically exist in the origin
    expect(localStorage.getItem("tldw-api-host")).toBe(
      JSON.stringify("http://localhost:8000")
    )
  })

  it("isolates areas so sync.set does not clobber local (H9)", async () => {
    await storage.local.set({ shared: "local-value" })
    await storage.sync.set({ shared: "sync-value" })

    const local = await storage.local.get("shared")
    const sync = await storage.sync.get("shared")
    expect(local.shared).toBe("local-value")
    expect(sync.shared).toBe("sync-value")
    // local stays UNPREFIXED for cross-shim / existing-data compatibility
    expect(localStorage.getItem("shared")).toBe(JSON.stringify("local-value"))
    expect(localStorage.getItem("plasmo-sync:shared")).toBe(
      JSON.stringify("sync-value")
    )
  })

  it("get(null) enumerates only the area's own keys", async () => {
    await storage.local.set({ a: 1 })
    await storage.sync.set({ b: 2 })

    const localAll = await storage.local.get(null)
    const syncAll = await storage.sync.get(null)

    expect(localAll).toEqual({ a: 1 })
    expect(syncAll).toEqual({ b: 2 })
  })

  it("does not persist session to disk (memory-only)", async () => {
    await storage.session.set({ token: "ephemeral" })

    // readable via the area API ...
    const read = await storage.session.get("token")
    expect(read.token).toBe("ephemeral")
    // ... but never written to localStorage
    expect(localStorage.getItem("token")).toBeNull()
    expect(localStorage.getItem("plasmo-session:token")).toBeNull()
    expect(localStorage.length).toBe(0)
  })

  it("does not emit onChanged nor resolve as success when set() fails (H9 #3)", async () => {
    const listener = vi.fn()
    storage.onChanged.addListener(listener)

    // Force the write to throw (simulate quota exceeded / serialization error).
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("QuotaExceededError")
      })

    await expect(storage.local.set({ big: "x" })).rejects.toThrow(
      "QuotaExceededError"
    )
    expect(listener).not.toHaveBeenCalled()

    setItemSpy.mockRestore()
    storage.onChanged.removeListener(listener)
  })

  it("emits onChanged with the area name after a successful set", async () => {
    const listener = vi.fn()
    storage.onChanged.addListener(listener)

    await storage.sync.set({ flag: true })

    expect(listener).toHaveBeenCalledTimes(1)
    const [changes, areaName] = listener.mock.calls[0]
    expect(areaName).toBe("sync")
    expect(changes.flag.newValue).toBe(true)

    storage.onChanged.removeListener(listener)
  })
})
