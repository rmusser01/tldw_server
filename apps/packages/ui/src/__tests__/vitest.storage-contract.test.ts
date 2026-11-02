import type { JSDOM } from "jsdom"
import { afterEach, describe, expect, it, vi } from "vitest"

const dom = (globalThis as unknown as { jsdom: JSDOM }).jsdom
const KEY = "vitest-storage-contract"

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.removeItem(KEY)
  dom.window.localStorage.removeItem(KEY)
})

describe("shared browser storage fixture", () => {
  it("uses the environment's real jsdom storage for window and global consumers", () => {
    expect(window.localStorage).toBe(dom.window.localStorage)
    expect(globalThis.localStorage).toBe(dom.window.localStorage)
  })

  it("enumerates and exposes stored keys rather than implementation fields", () => {
    localStorage.setItem(KEY, "owned value")
    expect(Object.keys(localStorage)).toContain(KEY)
    expect(KEY in localStorage).toBe(true)
    expect(localStorage.key(Object.keys(localStorage).indexOf(KEY))).toBe(KEY)
    localStorage.removeItem(KEY)
    expect(Object.keys(localStorage)).not.toContain(KEY)
    expect(KEY in localStorage).toBe(false)
  })

  it("intercepts physical reads and writes through Storage.prototype", () => {
    const write = vi.spyOn(Storage.prototype, "setItem")
    const read = vi.spyOn(Storage.prototype, "getItem")
    localStorage.setItem(KEY, "physical value")
    expect(localStorage.getItem(KEY)).toBe("physical value")
    expect(write).toHaveBeenCalledWith(KEY, "physical value")
    expect(read).toHaveBeenCalledWith(KEY)
  })

  it("accepts the real storage area without bypassing StorageEvent branding", () => {
    const event = new StorageEvent("storage", { storageArea: localStorage, key: KEY })
    expect(event.storageArea).toBe(dom.window.localStorage)
  })

  it("rejects a method-compatible storage impostor", () => {
    const impostor: Storage = {
      length: 0,
      clear() {},
      getItem: () => null,
      key: () => null,
      removeItem() {},
      setItem() {}
    }
    expect(() => new StorageEvent("storage", { storageArea: impostor })).toThrow(/not of type 'Storage'/)
  })
})
