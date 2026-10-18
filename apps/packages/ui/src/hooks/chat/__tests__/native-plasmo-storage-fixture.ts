import { afterAll, beforeEach, vi } from "vitest"
import { safeStorageSerde } from "@/utils/safe-storage"
import { browser } from "../../../../../../tldw-frontend/extension/shims/wxt-browser"

// Install the API before source imports construct real Plasmo storage instances.
vi.stubGlobal("chrome", { storage: browser.storage })
beforeEach(() => vi.stubGlobal("chrome", { storage: browser.storage }))
afterAll(() => vi.unstubAllGlobals())

// Fault controls reach the captured native API or the WebUI instance backend.
const { Storage } = await import("@plasmohq/storage")
const prototypeGet = Storage.prototype.get
const prototypeSet = Storage.prototype.set
type FaultArea = "local" | "sync"
const nativeAreas = {
  local: { get: browser.storage.local.get, set: browser.storage.local.set },
  sync: { get: browser.storage.sync.get, set: browser.storage.sync.set }
}
const decode = (raw: unknown): any => safeStorageSerde.deserializer!(raw)
export const storageOriginal = {
  get: async function (this: Storage, key: string): Promise<any> {
    if (prototypeGet) return prototypeGet.call(this, key)
    const area = this?.area === "sync" ? "sync" : "local"
    return decode((await nativeAreas[area].get(key))[key])
  },
  set: async function (this: Storage, key: string, value: any): Promise<any> {
    if (prototypeSet) return prototypeSet.call(this, key, value)
    const area = this?.area === "sync" ? "sync" : "local"
    await nativeAreas[area].set({ [key]: JSON.stringify(value) })
  }
}
const controlled = { ...storageOriginal }
export const spyOnStorageGet = (area: FaultArea = "local") => {
  const spy = vi.spyOn(controlled, "get")
  if (prototypeGet) {
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (key) {
      return controlled.get.call(this, key)
    })
  } else {
    const context = new Storage({ area })
    vi.spyOn(browser.storage[area], "get").mockImplementation(async (keys, callback) => {
      if (keys == null) return nativeAreas[area].get(keys, callback)
      const names = typeof keys === "string" ? [keys] : keys
      const result: Record<string, unknown> = {}
      for (const key of names) {
        const value = await controlled.get.call(context, key)
        if (value !== undefined) result[key] = JSON.stringify(value)
      }
      callback?.(result)
      return result
    })
  }
  return spy
}
export const spyOnStorageSet = (area: FaultArea = "local") => {
  const spy = vi.spyOn(controlled, "set")
  if (prototypeSet) {
    vi.spyOn(Storage.prototype, "set").mockImplementation(function (key, value) {
      return controlled.set.call(this, key, value)
    })
  } else {
    const context = new Storage({ area })
    vi.spyOn(browser.storage[area], "set").mockImplementation(async (items, callback) => {
      for (const [key, value] of Object.entries(items))
        await controlled.set.call(context, key, decode(value))
      callback?.()
    })
  }
  return spy
}
