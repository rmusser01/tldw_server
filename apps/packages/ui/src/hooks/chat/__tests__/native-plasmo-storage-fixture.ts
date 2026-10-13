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
const nativeGet = browser.storage.local.get
const nativeSet = browser.storage.local.set
const decode = (raw: unknown): any => safeStorageSerde.deserializer!(raw)
export const storageOriginal = {
  get: async function (this: Storage, key: string): Promise<any> {
    if (prototypeGet) return prototypeGet.call(this, key)
    return decode((await nativeGet(key))[key])
  },
  set: async function (this: Storage, key: string, value: any): Promise<any> {
    if (prototypeSet) return prototypeSet.call(this, key, value)
    await nativeSet({ [key]: JSON.stringify(value) })
  }
}
const controlled = { ...storageOriginal }
export const spyOnStorageGet = () => {
  const spy = vi.spyOn(controlled, "get")
  if (prototypeGet) {
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (key) {
      return controlled.get.call(this, key)
    })
  } else {
    vi.spyOn(browser.storage.local, "get").mockImplementation(async (keys, callback) => {
      if (keys == null) return nativeGet(keys, callback)
      const names = typeof keys === "string" ? [keys] : keys
      const result: Record<string, unknown> = {}
      for (const key of names) {
        const value = await controlled.get.call(undefined as never, key)
        if (value !== undefined) result[key] = JSON.stringify(value)
      }
      callback?.(result)
      return result
    })
  }
  return spy
}
export const spyOnStorageSet = () => {
  const spy = vi.spyOn(controlled, "set")
  if (prototypeSet) {
    vi.spyOn(Storage.prototype, "set").mockImplementation(function (key, value) {
      return controlled.set.call(this, key, value)
    })
  } else {
    vi.spyOn(browser.storage.local, "set").mockImplementation(async (items, callback) => {
      for (const [key, value] of Object.entries(items))
        await controlled.set.call(undefined as never, key, decode(value))
      callback?.()
    })
  }
  return spy
}
