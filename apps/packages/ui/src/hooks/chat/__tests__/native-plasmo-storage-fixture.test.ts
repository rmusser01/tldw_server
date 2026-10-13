import { spyOnStorageGet, spyOnStorageSet, storageOriginal } from "./native-plasmo-storage-fixture"
import { afterEach, expect, it, vi } from "vitest"
import { createSafeStorage } from "@/utils/safe-storage"
import { requirePersistentStorage } from "@/utils/persistent-storage"

const storage = createSafeStorage({ area: "local" })
afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})
it("roundtrips physical bytes through the real captured Plasmo client", async () => {
  requirePersistentStorage(storage)
  await storage.set("native-sdk-fixture", { scene: "original" })
  expect({ physical: window.localStorage.getItem("native-sdk-fixture") !== null,
    value: await storage.get("native-sdk-fixture") }).toEqual({ physical: true, value: { scene: "original" } })
})
it("rejects a physical write failure without recording the value", async () => {
  const prototype = Object.getPrototypeOf(window.localStorage) as globalThis.Storage
  vi.spyOn(prototype, "setItem").mockImplementation(() => { throw new Error("physical write denied") })
  await expect(storage.set("native-sdk-fixture", "unacknowledged")).rejects.toThrow("physical write denied")
  expect(window.localStorage.getItem("native-sdk-fixture")).toBeNull()
})
it("certifies the captured backend when the extension API retires", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome")!
  Object.defineProperty(globalThis, "chrome", { configurable: true, value: undefined })
  try {
    if ("hasPersistentBackend" in storage) expect(() => requirePersistentStorage(storage)).not.toThrow()
    else expect(() => requirePersistentStorage(storage)).toThrow("fork_chat_settings_unavailable")
  }
  finally { Object.defineProperty(globalThis, "chrome", descriptor) }
})

it("returns verified absence through actual native Plasmo", async () => {
  requirePersistentStorage(storage)
  expect(await storage.get("never-stored-native-scene")).toBeUndefined()
})

it("holds an actual captured storage read until the fault control releases", async () => {
  await storage.set("native-controlled-read", { original: true })
  let release!: () => void
  const held = new Promise<void>(resolve => { release = resolve })
  const read = spyOnStorageGet().mockImplementation(async function (key) {
    await held
    return storageOriginal.get.call(this, key)
  })
  let settled = false
  const pending = storage.get("native-controlled-read").then(value => { settled = true; return value })
  await Promise.resolve()
  expect(read).toHaveBeenCalledWith("native-controlled-read")
  expect(settled).toBe(false)
  release()
  expect(await pending).toEqual({ original: true })
})
it("holds physical storage publication before its acknowledgement", async () => {
  let release!: () => void
  const held = new Promise<void>(resolve => { release = resolve })
  const write = spyOnStorageSet().mockImplementation(async function (key, value) {
    await held
    return storageOriginal.set.call(this, key, value)
  })
  const pending = storage.set("native-controlled-write", { original: true })
  await Promise.resolve()
  expect(write).toHaveBeenCalledWith("native-controlled-write", { original: true })
  expect(window.localStorage.getItem("native-controlled-write")).toBeNull()
  release()
  await pending
  expect(await storage.get("native-controlled-write")).toEqual({ original: true })
})

it("preserves enumeration and plain legacy values during a read fault control", async () => {
  await storage.set("native-enumerated-record", { original: true })
  window.localStorage.setItem("native-legacy-string", "Legacy plain text")
  const originalRecords = await storage.getAll()
  const read = spyOnStorageGet()
  expect(await storage.getAll()).toEqual(originalRecords)
  expect(originalRecords).toHaveProperty("native-enumerated-record")
  expect(originalRecords).toHaveProperty("native-legacy-string", "Legacy plain text")
  expect(read).not.toHaveBeenCalled()
  expect(await storage.get("native-legacy-string")).toBe("Legacy plain text")
})
