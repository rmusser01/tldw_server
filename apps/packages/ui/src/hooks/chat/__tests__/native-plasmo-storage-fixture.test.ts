import "./native-plasmo-storage-fixture"
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
it("refuses persistent certification when its API retires", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome")!
  Object.defineProperty(globalThis, "chrome", { configurable: true, value: undefined })
  try { expect(() => requirePersistentStorage(storage)).toThrow("fork_chat_settings_unavailable") }
  finally { Object.defineProperty(globalThis, "chrome", descriptor) }
})

it("returns verified absence through actual native Plasmo", async () => {
  requirePersistentStorage(storage)
  expect(await storage.get("never-stored-native-scene")).toBeUndefined()
})
