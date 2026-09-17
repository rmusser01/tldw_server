import { afterEach, expect, it, vi } from "vitest"
import { createSafeStorage } from "../safe-storage"
import { requirePersistentStorage } from "../persistent-storage"

afterEach(() => vi.unstubAllGlobals())
it("rejects installed Plasmo's silent no-client get/set path while preserving ordinary use", async () => {
  vi.stubGlobal("browser", undefined)
  vi.stubGlobal("chrome", undefined)
  const storage = createSafeStorage({ area: "local" })
  await storage.set("required", "persona")
  expect(await storage.get("required")).toBeUndefined()
  expect(() => requirePersistentStorage(storage)).toThrow(
    "fork_chat_settings_unavailable",
  )
})
it("does not mistake a later global API for an instance's missing captured client", () => {
  vi.stubGlobal("browser", undefined)
  vi.stubGlobal("chrome", undefined)
  const storage = createSafeStorage({ area: "local" })
  vi.stubGlobal("browser", {
    storage: { local: { get: async () => ({}), set: async () => {} } },
  })
  expect(() => requirePersistentStorage(storage)).toThrow(
    "fork_chat_settings_unavailable",
  )
})
it("rejects inactive extension routing even with a previously captured client", () => {
  vi.stubGlobal("browser", {
    storage: { local: { get: async () => ({}), set: async () => {} } },
  })
  vi.stubGlobal("chrome", undefined)
  const storage = createSafeStorage({ area: "local" })
  vi.stubGlobal("browser", undefined)
  expect(() => requirePersistentStorage(storage)).toThrow(
    "fork_chat_settings_unavailable",
  )
})
it("accepts the captured persistent local client without requiring sync", async () => {
  const values: Record<string, unknown> = {}
  vi.stubGlobal("browser", {
    storage: {
      local: {
        get: async (keys: string[]) =>
          Object.fromEntries(keys.map((key) => [key, values[key]])),
        set: async (patch: object) => {
          Object.assign(values, patch)
        },
      },
    },
  })
  const storage = createSafeStorage({ area: "local" })
  requirePersistentStorage(storage)
  await storage.set("settings", { persona: "required" })
  expect(await storage.get("settings")).toEqual({ persona: "required" })
})
