import { beforeEach, describe, expect, it, vi } from "vitest"

const stores = vi.hoisted(() => ({
  local: new Map<string, unknown>(),
  sync: new Map<string, unknown>()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: ({ area }: { area?: string } = {}) => {
    const values = area === "sync" ? stores.sync : stores.local
    return {
      get: vi.fn(async (key: string) => values.get(key)),
      set: vi.fn(async (key: string, value: unknown) => {
        values.set(key, value)
      }),
      remove: vi.fn(async (key: string) => {
        values.delete(key)
      })
    }
  }
}))

describe("tldw settings storage migration", () => {
  beforeEach(() => {
    stores.local.clear()
    stores.sync.clear()
    vi.resetModules()
  })

  it("copies a legacy sync setting to local and removes the migrated value", async () => {
    stores.sync.set("tldwServerUrl", "https://legacy.example.test")
    const { readTldwSetting } = await import("@/services/tldw-settings-storage")

    await expect(readTldwSetting("tldwServerUrl")).resolves.toBe(
      "https://legacy.example.test"
    )
    expect(stores.local.get("tldwServerUrl")).toBe("https://legacy.example.test")
    expect(stores.sync.has("tldwServerUrl")).toBe(false)
  })

  it("preserves an existing local setting", async () => {
    stores.local.set("pageShareUrl", "https://local.example.test")
    stores.sync.set("pageShareUrl", "https://legacy.example.test")
    const { readTldwSetting } = await import("@/services/tldw-settings-storage")

    await expect(readTldwSetting("pageShareUrl")).resolves.toBe(
      "https://local.example.test"
    )
    expect(stores.sync.get("pageShareUrl")).toBe("https://legacy.example.test")
  })
})
