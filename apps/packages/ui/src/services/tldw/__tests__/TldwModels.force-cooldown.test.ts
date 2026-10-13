import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
  getCurrentUserProfile: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  // Shared, Map-backed storage: survives across service instances exactly
  // like chrome.storage.local survives an MV3 worker suspension.
  storageMap: new Map<string, unknown>()
}))

const config = {
  serverUrl: "https://tldw.example",
  authMode: "single-user",
  apiKey: "key-1"
}

vi.mock("@/services/tldw/TldwApiClient", () => ({
  isActiveCookieSessionConfig: () => false,
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (mocks.getConfig as (...args: unknown[]) => unknown)(...args),
    initialize: (...args: unknown[]) =>
      (mocks.initialize as (...args: unknown[]) => unknown)(...args),
    getModels: (...args: unknown[]) =>
      (mocks.getModels as (...args: unknown[]) => unknown)(...args),
    getCurrentUserProfile: (...args: unknown[]) =>
      (mocks.getCurrentUserProfile as (...args: unknown[]) => unknown)(
        ...args
      )
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => mocks.storageMap.get(key),
    set: async (key: string, value: unknown) => {
      mocks.storageMap.set(key, value)
    },
    remove: async (key: string) => {
      mocks.storageMap.delete(key)
    },
    watch: () => () => undefined
  })
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () =>
    (mocks.getRuntimeSingleUserApiKeyOverride as () => unknown)()
}))

import { TldwModelsService } from "@/services/tldw/TldwModels"

describe("TldwModelsService persisted force cooldown + freshness", () => {
  beforeEach(() => {
    mocks.storageMap.clear()
    mocks.getConfig.mockReset()
    mocks.getConfig.mockResolvedValue(config)
    mocks.initialize.mockReset()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getModels.mockReset()
    mocks.getModels.mockResolvedValue([{ id: "m1", name: "Model One" }])
    mocks.getCurrentUserProfile.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(null)
  })

  it("persists the force-cooldown so a new worker instance honors it without refetching", async () => {
    const first = new TldwModelsService()
    await first.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)

    // Simulate MV3 worker suspension: a brand-new service instance that only
    // sees the persisted cache + persisted cooldown timestamp.
    const second = new TldwModelsService()
    const models = await second.getModels(true)

    expect(models).toHaveLength(1)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("allows a forced refresh again after the cooldown elapses", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(Date.now())
    try {
      const first = new TldwModelsService()
      await first.getModels(true)
      expect(mocks.getModels).toHaveBeenCalledTimes(1)

      vi.advanceTimersByTime(31_000)

      const second = new TldwModelsService()
      await second.getModels(true)
      expect(mocks.getModels).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("reports catalog freshness for the model-warm alarm no-op gate", async () => {
    const empty = new TldwModelsService()
    expect(await empty.isCatalogFresh()).toBe(false)

    const warm = new TldwModelsService()
    await warm.getModels(false)
    expect(await warm.isCatalogFresh()).toBe(true)

    // A restarted worker hydrates freshness from persisted state.
    const restarted = new TldwModelsService()
    expect(await restarted.isCatalogFresh()).toBe(true)
  })
})
