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

describe("TldwModelsService warm-cache TTL (model-warm alarm no-op)", () => {
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

  it("warmCache(false) on a fresh catalog issues zero fetches on the next wake", async () => {
    // Cold start: one catalog fetch, persisted to storage.
    const first = new TldwModelsService()
    await first.warmCache(false)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    const initializeAfterCold = mocks.initialize.mock.calls.length

    // The periodic model-warm alarm fires in a restarted worker: the alarm
    // handler's isCatalogFresh() gate is true and warmModels(false) must
    // return the cached catalog without any network round trip.
    const second = new TldwModelsService()
    expect(await second.isCatalogFresh()).toBe(true)
    const models = await second.warmCache(false)
    expect(models).toHaveLength(1)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    expect(mocks.initialize.mock.calls.length).toBe(initializeAfterCold)
  })

  it("the no-op is TTL-bound: warm refetches only after the 5-minute TTL", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(Date.now())
    try {
      const cold = new TldwModelsService()
      await cold.warmCache(false)
      expect(mocks.getModels).toHaveBeenCalledTimes(1)

      // 4 minutes later: still inside the 5-minute TTL (the backend inventory TTL) → zero fetches.
      vi.advanceTimersByTime(4 * 60 * 1000)
      const warm = new TldwModelsService()
      expect(await warm.isCatalogFresh()).toBe(true)
      await warm.warmCache(false)
      expect(mocks.getModels).toHaveBeenCalledTimes(1)

      // Past the TTL the same non-forced warm refetches exactly once.
      vi.advanceTimersByTime(2 * 60 * 1000)
      const expired = new TldwModelsService()
      expect(await expired.isCatalogFresh()).toBe(false)
      await expired.warmCache(false)
      expect(mocks.getModels).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })
})
