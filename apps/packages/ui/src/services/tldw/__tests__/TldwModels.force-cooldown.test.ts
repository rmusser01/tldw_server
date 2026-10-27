import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
  getCurrentUserProfile: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  // Shared, Map-backed storage: survives across service instances exactly
  // like chrome.storage.local survives an MV3 worker suspension.
  storageMap: new Map<string, unknown>(),
  storageGet: vi.fn<(key: string) => Promise<unknown>>(),
  storageSet: vi.fn<(key: string, value: unknown) => Promise<void>>(),
  storageRemove: vi.fn<(key: string) => Promise<void>>()
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
      (mocks.getCurrentUserProfile as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (key: string) => mocks.storageGet(key),
    set: (key: string, value: unknown) => mocks.storageSet(key, value),
    remove: (key: string) => mocks.storageRemove(key),
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
    mocks.storageGet
      .mockReset()
      .mockImplementation(async (key) => mocks.storageMap.get(key))
    mocks.storageSet.mockReset().mockImplementation(async (key, value) => {
      mocks.storageMap.set(key, value)
    })
    mocks.storageRemove.mockReset().mockImplementation(async (key) => {
      mocks.storageMap.delete(key)
    })
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

  it("does not restore a retired cooldown from a held startup read", async () => {
    await new TldwModelsService().getModels(true)
    const retiredAt = mocks.storageMap.get("tldwModelsForceCooldown")
    let finishRead!: (value: unknown) => void
    let entered!: () => void
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    const held = new Promise<unknown>((resolve) => {
      finishRead = resolve
    })
    mocks.storageGet.mockImplementation(async (key) => {
      if (key === "tldwModelsForceCooldown") {
        entered()
        return held
      }
      return mocks.storageMap.get(key)
    })
    const service = new TldwModelsService()
    const hydration = service.getModels(false)
    await started
    await service.clearCache()
    finishRead(retiredAt)
    await hydration
    await service.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(3)
  })

  it("drops an unissued forced timestamp when invalidation overtakes its write queue", async () => {
    let releaseCache!: () => void
    let entered!: () => void
    const held = new Promise<void>((resolve) => {
      releaseCache = resolve
    })
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    mocks.storageSet.mockImplementationOnce(async (key, value) => {
      entered()
      await held
      mocks.storageMap.set(key, value)
    })
    const service = new TldwModelsService()
    const oldCatalog = service.getModels(false)
    await started
    let fetched!: () => void
    const fetchedPromise = new Promise<void>((resolve) => {
      fetched = resolve
    })
    mocks.getModels.mockImplementationOnce(async () => {
      fetched()
      return [{ id: "m2", name: "Second Model" }]
    })
    const oldForce = service.getModels(true, { requireFresh: true })
    await fetchedPromise
    // Let the resolved network read enqueue its timestamp behind the held cache write.
    await Promise.resolve()
    const invalidation = service.clearCache()
    releaseCache()
    await Promise.all([oldCatalog, oldForce, invalidation])
    expect(
      mocks.storageSet.mock.calls.filter(
        ([key, value]) => key === "tldwModelsForceCooldown" && Number(value) > 0
      )
    ).toEqual([])
  })

  it("clears a retired cooldown even when a non-forced fresh read overtakes the queued clear", async () => {
    const service = new TldwModelsService()
    await service.getModels(true)
    let releaseCache!: () => void
    let entered!: () => void
    const held = new Promise<void>((resolve) => {
      releaseCache = resolve
    })
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    mocks.storageSet.mockImplementationOnce(async (key, value) => {
      entered()
      await held
      mocks.storageMap.set(key, value)
    })
    const older = service.getModels(false, { requireFresh: true })
    await started
    const clear = service.clearCache()
    const successor = service.getModels(false, { requireFresh: true })
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(3))
    releaseCache()
    await Promise.all([older, clear, successor])
    await new TldwModelsService().getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(4)
  })

  it("keeps the new forced cooldown after an older asynchronous clear finishes", async () => {
    const service = new TldwModelsService()
    await service.getModels(true)
    let finishRemove!: () => void
    const held = new Promise<void>((resolve) => {
      finishRemove = resolve
    })
    mocks.storageRemove.mockImplementationOnce(async (key) => {
      await held
      mocks.storageMap.delete(key)
    })
    const clear = service.clearCache()
    const fresh = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(2))
    finishRemove()
    await Promise.all([clear, fresh])
    await new TldwModelsService().getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
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
