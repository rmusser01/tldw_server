import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getChatModels: vi.fn(),
  getCachedChatModels: vi.fn(),
  clearModelsCache: vi.fn(),
  subscribeInvalidation: vi.fn(),
  invalidationListener: null as ((token: string) => void) | null,
  invalidationSequence: 0,
  bgRequest: vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (mocks.getConfig as (...args: unknown[]) => unknown)(...args)
  },
  tldwModels: {
    getChatModels: (...args: unknown[]) =>
      (mocks.getChatModels as (...args: unknown[]) => unknown)(...args),
    getCachedChatModels: (...args: unknown[]) =>
      (mocks.getCachedChatModels as (...args: unknown[]) => unknown)(...args),
    clearCache: (...args: unknown[]) =>
      (mocks.clearModelsCache as (...args: unknown[]) => unknown)(...args),
    subscribeInvalidation: (listener: (token: string) => void) =>
      mocks.subscribeInvalidation(listener)
  }
}))

vi.mock("@/services/app", () => ({
  setNoOfRetrievedDocs: vi.fn(),
  setTotalFilePerKB: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => undefined),
    set: vi.fn(async () => undefined)
  })
}))

const importService = async () => import("@/services/tldw-server")

const deferred = <T>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

describe("fetchChatModels", () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.getChatModels.mockReset()
    mocks.getCachedChatModels.mockReset()
    mocks.clearModelsCache.mockReset()
    mocks.subscribeInvalidation.mockReset()
    mocks.invalidationListener = null
    mocks.invalidationSequence = 0
    mocks.bgRequest.mockReset()

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:3000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.getCachedChatModels.mockResolvedValue([])
    mocks.subscribeInvalidation.mockImplementation(
      (listener: (token: string) => void) => {
        mocks.invalidationListener = listener
        return () => undefined
      }
    )
    mocks.clearModelsCache.mockImplementation(async () => {
      mocks.invalidationSequence += 1
      mocks.invalidationListener?.(`test-token-${mocks.invalidationSequence}`)
    })
  })

  it("does not cache an empty startup result over later configured models", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "openai/gpt-4o",
          name: "GPT-4o",
          provider: "openai",
          type: "chat"
        }
      ])

    const { fetchChatModels } = await importService()

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([])

    const models = await fetchChatModels({ returnEmpty: true })

    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
    expect(models).toEqual([
      expect.objectContaining({
        model: "tldw:openai/gpt-4o",
        nickname: "GPT-4o",
        provider: "openai"
      })
    ])
  })

  it("applies each inner invalidation token to the outer cache once", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([
        { id: "llama/old", name: "Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "llama/fresh", name: "Fresh", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "llama/newer", name: "Newer", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "llama/unexpected", name: "Unexpected", provider: "llama", type: "chat" }
      ])

    const { fetchChatModels } = await importService()
    expect(mocks.subscribeInvalidation).toHaveBeenCalledTimes(1)

    await fetchChatModels({ returnEmpty: true })
    mocks.invalidationListener?.("shared-token")
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/fresh" })
    ])

    mocks.invalidationListener?.("shared-token")
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/fresh" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)

    mocks.invalidationListener?.("next-token")
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/newer" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(3)

    mocks.invalidationListener?.("shared-token")
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/newer" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(3)
  })

  it("subscribes to invalidations in an extension-like background context", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([
        { id: "llama/old", name: "Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "llama/fresh", name: "Fresh", provider: "llama", type: "chat" }
      ])

    const currentWindow = window
    vi.stubGlobal("window", undefined)
    try {
      const { fetchChatModels } = await importService()
      expect(mocks.subscribeInvalidation).toHaveBeenCalledTimes(1)

      await fetchChatModels({ returnEmpty: true })
      mocks.invalidationListener?.("background-token")
      await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
        expect.objectContaining({ model: "tldw:llama/fresh" })
      ])
    } finally {
      vi.stubGlobal("window", currentWindow)
    }
  })

  it("clears cached chat models when tldw settings update", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([
        {
          id: "openai/old-model",
          name: "Old Model",
          provider: "openai",
          type: "chat"
        }
      ])
      .mockResolvedValueOnce([
        {
          id: "openai/new-model",
          name: "New Model",
          provider: "openai",
          type: "chat"
        }
      ])

    const { fetchChatModels } = await importService()

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:openai/old-model" })
    ])

    window.dispatchEvent(new CustomEvent("tldw:config-updated"))

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:openai/new-model" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
    expect(mocks.clearModelsCache).toHaveBeenCalledTimes(1)
  })

  it("refetches warmed model caches after a successful provider save", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([
        { id: "llama/old", name: "Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "llama/new", name: "New", provider: "llama", type: "chat" }
      ])
    mocks.bgRequest.mockResolvedValueOnce({
      provider_key: "llama",
      status: "saved"
    })

    const { fetchChatModels } = await importService()
    const { setupOnboardingMethods } = await import(
      "@/services/tldw/domains/setup-onboarding"
    )

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/old" })
    ])
    await setupOnboardingMethods.saveSetupProvider.call(
      {},
      { provider_key: "llama", base_url: "http://192.168.2.216:18080/v1" }
    )
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/new" })
    ])

    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
    expect(mocks.clearModelsCache).toHaveBeenCalledTimes(1)
  })

  it("keeps warmed caches after a failed provider save", async () => {
    mocks.getChatModels.mockResolvedValueOnce([
      { id: "llama/current", name: "Current", provider: "llama", type: "chat" }
    ])
    mocks.bgRequest.mockResolvedValueOnce({
      provider_key: "llama",
      status: "failed"
    })

    const { fetchChatModels } = await importService()
    const { setupOnboardingMethods } = await import(
      "@/services/tldw/domains/setup-onboarding"
    )

    await fetchChatModels({ returnEmpty: true })
    await setupOnboardingMethods.saveSetupProvider.call(
      {},
      { provider_key: "llama", base_url: "http://192.168.2.216:18080/v1" }
    )
    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/current" })
    ])

    expect(mocks.getChatModels).toHaveBeenCalledTimes(1)
    expect(mocks.clearModelsCache).not.toHaveBeenCalled()
  })

  it("does not let a pre-update fetch overwrite or release the post-update fetch", async () => {
    const stale = deferred<Array<Record<string, unknown>>>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getChatModels
      .mockImplementationOnce(() => stale.promise)
      .mockImplementationOnce(() => fresh.promise)

    const { fetchChatModels } = await importService()

    const preUpdate = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledTimes(1))

    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    const postUpdate = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledTimes(2))

    stale.resolve([
      { id: "llama/stale", name: "Stale", provider: "llama", type: "chat" }
    ])
    await expect(preUpdate).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/stale" })
    ])

    const postUpdateFollower = fetchChatModels({ returnEmpty: true })
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)

    fresh.resolve([
      { id: "llama/fresh", name: "Fresh", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([postUpdate, postUpdateFollower])).resolves.toEqual([
      [expect.objectContaining({ model: "tldw:llama/fresh" })],
      [expect.objectContaining({ model: "tldw:llama/fresh" })]
    ])

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/fresh" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
  })
})
