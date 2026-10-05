import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getModels: vi.fn(),
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
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
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
    vi.unstubAllEnvs()
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.getModels.mockReset()
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

  it.each(["expired", "cache-only", "fetch-failure"])("withholds cookie models in the outer %s cache path", async (path) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.getChatModels.mockResolvedValueOnce([
      { id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }
    ])
    const { fetchChatModels } = await importService()
    expect(await fetchChatModels()).toHaveLength(1)
    if (path !== "cache-only") {
      mocks.getChatModels.mockRejectedValueOnce(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    } else {
      mocks.getChatModels.mockResolvedValueOnce([])
    }
    await expect(fetchChatModels({
      returnEmpty: true,
      allowNetwork: path !== "cache-only",
      forceRefresh: path === "fetch-failure"
    })).resolves.toEqual([])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(path === "cache-only" ? 1 : 2)
  })

  it.each([false, true])("does not share in-flight requests across cookie and key auth (cookie first=%s)", async (cookieFirst) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    const keyConfig = { serverUrl: "http://localhost:3000", authMode: "single-user", apiKey: "test-key" }
    const cookieConfig = { serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" }
    mocks.getConfig.mockResolvedValue(cookieFirst ? cookieConfig : keyConfig)
    const pending = deferred<Array<Record<string, unknown>>>()
    mocks.getChatModels.mockImplementationOnce(() => pending.promise).mockResolvedValueOnce(
      cookieFirst ? [{ id: "key-model", name: "Key Model", provider: "llama", type: "chat" }] : []
    )
    const { fetchChatModels } = await importService()
    const first = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledOnce())
    mocks.getConfig.mockResolvedValue(cookieFirst ? keyConfig : cookieConfig)
    const second = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(2))
    pending.resolve(cookieFirst ? [] : [{ id: "key-model", name: "Key Model", provider: "llama", type: "chat" }])
    await first
    await expect(second).resolves.toEqual(cookieFirst ? [expect.objectContaining({ model: "tldw:key-model" })] : [])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
  })

  it("shares simultaneous cookie requests with the same authentication mode", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    const pending = deferred<Array<Record<string, unknown>>>()
    mocks.getChatModels.mockImplementation(() => pending.promise)
    const { fetchChatModels } = await importService()
    const first = fetchChatModels()
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledOnce())
    const second = fetchChatModels()
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(2))
    pending.resolve([{ id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }])
    const [a, b] = await Promise.all([first, second])
    expect(a).toEqual(b)
    expect(a).toHaveLength(1)
    expect(mocks.getChatModels).toHaveBeenCalledOnce()
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

  it("does not fall back to a warmed wrapper catalog when discovery throws", async () => {
    mocks.getChatModels
      .mockResolvedValueOnce([{ id: "retired-model", provider: "openai", type: "chat" }])
      .mockRejectedValueOnce(new Error("Discovery unavailable"))
    const { fetchChatModels } = await importService()
    expect(await fetchChatModels()).toHaveLength(1)

    await expect(fetchChatModels({ forceRefresh: true, returnEmpty: true })).resolves.toEqual([])
    await expect(fetchChatModels({ allowNetwork: false })).resolves.toEqual([])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
  })

  it.each(["success", "empty", "failure"])("joins the post-update fetch after a pre-update %s without overwriting or releasing it", async (outcome) => {
    const stale = deferred<Array<Record<string, unknown>> | Error>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getChatModels
      .mockImplementationOnce(async () => {
        const result = await stale.promise
        if (result instanceof Error) throw result
        return result
      })
      .mockImplementationOnce(() => fresh.promise)

    const { fetchChatModels } = await importService()

    const preUpdate = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledTimes(1))

    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    const postUpdate = fetchChatModels({ returnEmpty: true })
    await vi.waitFor(() => expect(mocks.getChatModels).toHaveBeenCalledTimes(2))

    stale.resolve(outcome === "failure" ? new Error("Older discovery failed") : outcome === "empty" ? [] : [
      { id: "llama/stale", name: "Stale", provider: "llama", type: "chat" }
    ])

    const postUpdateFollower = fetchChatModels({ returnEmpty: true })
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)

    fresh.resolve([
      { id: "llama/fresh", name: "Fresh", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([preUpdate, postUpdate, postUpdateFollower])).resolves.toEqual([
      [expect.objectContaining({ model: "tldw:llama/fresh" })],
      [expect.objectContaining({ model: "tldw:llama/fresh" })],
      [expect.objectContaining({ model: "tldw:llama/fresh" })]
    ])

    await expect(fetchChatModels({ returnEmpty: true })).resolves.toEqual([
      expect.objectContaining({ model: "tldw:llama/fresh" })
    ])
    expect(mocks.getChatModels).toHaveBeenCalledTimes(2)
  })

  describe("with the real model cache", () => {
    beforeEach(async () => {
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      vi.spyOn(tldwClient, "getConfig").mockImplementation(mocks.getConfig)
      vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
      vi.spyOn(tldwClient, "getModels").mockImplementation(mocks.getModels)
      const { TldwModelsService } = await import("@/services/tldw/TldwModels")
      const models = new TldwModelsService()
      mocks.getChatModels.mockImplementation(models.getChatModels)
      mocks.getCachedChatModels.mockImplementation(models.getCachedChatModels)
      mocks.clearModelsCache.mockImplementation(() => models.clearCache())
      mocks.subscribeInvalidation.mockImplementation((listener) => models.subscribeInvalidation(listener))
    })

    it.each([
      ["Error", false], ["AbortError", false], ["TypeError", false],
      ["Error", true], ["AbortError", true], ["TypeError", true]
    ] as const)(
      "settles a %s failure tombstone without recursive discovery (warmed: %s)",
      async (name, warmed) => {
        if (warmed) mocks.getModels.mockResolvedValueOnce([
          { id: "retired-model", name: "Retired Model", provider: "openai", type: "chat" }
        ])
        const error = Object.assign(new Error("Discovery unavailable"), { name })
        mocks.getModels
          .mockRejectedValueOnce(error)
          .mockRejectedValueOnce(error)
          .mockRejectedValueOnce(error)
          .mockResolvedValue([
            { id: "unexpected-retry", name: "Unexpected Retry", provider: "openai", type: "chat" }
          ])
        const { fetchChatModels } = await importService()
        if (warmed) expect(await fetchChatModels()).toHaveLength(1)

        await expect(fetchChatModels({ forceRefresh: true })).resolves.toEqual([])
        await expect(fetchChatModels({ allowNetwork: false })).resolves.toEqual([])
        expect(mocks.getModels).toHaveBeenCalledTimes(warmed ? 2 : 1)
      }
    )

    it("replaces a warmed wrapper catalog after a successful empty discovery", async () => {
      mocks.getModels
        .mockResolvedValueOnce([
          { id: "retired-model", name: "Retired Model", provider: "openai", type: "chat" }
        ])
        .mockResolvedValue([])
      const { fetchChatModels } = await importService()
      await expect(fetchChatModels()).resolves.toEqual([
        expect.objectContaining({ model: "tldw:retired-model" })
      ])

      await expect(fetchChatModels({ forceRefresh: true })).resolves.toEqual([])
      await expect(fetchChatModels()).resolves.toEqual([])
      await expect(fetchChatModels({ allowNetwork: false })).resolves.toEqual([])
      expect(mocks.getModels).toHaveBeenCalledTimes(2)
    })

    it("withholds an expired wrapper and inner catalog when network access is disabled", async () => {
      const now = Date.now()
      const clock = vi.spyOn(Date, "now").mockReturnValue(now)
      mocks.getModels.mockResolvedValue([
        { id: "retired-model", name: "Retired Model", provider: "openai", type: "chat" }
      ])
      const { fetchChatModels } = await importService()
      expect(await fetchChatModels()).toHaveLength(1)
      clock.mockReturnValue(now + 5 * 60 * 1000)

      await expect(fetchChatModels({ allowNetwork: false })).resolves.toEqual([])
      expect(mocks.getModels).toHaveBeenCalledTimes(1)
    })
  })
})
