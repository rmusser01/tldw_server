import { beforeEach, describe, expect, it, vi } from "vitest"
import { derivePromptAssistAuthorizationRevision } from "@/services/chat-surface-scope"

const cacheScope = (url: string, apiKey = "test-key") => `${url}|single-user|key|none|manual|${derivePromptAssistAuthorizationRevision({ authMode: "single-user", apiKey })}`

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
  getCurrentUserProfile: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  storageGet: vi.fn(async () => null),
  storageSet: vi.fn(async () => undefined),
  storageValue: null as unknown,
  storageWatchers: new Set<
    (change: { oldValue?: unknown; newValue?: unknown }) => void
  >()
}))

vi.mock("@/services/tldw/TldwApiClient", async (importOriginal) => ({
  isActiveCookieSessionConfig: (
    await importOriginal<typeof import("@/services/tldw/TldwApiClient")>()
  ).isActiveCookieSessionConfig,
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

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    get: (...args: unknown[]) =>
      (mocks.storageGet as (...args: unknown[]) => unknown)(...args),
    set: (...args: unknown[]) =>
      (mocks.storageSet as (...args: unknown[]) => unknown)(...args),
    watch: (
      callbacks: Record<
        string,
        (change: { oldValue?: unknown; newValue?: unknown }) => void
      >
    ) => {
      const callback = callbacks.tldwModelsCache
      if (!callback) return () => undefined
      mocks.storageWatchers.add(callback)
      return () => mocks.storageWatchers.delete(callback)
    }
  })
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: (...args: unknown[]) =>
    (mocks.getRuntimeSingleUserApiKeyOverride as (...args: unknown[]) => unknown)(...args)
}))

const importService = async () => import("@/services/tldw/TldwModels")

const deferred = <T>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((next, fail) => {
    resolve = next
    reject = fail
  })
  return { promise, resolve, reject }
}

describe("TldwModelsService caching", () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.initialize.mockReset()
    mocks.getModels.mockReset()
    mocks.getCurrentUserProfile.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReset()
    mocks.storageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageValue = null
    mocks.storageWatchers.clear()

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getCurrentUserProfile.mockResolvedValue({ id: "user-1" })
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(null)
    mocks.storageGet.mockImplementation(async () => mocks.storageValue)
    mocks.storageSet.mockImplementation(async (_key, value) => {
      const oldValue = mocks.storageValue
      mocks.storageValue = value
      mocks.storageWatchers.forEach((callback) =>
        callback({ oldValue, newValue: value })
      )
    })
  })

  it.each(["advisory", "strict", "cached"])("retires a %s caller invalidated during config lookup before borrowing a new catalog", async mode => {
    const config = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "old-key" };
    const gate = deferred<typeof config>();
    mocks.getConfig.mockReturnValueOnce(gate.promise).mockResolvedValue({ ...config, apiKey: "new-key" });
    mocks.getModels.mockResolvedValue([{ id: "new-owner", name: "New", provider: "llama", type: "chat" }]);
    const { TldwModelsService } = await importService();
    const service = new TldwModelsService();
    const retired = mode === "cached" ? service.getCachedChatModels() : service.getModels(false, { throwOnError: mode === "strict" });
    const assertion = mode === "strict" ? expect(retired).rejects.toMatchObject({ status: 412 }) : expect(retired).resolves.toEqual([]);
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1));
    await service.clearCache();
    await expect(service.getModels()).resolves.toEqual([expect.objectContaining({ id: "new-owner" })]);
    gate.resolve(config);
    await assertion;
    expect(mocks.getModels).toHaveBeenCalledTimes(1);
  })

  it("keeps a cookie discovery joinable after authenticating its principal", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    mocks.getCurrentUserProfile.mockResolvedValue({ id: "current-principal" })
    const held = deferred<Array<{ id: string; name: string; provider: string; type?: string }>>()
    mocks.getModels.mockReturnValueOnce(held.promise).mockResolvedValueOnce([])
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const strict = service.getModels(true, { requireFresh: true, throwOnError: true })
    const assertion = expect(strict).resolves.toEqual([expect.objectContaining({ id: "current" })])
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    const advisory = service.getModels(true, { requireFresh: true })
    await Promise.resolve(); await Promise.resolve()
    held.resolve([{ id: "current", name: "Current", provider: "llama" }])
    await assertion
    expect((await advisory)[0]?.id).toBe("current")
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("shares mixed fresh discovery without converting a strict success to an empty catalog", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const held = deferred<Array<{ id: string; name: string; provider: string; type?: string }>>()
    mocks.getModels.mockReturnValueOnce(held.promise).mockResolvedValueOnce([])
    const strict = service.getModels(true, { requireFresh: true, throwOnError: true })
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    const advisory = service.getModels(true, { requireFresh: true })
    await Promise.resolve(); await Promise.resolve()
    held.resolve([{ id: "current", name: "Current", provider: "llama" }])
    expect((await strict)[0]?.id).toBe("current")
    expect((await advisory)[0]?.id).toBe("current")
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("preserves each caller's failure policy when discovery is shared", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const held = deferred<Array<{ id: string; name: string; provider: string; type?: string }>>()
    mocks.getModels.mockReturnValueOnce(held.promise)
    const strict = service.getModels(true, { requireFresh: true, throwOnError: true })
    const strictAssertion = expect(strict).rejects.toThrow("Metadata unavailable")
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    const advisory = service.getModels()
    const advisoryAssertion = expect(advisory).resolves.toEqual([])
    await Promise.resolve(); await Promise.resolve()
    held.reject(new Error("Metadata unavailable"))
    await Promise.all([strictAssertion, advisoryAssertion])
  })

  it("never lends a new owner's catalog to a retired failing direct request", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const held = deferred<Array<{ id: string; name: string; provider: string; type?: string }>>()
    mocks.getModels.mockReturnValueOnce(held.promise)
      .mockResolvedValueOnce([{ id: "new-owner", name: "New", provider: "llama" }])
    const old = service.getModels()
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    await service.clearCache()
    await service.getModels()
    held.reject(new Error("Old request failed"))
    await expect(old).resolves.toEqual([])
  })

  it("does not reuse the same-server cache after credentials change without an event", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    mocks.getModels.mockResolvedValueOnce([{ id: "account-a", name: "A", provider: "llama" }])
      .mockResolvedValueOnce([{ id: "account-b", name: "B", provider: "llama" }])
    await service.getModels()
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "other-key" })
    expect((await service.getModels())[0].id).toBe("account-b")
  })

  it("does not offer a previous cookie principal catalog after the new principal's discovery fails", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    mocks.getCurrentUserProfile.mockResolvedValueOnce({ id: "a" }).mockResolvedValueOnce({ id: "b" })
    mocks.getModels.mockResolvedValueOnce([{ id: "account-a", name: "A", provider: "llama" }])
      .mockRejectedValueOnce(new Error("Metadata unavailable"))
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    await service.getModels()
    await expect(service.getModels()).resolves.toEqual([])
  })

  it("reports failed authoritative discovery without substituting the offline catalog", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    mocks.getModels.mockResolvedValueOnce([{ id: "cached", name: "Cached", provider: "llama", type: "chat" }])
    await service.getModels()
    const failure = new Error("Metadata unavailable")
    mocks.getModels.mockRejectedValueOnce(failure)
    await expect(service.getModels(true, { requireFresh: true, throwOnError: true })).rejects.toBe(failure)
    expect((await service.getModels())[0].id).toBe("cached")
  })

  it("distinguishes an authoritative empty catalog from a failed read", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    mocks.getModels.mockResolvedValueOnce([])
    await expect(service.getModels(true, { requireFresh: true, throwOnError: true })).resolves.toEqual([])
    mocks.getModels.mockRejectedValueOnce(new Error("Unavailable"))
    await expect(service.getModels(true, { requireFresh: true, throwOnError: true })).rejects.toThrow("Unavailable")
  })

  it("does not return a retired discovery to a direct caller", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const held = deferred<Array<{ id: string; name: string; provider: string; type?: string }>>()
    mocks.getModels.mockReturnValueOnce(held.promise)
    const read = service.getModels()
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    await service.clearCache()
    held.resolve([{ id: "retired", name: "Retired", provider: "llama", type: "chat" }])
    await expect(read).resolves.toEqual([])
  })

  it("fetches models when single-user auth is provided by the WebUI runtime", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue("runtime-key")
    mocks.getModels.mockResolvedValue([
      { id: "local-model", name: "Local Model", provider: "llama", type: "chat" }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const models = await service.getModels()

    expect(models.map((model) => model.id)).toEqual(["local-model"])
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    expect(mocks.storageSet).toHaveBeenCalledWith(
      "tldwModelsCache",
      expect.objectContaining({
        scope: cacheScope("http://127.0.0.1:8000", "runtime-key")
      })
    )
  })

  it.each([
    ["quickstart", "http://localhost:3000", "single-user", "cookie-session", true],
    ["quickstart", "http://127.0.0.1:8000", "single-user", "cookie-session", false],
    ["advanced", "http://localhost:3000", "single-user", "cookie-session", false],
    ["quickstart", "http://localhost:3000", "multi-user", "cookie-session", false],
    ["quickstart", "http://localhost:3000", "single-user", undefined, false]
  ])(
    "admits models only for an active cookie session (%s, %s, %s, %s)",
    async (deploymentMode, serverUrl, authMode, authSource, allowed) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", deploymentMode)
      vi.stubEnv("NEXT_PUBLIC_API_URL", "")
      mocks.getConfig.mockResolvedValue({ serverUrl, authMode, authSource })
      mocks.getModels.mockResolvedValue([
        { id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }
      ])
      const { TldwModelsService } = await importService()
      const models = await new TldwModelsService().getModels()
      expect(models.map((model) => model.id)).toEqual(allowed ? ["cookie-model"] : [])
      expect(mocks.getModels).toHaveBeenCalledTimes(allowed ? 1 : 0)
      if (allowed) {
        expect(mocks.storageSet).not.toHaveBeenCalled()
      }
    }
  )

  it.each(["CHANGE_ME_TO_SECURE_API_KEY", "   "])(
    "does not treat invalid runtime single-user auth %s as model-ready",
    async (runtimeKey) => {
      mocks.getConfig.mockResolvedValue({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user"
      })
      mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(runtimeKey)
      mocks.getModels.mockResolvedValue([
        { id: "local-model", name: "Local Model", provider: "llama", type: "chat" }
      ])

      const { TldwModelsService } = await importService()
      const service = new TldwModelsService()

      const models = await service.getModels()

      expect(models).toEqual([])
      expect(mocks.getModels).not.toHaveBeenCalled()
      expect(mocks.storageSet).not.toHaveBeenCalled()
    }
  )

  it.each([
    [0, false, 401],
    [16 * 60 * 1000, false, 401],
    [0, true, 403]
  ])("withholds cookie catalogs after authentication fails (age=%s, forced=%s, status=%s)", async (age, forced, status) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.storageValue = {
      version: 4,
      timestamp: Date.now() - age,
      scope: "http://localhost:3000|single-user|cookie|none",
      models: [{ id: "expired-cookie-model", name: "Old Model", provider: "llama", type: "chat" }]
    }
    mocks.getModels.mockRejectedValue(Object.assign(new Error("Expired cookie session"), { status }))
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    await expect(service.getModels(forced)).resolves.toEqual([])
    expect(mocks.getModels).toHaveBeenCalledOnce()
    expect(mocks.storageSet).not.toHaveBeenCalled()
    await expect(new TldwModelsService().getCachedChatModels()).resolves.toEqual([])
  })

  it("revalidates a cookie session instead of returning the forced-refresh cooldown cache", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.getModels.mockResolvedValueOnce([
      { id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }
    ]).mockRejectedValueOnce(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    expect(await service.getModels(true)).toHaveLength(1)
    await expect(service.getModels(true)).resolves.toEqual([])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
  })

  it("requires live cookie authentication even when public model metadata is available", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.getCurrentUserProfile.mockRejectedValue(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    mocks.getModels.mockResolvedValue([
      { id: "public-model", name: "Public Model", provider: "llama", type: "chat" }
    ])
    const { TldwModelsService } = await importService()
    await expect(new TldwModelsService().getModels()).resolves.toEqual([])
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledOnce()
    expect(mocks.getModels).not.toHaveBeenCalled()
  })

  it("does not hydrate a cookie catalog without a live authenticated request", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.storageValue = {
      version: 4,
      timestamp: Date.now(),
      scope: "http://localhost:3000|single-user|cookie|none",
      models: [{ id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }]
    }
    const { TldwModelsService } = await importService()
    await expect(new TldwModelsService().getCachedChatModels()).resolves.toEqual([])
    expect(mocks.getModels).not.toHaveBeenCalled()
  })

  it.each([false, true])("dedupes concurrent in-flight model fetches (cookie=%s)", async (cookie) => {
    if (cookie) {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      vi.stubEnv("NEXT_PUBLIC_API_URL", "")
      mocks.getConfig.mockResolvedValue({
        serverUrl: "http://localhost:3000",
        authMode: "single-user",
        authSource: "cookie-session"
      })
    }
    vi.useFakeTimers()
    try {
      mocks.getModels.mockImplementation(async () => {
        await new Promise((resolve) => setTimeout(resolve, 25))
        return [
          { id: "model-a", name: "Model A", provider: "openai", type: "chat" }
        ]
      })

      const { TldwModelsService } = await importService()
      const service = new TldwModelsService()

      const first = service.getModels(true)
      const second = service.getModels(true)

      await vi.advanceTimersByTimeAsync(26)

      const [a, b] = await Promise.all([first, second])

      expect(a).toHaveLength(1)
      expect(b).toHaveLength(1)
      expect(mocks.getModels).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  }, 10_000)

  it("does not let a late cookie authentication failure clear a newer key catalog", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "cookie-session"
    })
    const failure = deferred<Error>()
    mocks.getModels
      .mockImplementationOnce(async () => { throw await failure.promise })
      .mockResolvedValueOnce([
        { id: "new-key-model", name: "New Model", provider: "llama", type: "chat" }
      ])
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const oldCookieFetch = service.getModels()
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledOnce())

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://key-server:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    await service.getModels()
    failure.resolve(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    await expect(oldCookieFetch).resolves.toEqual([])
    await expect(service.getModels()).resolves.toEqual([
      expect.objectContaining({ id: "new-key-model" })
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
    expect(mocks.storageValue).toEqual(expect.objectContaining({
      scope: cacheScope("http://key-server:8000"),
      models: [expect.objectContaining({ id: "new-key-model" })]
    }))
  })

  it("does not let a cookie failure overwrite another instance's persisted key catalog", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    const failure = deferred<Error>()
    mocks.getCurrentUserProfile.mockImplementationOnce(async () => { throw await failure.promise })
    mocks.getModels.mockResolvedValue([{ id: "key-model", name: "Key Model", provider: "llama", type: "chat" }])
    const { TldwModelsService } = await importService()
    const oldCookieFetch = new TldwModelsService().getModels()
    await vi.waitFor(() => expect(mocks.getCurrentUserProfile).toHaveBeenCalledOnce())
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://key-server:8000", authMode: "single-user", apiKey: "test-key" })
    await new TldwModelsService().getModels()
    const keyRecord = mocks.storageValue
    failure.resolve(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    await expect(oldCookieFetch).resolves.toEqual([])
    expect(mocks.storageValue).toEqual(keyRecord)
    await expect(new TldwModelsService().getCachedChatModels()).resolves.toEqual([expect.objectContaining({ id: "key-model" })])
  })

  it.each(["AbortError", "TypeError"])("retains cookie models on a catalog %s only after live authentication", async (name) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    mocks.getModels.mockResolvedValueOnce([{ id: "cookie-model", name: "Cookie Model", provider: "llama", type: "chat" }])
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    await service.getModels()
    const transient = Object.assign(new Error("Request interrupted"), { name })
    mocks.getCurrentUserProfile.mockRejectedValueOnce(transient)
    await expect(service.getModels()).resolves.toEqual([])
    mocks.getModels.mockRejectedValueOnce(transient)
    await expect(service.getModels()).resolves.toEqual([expect.objectContaining({ id: "cookie-model" })])
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(3)
    expect(mocks.storageSet).not.toHaveBeenCalled()
  })

  it.each(["AbortError", "TypeError"])("withholds stale fresh cookie capabilities after a catalog %s without discarding authenticated memory", async (name) => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    mocks.getModels.mockResolvedValueOnce([{ id: "cookie-model", name: "Cookie Model", provider: "llama", vision: false }])
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    await service.getModels()
    const transient = Object.assign(new Error("Request interrupted"), { name })
    mocks.getModels.mockRejectedValue(transient)
    await expect(service.getModels(true, { requireFresh: true })).resolves.toEqual([])
    await expect(service.getModels()).resolves.toEqual([expect.objectContaining({ id: "cookie-model" })])
    expect(mocks.getCurrentUserProfile).toHaveBeenCalledTimes(3)
    expect(mocks.storageSet).not.toHaveBeenCalled()
  })

  it("clears expired cookie memory even when fresh capability discovery was requested", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "")
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "cookie-session" })
    mocks.getModels.mockResolvedValueOnce([{ id: "cookie-model", name: "Cookie Model", provider: "llama", vision: true }])
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    await service.getModels()
    mocks.getCurrentUserProfile.mockRejectedValueOnce(Object.assign(new Error("Expired cookie session"), { status: 401 }))
    await expect(service.getModels(true, { requireFresh: true })).resolves.toEqual([])
    mocks.getModels.mockRejectedValueOnce(new TypeError("Network unavailable"))
    await expect(service.getModels()).resolves.toEqual([])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
    expect(mocks.storageSet).not.toHaveBeenCalled()
  })

  it("resets cached models when server scope changes", async () => {
    mocks.getModels
      .mockResolvedValueOnce([
        { id: "model-a", name: "Model A", provider: "openai", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "model-b", name: "Model B", provider: "anthropic", type: "chat" }
      ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    await service.getModels(true)

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8100",
      authMode: "single-user",
      apiKey: "test-key"
    })

    const next = await service.getModels()

    expect(next[0]?.id).toBe("model-b")
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
  })

  it("forwards refreshOpenRouter flag when explicitly requested", async () => {
    mocks.getModels.mockResolvedValue([
      { id: "openrouter/model-a", name: "Model A", provider: "openrouter", type: "chat" }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    await service.getModels(true, { refreshOpenRouter: true })

    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    expect(mocks.getModels).toHaveBeenCalledWith({ refreshOpenRouter: true })
  })

  it("resolves in-flight model metadata failures through the fallback path", async () => {
    mocks.getModels.mockRejectedValueOnce(
      new Error("Failed to fetch (GET /api/v1/llm/models/metadata)")
    )

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const first = service.getModels(true)
    const second = service.getModels(true)

    await expect(Promise.all([first, second])).resolves.toEqual([[], []])
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("does not retry or log model metadata requests aborted by page lifecycle", async () => {
    const abortError = Object.assign(
      new Error("signal is aborted without reason"),
      {
        name: "AbortError",
        code: "REQUEST_ABORTED",
        status: 0
      }
    )
    mocks.getModels.mockRejectedValue(abortError)
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)

    try {
      const { TldwModelsService } = await importService()
      const service = new TldwModelsService()

      await expect(service.getModels(true)).resolves.toEqual([])
      expect(mocks.getModels).toHaveBeenCalledTimes(1)
      expect(consoleError).not.toHaveBeenCalled()
    } finally {
      consoleError.mockRestore()
    }
  })

  it("reuses cached models during the forced refresh cooldown", async () => {
    mocks.getModels.mockResolvedValue([
      { id: "openrouter/model-a", name: "Model A", provider: "openrouter", type: "chat" }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    await service.getModels(true, { refreshOpenRouter: true })
    await service.getModels(true, { refreshOpenRouter: true })

    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("refreshes capability data without waiting for cooldown or clearing durable records on failure", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    mocks.getModels.mockResolvedValue([{ id: "model-a", name: "Model A", provider: "llama", vision: false }])
    await service.getModels(true)
    const durable = structuredClone(mocks.storageValue)
    mocks.getModels.mockRejectedValue(new Error("Discovery unavailable"))
    expect(await service.getModels(true, { requireFresh: true })).toEqual([])
    expect(mocks.storageValue).toEqual(durable)
    mocks.getModels.mockResolvedValue([{ id: "model-a", name: "Model A", provider: "llama", vision: true }])
    expect((await service.getModels(true, { requireFresh: true }))[0].capabilities).toContain("vision")
  })

  it("coalesces fresh capability requests and prevents an older discovery from replacing them", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const old = deferred<Array<{ id: string; name: string; provider: string; vision: boolean }>>()
    const fresh = deferred<Array<{ id: string; name: string; provider: string; vision: boolean }>>()
    mocks.getModels.mockReturnValueOnce(old.promise).mockReturnValueOnce(fresh.promise)
    const oldRead = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    const first = service.getModels(true, { requireFresh: true })
    const second = service.getModels(true, { requireFresh: true })
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(2))
    fresh.resolve([{ id: "model-a", name: "Model A", provider: "llama", vision: true }])
    const [a, b] = await Promise.all([first, second])
    expect(a[0].capabilities).toContain("vision")
    expect(b).toEqual(a)
    old.resolve([{ id: "model-a", name: "Model A", provider: "llama", vision: false }])
    await oldRead
    expect((await service.getModel("model-a"))?.capabilities).toContain("vision")
  })

  it("does not return a fresh capability result invalidated while the read was pending", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const pending = deferred<Array<{ id: string; name: string; provider: string; vision: boolean }>>()
    mocks.getModels.mockReturnValue(pending.promise)
    const read = service.getModels(true, { requireFresh: true })
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    await service.clearCache()
    pending.resolve([{ id: "model-a", name: "Model A", provider: "llama", vision: true }])
    expect(await read).toEqual([])
    expect(await service.getCachedChatModels()).toEqual([])
  })

  it("does not return fresh capabilities invalidated while their durable write is pending", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const write = deferred<void>()
    mocks.getModels.mockResolvedValue([{ id: "model-a", name: "Model A", provider: "llama", vision: true }])
    mocks.storageSet.mockImplementationOnce(() => write.promise)
    const read = service.getModels(true, { requireFresh: true })
    await vi.waitFor(() => expect(mocks.storageSet).toHaveBeenCalledTimes(1))
    const clear = service.clearCache()
    write.resolve()
    expect(await read).toEqual([])
    await clear
  })

  it("ignores legacy cache entries without schema version and refetches models", async () => {
    mocks.storageGet.mockResolvedValue({
      timestamp: Date.now(),
      scope: cacheScope("http://127.0.0.1:8000"),
      models: [
        {
          id: "z-ai/glm-4.6",
          name: "deepseek/deepseek-r1",
          provider: "openrouter",
          type: "chat"
        }
      ]
    })
    mocks.getModels.mockResolvedValue([
      { id: "deepseek/deepseek-r1", name: "deepseek/deepseek-r1", provider: "openrouter", type: "chat" }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const models = await service.getModels()

    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    expect(models[0]?.id).toBe("deepseek/deepseek-r1")
  })

  it("keeps image-generation models out of chat models", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openrouter"
      },
      {
        id: "black-forest-labs/flux.1-schnell",
        name: "black-forest-labs/flux.1-schnell",
        provider: "openrouter"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)
    const chatIds = chatModels.map((m) => m.id)

    expect(chatIds).toContain("openai/gpt-4o-mini")
    expect(chatIds).not.toContain("black-forest-labs/flux.1-schnell")
  })

  it("keeps chat-model filtering bound when the method is used as a callback", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openai",
        type: "chat"
      },
      {
        id: "black-forest-labs/flux.1-schnell",
        name: "black-forest-labs/flux.1-schnell",
        provider: "openrouter",
        type: "image"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const loadChatModels = service.getChatModels

    const chatModels = await loadChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
  })

  it("filters chat models from explicitly unconfigured providers", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openai",
        type: "chat",
        is_configured: true
      },
      {
        id: "anthropic/claude-sonnet-4",
        name: "anthropic/claude-sonnet-4",
        provider: "anthropic",
        type: "chat",
        is_configured: false
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
  })

  it("filters chat models when provider-level configuration is unavailable", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openai",
        type: "chat",
        provider_is_configured: true
      },
      {
        id: "anthropic/claude-sonnet-4",
        name: "anthropic/claude-sonnet-4",
        provider: "anthropic",
        type: "chat",
        provider_is_configured: false
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
  })

  it("filters chat models from explicitly disabled providers", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openai",
        type: "chat",
        is_configured: true,
        provider_enabled: true
      },
      {
        id: "openrouter/meta-llama/llama-3.1-8b-instruct",
        name: "openrouter/meta-llama/llama-3.1-8b-instruct",
        provider: "openrouter",
        type: "chat",
        is_configured: true,
        provider_enabled: false
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
  })

  it("filters chat models from providers with failed availability", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openai",
        type: "chat",
        is_configured: true,
        availability: "enabled"
      },
      {
        id: "vllm/local-model",
        name: "vllm/local-model",
        provider: "vllm",
        type: "chat",
        is_configured: true,
        availability: "failed"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
  })

  it("preserves readiness details on unavailable models for Studio prerequisites", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "ollama/gemma3:1b",
        name: "gemma3:1b",
        provider: "ollama",
        type: "chat",
        is_configured: true,
        provider_enabled: false,
        availability: "unavailable",
        readiness_reason_code: "egress_blocked",
        readiness_message: "Port not allowed: 11434",
        chat_provider: "ollama"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const models = await service.getModels(true)

    expect(models[0]).toEqual(
      expect.objectContaining({
        id: "ollama/gemma3:1b",
        providerEnabled: false,
        availability: "unavailable",
        readinessReasonCode: "egress_blocked",
        readinessMessage: "Port not allowed: 11434",
        chatProvider: "ollama"
      })
    )
  })

  it("selects the exact enabled llama metadata record and excludes its blocked pair", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "manual-llama.gguf",
        name: "manual-llama.gguf",
        provider: "llama",
        type: "chat",
        is_configured: true,
        provider_is_configured: true,
        provider_enabled: true,
        availability: "enabled",
        chat_provider: "llama.cpp",
        catalog_only: false
      },
      {
        id: "blocked-llama.gguf",
        name: "blocked-llama.gguf",
        provider: "llama",
        type: "chat",
        is_configured: true,
        provider_is_configured: true,
        provider_enabled: false,
        availability: "unavailable",
        readiness_reason_code: "egress_blocked",
        chat_provider: "llama.cpp",
        catalog_only: false
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels).toEqual([
      expect.objectContaining({
        id: "manual-llama.gguf",
        provider: "llama",
        providerEnabled: true,
        availability: "enabled",
        chatProvider: "llama.cpp"
      })
    ])
  })

  it("invalidates model and subscriber caches once across watched storage contexts", async () => {
    const sourceFresh = deferred<Array<Record<string, unknown>>>()
    const tombstoneWrite = deferred<void>()
    mocks.getModels
      .mockResolvedValueOnce([
        { id: "source-old", name: "Source Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "remote-old", name: "Remote Old", provider: "llama", type: "chat" }
      ])
      .mockImplementationOnce(() => sourceFresh.promise)
      .mockResolvedValueOnce([
        { id: "remote-fresh", name: "Remote Fresh", provider: "llama", type: "chat" }
      ])

    const { TldwModelsService } = await importService()
    const source = new TldwModelsService()
    const remote = new TldwModelsService()
    const sourceInvalidations: string[] = []
    const remoteInvalidations: string[] = []
    source.subscribeInvalidation((token) => sourceInvalidations.push(token))
    remote.subscribeInvalidation((token) => remoteInvalidations.push(token))

    await source.getModels(true)
    await remote.getModels(true)

    mocks.storageSet.mockImplementation(async (_key, value) => {
      const record = value as {
        models?: unknown
        invalidationToken?: unknown
      }
      if (
        record.models === null &&
        typeof record.invalidationToken === "string"
      ) {
        await tombstoneWrite.promise
      }
      const oldValue = mocks.storageValue
      mocks.storageValue = value
      mocks.storageWatchers.forEach((callback) =>
        callback({ oldValue, newValue: value })
      )
    })

    const clear = source.clearCache()
    expect(sourceInvalidations).toHaveLength(1)

    const sourceRequest = source.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(3))
    const sourceFollower = source.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(3)

    tombstoneWrite.resolve(undefined)
    await clear

    expect(remoteInvalidations).toEqual(sourceInvalidations)
    expect(sourceInvalidations).toHaveLength(1)
    await expect(source.getCachedChatModels()).resolves.toEqual([])
    await expect(remote.getCachedChatModels()).resolves.toEqual([])

    sourceFresh.resolve([
      { id: "source-fresh", name: "Source Fresh", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([sourceRequest, sourceFollower])).resolves.toEqual([
      [expect.objectContaining({ id: "source-fresh" })],
      [expect.objectContaining({ id: "source-fresh" })]
    ])

    const writes = mocks.storageSet.mock.calls.map((call) => call[1])
    const tombstoneIndex = writes.findIndex((value) => {
      const record = value as {
        models?: unknown
        invalidationToken?: unknown
      }
      return (
        record.models === null &&
        typeof record.invalidationToken === "string"
      )
    })
    const freshIndex = writes.findIndex((value) => {
      const record = value as { models?: unknown }
      return (
        Array.isArray(record.models) &&
        record.models.some(
          (model) =>
            typeof model === "object" &&
            model !== null &&
            "id" in model &&
            model.id === "source-fresh"
        )
      )
    })
    expect(tombstoneIndex).toBeGreaterThanOrEqual(0)
    expect(freshIndex).toBeGreaterThan(tombstoneIndex)

    await expect(remote.getModels(true)).resolves.toEqual([
      expect.objectContaining({ id: "remote-fresh" })
    ])
    expect(sourceInvalidations).toHaveLength(1)
    expect(remoteInvalidations).toHaveLength(1)
  })

  it("isolates invalidation listener failures and still persists the tombstone", async () => {
    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const observedTokens: string[] = []
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    service.subscribeInvalidation(() => {
      throw new Error("listener failed")
    })
    service.subscribeInvalidation((token) => observedTokens.push(token))

    await expect(service.clearCache()).resolves.toBeUndefined()

    expect(observedTokens).toHaveLength(1)
    expect(mocks.storageSet).toHaveBeenCalledWith(
      "tldwModelsCache",
      expect.objectContaining({
        models: null,
        invalidationToken: observedTokens[0]
      })
    )
    expect(consoleError).toHaveBeenCalled()
    consoleError.mockRestore()
  })

  it("ignores a delayed first-clear echo after a second clear owns the cache", async () => {
    const firstTombstoneStarted = deferred<void>()
    const releaseFirstTombstone = deferred<void>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getModels
      .mockResolvedValueOnce([
        { id: "old-model", name: "Old Model", provider: "llama", type: "chat" }
      ])
      .mockImplementationOnce(() => fresh.promise)
      .mockResolvedValueOnce([
        { id: "unexpected-model", name: "Unexpected", provider: "llama", type: "chat" }
      ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const invalidations: string[] = []
    service.subscribeInvalidation((token) => invalidations.push(token))
    await service.getModels(true)

    let tombstoneCount = 0
    mocks.storageSet.mockImplementation(async (_key, value) => {
      const record = value as {
        models?: unknown
        invalidationToken?: unknown
      }
      if (
        record.models === null &&
        typeof record.invalidationToken === "string"
      ) {
        tombstoneCount += 1
        if (tombstoneCount === 1) {
          firstTombstoneStarted.resolve(undefined)
          await releaseFirstTombstone.promise
        }
      }
      const oldValue = mocks.storageValue
      mocks.storageValue = value
      mocks.storageWatchers.forEach((callback) =>
        callback({ oldValue, newValue: value })
      )
    })

    const clearA = service.clearCache()
    await firstTombstoneStarted.promise
    const tokenA = invalidations[0]

    const clearB = service.clearCache()
    const tokenB = invalidations[1]
    expect(tokenA).not.toBe(tokenB)

    const requestB = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(2))

    releaseFirstTombstone.resolve(undefined)
    await Promise.all([clearA, clearB])

    expect(invalidations).toEqual([tokenA, tokenB])
    const followerB = service.getModels(true)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(mocks.getModels).toHaveBeenCalledTimes(2)

    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([requestB, followerB])).resolves.toEqual([
      [expect.objectContaining({ id: "fresh-model" })],
      [expect.objectContaining({ id: "fresh-model" })]
    ])

    const writes = mocks.storageSet.mock.calls.map((call) => call[1])
    const tombstones = writes.filter((value) => {
      const record = value as {
        models?: unknown
        invalidationToken?: unknown
      }
      return (
        record.models === null &&
        typeof record.invalidationToken === "string"
      )
    })
    const freshIndex = writes.findIndex((value) => {
      const record = value as { models?: unknown }
      return (
        Array.isArray(record.models) &&
        record.models.some(
          (model) =>
            typeof model === "object" &&
            model !== null &&
            "id" in model &&
            model.id === "fresh-model"
        )
      )
    })
    expect(tombstones).toHaveLength(2)
    expect(
      (tombstones[1] as { invalidationToken?: unknown }).invalidationToken
    ).toBe(tokenB)
    expect(freshIndex).toBeGreaterThan(writes.indexOf(tombstones[1]))
    await expect(service.getModels()).resolves.toEqual([
      expect.objectContaining({ id: "fresh-model" })
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
  })

  it("skips stale persistent hydration when cleared before the first read", async () => {
    const tombstoneStarted = deferred<void>()
    const releaseTombstone = deferred<void>()
    mocks.storageValue = {
      version: 4,
      timestamp: Date.now(),
      scope: cacheScope("http://127.0.0.1:8000"),
      models: [
        { id: "stale-model", name: "Stale Model", provider: "llama", type: "chat" }
      ]
    }
    mocks.getModels.mockResolvedValueOnce([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])
    mocks.storageSet.mockImplementation(async (_key, value) => {
      const record = value as {
        models?: unknown
        invalidationToken?: unknown
      }
      if (
        record.models === null &&
        typeof record.invalidationToken === "string"
      ) {
        tombstoneStarted.resolve(undefined)
        await releaseTombstone.promise
      }
      const oldValue = mocks.storageValue
      mocks.storageValue = value
      mocks.storageWatchers.forEach((callback) =>
        callback({ oldValue, newValue: value })
      )
    })

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const clear = service.clearCache()
    await tombstoneStarted.promise

    try {
      const models = service.getModels()
      await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
      expect(mocks.storageGet).not.toHaveBeenCalled()

      releaseTombstone.resolve(undefined)
      await expect(models).resolves.toEqual([
        expect.objectContaining({ id: "fresh-model" })
      ])
      await clear
    } finally {
      releaseTombstone.resolve(undefined)
      await clear
    }
  })

  it("invalidates an extension-like background context without window access", async () => {
    mocks.getModels
      .mockResolvedValueOnce([
        { id: "webui-old", name: "WebUI Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "extension-old", name: "Extension Old", provider: "llama", type: "chat" }
      ])
      .mockResolvedValueOnce([
        { id: "extension-fresh", name: "Extension Fresh", provider: "llama", type: "chat" }
      ])

    const { TldwModelsService } = await importService()
    const source = new TldwModelsService()
    await source.getModels(true)

    const currentWindow = window
    vi.stubGlobal("window", undefined)
    try {
      const background = new TldwModelsService()
      const backgroundInvalidations: string[] = []
      background.subscribeInvalidation((token) =>
        backgroundInvalidations.push(token)
      )

      await background.getModels(true)
      await source.clearCache()

      expect(backgroundInvalidations).toHaveLength(1)
      await expect(background.getCachedChatModels()).resolves.toEqual([])
      await expect(background.getModels(true)).resolves.toEqual([
        expect.objectContaining({ id: "extension-fresh" })
      ])
    } finally {
      vi.stubGlobal("window", currentWindow)
    }
  })

  it("applies a startup tombstone without hydrating stale models", async () => {
    mocks.storageValue = {
      version: 4,
      models: null,
      timestamp: 0,
      scope: null,
      invalidationToken: "startup-token"
    }
    mocks.getModels.mockResolvedValueOnce([
      { id: "startup-fresh", name: "Startup Fresh", provider: "llama", type: "chat" }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()
    const invalidations: string[] = []
    service.subscribeInvalidation((token) => invalidations.push(token))

    await expect(service.getModels()).resolves.toEqual([
      expect.objectContaining({ id: "startup-fresh" })
    ])
    expect(invalidations).toEqual(["startup-token"])
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("prevents a pre-clear fetch from repopulating cache or taking post-clear ownership", async () => {
    const stale = deferred<Array<Record<string, unknown>>>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getModels
      .mockImplementationOnce(() => stale.promise)
      .mockImplementationOnce(() => fresh.promise)

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const preClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))

    await service.clearCache()
    const postClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(2))

    stale.resolve([
      { id: "stale-model", name: "Stale Model", provider: "llama", type: "chat" }
    ])
    await expect(preClear).resolves.toEqual([])

    const postClearFollower = service.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(2)

    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([postClear, postClearFollower])).resolves.toEqual([
      [expect.objectContaining({ id: "fresh-model" })],
      [expect.objectContaining({ id: "fresh-model" })]
    ])

    await expect(service.getModels()).resolves.toEqual([
      expect.objectContaining({ id: "fresh-model" })
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
    expect(
      mocks.storageSet.mock.calls.some((call) =>
        JSON.stringify((call as unknown[])[1]).includes("stale-model")
      )
    ).toBe(false)
  })

  it("does not let a fetch invalidated during config lookup own the next generation", async () => {
    const config = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const configGate = deferred<typeof config>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getConfig
      .mockImplementationOnce(() => configGate.promise)
      .mockResolvedValue(config)
    mocks.getModels.mockImplementationOnce(() => fresh.promise)

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const preClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1))
    await service.clearCache()

    configGate.resolve(config)
    await expect(preClear).resolves.toEqual([])
    expect(mocks.getModels).not.toHaveBeenCalled()
    const postClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))

    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])

    await expect(preClear).resolves.toEqual([])
    await expect(postClear).resolves.toEqual([
      expect.objectContaining({ id: "fresh-model" })
    ])
  })

  it("does not let an old-scope fetch invalidate a newer in-flight scope", async () => {
    const oldConfig = {
      serverUrl: "http://old-server:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const newConfig = {
      serverUrl: "http://new-server:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const oldConfigGate = deferred<typeof oldConfig>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getConfig
      .mockImplementationOnce(() => oldConfigGate.promise)
      .mockResolvedValue(newConfig)
    mocks.getModels.mockImplementationOnce(() => fresh.promise)

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const oldScopeFetch = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1))
    await service.clearCache()

    const newScopeFetch = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))

    oldConfigGate.resolve(oldConfig)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(mocks.getModels).toHaveBeenCalledTimes(1)

    const newScopeFollower = service.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)

    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])
    await expect(
      Promise.all([oldScopeFetch, newScopeFetch, newScopeFollower])
    ).resolves.toEqual([
      [],
      [expect.objectContaining({ id: "fresh-model" })],
      [expect.objectContaining({ id: "fresh-model" })]
    ])

    await expect(service.getModels()).resolves.toEqual([
      expect.objectContaining({ id: "fresh-model" })
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
    expect(mocks.storageSet).toHaveBeenLastCalledWith(
      "tldwModelsCache",
      expect.objectContaining({
        models: [expect.objectContaining({ id: "fresh-model" })],
        scope: cacheScope("http://new-server:8000"),
        timestamp: expect.any(Number)
      })
    )
    expect(JSON.stringify(mocks.storageSet.mock.calls)).not.toContain(
      "old-server"
    )
  })

  it("keeps a newer fetch owned when an old cached lookup resolves", async () => {
    const oldConfig = {
      serverUrl: "http://old-server:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const newConfig = {
      serverUrl: "http://new-server:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const oldConfigGate = deferred<typeof oldConfig>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getConfig
      .mockImplementationOnce(() => oldConfigGate.promise)
      .mockResolvedValue(newConfig)
    mocks.getModels.mockImplementationOnce(() => fresh.promise)

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const oldCachedLookup = service.getCachedChatModels()
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1))
    await service.clearCache()

    const newScopeFetch = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))

    oldConfigGate.resolve(oldConfig)
    await expect(oldCachedLookup).resolves.toEqual([])

    const newScopeFollower = service.getModels(true)
    expect(mocks.getModels).toHaveBeenCalledTimes(1)

    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])
    await expect(Promise.all([newScopeFetch, newScopeFollower])).resolves.toEqual([
      [expect.objectContaining({ id: "fresh-model" })],
      [expect.objectContaining({ id: "fresh-model" })]
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(1)
  })

  it("keeps legacy chat models when provider availability metadata is absent", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "legacy/model-a",
        name: "legacy/model-a",
        provider: "custom",
        type: "chat"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getChatModels(true)

    expect(chatModels.map((model) => model.id)).toEqual(["legacy/model-a"])
  })

  it("returns cached chat models without fetching provider metadata again", async () => {
    mocks.storageGet.mockResolvedValue({
      version: 4,
      timestamp: Date.now(),
      scope: cacheScope("http://127.0.0.1:8000"),
      models: [
        {
          id: "openai/gpt-4o-mini",
          name: "openai/gpt-4o-mini",
          provider: "openai",
          type: "chat"
        },
        {
          id: "anthropic/claude-sonnet-4",
          name: "anthropic/claude-sonnet-4",
          provider: "anthropic",
          type: "chat",
          isConfigured: false
        },
        {
          id: "black-forest-labs/flux.1-schnell",
          name: "black-forest-labs/flux.1-schnell",
          provider: "openrouter",
          type: "image"
        }
      ]
    })

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const chatModels = await service.getCachedChatModels()

    expect(chatModels.map((model) => model.id)).toEqual(["openai/gpt-4o-mini"])
    expect(mocks.getModels).not.toHaveBeenCalled()
  })

  it("includes image-generation models in image models", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "openai/gpt-4o-mini",
        name: "openai/gpt-4o-mini",
        provider: "openrouter"
      },
      {
        id: "black-forest-labs/flux.1-schnell",
        name: "black-forest-labs/flux.1-schnell",
        provider: "openrouter"
      }
    ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const imageModels = await service.getImageModels(true)
    const imageIds = imageModels.map((m) => m.id)

    expect(imageIds).toContain("black-forest-labs/flux.1-schnell")
    expect(imageIds).not.toContain("openai/gpt-4o-mini")
  })
})
