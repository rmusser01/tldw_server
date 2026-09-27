import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  storageGet: vi.fn(async () => null),
  storageSet: vi.fn(async () => undefined),
  storageValue: null as unknown,
  storageWatchers: new Set<
    (change: { oldValue?: unknown; newValue?: unknown }) => void
  >()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (mocks.getConfig as (...args: unknown[]) => unknown)(...args),
    initialize: (...args: unknown[]) =>
      (mocks.initialize as (...args: unknown[]) => unknown)(...args),
    getModels: (...args: unknown[]) =>
      (mocks.getModels as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/utils/safe-storage", () => ({
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
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

describe("TldwModelsService caching", () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.initialize.mockReset()
    mocks.getModels.mockReset()
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
        scope: "http://127.0.0.1:8000|single-user|key|none"
      })
    )
  })

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

  it("dedupes concurrent in-flight model fetches", async () => {
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

  it("ignores legacy cache entries without schema version and refetches models", async () => {
    mocks.storageGet.mockResolvedValue({
      timestamp: Date.now(),
      scope: "http://127.0.0.1:8000|single-user|key|none",
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
      scope: "http://127.0.0.1:8000|single-user|key|none",
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
    await expect(preClear).resolves.toEqual([
      expect.objectContaining({ id: "stale-model" })
    ])

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
    const stale = deferred<Array<Record<string, unknown>>>()
    const fresh = deferred<Array<Record<string, unknown>>>()
    mocks.getConfig
      .mockImplementationOnce(() => configGate.promise)
      .mockResolvedValue(config)
    mocks.getModels
      .mockImplementationOnce(() => stale.promise)
      .mockImplementationOnce(() => fresh.promise)

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    const preClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1))
    await service.clearCache()

    configGate.resolve(config)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(1))
    const postClear = service.getModels(true)
    await vi.waitFor(() => expect(mocks.getModels).toHaveBeenCalledTimes(2))

    stale.resolve([
      { id: "stale-model", name: "Stale Model", provider: "llama", type: "chat" }
    ])
    fresh.resolve([
      { id: "fresh-model", name: "Fresh Model", provider: "llama", type: "chat" }
    ])

    await expect(preClear).resolves.toEqual([
      expect.objectContaining({ id: "stale-model" })
    ])
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
      [expect.objectContaining({ id: "fresh-model" })],
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
        scope: "http://new-server:8000|single-user|key|none",
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
      scope: "http://127.0.0.1:8000|single-user|key|none",
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
