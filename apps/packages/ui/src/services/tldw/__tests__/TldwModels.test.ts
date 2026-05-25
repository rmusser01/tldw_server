import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
  storageGet: vi.fn(async () => null),
  storageSet: vi.fn(async () => undefined)
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
      (mocks.storageSet as (...args: unknown[]) => unknown)(...args)
  })
}))

const importService = async () => import("@/services/tldw/TldwModels")

describe("TldwModelsService caching", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.initialize.mockReset()
    mocks.getModels.mockReset()
    mocks.storageGet.mockReset()
    mocks.storageSet.mockReset()

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.initialize.mockResolvedValue(undefined)
    mocks.storageGet.mockResolvedValue(null)
    mocks.storageSet.mockResolvedValue(undefined)
  })

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

  it("retries aborted model metadata requests before returning an empty model list", async () => {
    const abortError = Object.assign(
      new Error("signal is aborted without reason"),
      {
        name: "AbortError",
        code: "REQUEST_ABORTED",
        status: 0
      }
    )
    mocks.getModels
      .mockRejectedValueOnce(abortError)
      .mockResolvedValueOnce([
        { id: "gpt-4o", name: "gpt-4o", provider: "openai", type: "chat" }
      ])

    const { TldwModelsService } = await importService()
    const service = new TldwModelsService()

    await expect(service.getModels(true)).resolves.toEqual([
      expect.objectContaining({
        id: "gpt-4o",
        provider: "openai",
        type: "chat"
      })
    ])
    expect(mocks.getModels).toHaveBeenCalledTimes(2)
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
      version: 3,
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
