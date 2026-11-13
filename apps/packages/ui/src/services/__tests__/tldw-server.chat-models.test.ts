import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getChatModels: vi.fn(),
  getModels: vi.fn()
}))

vi.mock("@plasmohq/storage", () => ({
  Storage: vi.fn()
}))

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  })
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getConfig: vi.fn(async () => ({
      serverUrl: "http://127.0.0.1:8000",
      apiKey: "test-key",
      authMode: "single-user"
    })),
    updateConfig: vi.fn()
  },
  tldwModels: {
    getChatModels: (...args: unknown[]) =>
      (mocks.getChatModels as (...args: unknown[]) => unknown)(...args),
    getModels: (...args: unknown[]) =>
      (mocks.getModels as (...args: unknown[]) => unknown)(...args)
  }
}))

describe("fetchChatModels", () => {
  beforeEach(async () => {
    vi.resetModules()
    mocks.getChatModels.mockReset()
    const { clearChatModelsCache } = await import("../tldw-server")
    clearChatModelsCache()
  })

  it("keeps identical model ids from different providers as distinct choices", async () => {
    mocks.getChatModels.mockResolvedValue([
      {
        id: "shared-model",
        name: "shared-model",
        provider: "openai",
        type: "chat"
      },
      {
        id: "shared-model",
        name: "shared-model",
        provider: "anthropic",
        type: "chat"
      }
    ])

    const { fetchChatModels } = await import("../tldw-server")
    const models = await fetchChatModels({
      returnEmpty: true,
      forceRefresh: true
    })

    expect(models).toHaveLength(2)
    expect(
      models.map((model: any) => `${model.provider}:${model.model}`)
    ).toEqual(["openai:tldw:shared-model", "anthropic:tldw:shared-model"])
  })

  it("retains the configured server default in the UI catalog", async () => {
    mocks.getChatModels.mockResolvedValue([
      {
        id: "first",
        name: "First",
        provider: "ollama",
        type: "chat",
        isServerDefault: false
      },
      {
        id: "second",
        name: "Second",
        provider: "openai",
        type: "chat",
        isServerDefault: true
      }
    ])
    const { fetchChatModels } = await import("../tldw-server")
    const models = await fetchChatModels({ requireFresh: true })
    expect(models.map((model) => model.is_server_default)).toEqual([
      false,
      true
    ])
  })
})

// #3343: TldwModels.getModels() answers [] when the read fails, which left the
// picker's "All models" scope silently short. The catalog is read strictly.
describe("fetchChatModelCatalog", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.getModels.mockReset()
  })

  it("reports a failed catalog read instead of an empty catalog", async () => {
    mocks.getModels.mockImplementation(async (_forceRefresh?: boolean, options?: { throwOnError?: boolean }) => {
      if (options?.throwOnError) throw new Error("Request failed: HTTP 502")
      return []
    })
    const { fetchChatModelCatalog } = await import("../tldw-server")
    await expect(fetchChatModelCatalog()).rejects.toThrow("HTTP 502")
  })
})
