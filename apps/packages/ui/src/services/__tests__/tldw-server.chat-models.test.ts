import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getChatModels: vi.fn()
}))

vi.mock("@plasmohq/storage", () => ({
  Storage: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
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
      (mocks.getChatModels as (...args: unknown[]) => unknown)(...args)
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
      { id: "shared-model", name: "shared-model", provider: "openai", type: "chat" },
      { id: "shared-model", name: "shared-model", provider: "anthropic", type: "chat" }
    ])

    const { fetchChatModels } = await import("../tldw-server")
    const models = await fetchChatModels({ returnEmpty: true, forceRefresh: true })

    expect(models).toHaveLength(2)
    expect(models.map((model: any) => `${model.provider}:${model.model}`)).toEqual([
      "openai:tldw:shared-model",
      "anthropic:tldw:shared-model"
    ])
  })
})
