import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getChatModels: vi.fn(),
  getCachedChatModels: vi.fn()
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
      (mocks.getCachedChatModels as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/services/app", () => ({
  setNoOfRetrievedDocs: vi.fn(),
  setTotalFilePerKB: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => undefined),
    set: vi.fn(async () => undefined)
  })
}))

const importService = async () => import("@/services/tldw-server")

describe("fetchChatModels", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.getConfig.mockReset()
    mocks.getChatModels.mockReset()
    mocks.getCachedChatModels.mockReset()

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:3000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.getCachedChatModels.mockResolvedValue([])
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
})
