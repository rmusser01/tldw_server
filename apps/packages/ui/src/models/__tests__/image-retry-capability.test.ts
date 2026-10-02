import type { TldwModelsService } from "@/services/tldw/TldwModels"
import type { HumanMessage, MessageContent } from "@/types/messages"
import { beforeEach, describe, expect, it, vi } from "vitest"
const io = vi.hoisted(() => ({
  metadata: vi.fn(),
  stream: vi.fn(),
  send: vi.fn(),
  stored: null as unknown,
  service: null as TldwModelsService | null,
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: async () => ({
      serverUrl: "http://test.invalid",
      authMode: "single-user",
      apiKey: "synthetic-key",
    }),
    initialize: async () => {},
    getModels: (...args: unknown[]) => io.metadata(...args),
  },
}))
vi.mock("@/services/tldw", async () => ({
  tldwModels: {
    getModel: (...args: Parameters<TldwModelsService["getModel"]>) =>
      io.service!.getModel(...args),
    getModels: (...args: Parameters<TldwModelsService["getModels"]>) =>
      io.service!.getModels(...args),
  },
  tldwChat: { sendMessage: io.send, streamMessage: io.stream },
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async () => io.stored,
    set: async (_key: string, value: unknown) => {
      io.stored = value
    },
    watch: () => () => {},
  }),
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
}))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => ({}),
  getModelSettings: async () => ({}),
}))
vi.mock("@/services/tldw-server", () => ({
  getDefaultApiProvider: async () => "llama.cpp",
}))
vi.mock("@/utils/resolve-api-provider", async () => ({
  ...(await vi.importActual<typeof import("@/utils/resolve-api-provider")>(
    "@/utils/resolve-api-provider",
  )),
  resolveApiProviderForModel: async () => "llama.cpp",
}))
const catalog = (vision: boolean) => [
  {
    id: "test-model",
    name: "Test Model",
    provider: "llama",
    type: "chat",
    vision,
    capabilities: { vision, streaming: true },
  },
]
const image = "data:image/png;base64,aW1hZ2U="
const content: MessageContent = [
  { type: "text", text: "Keep original" },
  { type: "image_url", image_url: image },
]
let message: () => HumanMessage
const load = async () => {
  const { HumanMessage } = await import("@/types/messages")
  message = () => new HumanMessage({ content })
  io.service = (await import("@/services/tldw/TldwModels")).tldwModels
  return {
    service: (await import("@/services/tldw/TldwModels")).tldwModels,
    factory: (await import("@/models")).pageAssistModel,
  }
}
describe("image Retry with the real model cache and factory", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    io.stored = null
    io.metadata.mockResolvedValue(catalog(false))
    io.send.mockResolvedValue("Recovered")
    io.stream.mockImplementation(async function* () {
      yield "Recovered"
    })
  })
  it.each(
    ["tldw:test-model", "tldw:llama.cpp:test-model"].flatMap((model) =>
      [false, true].map((retryFailedTurn) => ({ model, retryFailedTurn })),
    ),
  )(
    "refreshes a cached negative for explicit Retry: $model / server Retry $retryFailedTurn",
    async ({ model, retryFailedTurn }) => {
      const { service, factory } = await load()
      await service.getModels(true)
      io.metadata.mockResolvedValue(catalog(true))
      const client = await factory({
        model,
        refreshImageCapability: true,
        retryFailedTurn,
        clientMessageId: "original-user",
        conversationId: "original-chat",
      })
      for await (const _ of await client.stream([message()])) {
        /* consume the real stream */
      }
      expect(io.metadata).toHaveBeenCalledTimes(2)
      expect(io.stream.mock.calls[0][0]).toEqual([
        {
          role: "user",
          content: [
            { type: "text", text: "Keep original" },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ])
      expect(io.stream.mock.calls[0][1]).toMatchObject({
        model: "test-model",
        retryFailedTurn,
        clientMessageId: "original-user",
        conversationId: "original-chat",
      })
    },
  )
  it("keeps ordinary model creation cached without discovery", async () => {
    const { service, factory } = await load()
    await service.getModels()
    io.metadata.mockResolvedValue(catalog(true))
    const client = await factory({ model: "tldw:test-model" })
    await client.invoke([
      new (await import("@/types/messages")).HumanMessage("Text only"),
    ])
    expect(io.metadata).toHaveBeenCalledTimes(1)
  })
  it.each(["current-negative", "missing-model", "metadata-outage"])(
    "keeps the guard closed for fresh %s",
    async (state) => {
      const { service, factory } = await load()
      await service.getModels(true)
      const durable = structuredClone(io.stored)
      if (state === "metadata-outage")
        io.metadata.mockRejectedValue(new Error("Catalog unreachable"))
      if (state === "missing-model") io.metadata.mockResolvedValue([])
      const client = await factory({
        model: "tldw:test-model",
        refreshImageCapability: true,
        retryFailedTurn: true,
      })
      await expect(client.invoke([message()])).rejects.toThrow(
        /image support.*not confirmed/i,
      )
      expect(io.metadata).toHaveBeenCalledTimes(2)
      expect(io.send).not.toHaveBeenCalled()
      if (state === "metadata-outage") expect(io.stored).toEqual(durable)
    },
  )
  it.each(["llama.cpp", undefined])(
    "binds refreshed unqualified capability to the selected transport provider: %s",
    async (apiProvider) => {
      const { service, factory } = await load()
      await service.getModels(true)
      io.metadata.mockResolvedValue([
        { ...catalog(true)[0], provider: "openai" },
        ...catalog(false),
      ])
      const client = await factory({
        model: "tldw:test-model",
        apiProvider,
        refreshImageCapability: true,
      })
      await expect(client.invoke([message()])).rejects.toThrow(
        /image support.*not confirmed/i,
      )
      expect(io.send).not.toHaveBeenCalled()
    },
  )

  it.each(["llama.cpp", undefined])(
    "rejects a foreign cached positive before deciding whether Retry needs discovery: %s",
    async (apiProvider) => {
      const { service, factory } = await load()
      io.metadata.mockResolvedValue([
        { ...catalog(true)[0], provider: "openai" },
        ...catalog(false),
      ])
      await service.getModels(true)
      const client = await factory({
        model: "tldw:test-model",
        apiProvider,
        refreshImageCapability: true,
      })
      await expect(client.invoke([message()])).rejects.toThrow(
        /image support.*not confirmed/i,
      )
      expect(io.send).not.toHaveBeenCalled()
    },
  )

  it("does not borrow vision from a different provider with the same model ID", async () => {
    const { service, factory } = await load()
    await service.getModels(true)
    io.metadata.mockResolvedValue([
      { ...catalog(true)[0], provider: "openai" },
      ...catalog(false),
    ])
    const client = await factory({
      model: "tldw:llama.cpp:test-model",
      refreshImageCapability: true,
    })
    await expect(client.invoke([message()])).rejects.toThrow(
      /image support.*not confirmed/i,
    )
    expect(io.send).not.toHaveBeenCalled()
  })
})
