import { beforeEach, describe, expect, it, vi } from "vitest"
import { pageAssistModel } from "../index"
import { HumanMessage } from "@/types/messages"
import { useStoreChatModelSettings } from "@/store/model"

const mocks = vi.hoisted(() => ({ getModel: vi.fn(), getModels: vi.fn(), getModelSettings: vi.fn(), stream: vi.fn() }))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => ({}), getModelSettings: mocks.getModelSettings
}))
vi.mock("@/services/tldw-server", () => ({ getDefaultApiProvider: async () => "openai" }))
vi.mock("@/services/tldw", () => ({
  tldwModels: { getModel: mocks.getModel, getModels: mocks.getModels },
  tldwChat: { streamMessage: mocks.stream }
}))

const image = new HumanMessage({ content: [{ type: "image_url", image_url: { url: "data:image/png;base64,owned" } }] })
const catalog = [
  { id: "shared", provider: "llama", capabilities: ["vision"] },
  { id: "shared", provider: "custom_openai_api", capabilities: [] }
]

describe("provider-qualified Chat model construction", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStoreChatModelSettings.getState().reset()
    mocks.getModelSettings.mockResolvedValue({})
    mocks.getModels.mockResolvedValue(catalog)
    mocks.getModel.mockResolvedValue(catalog[0])
    mocks.stream.mockImplementation(async function* () { yield "Hello" })
  })

  it.each(["custom-openai-api:shared", "tldw:custom_openai_api:shared"])("dispatches the bare model to the selected provider: %s", async model => {
    const chat = await pageAssistModel({ model, saveToDb: false })
    for await (const _ of await chat.stream([new HumanMessage({ content: "Hello" })])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ model: "shared", apiProvider: "custom-openai-api" })
  })

  it.each(["custom-openai-api:shared", "custom-openai-api:missing"])("does not borrow another provider's vision capability: %s", async model => {
    const chat = await pageAssistModel({ model, saveToDb: false })
    await expect(chat.stream([image])).rejects.toThrow(/image support.*not confirmed/i)
    expect(mocks.stream).not.toHaveBeenCalled()
  })

  it.each(["llama.cpp:shared", "tldw:llamacpp:shared"])("uses the selected provider's confirmed vision capability: %s", async model => {
    const chat = await pageAssistModel({ model, saveToDb: false })
    for await (const _ of await chat.stream([image])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ model: "shared", apiProvider: "llama.cpp" })
  })

  it("retains the existing unqualified model and default-provider behavior", async () => {
    const chat = await pageAssistModel({ model: "tldw:shared", saveToDb: false })
    for await (const _ of await chat.stream([image])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ model: "shared", apiProvider: "openai" })
  })

  it("matches the local catalogue alias without changing a model ID containing colons", async () => {
    mocks.getModels.mockResolvedValue([{ id: "Gemma:Q4", provider: "local", capabilities: ["vision"] }])
    const chat = await pageAssistModel({ model: "local-llm:Gemma:Q4", saveToDb: false })
    for await (const _ of await chat.stream([image])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ model: "Gemma:Q4", apiProvider: "local-llm" })
  })

  it("retains previously saved bare-model generation settings", async () => {
    mocks.getModelSettings.mockImplementation(async key => key === "shared"
      ? { temperature: 0.25, numPredict: 123, reasoningEffort: "high" } : {})
    const chat = await pageAssistModel({ model: "custom-openai-api:shared", saveToDb: false })
    for await (const _ of await chat.stream([new HumanMessage({ content: "Hello" })])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ apiProvider: "custom-openai-api", temperature: 0.25, maxTokens: 123, reasoningEffort: "high" })
  })

  it("prefers settings saved for the qualified model over legacy settings", async () => {
    mocks.getModelSettings.mockImplementation(async key => key === "custom-openai-api:shared"
      ? { temperature: 0.4, numPredict: 321, reasoningEffort: "low" }
      : { temperature: 0.25, numPredict: 123, reasoningEffort: "high" })
    const chat = await pageAssistModel({ model: "custom-openai-api:shared", saveToDb: false })
    for await (const _ of await chat.stream([new HumanMessage({ content: "Hello" })])) { /* consume */ }
    expect(mocks.stream.mock.calls[0][1]).toMatchObject({ apiProvider: "custom-openai-api", temperature: 0.4, maxTokens: 321, reasoningEffort: "low" })
  })
})
