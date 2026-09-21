import { beforeEach, describe, expect, it, vi } from "vitest"
import { humanMessageFormatter } from "@/utils/human-message"
import { generateHistory } from "@/utils/generate-history"
import { ChatTldw } from "../ChatTldw"
import { pageAssistModel } from "../index"

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  streamMessage: vi.fn(),
  getModel: vi.fn(),
  ocr: vi.fn()
}))
vi.mock("@/services/tldw", () => ({
  tldwChat: { sendMessage: mocks.sendMessage, streamMessage: mocks.streamMessage },
  tldwModels: { getModel: async (model: string) => {
    const info = await mocks.getModel(model)
    return info ? { id: model, provider: "openai", ...info } : info
  } }
}))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => ({}),
  getModelSettings: async () => ({})
}))
vi.mock("@/services/tldw-server", () => ({ getDefaultApiProvider: async () => "openai" }))
vi.mock("@/utils/resolve-api-provider", async () => ({
  ...await vi.importActual<typeof import("@/utils/resolve-api-provider")>("@/utils/resolve-api-provider"),
  resolveApiProviderForModel: async () => "openai"
}))
vi.mock("@/utils/ocr", () => ({ processImageForOCR: mocks.ocr }))
vi.mock("@/libs/openai", () => ({ getAllOpenAIModels: vi.fn() }))
vi.mock("@/db/dexie/models", async () => {
  const actual = await vi.importActual<typeof import("@/db/model-provider-utils")>("@/db/model-provider-utils")
  return { isCustomModel: actual.isCustomModel }
})

const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
const customModel = "custom_model-abcd-abcd-abc-abcd"
const format = (text = "  Exact question  ", model = customModel, useOCR = false, imageUrl = image) =>
  humanMessageFormatter({
    content: [{ type: "text", text }, { type: "image_url", image_url: imageUrl }],
    model,
    useOCR
  })

describe("image input at the real formatter and model boundary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.sendMessage.mockResolvedValue("Answer")
    mocks.streamMessage.mockImplementation(async function* () { yield "Answer" })
    mocks.getModel.mockResolvedValue({ capabilities: [] })
    mocks.ocr.mockResolvedValue("Explicitly extracted text")
  })

  it.each([false, undefined])("blocks image send and Retry before either transport when capability is %s", async supportsMultimodal => {
    const message = await format()
    const original = structuredClone(message.content)
    const model = new ChatTldw({ model: "unconfirmed", supportsMultimodal, retryFailedTurn: true })
    await expect(model.stream([message])).rejects.toThrow(/image support.*not confirmed/i)
    await expect(model.invoke([message])).rejects.toThrow(/image support.*not confirmed/i)
    expect(mocks.streamMessage).not.toHaveBeenCalled()
    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(message.content).toEqual(original)
  })

  it.each(["ordinary", customModel])("keeps image-only formatter input and blocks an unsupported model: %s", async modelId => {
    const message = await format("", modelId)
    expect(message.content).toEqual([{ type: "text", text: "" }, { type: "image_url", image_url: image }])
    await expect(new ChatTldw({ model: modelId }).stream([message])).rejects.toThrow(/image support.*not confirmed/i)
    expect(mocks.streamMessage).not.toHaveBeenCalled()
  })

  it.each(["image/png", "image/jpeg", "image/webp"])("preserves supported %s payload exactly in stream and invoke", async mime => {
    const url = `data:${mime};base64,c3ludGhldGljLWJ5dGVz`
    const message = await format("  Exact question  ", customModel, false, url)
    const model = new ChatTldw({ model: "vision", supportsMultimodal: true })
    for await (const _token of await model.stream([message])) { /* consume */ }
    await model.invoke([message])
    const expected = [{ role: "user", content: [{ type: "text", text: "  Exact question  " }, { type: "image_url", image_url: { url } }] }]
    expect(mocks.streamMessage.mock.calls[0][0]).toEqual(expected)
    expect(mocks.sendMessage.mock.calls[0][0]).toEqual(expected)
  })

  it("preserves supported image-only input through the real custom formatter", async () => {
    const message = await format("")
    await new ChatTldw({ model: "vision", supportsMultimodal: true }).invoke([message])
    expect(mocks.sendMessage.mock.calls[0][0]).toEqual([{ role: "user", content: [{ type: "text", text: "" }, { type: "image_url", image_url: { url: image } }] }])
  })

  it("blocks images in restored history even when the new question is text-only", async () => {
    const history = generateHistory([{ role: "user", content: "Earlier image", image }, { role: "assistant", content: "Earlier answer" }], customModel)
    const text = await humanMessageFormatter({ content: "Follow-up", model: customModel, useOCR: false })
    await expect(new ChatTldw({ model: "text-only" }).invoke([...history, text])).rejects.toThrow(/image support.*not confirmed/i)
    expect(mocks.sendMessage).not.toHaveBeenCalled()
  })

  it("allows an explicit OCR conversion without enabling image support or mutating original input", async () => {
    const message = await format("Read this", customModel, true)
    await new ChatTldw({ model: "text-only" }).invoke([message])
    expect(mocks.ocr).toHaveBeenCalledWith(image)
    expect(mocks.sendMessage.mock.calls[0][0]).toEqual([{ role: "user", content: "Read this\n\n[IMAGE OCR TEXT]\nExplicitly extracted text" }])
  })

  it("keeps ordinary text-only input available with unknown capabilities", async () => {
    const message = await humanMessageFormatter({ content: "  plain text  ", model: "ordinary", useOCR: false })
    await new ChatTldw({ model: "unknown" }).invoke([message])
    expect(mocks.sendMessage.mock.calls[0][0]).toEqual([{ role: "user", content: "  plain text  " }])
  })

  it.each(["no-vision", "missing-model", "catalog-failure"])("uses the actual model factory to block unconfirmed vision: %s", async state => {
    if (state === "missing-model") mocks.getModel.mockResolvedValue(null)
    if (state === "catalog-failure") mocks.getModel.mockRejectedValue(new Error("Catalog unavailable"))
    const model = await pageAssistModel({ model: "local", saveToDb: false })
    await expect(model.invoke([await format()])).rejects.toThrow(/image support.*not confirmed/i)
    expect(mocks.sendMessage).not.toHaveBeenCalled()
  })

  it("uses catalog-confirmed vision through the actual model factory", async () => {
    mocks.getModel.mockResolvedValue({ capabilities: ["vision"] })
    const model = await pageAssistModel({ model: "vision", saveToDb: false })
    await model.invoke([await format()])
    expect(mocks.sendMessage.mock.calls[0][0][0].content).toContainEqual({ type: "image_url", image_url: { url: image } })
  })
})
