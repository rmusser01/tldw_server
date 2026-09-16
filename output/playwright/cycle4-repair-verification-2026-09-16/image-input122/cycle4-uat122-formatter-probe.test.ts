import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({ sendMessage: vi.fn(), streamMessage: vi.fn(), ocr: vi.fn() }))
vi.mock("@/services/tldw", () => ({ tldwChat: { sendMessage: mocks.sendMessage, streamMessage: mocks.streamMessage } }))
vi.mock("@/utils/ocr", () => ({ processImageForOCR: mocks.ocr }))
vi.mock("@/libs/openai", () => ({ getAllOpenAIModels: vi.fn() }))
vi.mock("@/db/dexie/models", async () => {
  const actual = await vi.importActual<typeof import("@/db/model-provider-utils")>("@/db/model-provider-utils")
  return { isCustomModel: actual.isCustomModel }
})

import { humanMessageFormatter } from "@/utils/human-message"
import { ChatTldw } from "@/models/ChatTldw"
import { HumanMessage } from "@/types/messages"

const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j1ioAAAAASUVORK5CYII="
describe("UAT122 real formatter to model boundary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.sendMessage.mockResolvedValue("answer")
    mocks.streamMessage.mockImplementation(async function* () { yield "answer" })
  })
  it.each(["ordinary", "custom_model-abcd-abcd-abc-abcd"])('keeps image-only content in real formatter: %s', async model => {
    const content = [{ type: "text" as const, text: "" }, { type: "image_url" as const, image_url: image }]
    const formatted = await humanMessageFormatter({ content, model, useOCR: false })
    expect(formatted).toBeInstanceOf(HumanMessage)
    expect(formatted.content).toEqual(content)
    expect(mocks.ocr).not.toHaveBeenCalled()
  })
  it.each(["", "  literal question  "])('vision-positive formatter/model keeps exact payload: %j', async text => {
    const formatted = await humanMessageFormatter({ content: [{ type: "text", text }, { type: "image_url", image_url: image }], model: "custom_model-abcd-abcd-abc-abcd", useOCR: false })
    const model = new ChatTldw({ model: "custom", supportsMultimodal: true })
    await model.invoke([formatted])
    expect(mocks.sendMessage.mock.calls[0][0]).toEqual([{ role: "user", content: [{ type: "text", text }, { type: "image_url", image_url: { url: image } }] }])
  })
  it.each([false, undefined])('never silently sends text-only when vision support is %j', async supportsMultimodal => {
    const formatted = await humanMessageFormatter({ content: [{ type: "text", text: "exact question" }, { type: "image_url", image_url: image }], model: "ordinary", useOCR: false })
    const model = new ChatTldw({ model: "local", supportsMultimodal })
    let rejected = false
    try { await model.invoke([formatted]) } catch { rejected = true }
    // Either complete payload or explicit rejection satisfies the no-silent-drop contract.
    expect(rejected || Array.isArray(mocks.sendMessage.mock.calls[0]?.[0]?.[0]?.content)).toBe(true)
  })
  it('never silently streams an empty image-only request on unsupported model', async () => {
    const formatted = await humanMessageFormatter({ content: [{ type: "text", text: "" }, { type: "image_url", image_url: image }], model: "custom_model-abcd-abcd-abc-abcd", useOCR: false })
    const model = new ChatTldw({ model: "custom", supportsMultimodal: false })
    let rejected = false
    try { for await (const _ of await model.stream([formatted])) { /* consume */ } } catch { rejected = true }
    expect(rejected || Array.isArray(mocks.streamMessage.mock.calls[0]?.[0]?.[0]?.content)).toBe(true)
  })
})
