import { beforeEach, describe, expect, it, vi } from "vitest"
import { pageAssistModel } from "../index"
import { resolveInheritedSampling } from "../chat-sampling-defaults"
import { HumanMessage } from "@/types/messages"
import { useStoreChatModelSettings } from "@/store/model"

// XS-16: the side panel shows "Default (x)" for an unset sampling parameter.
// x must be what the chat request actually sends when nothing in the current
// chat sets the parameter, so this pins resolveInheritedSampling (what the
// panel shows) to the request pageAssistModel builds.

const mocks = vi.hoisted(() => ({
  userDefaults: {} as Record<string, unknown>,
  getModel: vi.fn(),
  getModels: vi.fn(),
  stream: vi.fn()
}))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => mocks.userDefaults,
  getModelSettings: async () => ({})
}))
vi.mock("@/services/tldw-server", () => ({ getDefaultApiProvider: async () => "openai" }))
vi.mock("@/services/tldw", () => ({
  tldwModels: { getModel: mocks.getModel, getModels: mocks.getModels },
  tldwChat: { streamMessage: mocks.stream }
}))

const sentSampling = async () => {
  const chat = await pageAssistModel({ model: "tldw:shared", saveToDb: false })
  for await (const _ of await chat.stream([new HumanMessage("Hello")])) {
    /* consume */
  }
  const options = mocks.stream.mock.calls[0][1]
  return {
    temperature: options.temperature,
    topP: options.topP,
    frequencyPenalty: options.frequencyPenalty,
    presencePenalty: options.presencePenalty
  }
}

/** The request options pageAssistModel hands the transport. */
const sentOptions = async () => {
  const chat = await pageAssistModel({ model: "tldw:shared", saveToDb: false })
  for await (const _ of await chat.stream([new HumanMessage("Hello")])) {
    /* consume */
  }
  return mocks.stream.mock.calls[0][1]
}

const modelWith = (sampling_fields?: string[]) => {
  // The provider only has to match the default one; what it accepts comes from the server.
  const model = { id: "shared", provider: "openai", capabilities: [], ...(sampling_fields ? { samplingFields: sampling_fields } : {}) }
  mocks.getModels.mockResolvedValue([model])
  mocks.getModel.mockResolvedValue(model)
}

describe("sampling defaults shown in the side panel (XS-16)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.userDefaults = {}
    useStoreChatModelSettings.getState().reset()
    mocks.getModels.mockResolvedValue([{ id: "shared", provider: "openai", capabilities: [] }])
    mocks.getModel.mockResolvedValue({ id: "shared", provider: "openai", capabilities: [] })
    mocks.stream.mockImplementation(async function* () {
      yield "Hello"
    })
  })

  it("matches what is sent when nothing sets the parameters", async () => {
    expect(await sentSampling()).toEqual(resolveInheritedSampling({}))
    expect(resolveInheritedSampling({})).toEqual({
      temperature: 0.7,
      topP: 1,
      frequencyPenalty: 0,
      presencePenalty: 0
    })
  })

  it("matches what is sent when the user has saved default model settings", async () => {
    mocks.userDefaults = {
      temperature: 0.4,
      topP: 0.8,
      frequencyPenalty: 0.9,
      presencePenalty: 0.5
    }
    expect(await sentSampling()).toEqual(
      resolveInheritedSampling(mocks.userDefaults)
    )
  })
})

// #3283: these settings were stored and shown, but never sent.
describe("sampling settings reach the request (#3283)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.userDefaults = {}
    useStoreChatModelSettings.getState().reset()
    mocks.stream.mockImplementation(async function* () {
      yield "Hello"
    })
  })

  it("sends the chat's frequency and presence penalties", async () => {
    modelWith(["top_k", "min_p"])
    useStoreChatModelSettings.getState().updateSetting("frequencyPenalty", 0.4)
    useStoreChatModelSettings.getState().updateSetting("presencePenalty", 0.2)
    expect(await sentOptions()).toMatchObject({ frequencyPenalty: 0.4, presencePenalty: 0.2 })
  })

  it("sends Top K and Min P when the model's provider accepts them", async () => {
    modelWith(["top_k", "min_p"])
    useStoreChatModelSettings.getState().updateSetting("topK", 40)
    useStoreChatModelSettings.getState().updateSetting("minP", 0.05)
    expect(await sentOptions()).toMatchObject({ topK: 40, minP: 0.05 })
  })

  it("leaves out what the provider would reject", async () => {
    modelWith(["top_k"])
    useStoreChatModelSettings.getState().updateSetting("topK", 40)
    useStoreChatModelSettings.getState().updateSetting("minP", 0.05)
    const options = await sentOptions()
    expect(options.topK).toBe(40)
    expect(options.minP).toBeUndefined()
  })

  it("sends neither when the server doesn't say what the provider accepts", async () => {
    modelWith(undefined)
    useStoreChatModelSettings.getState().updateSetting("topK", 40)
    useStoreChatModelSettings.getState().updateSetting("minP", 0.05)
    const options = await sentOptions()
    expect(options.topK).toBeUndefined()
    expect(options.minP).toBeUndefined()
  })
})
