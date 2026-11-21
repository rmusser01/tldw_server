import { beforeEach, describe, expect, it, vi } from "vitest"
import { HumanMessage, AIMessage, SystemMessage } from "@/types/messages"

const mocks = vi.hoisted(() => ({
  createChatCompletion: vi.fn(),
  streamChatCompletion: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getConfig: vi.fn(async () => null),
    createChatCompletion: (...args: unknown[]) => mocks.createChatCompletion(...args),
    streamChatCompletion: (...args: unknown[]) => mocks.streamChatCompletion(...args)
  }
}))
vi.mock("@/services/tldw", async () => {
  const { TldwChatService } = await import("@/services/tldw/TldwChat")
  return { tldwChat: new TldwChatService() }
})

import { ChatTldw } from "../ChatTldw"

const turnId = "12409645-7bce-4cba-b03b-bc4b0b27cc68"

describe("durable user turn transport", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.createChatCompletion.mockResolvedValue({
      json: async () => ({ choices: [{ message: { content: "37 seconds" } }] })
    })
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield { choices: [{ delta: { content: "37 seconds" } }] }
    })
  })

  it("rejects server-owned durable identity on an owner-selected request", () => {
    expect(() => new ChatTldw({
      model: "gemma", clientManagedHistory: true, saveToDb: true,
      conversationId: "workspace-chat", tldwTurn: { user_message_id: turnId },
      originalUserMessage: "Hello"
    })).toThrow("unsupported_history_durable_turn")
  })

  it.each([
    { tldw_conversation_id: "other-chat", tldw_user_message_id: turnId },
    { tldw_conversation_id: "workspace-chat", tldw_user_message_id: "other-user" }
  ])("ignores a durable receipt outside the captured turn (%j)", async (receipt) => {
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield { ...receipt, tldw_message_id: "other-assistant" }
      yield { choices: [{ delta: { content: "Answer" } }] }
    })
    const model = new ChatTldw({
      model: "gemma", saveToDb: true, conversationId: "workspace-chat",
      tldwTurn: { user_message_id: turnId }, originalUserMessage: "Hello"
    })
    const tokens: string[] = []
    for await (const token of await model.stream([new HumanMessage("Hello")])) tokens.push(token)
    expect(tokens).toEqual(["Answer"])
    expect(model.conversationId).toBe("workspace-chat")
    expect(model.serverUserMessageId).toBeUndefined()
    expect(model.userServerMessageId).toBeUndefined()
    expect(model.serverMessageId).toBeUndefined()
    expect(model.serverMessagesAlreadyPersisted).toBe(false)
  })

  it.each([false, true])("sends one original user with request-local grounding (stream=%s)", async (stream) => {
    const model = new ChatTldw({
      model: "gemma",
      saveToDb: true,
      conversationId: "workspace-chat",
      tldwTurn: { user_message_id: turnId },
      originalUserMessage: "What is the sampling interval?"
    })
    const messages = [
      new SystemMessage("Answer from the supplied evidence."),
      new HumanMessage("Earlier question"),
      new AIMessage("Earlier answer"),
      new HumanMessage("Evidence: the sensor samples every 37 seconds.")
    ]
    if (stream) {
      for await (const _token of await model.stream(messages)) { /* drain */ }
    } else {
      await model.generateOnce(messages)
    }
    const request = (stream ? mocks.streamChatCompletion : mocks.createChatCompletion).mock.calls[0][0]
    expect(request).toMatchObject({
      conversation_id: "workspace-chat",
      save_to_db: true,
      tldw_turn: { user_message_id: turnId },
      messages: [
        { role: "system", content: "Answer from the supplied evidence." },
        { role: "system", content: "Evidence: the sensor samples every 37 seconds." },
        { role: "user", content: "What is the sampling interval?" }
      ]
    })
  })

  it("does not duplicate the raw user as system context", async () => {
    const model = new ChatTldw({
      model: "gemma", saveToDb: true, conversationId: "workspace-chat",
      tldwTurn: { user_message_id: turnId }, originalUserMessage: "Hello"
    })
    await model.generateOnce([new HumanMessage("Hello")])
    expect(mocks.createChatCompletion.mock.calls[0][0].messages).toEqual([{ role: "user", content: "Hello" }])
  })

  it("leaves legacy callers' history unchanged", async () => {
    const model = new ChatTldw({ model: "gemma" })
    await model.generateOnce([new HumanMessage("Earlier"), new AIMessage("Answer"), new HumanMessage("Next")])
    expect(mocks.createChatCompletion.mock.calls[0][0]).toMatchObject({
      messages: [{ role: "user", content: "Earlier" }, { role: "assistant", content: "Answer" }, { role: "user", content: "Next" }]
    })
    expect(mocks.createChatCompletion.mock.calls[0][0]).not.toHaveProperty("tldw_turn")
  })

  it.each([
    { saveToDb: false, conversationId: "workspace-chat" },
    { saveToDb: true, conversationId: undefined }
  ])("rejects an unbound durable turn before transport (%j)", async (binding) => {
    const model = new ChatTldw({
      model: "gemma", ...binding,
      tldwTurn: { user_message_id: turnId }, originalUserMessage: "Hello"
    })
    await expect(model.generateOnce([new HumanMessage("Hello")])).rejects.toThrow(/persisted conversation/i)
    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
  })

  it("keeps a user receipt separate from assistant persistence", async () => {
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield { tldw_conversation_id: "workspace-chat", tldw_user_message_id: turnId }
      yield { choices: [{ delta: { content: "partial" } }] }
    })
    const model = new ChatTldw({
      model: "gemma", saveToDb: true, conversationId: "workspace-chat",
      tldwTurn: { user_message_id: turnId }, originalUserMessage: "Hello"
    })
    for await (const _token of await model.stream([new HumanMessage("Hello")])) { /* drain */ }
    expect(model.serverUserMessageId).toBe(turnId)
    expect(model.serverMessagesAlreadyPersisted).toBe(false)
  })
})
