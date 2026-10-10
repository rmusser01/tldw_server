import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  streamMessage: vi.fn()
}))

vi.mock("@/services/tldw", async () => {
  const actual =
    await vi.importActual<typeof import("@/services/tldw")>("@/services/tldw")
  return {
    ...actual,
    tldwChat: {
      ...actual.tldwChat,
      streamMessage: mocks.streamMessage
    }
  }
})

import { ChatTldw } from "@/models/ChatTldw"
import {
  beginServerChatWrite,
  getServerChatSaveStatus,
  resetServerChatSaveStatus,
  serverChatSaveStatusStore
} from "@/store/server-chat-save-status"
import { HumanMessage } from "@/types/messages"

const consume = async (stream: AsyncGenerator<unknown>) => {
  for await (const _token of stream) {
    // drain
  }
}

// CS-03 / XS-05 (#3104): a completion that the server persists into an
// existing server chat is a server write, so its outcome feeds the labels.
describe("ChatTldw server-persisted completions record the save status", () => {
  beforeEach(() => {
    mocks.streamMessage.mockReset()
    resetServerChatSaveStatus()
  })

  it("marks the server chat saving while streaming and saved after its persistence acknowledgment", async () => {
    let statusDuringStream: string | null = null
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      statusDuringStream = getServerChatSaveStatus("server-chat")
      yield "Saved reply"
      onChunk?.({ tldw_conversation_id: "server-chat", tldw_message_id: "saved-assistant" })
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await consume(await model.stream([new HumanMessage("Hi")]))

    expect(statusDuringStream).toBe("saving")
    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
  })

  it.each([
    { name: "no receipt", chunk: undefined },
    { name: "conversation metadata only", chunk: { tldw_conversation_id: "server-chat" } },
    { name: "provider completion only", chunk: { id: "provider-id", choices: [{ delta: {}, finish_reason: "stop" }] } },
    { name: "another conversation's receipt", chunk: { tldw_conversation_id: "other-chat", tldw_message_id: "saved-assistant" } }
  ])("does not mark the chat saved at EOF with $name", async ({ chunk }) => {
    beginServerChatWrite("server-chat")("saved")
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      yield "Unconfirmed reply"
      if (chunk) onChunk?.(chunk)
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })
    const tokens = []
    for await (const token of await model.stream([new HumanMessage("Hi")])) tokens.push(token)

    expect(tokens).toEqual(["Unconfirmed reply"])
    expect(getServerChatSaveStatus("server-chat")).toBe("failed")
    expect(serverChatSaveStatusStore.getState().entries["server-chat"].inFlight).toBe(0)
  })

  it("accepts an ordinary native durable-turn acknowledgment for the current input", async () => {
    const inputId = "12345678-1234-4321-8123-123456789abc"
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      yield "Saved reply"
      onChunk?.({ tldw_conversation_id: "server-chat", tldw_user_message_id: inputId,
        tldw_message_id: "saved-assistant" })
    })
    const model = new ChatTldw({ model: "gpt-test", saveToDb: true,
      conversationId: "server-chat", tldwTurn: { user_message_id: inputId }, originalUserMessage: "Hi" })

    await consume(await model.stream([new HumanMessage("Hi")]))

    expect(model.serverMessagesAlreadyPersisted).toBe(true)
    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
    expect(serverChatSaveStatusStore.getState().entries["server-chat"].inFlight).toBe(0)
  })

  it("preserves the previous outcome and partial output on transport interruption", async () => {
    beginServerChatWrite("server-chat")("saved")
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      yield "Partial"
      onChunk?.({ event: "stream_transport_interrupted", detail: "port dropped" })
    })
    const model = new ChatTldw({ model: "gpt-test", saveToDb: true, conversationId: "server-chat" })
    const chunks = []
    for await (const chunk of await model.stream([new HumanMessage("Hi")])) chunks.push(chunk)

    expect(chunks).toEqual(["Partial", { event: "stream_transport_interrupted", detail: "port dropped" }])
    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
    expect(serverChatSaveStatusStore.getState().entries["server-chat"].inFlight).toBe(0)
  })

  it("marks the server chat failed when the persisted completion errors", async () => {
    mocks.streamMessage.mockImplementationOnce(async function* () {
      yield "Partial"
      throw new Error("provider failed")
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await expect(
      consume(await model.stream([new HumanMessage("Hi")]))
    ).rejects.toThrow("provider failed")

    expect(getServerChatSaveStatus("server-chat")).toBe("failed")
  })

  it("keeps the previous outcome when the user stops the reply", async () => {
    beginServerChatWrite("server-chat")("saved")
    const controller = new AbortController()
    mocks.streamMessage.mockImplementationOnce(async function* () {
      yield "Partial"
      controller.abort()
      yield "ignored"
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await consume(
      await model.stream([new HumanMessage("Hi")], { signal: controller.signal })
    )

    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
  })

  it("does not track completions that are not saved to a server chat", async () => {
    mocks.streamMessage.mockImplementation(async function* () {
      yield "Local reply"
    })

    await consume(
      await new ChatTldw({
        model: "gpt-test",
        saveToDb: false,
        conversationId: "server-chat"
      }).stream([new HumanMessage("Hi")])
    )
    await consume(
      await new ChatTldw({
        model: "gpt-test",
        clientManagedHistory: true,
        saveToDb: true,
        conversationId: "server-chat"
      }).stream([new HumanMessage("Hi")])
    )

    expect(getServerChatSaveStatus("server-chat")).toBe("unknown")
  })
})
