import { describe, expect, it, vi } from "vitest"

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
import { HumanMessage } from "@/types/messages"

describe("ChatTldw stream metadata handoff", () => {
  it("captures only the current saved assistant acknowledgement", async () => {
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      yield "Saved reply"
      onChunk?.({ tldw_conversation_id: "server-chat", tldw_message_id: " saved-assistant ", tldw_user_message_id: " saved-user " })
    })
    const model = new ChatTldw({ model: "gpt-test", saveToDb: true })
    for await (const _token of await model.stream([new HumanMessage("Hi")])) { /* consume */ }
    expect(model.serverMessageId).toBe("saved-assistant")
    expect(model.userServerMessageId).toBe("saved-user")

    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      onChunk?.({ tldw_conversation_id: "server-chat" })
      yield "Unacknowledged reply"
    })
    for await (const _token of await model.stream([new HumanMessage("Again")])) { /* consume */ }
    expect(model.serverMessageId).toBeUndefined()
    expect(model.userServerMessageId).toBeUndefined()
  })

  it("ignores late acknowledgements after the request is aborted", async () => {
    const controller = new AbortController()
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      yield "Old account reply"
      controller.abort()
      onChunk?.({ tldw_conversation_id: "old-chat", tldw_message_id: "old-assistant", tldw_user_message_id: "old-user" })
    })
    const model = new ChatTldw({ model: "gpt-test", saveToDb: true })
    for await (const _token of await model.stream([new HumanMessage("Hi")], { signal: controller.signal })) { /* consume */ }
    expect(model.serverMessageId).toBeUndefined()
    expect(model.userServerMessageId).toBeUndefined()
    expect(model.conversationId).toBeUndefined()
  })

  it("keeps ephemeral stream IDs out of saved conversation linking", async () => {
    mocks.streamMessage.mockImplementation(async function* (_messages, _options, onChunk) {
      onChunk?.({ event: "tldw_metadata", tldw_conversation_id: "ephemeral-id", tldw_message_id: "ephemeral-message", tldw_user_message_id: "ephemeral-user" })
      yield "Local answer"
    })
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true, saveToDb: false })
    const tokens: string[] = []
    for await (const token of await model.stream([new HumanMessage("Hi")])) tokens.push(token)
    expect({ tokens, conversationId: model.conversationId, saveToDb: model.saveToDb }).toEqual({
      tokens: ["Local answer"], conversationId: undefined, saveToDb: false,
    })
    expect(model.serverMessageId).toBeUndefined()
    expect(model.userServerMessageId).toBeUndefined()
  })
  it("captures streamed conversation metadata without yielding it as assistant text", async () => {
    mocks.streamMessage.mockImplementation(
      async function* (
        _messages: unknown[],
        _options: unknown,
        onChunk?: (chunk: unknown) => void
      ) {
        onChunk?.({
          event: "tldw_metadata",
          tldw_conversation_id: "server-chat-99"
        })
        yield "hello"
      }
    )

    const model = new ChatTldw({
      model: "tldw:gpt-test",
      streaming: true
    })
    const tokens: string[] = []

    for await (const token of await model.stream([new HumanMessage("Hi")])) {
      tokens.push(token)
    }

    expect(tokens).toEqual(["hello"])
    expect(model.conversationId).toBe("server-chat-99")
    expect(model.saveToDb).toBe(true)
  })
})
