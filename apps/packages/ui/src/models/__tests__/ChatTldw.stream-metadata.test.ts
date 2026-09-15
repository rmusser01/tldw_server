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
  it("keeps ephemeral stream IDs out of saved conversation linking", async () => {
    mocks.streamMessage.mockImplementation(async function* (_messages, _options, onChunk) {
      onChunk?.({ event: "tldw_metadata", tldw_conversation_id: "ephemeral-id" })
      yield "Local answer"
    })
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true, saveToDb: false })
    const tokens: string[] = []
    for await (const token of await model.stream([new HumanMessage("Hi")])) tokens.push(token)
    expect({ tokens, conversationId: model.conversationId, saveToDb: model.saveToDb }).toEqual({
      tokens: ["Local answer"], conversationId: undefined, saveToDb: false,
    })
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
