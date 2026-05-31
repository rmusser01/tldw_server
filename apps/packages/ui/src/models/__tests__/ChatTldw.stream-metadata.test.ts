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
