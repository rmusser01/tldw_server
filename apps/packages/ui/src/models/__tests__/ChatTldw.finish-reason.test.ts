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

const streamEndingWith = (finishReason: string | null) =>
  async function* (
    _messages: unknown[],
    _options: unknown,
    onChunk?: (chunk: unknown) => void
  ) {
    onChunk?.({ choices: [{ index: 0, delta: { content: "Half an" } }] })
    yield "Half an"
    // The final chunk carries no text, only why the model stopped.
    onChunk?.({ choices: [{ index: 0, delta: {}, finish_reason: finishReason }] })
  }

const drain = async (model: ChatTldw) => {
  for await (const _chunk of await model.stream([new HumanMessage("Hi")])) {
    // consume
  }
}

describe("ChatTldw stream finish reason (CM-03)", () => {
  it("records that the model stopped at its output limit", async () => {
    mocks.streamMessage.mockImplementation(streamEndingWith("length"))
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })

    await drain(model)

    expect(model.finishReason).toBe("length")
  })

  it("forgets the previous stream's finish reason on the next stream", async () => {
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })
    mocks.streamMessage.mockImplementation(streamEndingWith("length"))
    await drain(model)

    mocks.streamMessage.mockImplementation(streamEndingWith(null))
    await drain(model)

    expect(model.finishReason).toBeUndefined()
  })

  it("keeps the generation info unchanged for a normal reply", async () => {
    mocks.streamMessage.mockImplementation(streamEndingWith("stop"))
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })
    const handleLLMEnd = vi.fn()

    for await (const _chunk of await model.stream([new HumanMessage("Hi")], {
      callbacks: [{ handleLLMEnd }]
    })) {
      // consume
    }

    expect(model.finishReason).toBe("stop")
    // Native saved turns refuse client generation metadata, so a normal reply
    // must not gain any.
    expect(handleLLMEnd.mock.calls[0][0].generations[0][0].generationInfo).toBeUndefined()
  })
})
