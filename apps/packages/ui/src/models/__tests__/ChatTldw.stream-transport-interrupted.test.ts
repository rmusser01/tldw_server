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

describe("ChatTldw stream transport interruption", () => {
  it("re-emits the stream_transport_interrupted sentinel after the token stream", async () => {
    mocks.streamMessage.mockImplementation(
      async function* (
        _messages: unknown[],
        _options: unknown,
        onChunk?: (chunk: unknown) => void
      ) {
        yield "partial"
        // Mirrors TldwChat: the sentinel arrives via onChunk (it carries no
        // assistant text, so it is never yielded as a token).
        onChunk?.({
          event: "stream_transport_interrupted",
          detail: "port dropped",
          partial_response_saved: true
        })
      }
    )

    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })
    const chunks: unknown[] = []

    for await (const chunk of await model.stream([new HumanMessage("Hi")])) {
      chunks.push(chunk)
    }

    expect(chunks[0]).toBe("partial")
    expect(chunks[chunks.length - 1]).toMatchObject({
      event: "stream_transport_interrupted",
      detail: "port dropped"
    })
  })

  it("does not re-emit the sentinel when the caller aborted the stream", async () => {
    const controller = new AbortController()
    mocks.streamMessage.mockImplementation(
      async function* (
        _messages: unknown[],
        _options: unknown,
        onChunk?: (chunk: unknown) => void
      ) {
        yield "partial"
        controller.abort()
        onChunk?.({
          event: "stream_transport_interrupted",
          detail: "port dropped",
          partial_response_saved: true
        })
      }
    )

    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })
    const chunks: unknown[] = []

    for await (const chunk of await model.stream([new HumanMessage("Hi")], {
      signal: controller.signal
    })) {
      chunks.push(chunk)
    }

    // Abort takes precedence: no interruption sentinel is surfaced.
    expect(
      chunks.some(
        (chunk) =>
          chunk &&
          typeof chunk === "object" &&
          (chunk as Record<string, unknown>).event ===
            "stream_transport_interrupted"
      )
    ).toBe(false)
  })
})
