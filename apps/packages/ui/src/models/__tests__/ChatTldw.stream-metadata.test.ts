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
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  FunctionMessage
} from "@/types/messages"

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


describe("selected history provider fidelity", () => {
  it("preserves tool-call-only assistant and its exact result identity", async () => {
    mocks.streamMessage.mockImplementation(async function* () {
      yield "ok"
    })
    const model = new ChatTldw({ model: "test" })
    const calls = [
      {
        id: "call-1",
        type: "function",
        function: { name: "lookup", arguments: "{}" }
      }
    ]
    for await (const _ of await model.stream([
      new AIMessage({ content: "", additional_kwargs: { tool_calls: calls } }),
      new ToolMessage({ content: "found", tool_call_id: "call-1" })
    ])) {
    }
    expect(mocks.streamMessage.mock.lastCall?.[0]).toEqual([
      { role: "assistant", content: "", tool_calls: calls },
      { role: "tool", content: "found", tool_call_id: "call-1" }
    ])
  })

  it("rejects unsupported legacy function rows instead of turning them into users", async () => {
    const model = new ChatTldw({ model: "test" })
    await expect(
      model.stream([new FunctionMessage({ name: "lookup", content: "found" })])
    ).rejects.toThrow("unsupported_history_function_message")
  })

  it("keeps explicitly client-managed inference stateless after server metadata", async () => {
    mocks.streamMessage.mockImplementation(
      async function* (_messages, _options, onChunk) {
        onChunk({ conversation_id: "unrequested-storage" })
        yield "ok"
      }
    )
    const model = new ChatTldw({
      model: "test",
      clientManagedHistory: true,
      conversationId: "old",
      saveToDb: true
    })
    for await (const _ of await model.stream([new HumanMessage("Hi")])) {
    }
    expect(model.saveToDb).toBe(false)
    expect(model.conversationId).toBeUndefined()
    expect(mocks.streamMessage.mock.lastCall?.[1]).toMatchObject({
      saveToDb: false
    })
  })
})

it("prepared client history preserves injected system roles and rejects image loss", () => {
  const model = new ChatTldw({ model: "test", clientManagedHistory: true })
  const request = model.prepareClientManagedRequest([
    { role: "system", content: "dynamic UI" },
    new HumanMessage("next")
  ] as any)
  expect(request.messages[0]).toEqual({ role: "system", content: "dynamic UI" })
  expect(() =>
    model.prepareClientManagedRequest([
      new HumanMessage({
        content: [
          { type: "image_url", image_url: "data:image/png;base64,AAAA" }
        ]
      })
    ])
  ).toThrow("unsupported_history_model_images")
})
