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


describe("selected history provider fidelity", () => {
  it("round-trips approved Moonshot reasoning through the app API only for browser-local history", async () => {
    const model = new ChatTldw({ model: "kimi-k2.6", apiProvider: "moonshot", clientManagedHistory: true })
    const user = new HumanMessage("Question")
    const preparedRequest = model.prepareClientManagedRequest([user], true)
    expect(preparedRequest.metadata).toMatchObject({ tldw_moonshot_reasoning_history_v1: true })
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      onChunk({ choices: [{ delta: { reasoning_content: "unapproved" } }] })
      yield "Answer"
      onChunk({ event: "moonshot_history_reasoning", data: { reasoning_content: "  exact </think> bytes\n" } })
    })
    const handleLLMEnd = vi.fn()
    for await (const _ of await model.stream([user], { preparedRequest, callbacks: [{ handleLLMEnd }] })) {
    }
    const { moonshotReasoning } = handleLLMEnd.mock.lastCall![0].generations[0][0].generationInfo
    expect(moonshotReasoning).toBe("  exact </think> bytes\n")
    const history = [new AIMessage({ content: "Answer", additional_kwargs: { moonshotReasoning } }), user]
    expect(model.prepareClientManagedRequest(history, true).messages[0]).toMatchObject({
      moonshot_reasoning_content: moonshotReasoning
    })
    expect(model.prepareClientManagedRequest(history).messages[0]).not.toHaveProperty("moonshot_reasoning_content")
    for (const route of [{ model: "auto" }, { model: "test", apiProvider: "openai" }]) {
      const routeClient = new ChatTldw({ ...route, clientManagedHistory: true })
      const localRequest = routeClient.prepareClientManagedRequest(history, true)
      expect(localRequest.metadata).toMatchObject({ tldw_moonshot_reasoning_history_v1: true })
      expect(localRequest.messages[0]).toMatchObject({ moonshot_reasoning_content: moonshotReasoning })
      const serverOwnedRequest = routeClient.prepareClientManagedRequest(history)
      expect(serverOwnedRequest.metadata?.tldw_moonshot_reasoning_history_v1).toBeUndefined()
      expect(serverOwnedRequest.messages[0]).not.toHaveProperty("moonshot_reasoning_content")
    }
  })

  it("replays the Gemini signature captured from a streamed function call", async () => {
    const extraContent = { google: { thought_signature: "c2lnbmF0dXJl" } }
    mocks.streamMessage.mockImplementationOnce(async function* (_messages, _options, onChunk) {
      onChunk({
        choices: [{ delta: { tool_calls: [{
          index: 0,
          id: "call-1",
          function: { name: "lookup", arguments: "{" },
          extra_content: { ...extraContent, ignored_provider: { extra: "ignored" } }
        }] } }]
      })
      onChunk({
        choices: [{ delta: { tool_calls: [{
          index: 0,
          function: { arguments: "}" },
          extra_content: { google: { thought_signature: "a".repeat(100_001) } }
        }] } }]
      })
      yield ""
    })
    const model = new ChatTldw({ model: "gemini-3-flash-preview", clientManagedHistory: true })
    const handleLLMEnd = vi.fn()
    for await (const _ of await model.stream([new HumanMessage("Look it up")], {
      callbacks: [{ handleLLMEnd }]
    })) {
      // Allow the completion callback to run before checking metadata.
    }
    const { tool_calls } = handleLLMEnd.mock.lastCall![0].generations[0][0].generationInfo
    const request = model.prepareClientManagedRequest([
      new AIMessage({ content: "", additional_kwargs: { tool_calls } }),
      new ToolMessage({ content: "found", tool_call_id: "call-1" })
    ])
    expect(request.messages).toEqual([
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call-1",
          type: "function",
          function: { name: "lookup", arguments: "{}" },
          extra_content: extraContent
        }]
      },
      { role: "tool", content: "found", tool_call_id: "call-1" }
    ])
  })

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
      // Consume the response before checking the recorded request.
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
      // Consume the response before checking client state.
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
