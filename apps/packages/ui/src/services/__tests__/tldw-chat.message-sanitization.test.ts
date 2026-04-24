import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(async () => undefined),
  createChatCompletion: vi.fn(),
  streamChatCompletion: vi.fn(),
  getConfig: vi.fn(async () => null)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    createChatCompletion: mocks.createChatCompletion,
    streamChatCompletion: mocks.streamChatCompletion,
    getConfig: mocks.getConfig
  }
}))

import {
  TldwChatService,
  getLastChatCompletionDebugSnapshot
} from "@/services/tldw/TldwChat"
import type { ChatMessage } from "@/services/tldw/TldwApiClient"

const makeDefaultResponse = () =>
  new Response(
    JSON.stringify({
      choices: [{ message: { content: "ok" } }]
    }),
    {
      status: 200,
      headers: { "content-type": "application/json" }
    }
  )

describe("TldwChatService message sanitization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    mocks.createChatCompletion.mockResolvedValue(makeDefaultResponse())
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield {
        choices: [{ index: 0, delta: { content: "hello" }, finish_reason: null }]
      }
    })
    mocks.getConfig.mockResolvedValue(null)
  })

  it("falls back to system prompt when request messages are empty (send)", async () => {
    const service = new TldwChatService()

    await service.sendMessage([], {
      model: "gpt-test",
      systemPrompt: "  Be concise.  "
    })

    const request = mocks.createChatCompletion.mock.calls[0][0] as {
      messages: ChatMessage[]
    }
    expect(request.messages).toEqual([
      { role: "system", content: "Be concise." }
    ])
  })

  it("fails locally when no usable messages or system prompt exist", async () => {
    const service = new TldwChatService()

    await expect(
      service.sendMessage(
        [{ role: "user", content: "   " }],
        { model: "gpt-test" }
      )
    ).rejects.toMatchObject({
      message: "Chat completion failed",
      cause: expect.objectContaining({
        message: expect.stringContaining("Cannot send chat request without any messages")
      })
    })
    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
  })

  it("falls back to system prompt when request messages are empty (stream)", async () => {
    const service = new TldwChatService()
    const tokens: string[] = []

    for await (const token of service.streamMessage([], {
      model: "gpt-test",
      systemPrompt: "System prompt"
    })) {
      tokens.push(token)
    }

    const request = mocks.streamChatCompletion.mock.calls[0][0] as {
      messages: ChatMessage[]
    }
    expect(request.messages).toEqual([
      { role: "system", content: "System prompt" }
    ])
    expect(tokens).toEqual(["hello"])
  })

  it("preserves image-only user turns while pruning empty text parts", async () => {
    const service = new TldwChatService()
    const messages: ChatMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "   " },
          {
            type: "image_url",
            image_url: {
              url: "https://example.com/image.png"
            }
          }
        ]
      }
    ]

    await service.sendMessage(messages, { model: "gpt-test" })

    const request = mocks.createChatCompletion.mock.calls[0][0] as {
      messages: ChatMessage[]
    }
    expect(request.messages).toEqual([
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: {
              url: "https://example.com/image.png"
            }
          }
        ]
      }
    ])
  })

  it("captures the last non-stream payload for debugging", async () => {
    const service = new TldwChatService()
    await service.sendMessage([{ role: "user", content: "hello there" }], {
      model: "gpt-test"
    })

    const snapshot = getLastChatCompletionDebugSnapshot()
    expect(snapshot).toMatchObject({
      endpoint: "/api/v1/chat/completions",
      mode: "non-stream",
      request: expect.objectContaining({
        model: "gpt-test",
        stream: false
      })
    })
    expect(snapshot?.request.messages).toEqual([
      { role: "user", content: "hello there" }
    ])
  })

  it("captures the last stream payload for debugging", async () => {
    const service = new TldwChatService()
    for await (const _ of service.streamMessage(
      [{ role: "user", content: "stream hello" }],
      { model: "gpt-test" }
    )) {
      break
    }

    const snapshot = getLastChatCompletionDebugSnapshot()
    expect(snapshot).toMatchObject({
      endpoint: "/api/v1/chat/completions",
      mode: "stream",
      request: expect.objectContaining({
        model: "gpt-test",
        stream: true
      })
    })
    expect(snapshot?.request.messages).toEqual([
      { role: "user", content: "stream hello" }
    ])
  })

  it("includes bounded research context in non-stream requests", async () => {
    const service = new TldwChatService()

    await service.sendMessage([{ role: "user", content: "hello there" }], {
      model: "gpt-test",
      researchContext: {
        run_id: "run_123",
        query: "Battery recycling supply chain",
        question: "What changed in the battery recycling market?",
        outline: [{ title: "Overview" }],
        key_claims: [{ text: "Claim one" }],
        unresolved_questions: ["What changed in Europe?"],
        verification_summary: { unsupported_claim_count: 0 },
        source_trust_summary: { high_trust_count: 3 },
        research_url: "/research?run=run_123"
      }
    } as any)

    const request = mocks.createChatCompletion.mock.calls[0][0] as {
      research_context?: Record<string, unknown>
    }
    expect(request.research_context).toMatchObject({
      run_id: "run_123",
      question: "What changed in the battery recycling market?",
      research_url: "/research?run=run_123"
    })
  })

  it("includes bounded research context in stream requests", async () => {
    const service = new TldwChatService()

    for await (const _ of service.streamMessage(
      [{ role: "user", content: "stream hello" }],
      {
        model: "gpt-test",
        researchContext: {
          run_id: "run_456",
          query: "Grid-scale recycling economics",
          question: "What is changing in the grid-scale recycling market?",
          outline: [{ title: "Market overview" }],
          key_claims: [{ text: "Claim one" }],
          unresolved_questions: ["What changed in the EU?"],
          verification_summary: { unsupported_claim_count: 0 },
          source_trust_summary: { high_trust_count: 2 },
          research_url: "/research?run=run_456"
        }
      } as any
    )) {
      break
    }

    const request = mocks.streamChatCompletion.mock.calls[0][0] as {
      research_context?: Record<string, unknown>
    }
    expect(request.research_context).toMatchObject({
      run_id: "run_456",
      research_url: "/research?run=run_456"
    })
  })

  it("times out when no visible assistant progress arrives", async () => {
    vi.useFakeTimers()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      chatRequestTimeoutMs: 50,
      chatStartupTimeoutMs: 50,
      chatStreamIdleTimeoutMs: 500
    })
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        let seq = 0
        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 20))
          if (options?.signal?.aborted) {
            return
          }
          seq += 1
          yield {
            event: "run_started",
            run_id: "run_1",
            seq,
            data: {}
          }
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "why is prompt engineering important?" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
      }
      return tokens
    })()
    const settled = streamRun.then(
      (value) => ({ status: "resolved" as const, value }),
      (error) => ({ status: "rejected" as const, error })
    )

    await vi.advanceTimersByTimeAsync(80)

    await expect(settled).resolves.toMatchObject({
      status: "rejected",
      error: {
        message: "Stream completion failed",
        cause: expect.objectContaining({
          message: expect.stringContaining("visible output")
        })
      }
    })
  })

  it("maps thrown aborts caused by startup timeout to the timeout error", async () => {
    vi.useFakeTimers()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      chatRequestTimeoutMs: 50,
      chatStartupTimeoutMs: 50,
      chatStreamIdleTimeoutMs: 500
    })
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        let seq = 0
        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 20))
          if (options?.signal?.aborted) {
            const abortError = new Error("The operation was aborted.")
            abortError.name = "AbortError"
            throw abortError
          }
          seq += 1
          yield {
            event: "run_started",
            run_id: "run_abort",
            seq,
            data: {}
          }
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "why did the stream abort?" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
      }
      return tokens
    })()
    const settled = streamRun.then(
      (value) => ({ status: "resolved" as const, value }),
      (error) => ({ status: "rejected" as const, error })
    )

    await vi.advanceTimersByTimeAsync(80)

    await expect(settled).resolves.toMatchObject({
      status: "rejected",
      error: {
        message: "Stream completion failed",
        cause: expect.objectContaining({
          message: expect.stringContaining("visible output")
        })
      }
    })
  })

  it("detects nested abort causes during stream cancellation", async () => {
    vi.useFakeTimers()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      chatRequestTimeoutMs: 500,
      chatStartupTimeoutMs: 500,
      chatStreamIdleTimeoutMs: 500
    })
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        yield {
          event: "run_started",
          run_id: "run_nested_abort",
          seq: 1,
          data: {}
        }
        await new Promise((resolve) => setTimeout(resolve, 20))
        if (options?.signal?.aborted) {
          const abortError = new Error("The operation was aborted.")
          abortError.name = "AbortError"
          const wrappedAbort = new Error("inner wrapper", { cause: abortError })
          throw new Error("outer wrapper", { cause: wrappedAbort })
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "cancel with nested abort" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
      }
      return tokens
    })()
    const settled = streamRun.then(
      (value) => ({ status: "resolved" as const, value }),
      (error) => ({ status: "rejected" as const, error })
    )

    await vi.advanceTimersByTimeAsync(5)
    service.cancelStream()
    await vi.advanceTimersByTimeAsync(40)

    await expect(settled).resolves.toMatchObject({
      status: "rejected",
      error: {
        name: "AbortError",
        message: expect.stringContaining("aborted")
      }
    })
  })

  it("rejects with AbortError when the stream exits cleanly after cancellation", async () => {
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        yield {
          choices: [{ index: 0, delta: { content: "hello" }, finish_reason: null }]
        }
        while (!options?.signal?.aborted) {
          await new Promise((resolve) => setTimeout(resolve, 5))
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "stop after one token" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
        service.cancelStream()
      }
      return tokens
    })()

    await expect(streamRun).rejects.toMatchObject({
      name: "AbortError",
      message: expect.stringMatching(/abort/i)
    })
  })

  it("uses dedicated startup timeout instead of chat request timeout", async () => {
    vi.useFakeTimers()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      chatRequestTimeoutMs: 50,
      chatStartupTimeoutMs: 500,
      chatStreamIdleTimeoutMs: 500
    })
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        await new Promise((resolve) => setTimeout(resolve, 20))
        if (options?.signal?.aborted) {
          return
        }
        yield {
          event: "run_started",
          run_id: "run_1",
          seq: 1,
          data: {}
        }

        await new Promise((resolve) => setTimeout(resolve, 20))
        if (options?.signal?.aborted) {
          return
        }
        yield {
          event: "tool_pending",
          run_id: "run_1",
          seq: 2,
          data: {}
        }

        await new Promise((resolve) => setTimeout(resolve, 30))
        if (options?.signal?.aborted) {
          return
        }
        yield {
          choices: [{ index: 0, delta: { content: "hello" }, finish_reason: null }]
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "say hello after warmup" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
      }
      return tokens
    })()

    await vi.advanceTimersByTimeAsync(120)

    await expect(streamRun).resolves.toEqual(["hello"])
  })

  it("keeps streaming when visible token progress arrives", async () => {
    vi.useFakeTimers()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      chatRequestTimeoutMs: 50,
      chatStreamIdleTimeoutMs: 50
    })
    mocks.streamChatCompletion.mockImplementation(
      async function* (
        _request: unknown,
        options?: { signal?: AbortSignal }
      ) {
        await new Promise((resolve) => setTimeout(resolve, 20))
        if (options?.signal?.aborted) {
          return
        }
        yield {
          choices: [{ index: 0, delta: { content: "hello" }, finish_reason: null }]
        }
        await new Promise((resolve) => setTimeout(resolve, 20))
        if (options?.signal?.aborted) {
          return
        }
        yield {
          choices: [{ index: 0, delta: { content: " world" }, finish_reason: null }]
        }
      }
    )

    const service = new TldwChatService()
    const streamRun = (async () => {
      const tokens: string[] = []
      for await (const token of service.streamMessage(
        [{ role: "user", content: "say hello" }],
        { model: "gpt-test" }
      )) {
        tokens.push(token)
      }
      return tokens
    })()

    await vi.advanceTimersByTimeAsync(80)

    await expect(streamRun).resolves.toEqual(["hello", " world"])
  })
})
