import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(async () => {}),
  getConfig: vi.fn(async () => null),
  createChatCompletion: vi.fn(),
  streamChatCompletion: vi.fn()
}))

vi.mock("../TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    createChatCompletion: (...args: unknown[]) =>
      mocks.createChatCompletion(...args),
    streamChatCompletion: (...args: unknown[]) =>
      mocks.streamChatCompletion(...args)
  }
}))

import { TldwChatService } from "../TldwChat"

const chunk = (content: string) => ({ choices: [{ delta: { content } }] })

describe("TldwChatService abort lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getConfig.mockResolvedValue(null)
  })

  it("identifies successful saved regeneration in both request transports", async () => {
    mocks.createChatCompletion.mockResolvedValue({ json: async () => ({ choices: [{ message: { content: "answer" } }] }) })
    mocks.streamChatCompletion.mockImplementation(async function* () { yield chunk("answer") })
    const service = new TldwChatService()
    const messages = [
      { role: "system" as const, content: "Current instructions" },
      { role: "user" as const, content: "Earlier question" },
      { role: "assistant" as const, content: "Earlier answer" },
      { role: "user" as const, content: "Repeated prompt" }
    ]
    const options = { model: "m", regenerateFromMessageId: "saved-reply" }
    await service.sendMessage(messages, options)
    for await (const _token of service.streamMessage(messages, options)) { /* consume */ }
    for (const [request] of [mocks.createChatCompletion.mock.calls[0], mocks.streamChatCompletion.mock.calls[0]]) {
      expect(request.metadata).toEqual({ tldw_regenerate_from_message_id: "saved-reply" })
      expect(request.messages).toEqual([messages[0], messages[3]])
    }
  })

  it("passes the caller signal to non-streaming chat completion", async () => {
    mocks.createChatCompletion.mockResolvedValue({
      json: async () => ({ choices: [{ message: { content: "answer" } }] })
    })
    const controller = new AbortController()
    const service = new TldwChatService()

    await expect(service.sendMessage(
      [{ role: "user", content: "hi" }],
      { model: "m", signal: controller.signal }
    )).resolves.toBe("answer")

    expect(mocks.createChatCompletion.mock.calls[0]?.[1]).toMatchObject({
      signal: controller.signal
    })
  })

  it.each([false, true])("keeps explicit retry intent in request metadata (retry=%s)", async (retryFailedTurn) => {
    mocks.createChatCompletion.mockResolvedValue({ json: async () => ({ choices: [{ message: { content: "answer" } }] }) })
    mocks.streamChatCompletion.mockImplementation(async function* () { yield chunk("answer") })
    const service = new TldwChatService()
    const messages = [{ role: "user" as const, content: "Retry question" }]
    await service.sendMessage(messages, { model: "m", retryFailedTurn })
    for await (const _token of service.streamMessage(messages, { model: "m", retryFailedTurn })) { /* consume */ }
    for (const request of [mocks.createChatCompletion.mock.calls[0][0], mocks.streamChatCompletion.mock.calls[0][0]]) {
      expect(request.metadata).toEqual(retryFailedTurn ? { tldw_retry_failed_turn: true } : undefined)
      expect(request.messages).toEqual(messages)
      expect(request.extra_body).toBeUndefined()
    }
  })

  it("carries the current local user correlation as app metadata on both transports", async () => {
    mocks.createChatCompletion.mockResolvedValue({ json: async () => ({ choices: [{ message: { content: "answer" } }] }) })
    mocks.streamChatCompletion.mockImplementation(async function* () { yield chunk("answer") })
    const service = new TldwChatService()
    const messages = [{ role: "user" as const, content: "Question" }]
    await service.sendMessage(messages, { model: "m", clientMessageId: "local-user" })
    for await (const _token of service.streamMessage(messages, { model: "m", clientMessageId: "local-user", retryFailedTurn: true })) { /* consume */ }
    expect(mocks.createChatCompletion.mock.calls[0][0].metadata).toEqual({ tldw_client_message_id: "local-user" })
    expect(mocks.streamChatCompletion.mock.calls[0][0].metadata).toEqual({ tldw_client_message_id: "local-user", tldw_retry_failed_turn: true })
    expect(mocks.createChatCompletion.mock.calls[0][0].messages).toEqual(messages)
  })

  it("passes the captured request scope to both completion transports", async () => {
    mocks.createChatCompletion.mockResolvedValue({
      json: async () => ({ choices: [{ message: { content: "answer" } }] })
    })
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield chunk("answer")
    })
    const requestScope = {
      config: {
        serverUrl: "https://research-one.test",
        authMode: "multi-user" as const
      },
      userId: 42
    }
    const service = new TldwChatService()

    await service.sendMessage(
      [{ role: "user", content: "hi" }],
      { model: "m", requestScope }
    )
    for await (const _token of service.streamMessage(
      [{ role: "user", content: "hi" }],
      { model: "m", stream: true, requestScope }
    )) {
      // consume the stream
    }

    expect(mocks.createChatCompletion.mock.calls[0]?.[1]).toMatchObject({
      requestScope
    })
    expect(mocks.streamChatCompletion.mock.calls[0]?.[1]).toMatchObject({
      requestScope
    })
  })

  it("preserves a non-streaming request-scope change error", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.createChatCompletion.mockRejectedValue(scopeChangedError)
    const service = new TldwChatService()

    await expect(service.sendMessage(
      [{ role: "user", content: "hi" }],
      { model: "m" }
    )).rejects.toBe(scopeChangedError)
  })

  it("preserves a streaming request-scope change error", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.streamChatCompletion.mockImplementation(async function* () {
      yield* []
      throw scopeChangedError
    })
    const service = new TldwChatService()
    const stream = service.streamMessage(
      [{ role: "user", content: "hi" }],
      { model: "m", stream: true }
    )

    await expect(stream.next()).rejects.toBe(scopeChangedError)
  })

  it("gives each streamMessage call its own controller so concurrent streams do not cancel each other", async () => {
    const receivedSignals: AbortSignal[] = []
    mocks.streamChatCompletion.mockImplementation(
      async function* (_req: unknown, opts: { signal: AbortSignal }) {
        receivedSignals.push(opts.signal)
        yield chunk("a")
        yield chunk("b")
      }
    )

    const service = new TldwChatService()
    const genA = service.streamMessage(
      [{ role: "user", content: "A" }],
      { model: "m", stream: true }
    )
    const genB = service.streamMessage(
      [{ role: "user", content: "B" }],
      { model: "m", stream: true }
    )

    // Enter both generator bodies so each registers its own controller.
    await genA.next()
    await genB.next()

    expect(receivedSignals).toHaveLength(2)
    expect(receivedSignals[0]).not.toBe(receivedSignals[1])
    // Starting B must not abort A (the old code called `this.cancelStream()`).
    expect(receivedSignals[0].aborted).toBe(false)
    expect(receivedSignals[1].aborted).toBe(false)

    // Drain both so their finally blocks run (clears internal timers).
    await genA.next()
    await genA.next()
    await genB.next()
    await genB.next()
  })

  it("aborts the internal request when the caller's signal fires", async () => {
    let capturedSignal: AbortSignal | undefined
    mocks.streamChatCompletion.mockImplementation(
      async function* (_req: unknown, opts: { signal: AbortSignal }) {
        capturedSignal = opts.signal
        yield chunk("x")
        yield chunk("y")
      }
    )

    const service = new TldwChatService()
    const caller = new AbortController()
    const gen = service.streamMessage(
      [{ role: "user", content: "hi" }],
      { model: "m", stream: true, signal: caller.signal }
    )

    const first = await gen.next()
    expect(first.value).toBe("x")
    expect(capturedSignal?.aborted).toBe(false)

    caller.abort()
    // The caller's signal is threaded into this call's internal controller.
    expect(capturedSignal?.aborted).toBe(true)

    await expect(gen.next()).rejects.toThrow(/abort|cancel/i)
  })

  it("cancelStream aborts every in-flight stream (global stop everything)", async () => {
    const receivedSignals: AbortSignal[] = []
    mocks.streamChatCompletion.mockImplementation(
      async function* (_req: unknown, opts: { signal: AbortSignal }) {
        receivedSignals.push(opts.signal)
        yield chunk("a")
        yield chunk("b")
      }
    )

    const service = new TldwChatService()
    const genA = service.streamMessage(
      [{ role: "user", content: "A" }],
      { model: "m", stream: true }
    )
    const genB = service.streamMessage(
      [{ role: "user", content: "B" }],
      { model: "m", stream: true }
    )
    await genA.next()
    await genB.next()

    service.cancelStream()

    expect(receivedSignals[0].aborted).toBe(true)
    expect(receivedSignals[1].aborted).toBe(true)

    await expect(genA.next()).rejects.toThrow(/abort|cancel/i)
    await expect(genB.next()).rejects.toThrow(/abort|cancel/i)
  })
})
