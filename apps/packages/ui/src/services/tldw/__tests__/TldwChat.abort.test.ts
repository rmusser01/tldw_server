import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(async () => {}),
  getConfig: vi.fn(async () => null),
  streamChatCompletion: vi.fn()
}))

vi.mock("../TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
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
