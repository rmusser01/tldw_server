// Shared streaming throttle helper — unit tests (TASK-13521, Batch W1 Stage 1).
//
// The helper coalesces high-frequency streamed text into at most one store
// update per STREAMING_UPDATE_INTERVAL_MS (80ms) window and lands the update
// via last-index replacement (slice) instead of a full-transcript `prev.map`
// clone. These tests pin the contract the three consumers rely on:
//
//   - cadence: 500 chunks paced <80ms apart produce at most
//     ceil(elapsed/80)+1 store updates,
//   - last chunk is never dropped (trailing flushNow lands the latest text),
//   - cancel drops a pending update (turn restored), disarm keeps it (abort),
//   - non-streaming messages keep referential identity and the array is not
//     cloned through `.map`.
//
// Run: cd apps/packages/ui && bun run vitest run src/hooks/chat/__tests__/streamingThrottle.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  STREAMING_UPDATE_INTERVAL_MS,
  createStreamingTextScheduler,
  replaceStreamingMessage,
} from "../streamingThrottle"
import type { Message } from "@/store/option"

const messageWithId = (id: string, text: string): Message =>
  ({
    id,
    isBot: true,
    role: "assistant",
    name: "Sender",
    message: text,
    sources: [],
    createdAt: 1
  }) as Message

const tokenForChunk = (index: number) => `token-${index} `

describe("replaceStreamingMessage", () => {
  it("replaces only the target message and keeps every other reference identical", () => {
    const seeds = [
      messageWithId("seed-0", "a"),
      messageWithId("seed-1", "b"),
      messageWithId("streaming", "partial"),
      messageWithId("seed-2", "c")
    ]
    const next = replaceStreamingMessage(seeds, "streaming", (message) => ({
      ...message,
      message: "complete"
    }))

    expect(next).not.toBe(seeds)
    expect(next[0]).toBe(seeds[0])
    expect(next[1]).toBe(seeds[1])
    expect(next[3]).toBe(seeds[3])
    expect(next[2]?.message).toBe("complete")
  })

  it("targets the last occurrence of the id", () => {
    const seeds = [
      messageWithId("dup", "first"),
      messageWithId("dup", "last")
    ]
    const next = replaceStreamingMessage(seeds, "dup", (message) => ({
      ...message,
      message: "updated"
    }))

    expect(next[0]?.message).toBe("first")
    expect(next[1]?.message).toBe("updated")
  })

  it("returns the previous array untouched when the id is absent", () => {
    const seeds = [messageWithId("seed-0", "a")]
    const next = replaceStreamingMessage(seeds, "gone", (message) => ({
      ...message,
      message: "updated"
    }))

    expect(next).toBe(seeds)
  })
})

describe("createStreamingTextScheduler", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const createStore = () => {
    const store = {
      messages: [
        messageWithId("seed-0", "a"),
        messageWithId("seed-1", "b"),
        messageWithId("streaming", "▋")
      ],
      updates: 0,
      setMessages: (updater: (prev: Message[]) => Message[]) => {
        const next = updater(store.messages)
        if (next !== store.messages) store.updates += 1
        store.messages = next
      }
    }
    return store
  }

  it("coalesces 500 chunks paced <80ms apart into at most ceil(elapsed/80)+1 updates", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler(
      "streaming",
      store.setMessages
    )

    const CHUNKS = 500
    const PACE_MS = 8
    const elapsedMs = CHUNKS * PACE_MS
    let fullText = ""
    for (let index = 0; index < CHUNKS; index += 1) {
      vi.advanceTimersByTime(PACE_MS)
      fullText += tokenForChunk(index)
      scheduler.schedule(`${fullText}▋`, 0)
    }

    const maxUpdates = Math.ceil(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) + 1
    expect(store.updates).toBeLessThanOrEqual(maxUpdates)
    // Coalescing must not degenerate into per-chunk updates either.
    expect(store.updates).toBeGreaterThanOrEqual(
      Math.floor(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) - 1
    )

    scheduler.flushNow()
    expect(store.messages[2]?.message).toBe(`${fullText}▋`)
  })

  it("never applies a store update through Array.prototype.map on the transcript", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler(
      "streaming",
      store.setMessages
    )

    const originalMap = Array.prototype.map
    let mapClonesOnTranscript = 0
    const spy = vi
      .spyOn(Array.prototype, "map")
      .mockImplementation(function (this: unknown[], mapper, thisArg) {
        if (this.length === store.messages.length) mapClonesOnTranscript += 1
        return originalMap.call(this, mapper as never, thisArg)
      })

    try {
      for (let index = 0; index < 200; index += 1) {
        vi.advanceTimersByTime(4)
        scheduler.schedule(`${tokenForChunk(index)}▋`, 0)
      }
      scheduler.flushNow()
    } finally {
      spy.mockRestore()
    }

    expect(mapClonesOnTranscript).toBe(0)
    expect(store.messages[0]?.message).toBe("a")
    expect(store.messages[1]?.message).toBe("b")
  })

  it("flushNow lands the last chunk scheduled inside an open window (last chunk not dropped)", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler(
      "streaming",
      store.setMessages
    )

    // All chunks land inside the very first 80ms window: no timer has fired.
    scheduler.schedule("one▋", 0)
    scheduler.schedule("one two▋", 0)
    scheduler.schedule("one two three▋", 0)
    expect(store.updates).toBe(0)

    scheduler.flushNow()
    expect(store.updates).toBe(1)
    expect(store.messages[2]?.message).toBe("one two three▋")

    // The trailing timer was disarmed: no second update fires later.
    vi.advanceTimersByTime(1_000)
    expect(store.updates).toBe(1)
  })

  it("cancel drops a pending update so a restored turn cannot be overwritten", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler(
      "streaming",
      store.setMessages
    )

    scheduler.schedule("partial▋", 0)
    scheduler.cancel()
    vi.advanceTimersByTime(1_000)
    scheduler.flushNow()

    expect(store.updates).toBe(0)
    expect(store.messages[2]?.message).toBe("▋")
  })

  it("disarm keeps the pending text so a later flushNow can still land it", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler(
      "streaming",
      store.setMessages
    )

    scheduler.schedule("partial▋", 0)
    scheduler.disarm()
    vi.advanceTimersByTime(1_000)
    expect(store.updates).toBe(0)

    scheduler.flushNow()
    expect(store.updates).toBe(1)
    expect(store.messages[2]?.message).toBe("partial▋")
  })

  it("shouldFlush=false drops the pending update at flush time", () => {
    const store = createStore()
    const scheduler = createStreamingTextScheduler("streaming", store.setMessages, {
      shouldFlush: () => false
    })

    scheduler.schedule("partial▋", 0)
    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS)

    expect(store.updates).toBe(0)
    expect(scheduler.hasPending()).toBe(false)
  })
})
