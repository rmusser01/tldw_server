import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  createStreamingUpdateScheduler,
  STREAMING_UPDATE_INTERVAL_MS,
} from "@/utils/streaming-update-scheduler"

describe("createStreamingUpdateScheduler", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("flushes at most once per interval while chunks stream", () => {
    const applied: string[] = []
    const scheduler = createStreamingUpdateScheduler<string>({
      apply: (value) => applied.push(value),
    })

    // 200 chunks arriving over 400ms (one every 2ms).
    const TOTAL_MS = 400
    const CHUNK_COUNT = 200
    for (let i = 0; i < CHUNK_COUNT; i++) {
      scheduler.schedule(`chunk-${i}`)
      vi.advanceTimersByTime(TOTAL_MS / CHUNK_COUNT)
    }
    scheduler.flushNow()

    const maxFlushes = Math.ceil(TOTAL_MS / STREAMING_UPDATE_INTERVAL_MS) + 1
    expect(applied.length).toBeLessThanOrEqual(maxFlushes)
    expect(applied.length).toBeGreaterThan(1)
    // Final content is complete: the last scheduled value landed.
    expect(applied[applied.length - 1]).toBe(`chunk-${CHUNK_COUNT - 1}`)
    // No trailing timer remains armed after flushNow.
    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS * 3)
    const lengthAfterSettle = applied.length
    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS * 3)
    expect(applied.length).toBe(lengthAfterSettle)
  })

  it("flushes a same-tick burst exactly once", () => {
    const apply = vi.fn()
    const scheduler = createStreamingUpdateScheduler<number>({ apply })

    for (let i = 0; i < 500; i++) {
      scheduler.schedule(i)
    }
    expect(apply).not.toHaveBeenCalled()

    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS)
    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenCalledWith(499)

    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS * 5)
    expect(apply).toHaveBeenCalledTimes(1)
  })

  it("flushNow writes pending content synchronously and disarms the timer", () => {
    const apply = vi.fn()
    const scheduler = createStreamingUpdateScheduler<string>({ apply })

    scheduler.schedule("partial-text")
    scheduler.flushNow()

    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenCalledWith("partial-text")

    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS * 2)
    expect(apply).toHaveBeenCalledTimes(1)
  })

  it("cancel drops pending content so a late flush cannot clobber restored state", () => {
    const apply = vi.fn()
    const scheduler = createStreamingUpdateScheduler<string>({ apply })

    scheduler.schedule("streamed-text")
    expect(scheduler.hasPending()).toBe(true)
    scheduler.cancel()
    expect(scheduler.hasPending()).toBe(false)

    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS * 3)
    scheduler.flushNow()
    expect(apply).not.toHaveBeenCalled()
  })

  it("drops pending content when shouldFlush returns false at flush time", () => {
    const apply = vi.fn()
    const shouldFlush = vi.fn(() => false)
    const scheduler = createStreamingUpdateScheduler<string>({
      apply,
      shouldFlush,
    })

    scheduler.schedule("stale-text")
    vi.advanceTimersByTime(STREAMING_UPDATE_INTERVAL_MS)

    expect(shouldFlush).toHaveBeenCalledTimes(1)
    expect(apply).not.toHaveBeenCalled()
    expect(scheduler.hasPending()).toBe(false)
  })

  it("honors a custom interval", () => {
    const apply = vi.fn()
    const scheduler = createStreamingUpdateScheduler<string>({
      apply,
      intervalMs: 300,
    })

    // The first chunk flushes immediately (no previous flush to wait out).
    scheduler.schedule("a")
    vi.advanceTimersByTime(0)
    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenCalledWith("a")

    // Later chunks wait out the full custom interval.
    scheduler.schedule("b")
    vi.advanceTimersByTime(299)
    expect(apply).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(1)
    expect(apply).toHaveBeenCalledTimes(2)
    expect(apply).toHaveBeenLastCalledWith("b")
  })
})
