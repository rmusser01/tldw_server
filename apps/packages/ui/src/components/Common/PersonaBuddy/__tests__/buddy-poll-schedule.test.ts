import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  BUDDY_POLL_TICK_MS,
  buddyPollTicksAfter,
  startBuddyPoll,
  type BuddyPoll
} from "../buddy-poll-schedule"

// XP-16: Buddy reads back off while nothing changes and pause in hidden tabs.

const setVisibility = (state: DocumentVisibilityState) => {
  vi.spyOn(document, "visibilityState", "get").mockReturnValue(state)
  document.dispatchEvent(new Event("visibilitychange"))
}

describe("Buddy poll schedule (XP-16)", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("doubles the wait after each quiet read, up to once a minute", () => {
    expect([0, 1, 2, 3, 4, 10].map(buddyPollTicksAfter)).toEqual([
      1, 2, 4, 8, 12, 12
    ])
  })

  it("reads at the times the schedule gives while nothing changes", () => {
    const readAt: number[] = []
    let now = 0
    let poll: BuddyPoll | undefined
    poll = startBuddyPoll(() => {
      readAt.push(now)
      poll?.settle("same")
    })
    // The first read runs before startBuddyPoll returns.
    poll.settle("same")
    for (let tick = 0; tick < 24; tick += 1) {
      now += BUDDY_POLL_TICK_MS
      vi.advanceTimersByTime(BUDDY_POLL_TICK_MS)
    }
    poll.stop()
    expect(readAt).toEqual([0, 5_000, 15_000, 35_000, 75_000])
  })

  it("does not re-read when the page is hidden and shown between ticks", () => {
    const read = vi.fn()
    const poll = startBuddyPoll(read)
    expect(read).toHaveBeenCalledTimes(1)
    setVisibility("hidden")
    setVisibility("visible")
    expect(read).toHaveBeenCalledTimes(1)
    poll.stop()
  })

  it("stops reading once stopped", () => {
    const read = vi.fn()
    const poll = startBuddyPoll(read)
    poll.stop()
    vi.advanceTimersByTime(60_000)
    setVisibility("hidden")
    setVisibility("visible")
    expect(read).toHaveBeenCalledTimes(1)
  })

  it("issues no reads at all while the page stays hidden", () => {
    setVisibility("hidden")
    const read = vi.fn()
    const poll = startBuddyPoll(read)
    vi.advanceTimersByTime(60_000)
    expect(read).not.toHaveBeenCalled()
    setVisibility("visible")
    expect(read).toHaveBeenCalledTimes(1)
    poll.stop()
  })

  it("reads only once a minute while the buddy panel is closed (WP-39)", () => {
    const readAt: number[] = []
    let now = 0
    let poll: BuddyPoll | undefined
    poll = startBuddyPoll(
      () => {
        readAt.push(now)
        poll?.settle("same")
      },
      { isActive: () => false }
    )
    for (let tick = 0; tick < 26; tick += 1) {
      now += BUDDY_POLL_TICK_MS
      vi.advanceTimersByTime(BUDDY_POLL_TICK_MS)
    }
    poll.stop()
    expect(readAt).toEqual([0, 60_000, 120_000])
  })

  it("keeps the 5 s cadence while the buddy panel is open and reads change", () => {
    const readAt: number[] = []
    let now = 0
    let poll: BuddyPoll | undefined
    poll = startBuddyPoll(
      () => {
        readAt.push(now)
        poll?.settle(`changed-${readAt.length}`)
      },
      { isActive: () => true }
    )
    for (let tick = 0; tick < 4; tick += 1) {
      now += BUDDY_POLL_TICK_MS
      vi.advanceTimersByTime(BUDDY_POLL_TICK_MS)
    }
    poll.stop()
    expect(readAt).toEqual([0, 5_000, 10_000, 15_000, 20_000])
  })

  it("re-reads when the window regains focus after a tick has passed", () => {
    const read = vi.fn()
    const poll = startBuddyPoll(read)
    // Report the initial read as quiet twice so the interval backs off
    // (two ticks of unchanged data => next read four ticks out).
    poll.settle("same")
    poll.settle("same")
    expect(read).toHaveBeenCalledTimes(1)
    window.dispatchEvent(new Event("focus"))
    expect(read).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(BUDDY_POLL_TICK_MS)
    // The quiet-read backoff postpones the next interval read...
    expect(read).toHaveBeenCalledTimes(1)
    // ...and focus brings it forward.
    window.dispatchEvent(new Event("focus"))
    expect(read).toHaveBeenCalledTimes(2)
    poll.stop()
  })

  it("stops listening for focus once stopped", () => {
    const read = vi.fn()
    const poll = startBuddyPoll(read)
    poll.stop()
    vi.advanceTimersByTime(BUDDY_POLL_TICK_MS)
    window.dispatchEvent(new Event("focus"))
    expect(read).toHaveBeenCalledTimes(1)
  })
})
