import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { startWorkspaceStatusPoll } from "../workspace-status-poll-scheduler"

// WP-42: one poller for Research Workspace source statuses, ticking only
// while sources are processing and the page is visible.

const INTERVAL_MS = 5_000

const setVisibility = (state: DocumentVisibilityState) => {
  vi.spyOn(document, "visibilityState", "get").mockReturnValue(state)
  document.dispatchEvent(new Event("visibilitychange"))
}

describe("workspace status poll scheduler (WP-42)", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("issues zero polls while no source is processing", () => {
    const poll = vi.fn()
    const scheduler = startWorkspaceStatusPoll({
      intervalMs: INTERVAL_MS,
      shouldPoll: () => false,
      poll
    })
    vi.advanceTimersByTime(60_000)
    expect(poll).not.toHaveBeenCalled()
    scheduler.stop()
  })

  it("polls on the interval while sources are processing", () => {
    const poll = vi.fn()
    const scheduler = startWorkspaceStatusPoll({
      intervalMs: INTERVAL_MS,
      shouldPoll: () => true,
      poll
    })
    vi.advanceTimersByTime(15_000)
    expect(poll).toHaveBeenCalledTimes(3)
    scheduler.stop()
  })

  it("issues no polls while the page is hidden and catches up once when visible", () => {
    const poll = vi.fn()
    const scheduler = startWorkspaceStatusPoll({
      intervalMs: INTERVAL_MS,
      shouldPoll: () => true,
      poll
    })
    setVisibility("hidden")
    vi.advanceTimersByTime(15_000)
    expect(poll).not.toHaveBeenCalled()

    setVisibility("visible")
    expect(poll).toHaveBeenCalledTimes(1)

    vi.advanceTimersByTime(5_000)
    expect(poll).toHaveBeenCalledTimes(2)
    scheduler.stop()
  })

  it("uses exactly one interval and one visibilitychange listener, both cleaned up", () => {
    const addListener = vi.spyOn(document, "addEventListener")
    const removeListener = vi.spyOn(document, "removeEventListener")
    const poll = vi.fn()
    const scheduler = startWorkspaceStatusPoll({
      intervalMs: INTERVAL_MS,
      shouldPoll: () => true,
      poll
    })

    const visibilityRegistrations = addListener.mock.calls.filter(
      ([type]) => type === "visibilitychange"
    )
    expect(visibilityRegistrations).toHaveLength(1)

    scheduler.stop()
    vi.advanceTimersByTime(60_000)
    setVisibility("visible")
    expect(poll).not.toHaveBeenCalled()
    expect(
      removeListener.mock.calls.filter(([type]) => type === "visibilitychange")
    ).toHaveLength(1)
  })
})
