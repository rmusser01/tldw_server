// @vitest-environment jsdom
/**
 * CO-07 (#3101, G12): whether the reader sees the latest messages is kept as
 * state that follows scrolling, resizing and new messages. An unmounted or
 * non-scrollable chat counts as "at the latest", so no floating button
 * covers it, and messages that land below the reader are flagged as new.
 */
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useSmartScroll } from "../useSmartScroll"
import { renderSmartScroll, type FixtureMessage } from "./smart-scroll-fixture"

const conversation: FixtureMessage[] = [
  { id: "u1", isBot: false },
  { id: "a1", isBot: true }
]
const atBottom = { scrollTop: 4400, scrollHeight: 5000, clientHeight: 600 }

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("useSmartScroll at-bottom state", () => {
  it("counts a chat whose scroller has not mounted as showing the latest", () => {
    const { result } = renderHook(() => useSmartScroll([], false, 100))
    expect(result.current.isAutoScrollToBottom).toBe(true)
  })

  it("counts a chat with nothing to scroll as showing the latest", () => {
    const view = renderSmartScroll({
      messages: conversation,
      metrics: { scrollTop: 0, scrollHeight: 500, clientHeight: 600 }
    })
    view.settle()
    // Jumping to a message (search, timeline) pauses following.
    act(() => {
      view.api.pauseAutoScroll()
    })
    view.rerender({ messages: conversation })
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(true)
  })

  it("tracks the reader scrolling away from and back to the latest", () => {
    const view = renderSmartScroll({ messages: conversation, metrics: atBottom })
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(true)

    view.userScrollTo(1000)
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(false)

    view.userScrollTo(4400)
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(true)
  })

  it("updates when the scroller is resized without new messages", () => {
    const view = renderSmartScroll({ messages: conversation, metrics: atBottom })
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(true)

    // The scroller got shorter (for example the composer grew), so the
    // latest lines are now below the fold.
    view.metrics.clientHeight = 300
    view.resize()
    expect(view.api.isAutoScrollToBottom).toBe(false)
  })

  it("flags messages that arrive below the reader as new until they return", () => {
    const view = renderSmartScroll({ messages: conversation, metrics: atBottom })
    view.settle()
    expect(view.api.hasUnseenMessages).toBe(false)

    view.userScrollTo(1000)
    view.settle()
    view.metrics.scrollHeight = 5400
    view.rerender({ messages: [...conversation, { id: "a2", isBot: true }] })
    view.settle()
    expect(view.api.hasUnseenMessages).toBe(true)

    view.userScrollTo(4800)
    view.settle()
    expect(view.api.hasUnseenMessages).toBe(false)
  })

  it("flags a reply that finishes while the reader is scrolled up", () => {
    const view = renderSmartScroll({
      messages: conversation,
      streaming: true,
      metrics: atBottom
    })
    view.settle()
    view.userScrollTo(1000)
    view.settle()
    expect(view.api.hasUnseenMessages).toBe(false)

    view.rerender({ messages: conversation, streaming: false })
    view.settle()
    expect(view.api.hasUnseenMessages).toBe(true)
  })
})
