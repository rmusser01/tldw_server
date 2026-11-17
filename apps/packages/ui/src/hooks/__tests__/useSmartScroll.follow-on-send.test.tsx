// @vitest-environment jsdom
/**
 * CM-11 (#3101, G17): sending while scrolled up must bring the reader to
 * their message and the reply, and a scroll position that drops only because
 * the content shrank must not count as the reader scrolling up.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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

describe("useSmartScroll follows the conversation on send", () => {
  it("brings the reader to the latest message when they send while scrolled up", () => {
    const view = renderSmartScroll({ messages: conversation, metrics: atBottom })
    view.settle()
    view.userScrollTo(1500)
    view.settle()
    expect(view.api.isAutoScrollToBottom).toBe(false)
    view.scrollTo.mockClear()

    view.metrics.scrollHeight = 5400
    view.rerender({
      messages: [
        ...conversation,
        { id: "u2", isBot: false },
        { id: "a2", isBot: true, message: "" }
      ],
      streaming: true
    })
    view.settle()
    expect(view.scrollTo).toHaveBeenLastCalledWith(
      expect.objectContaining({ top: 5400 })
    )

    // ...and keeps following the reply as it streams in.
    view.metrics.scrollHeight = 5800
    view.rerender({
      messages: [
        ...conversation,
        { id: "u2", isBot: false },
        { id: "a2", isBot: true, message: "Streaming reply" }
      ],
      streaming: true
    })
    view.settle()
    expect(view.scrollTo).toHaveBeenLastCalledWith(
      expect.objectContaining({ top: 5800 })
    )
  })

  it("keeps following a streaming reply when its content briefly shrinks", () => {
    const view = renderSmartScroll({
      messages: conversation,
      streaming: true,
      metrics: atBottom
    })
    // The reader is reading at the bottom.
    view.userScrollTo(4400)
    view.settle()
    view.scrollTo.mockClear()

    // The reply re-renders shorter for a frame; the browser clamps scrollTop.
    view.metrics.scrollHeight = 4800
    view.metrics.scrollTop = 4200
    view.fireScroll()

    view.metrics.scrollHeight = 5300
    view.rerender({
      messages: [conversation[0], { id: "a1", isBot: true, message: "longer" }],
      streaming: true
    })
    view.settle()
    expect(view.scrollTo).toHaveBeenLastCalledWith(
      expect.objectContaining({ top: 5300 })
    )
  })

  it("still pauses following when the reader really scrolls up", () => {
    const view = renderSmartScroll({
      messages: conversation,
      streaming: true,
      metrics: atBottom
    })
    view.settle()
    view.userScrollTo(3000)
    view.settle()
    view.scrollTo.mockClear()

    view.metrics.scrollHeight = 5300
    view.rerender({
      messages: [conversation[0], { id: "a1", isBot: true, message: "longer" }],
      streaming: true
    })
    view.settle()
    expect(view.scrollTo).not.toHaveBeenCalled()
  })

  it("does not pull the reader down when only an assistant message arrives", () => {
    const view = renderSmartScroll({ messages: conversation, metrics: atBottom })
    view.settle()
    view.userScrollTo(1500)
    view.settle()
    view.scrollTo.mockClear()

    view.metrics.scrollHeight = 5400
    view.rerender({ messages: [...conversation, { id: "a2", isBot: true }] })
    view.settle()
    expect(view.scrollTo).not.toHaveBeenCalled()
  })

  it("does not jump to the latest message when older messages load above", () => {
    const latest: FixtureMessage[] = [
      { id: "u2", isBot: false },
      { id: "a2", isBot: true }
    ]
    const view = renderSmartScroll({ messages: latest, metrics: atBottom })
    view.settle()
    view.userScrollTo(1500)
    view.settle()
    view.scrollTo.mockClear()

    view.metrics.scrollHeight = 5400
    view.rerender({ messages: [...conversation, ...latest] })
    view.settle()
    expect(view.scrollTo).not.toHaveBeenCalled()
  })
})
