import { act, fireEvent, renderHook } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { useSmartScroll } from "../useSmartScroll"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function setup() {
  const frames: FrameRequestCallback[] = []
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames.push(callback)
    return frames.length
  })
  const element = document.createElement("div")
  Object.defineProperties(element, {
    scrollHeight: { value: 2000 },
    clientHeight: { value: 600 },
    scrollTo: { value: vi.fn() }
  })
  const empty: Array<{ id: string }> = []
  const hook = renderHook(
    ({ messages, streaming, threshold }) =>
      useSmartScroll(messages, streaming, threshold),
    { initialProps: { messages: empty, streaming: false, threshold: 100 } }
  )
  hook.result.current.containerRef.current = element
  return {
    ...hook,
    element,
    frames,
    flush: () =>
      act(() => {
        frames.splice(0).forEach((callback) => callback(0))
      })
  }
}

it.each(["messages", "streaming"])(
  "explicit navigation revokes queued %s automatic scrolling",
  (kind) => {
    const view = setup()
    view.rerender({
      messages: [{ id: "m1" }],
      streaming: kind === "streaming",
      threshold: 100
    })
    expect(view.frames.length).toBeGreaterThan(0)
    act(() => {
      view.result.current.pauseAutoScroll()
    })
    view.flush()
    expect(view.element.scrollTo).not.toHaveBeenCalled()
  }
)

it("explicit return to bottom resumes automatic streaming after navigation", () => {
  const view = setup()
  view.rerender({ messages: [{ id: "m1" }], streaming: false, threshold: 100 })
  act(() => {
    view.result.current.pauseAutoScroll()
  })
  view.flush()
  act(() => {
    view.result.current.autoScrollToBottom()
  })
  expect(view.element.scrollTo).toHaveBeenLastCalledWith({
    top: 2000,
    behavior: "smooth"
  })
  vi.mocked(view.element.scrollTo).mockClear()
  view.rerender({ messages: [{ id: "m1" }], streaming: true, threshold: 100 })
  view.flush()
  expect(view.element.scrollTo).toHaveBeenCalledWith({
    top: 2000,
    behavior: "auto"
  })
})

it("explicit navigation clears a queued bottom re-enable from an earlier scroll", () => {
  vi.useFakeTimers()
  const view = setup()
  view.element.scrollTop = 1400
  view.rerender({ messages: [{ id: "m1" }], streaming: false, threshold: 101 })
  fireEvent.scroll(view.element)
  act(() => {
    view.result.current.pauseAutoScroll()
  })
  act(() => {
    vi.advanceTimersByTime(300)
  })
  expect(view.result.current.isAutoScrollToBottom).toBe(false)
})
