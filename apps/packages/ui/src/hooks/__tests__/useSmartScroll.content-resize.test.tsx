import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useSmartScroll } from "../useSmartScroll"

// XS-09: the last markdown, table or code render grows the transcript after
// streaming ends. While the view follows the reply it must stay pinned to the
// bottom instead of stopping short.

type ResizeCallback = () => void

const observers: { callback: ResizeCallback; targets: Element[] }[] = []
let observerCount = 0

class FakeResizeObserver {
  private entry: { callback: ResizeCallback; targets: Element[] }
  constructor(callback: ResizeCallback) {
    observerCount += 1
    this.entry = { callback, targets: [] }
    observers.push(this.entry)
  }
  observe(target: Element) {
    this.entry.targets.push(target)
  }
  unobserve() {}
  disconnect() {
    this.entry.targets = []
  }
}

// The hook also keeps a viewport observer on the scroller itself; the content
// observer is the one that watches the transcript's children.
const contentObservers = () =>
  observers.filter((entry) =>
    entry.targets.some((target) => target.getAttribute("data-testid") === "content")
  )

const notifyResize = () => {
  act(() => {
    observers
      .filter((entry) => entry.targets.length > 0)
      .forEach((entry) => entry.callback())
  })
}

const Transcript = ({ pin, tokens = 0 }: { pin: boolean; tokens?: number }) => {
  // A new array for every streamed token, like a reply's message list.
  const messages = React.useMemo(() => [{ id: "m-1", tokens }], [tokens])
  const { containerRef } = useSmartScroll(messages, tokens > 0, 100, {
    pinOnContentResize: pin
  })
  return (
    <div ref={containerRef} data-testid="scroller">
      <div data-testid="content" />
    </div>
  )
}

const measure = (element: HTMLElement, initial: { scrollHeight: number; clientHeight: number; scrollTop: number }) => {
  const box = { ...initial }
  Object.defineProperty(element, "scrollHeight", { configurable: true, get: () => box.scrollHeight })
  Object.defineProperty(element, "clientHeight", { configurable: true, get: () => box.clientHeight })
  Object.defineProperty(element, "scrollTop", {
    configurable: true,
    get: () => box.scrollTop,
    set: (value: number) => {
      box.scrollTop = Number(value)
    }
  })
  Object.defineProperty(element, "scrollTo", {
    configurable: true,
    value: ({ top }: { top: number }) => {
      box.scrollTop = Number(top)
    }
  })
  return box
}

describe("useSmartScroll content growth", () => {
  beforeEach(() => {
    observers.length = 0
    observerCount = 0
    vi.stubGlobal("ResizeObserver", FakeResizeObserver)
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0)
      return 1
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("re-pins to the bottom when the transcript grows while following the reply", () => {
    render(<Transcript pin />)
    const scroller = screen.getByTestId("scroller")
    const box = measure(scroller, { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 })

    box.scrollHeight = 1300
    notifyResize()

    expect(box.scrollTop).toBe(1300)
  })

  it("leaves the reader in place after they scroll up", () => {
    render(<Transcript pin />)
    const scroller = screen.getByTestId("scroller")
    const box = measure(scroller, { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 })
    fireEvent.scroll(scroller)
    box.scrollTop = 200
    fireEvent.scroll(scroller)

    box.scrollHeight = 1300
    notifyResize()

    expect(box.scrollTop).toBe(200)
  })

  it("does not watch content size unless asked to", () => {
    render(<Transcript pin={false} />)
    const scroller = screen.getByTestId("scroller")
    const box = measure(scroller, { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 })

    box.scrollHeight = 1300
    notifyResize()

    expect(box.scrollTop).toBe(600)
  })

  it("does not snap back when the reader had already left the bottom before the growth", () => {
    render(<Transcript pin />)
    const scroller = screen.getByTestId("scroller")
    const box = measure(scroller, { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 })
    notifyResize()
    // A short upward scroll made while a programmatic scroll was settling is
    // ignored by the scroll handler, so auto-follow is still on.
    box.scrollTop = 300

    box.scrollHeight = 1300
    notifyResize()

    expect(box.scrollTop).toBe(300)
  })

  it("keeps one content observer while a reply streams", () => {
    const { rerender } = render(<Transcript pin />)
    measure(screen.getByTestId("scroller"), { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 })
    const observersAfterMount = observerCount
    expect(contentObservers()).toHaveLength(1)
    for (let tokens = 1; tokens <= 5; tokens += 1) {
      rerender(<Transcript pin tokens={tokens} />)
    }
    expect(observerCount).toBe(observersAfterMount)
    expect(contentObservers()).toHaveLength(1)
  })

  it("watches transcript content that renders after mount", async () => {
    render(<Transcript pin />)
    const scroller = screen.getByTestId("scroller")
    const late = document.createElement("div")
    await act(async () => {
      scroller.appendChild(late)
      await Promise.resolve()
    })
    expect(contentObservers()[0].targets).toContain(late)
  })
})
