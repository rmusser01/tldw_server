import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useSmartScroll } from "../useSmartScroll"

// XS-09: the last markdown, table or code render grows the transcript after
// streaming ends. While the view follows the reply it must stay pinned to the
// bottom instead of stopping short.

type ResizeCallback = () => void

const observers: { callback: ResizeCallback; targets: Element[] }[] = []

class FakeResizeObserver {
  private entry: { callback: ResizeCallback; targets: Element[] }
  constructor(callback: ResizeCallback) {
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

const notifyResize = () => {
  act(() => {
    observers
      .filter((entry) => entry.targets.length > 0)
      .forEach((entry) => entry.callback())
  })
}

const Transcript = ({ pin }: { pin: boolean }) => {
  const { containerRef } = useSmartScroll([{ id: "m-1" }], false, 100, {
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
})
