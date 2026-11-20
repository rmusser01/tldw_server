// @vitest-environment jsdom
// AX-04 (#3124): /chat focuses the composer on load, while the dock is still
// small, so the dock scrolled the textarea's bottom into view. When the dock
// then grew, the old offset left the textarea above its visible area (at
// 320x256 nothing showed it). Revealing the focused control again whenever
// the dock resizes fixes that.
import { renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import {
  revealInScroller,
  useKeepFocusedComposerControlInView
} from "../composer-dock-scroll"

const rect = (top: number, height: number) =>
  ({ top, bottom: top + height, height }) as DOMRect

const fakeScroller = (scrollTop: number, top: number, height: number) => {
  const scroller = document.createElement("div")
  scroller.scrollTop = scrollTop
  scroller.getBoundingClientRect = () => rect(top, height)
  return scroller
}

const fakeElement = (top: number, height: number) => {
  const element = document.createElement("textarea")
  element.getBoundingClientRect = () => rect(top, height)
  return element
}

describe("revealInScroller", () => {
  it("scrolls back up when the element sits above the visible area", () => {
    // Dock visible 100..250; the textarea is 38px above it.
    const scroller = fakeScroller(83, 100, 150)
    revealInScroller(scroller, fakeElement(62, 40))
    expect(scroller.scrollTop).toBe(45)
  })

  it("scrolls down when the element sits below the visible area", () => {
    const scroller = fakeScroller(0, 100, 150)
    revealInScroller(scroller, fakeElement(240, 40))
    expect(scroller.scrollTop).toBe(30)
  })

  it("leaves a visible element alone", () => {
    const scroller = fakeScroller(20, 100, 150)
    revealInScroller(scroller, fakeElement(120, 40))
    expect(scroller.scrollTop).toBe(20)
  })

  it("keeps the top of an element taller than the area in view", () => {
    const scroller = fakeScroller(0, 100, 150)
    revealInScroller(scroller, fakeElement(300, 400))
    expect(scroller.scrollTop).toBe(200)
  })
})

// AX-04 review: reveal the whole composer row (textarea and Send), and leave
// a scroll the user made inside the dock alone.
describe("useKeepFocusedComposerControlInView", () => {
  const setup = () => {
    const scroller = fakeScroller(0, 100, 150)
    Object.defineProperty(scroller, "clientHeight", { value: 150 })
    const row = document.createElement("div")
    row.setAttribute("data-composer-row", "")
    const textarea = fakeElement(240, 40)
    row.getBoundingClientRect = () => rect(230, 58)
    row.appendChild(textarea)
    scroller.appendChild(row)
    document.body.appendChild(scroller)
    textarea.focus()
    const ref = { current: scroller }
    const hook = renderHook(
      ({ height }: { height: number }) =>
        useKeepFocusedComposerControlInView(ref, height),
      { initialProps: { height: 100 } }
    )
    return { scroller, hook }
  }

  it("brings the whole composer row into view, not just the textarea", () => {
    const { scroller } = setup()
    // The row ends at 288; the area ends at 250.
    expect(scroller.scrollTop).toBe(38)
    scroller.remove()
  })

  it("leaves a scroll the user made since the last resize alone", () => {
    const { scroller, hook } = setup()
    scroller.scrollTop = 5
    scroller.dispatchEvent(new Event("scroll"))

    hook.rerender({ height: 120 })
    expect(scroller.scrollTop).toBe(5)

    // A later resize without a user scroll reveals again.
    hook.rerender({ height: 140 })
    expect(scroller.scrollTop).toBe(43)
    scroller.remove()
  })
})
