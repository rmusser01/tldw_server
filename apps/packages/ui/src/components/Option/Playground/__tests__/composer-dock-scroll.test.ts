// @vitest-environment jsdom
// AX-04 (#3124): /chat focuses the composer on load, while the dock is still
// small, so the dock scrolled the textarea's bottom into view. When the dock
// then grew, the old offset left the textarea above its visible area (at
// 320x256 nothing showed it). Revealing the focused control again whenever
// the dock resizes fixes that.
import { describe, expect, it } from "vitest"

import { revealInScroller } from "../composer-dock-scroll"

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
