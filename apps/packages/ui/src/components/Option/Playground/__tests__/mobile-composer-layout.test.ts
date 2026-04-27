import { describe, expect, it } from "vitest"
import {
  computeKeyboardInsetPx,
  isKeyboardLikelyOpen,
  resolveMobileComposerViewportState,
  resolveStickyComposerTextareaMaxHeight
} from "../mobile-composer-layout"

describe("mobile-composer-layout", () => {
  it("computes keyboard inset from layout and visual viewport geometry", () => {
    expect(
      computeKeyboardInsetPx({
        layoutViewportHeight: 844,
        visualViewportHeight: 520,
        visualViewportOffsetTop: 18
      })
    ).toBe(306)

    expect(
      computeKeyboardInsetPx({
        layoutViewportHeight: 844,
        visualViewportHeight: 844,
        visualViewportOffsetTop: 0
      })
    ).toBe(0)
  })

  it("detects likely keyboard state with thresholding", () => {
    expect(isKeyboardLikelyOpen({ keyboardInsetPx: 45 })).toBe(false)
    expect(isKeyboardLikelyOpen({ keyboardInsetPx: 120 })).toBe(true)
    expect(
      isKeyboardLikelyOpen({ keyboardInsetPx: 70, thresholdPx: 64 })
    ).toBe(true)
  })

  it("resolves stable viewport state for invalid values", () => {
    expect(
      resolveMobileComposerViewportState({
        layoutViewportHeight: Number.NaN,
        visualViewportHeight: 500
      })
    ).toEqual({
      keyboardInsetPx: 0,
      keyboardOpen: false
    })
  })

  it("keeps sticky composer textarea height below the viewport cap", () => {
    expect(
      resolveStickyComposerTextareaMaxHeight({
        viewportHeightPx: 480,
        keyboardInsetPx: 0,
        isMobileViewport: true,
        defaultMaxHeightPx: 480
      })
    ).toBe(220)
  })
})
