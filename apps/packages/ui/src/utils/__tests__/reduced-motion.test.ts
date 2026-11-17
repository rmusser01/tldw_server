import { afterEach, describe, expect, it } from "vitest"
import { mockReducedMotion } from "@/test-utils/reducedMotion"
import { prefersReducedMotion, smoothScrollBehavior } from "../reduced-motion"

// AX-19 (#3124): scripted smooth scrolling must follow the OS preference.
describe("reduced motion helpers", () => {
  const original = window.matchMedia
  afterEach(() => {
    window.matchMedia = original
  })

  it.each([
    [true, "auto"],
    [false, "smooth"]
  ] as const)("reduce=%s gives %s scrolling", (reduce, behavior) => {
    mockReducedMotion(reduce)
    expect(prefersReducedMotion()).toBe(reduce)
    expect(smoothScrollBehavior()).toBe(behavior)
  })

  it("falls back to motion when matchMedia is unavailable", () => {
    window.matchMedia = undefined as unknown as typeof window.matchMedia
    expect(prefersReducedMotion()).toBe(false)
    expect(smoothScrollBehavior()).toBe("smooth")
  })
})
