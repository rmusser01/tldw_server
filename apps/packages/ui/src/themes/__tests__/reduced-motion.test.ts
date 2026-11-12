import { describe, expect, it } from "vitest"
import { elementFrom, rulesTargeting, sharedStylesheetRules } from "./shared-stylesheet-rules"

// AX-19 (#3124): chat's shake, pulse and bounce animations ignored
// prefers-reduced-motion; only a few page-specific classes were covered.

const reducedMotionDeclarations = (html: string) =>
  Object.fromEntries(
    rulesTargeting(elementFrom(html), { mediaIncludes: "prefers-reduced-motion: reduce" })
      .flatMap((rule) => rule.declarations)
      .map((decl) => [decl.prop, decl.value])
  )

describe("reduced motion", () => {
  it.each([
    '<div class="animate-shake">Message is empty</div>',
    '<span class="animate-pulse">Recording</span>',
    '<p class="anything">Any element</p>'
  ])("stops animations, transitions and smooth scrolling on %s", (html) => {
    expect(reducedMotionDeclarations(html)).toMatchObject({
      "animation-duration": "0.01ms",
      "animation-iteration-count": "1",
      "transition-duration": "0.01ms",
      "scroll-behavior": "auto"
    })
  })

  it("sets the shared transition duration token to zero", () => {
    const rootRule = sharedStylesheetRules().find(
      (rule) =>
        rule.selectors.includes(":root") &&
        rule.media.some((media) => media.includes("prefers-reduced-motion: reduce"))
    )
    expect(
      rootRule?.declarations.find((decl) => decl.prop === "--duration-base")?.value
    ).toBe("0ms")
  })
})
