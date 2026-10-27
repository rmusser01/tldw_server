import { describe, expect, it } from "vitest"
import { elementFrom, rulesTargeting } from "./shared-stylesheet-rules"

// AX-08 (#3124): the focus halo on text inputs, selects and switches was the
// focus colour at 35-40% opacity, which measures well under 3:1 against the
// surface. A keyboard user needs the halo itself to be visible.

const focusShadows = (html: string) =>
  rulesTargeting(elementFrom(html), {
    ignoredPseudoClasses: [":focus-visible", ":focus"]
  })
    .flatMap((rule) => rule.declarations)
    .filter((decl) => decl.prop === "box-shadow")
    .map((decl) => decl.value)

describe("form control focus halo", () => {
  it.each([
    '<input class="ant-input ant-input-focused" />',
    '<span class="ant-input-affix-wrapper ant-input-affix-wrapper-focused"></span>',
    '<div class="ant-input-number ant-input-number-focused"></div>',
    '<div class="ant-picker ant-picker-focused"></div>',
    '<button class="focus-ring ant-switch"></button>'
  ])("draws an opaque focus-colour halo: %s", (html) => {
    const shadows = focusShadows(html)
    expect(shadows.length).toBeGreaterThan(0)
    for (const shadow of shadows) {
      expect(shadow).toContain("rgb(var(--color-focus))")
    }
  })
})
