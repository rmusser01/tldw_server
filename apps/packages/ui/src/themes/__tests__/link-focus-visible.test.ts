import { describe, expect, it } from "vitest"
import { elementFrom, rulesTargeting } from "./shared-stylesheet-rules"

// AX-08 (#3124): antd's global link reset (a { outline: none } and
// a:focus { outline: 0 }) left links without their own focus classes, such as
// the side panel header links, with no visible keyboard focus.

const focusOutline = (html: string) =>
  rulesTargeting(elementFrom(html), { ignoredPseudoClasses: [":focus-visible"] })
    .filter((rule) => rule.matchedSelector.includes(":focus-visible"))
    .flatMap((rule) => rule.declarations)
    .find((decl) => decl.prop === "outline")

describe("link focus indicator", () => {
  it("outlines a focused link in the theme focus colour", () => {
    expect(focusOutline('<a href="/settings">Settings</a>')?.value).toBe(
      "2px solid rgb(var(--color-focus))"
    )
  })

  it.each([
    '<a href="/x" class="focus-visible:ring-2 focus-visible:ring-focus">Own ring</a>',
    '<a href="/x" class="focus:outline-none focus:ring-2">Own focus</a>',
    "<a>No href</a>"
  ])("leaves links that style their own focus alone: %s", (html) => {
    expect(focusOutline(html)).toBeUndefined()
  })
})
