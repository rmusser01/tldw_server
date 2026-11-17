import { describe, expect, it } from "vitest"
import { elementFrom, rulesTargeting } from "./shared-stylesheet-rules"

// XP-15 (#3124): a body-colour rule for .ant-typography (specificity 0,4,0,
// !important) outranked the muted rule for .ant-typography-secondary, so every
// type="secondary" hint rendered in the full text colour, louder than headings.

const colours = (html: string) =>
  rulesTargeting(elementFrom(html))
    .flatMap((rule) => rule.declarations)
    .filter((decl) => decl.prop === "color")
    .map((decl) => decl.value)

describe("antd secondary typography colour", () => {
  it("draws secondary text only in the muted colour", () => {
    expect(colours('<span class="ant-typography ant-typography-secondary">Hint</span>')).toEqual([
      "rgb(var(--color-text-muted))"
    ])
  })

  it("keeps ordinary typography in the body colour", () => {
    expect(colours('<span class="ant-typography">Body</span>')).toEqual([
      "rgb(var(--color-text))"
    ])
  })
})
