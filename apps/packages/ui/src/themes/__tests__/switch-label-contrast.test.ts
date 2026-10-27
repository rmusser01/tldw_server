import { describe, expect, it } from "vitest"
import { rulesTargeting } from "./shared-stylesheet-rules"

// AX-07 (#3124): antd writes a switch's checked label (checkedChildren) in
// white on the primary track, 3.13:1 in the default dark theme. antd has no
// token for that label alone (the unchecked label shares it, on a grey
// track), so the stylesheet gives the checked label the primary foreground.

const labelColour = (switchClass: string, labelClass: string) => {
  const host = document.createElement("div")
  host.innerHTML = `<button class="${switchClass}"><span class="ant-switch-inner"><span class="${labelClass}">On</span></span></button>`
  const label = host.querySelector(`.${labelClass}`)!
  return rulesTargeting(label)
    .flatMap((rule) => rule.declarations)
    .filter((decl) => decl.prop === "color")
    .map((decl) => decl.value)
}

describe("switch label contrast", () => {
  it("writes the checked label in the primary foreground colour", () => {
    expect(labelColour("ant-switch ant-switch-checked", "ant-switch-inner-checked")).toEqual([
      "rgb(var(--color-primary-foreground))"
    ])
  })

  it("leaves the unchecked label on antd's own colour", () => {
    expect(labelColour("ant-switch", "ant-switch-inner-unchecked")).toEqual([])
  })
})
