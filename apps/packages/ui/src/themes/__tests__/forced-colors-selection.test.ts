import { describe, expect, it } from "vitest"
import { elementFrom, rulesTargeting } from "./shared-stylesheet-rules"

// AX-15 (#3124): in Windows High Contrast / forced-colours mode the browser
// replaces fill colours, so a selected toggle that differs only by fill looks
// identical to its neighbours. Selected states need a cue that survives.

const SELECTED_STATES = [
  '<button aria-pressed="true">List</button>',
  '<div role="radio" aria-checked="true">Markdown</div>',
  '<li class="ant-pagination-item ant-pagination-item-active"><a>2</a></li>',
  '<div class="ant-segmented-item ant-segmented-item-selected">Split</div>'
]

describe("forced-colours selected state", () => {
  it.each(SELECTED_STATES)("outlines %s with a system colour", (html) => {
    const rules = rulesTargeting(elementFrom(html), {
      mediaIncludes: "forced-colors: active"
    })
    const outline = rules
      .flatMap((rule) => rule.declarations)
      .find((decl) => decl.prop === "outline")
    expect(outline?.value).toMatch(/solid (Highlight|CanvasText|ButtonText)/)
  })

  it("leaves unselected toggles alone", () => {
    const rules = rulesTargeting(
      elementFrom('<button aria-pressed="false">Timeline</button>'),
      { mediaIncludes: "forced-colors: active" }
    )
    expect(rules).toEqual([])
  })
})
