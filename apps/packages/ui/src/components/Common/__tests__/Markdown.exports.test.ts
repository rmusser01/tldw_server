import { describe, expect, it } from "vitest"

import * as MarkdownModule from "../Markdown"

describe("Markdown module exports", () => {
  it("supports both named and default imports for shared consumers", () => {
    // React.memo component (TASK-15622): an object wrapping the render function.
    expect(MarkdownModule.Markdown).toMatchObject({
      $$typeof: Symbol.for("react.memo"),
      type: expect.any(Function)
    })
    expect(MarkdownModule.default).toBe(MarkdownModule.Markdown)
  })
})
