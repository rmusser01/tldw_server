import { createRequire } from "node:module"
import { describe, expect, it } from "vitest"

const require = createRequire(import.meta.url)
const { flatten } = require("../../scripts/sync-public-locales.js") as {
  flatten: (value: Record<string, unknown>) => Record<string, string>
}

describe("public locale sync", () => {
  it("rejects distinct source paths that sanitize to one Chrome key", () => {
    expect(() => flatten({ "foo-bar": "first", foo_bar: "second" })).toThrow(
      /foo-bar.*foo_bar|foo_bar.*foo-bar/
    )
  })
})
