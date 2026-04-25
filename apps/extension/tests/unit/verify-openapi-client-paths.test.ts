import { describe, expect, it } from "vitest"

import { extractFallbackFieldNamesFromSource } from "../../scripts/verify-openapi-client-paths.mjs"

describe("extractFallbackFieldNamesFromSource", () => {
  it("ignores brackets inside strings and comments while parsing the fallback array", () => {
    const src = `
      export const MEDIA_ADD_SCHEMA_FALLBACK = [
        { name: "field[0]" },
        // ] ignored because it is in a line comment
        { name: "still-valid", note: "value ] stays inside the string" },
        /* [ ignored because it is in a block comment ] */
        { name: 'final-field' }
      ]

      export const OTHER = ["outside", "]"]
    `

    expect(extractFallbackFieldNamesFromSource(src)).toEqual([
      "field[0]",
      "still-valid",
      "final-field",
    ])
  })
})
