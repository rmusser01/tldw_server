import { describe, expect, it } from "vitest"

import {
  extractFallbackFieldNamesFromFallbackSource,
  extractFallbackFieldNamesFromSource,
} from "../../scripts/verify-openapi-client-paths.mjs"

describe("extractFallbackFieldNamesFromSource", () => {
  it("parses typed fallback declarations without treating annotation brackets as the array", () => {
    const src = `
      export const MEDIA_ADD_SCHEMA_FALLBACK: Array<{
        name: string
        enum?: unknown[]
      }> = [
        { name: "typed-field" },
        { name: 'second-field' }
      ]
    `

    expect(extractFallbackFieldNamesFromSource(src)).toEqual([
      "typed-field",
      "second-field",
    ])
  })

  it("exercises the fallback-only parser for typed fallback declarations", () => {
    const src = `
      export const MEDIA_ADD_SCHEMA_FALLBACK: Array<{
        name: string
        enum?: unknown[]
      }> = [
        { name: "typed-field" },
        { name: 'second-field' }
      ]
    `

    expect(extractFallbackFieldNamesFromFallbackSource(src)).toEqual([
      "typed-field",
      "second-field",
    ])
  })

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

  it("skips type annotation brackets before the fallback initializer", () => {
    const src = `
      export const MEDIA_ADD_SCHEMA_FALLBACK: Array<{
        name: string
        enum?: unknown[]
      }> = [
        { name: "api_name" },
        { name: "chunk_method", enum: ["semantic", "tokens"] }
      ]
    `

    expect(extractFallbackFieldNamesFromSource(src)).toEqual([
      "api_name",
      "chunk_method",
    ])
  })
})
