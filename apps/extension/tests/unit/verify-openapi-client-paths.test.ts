import fs from "node:fs"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  extractClientPaths,
  extractFallbackFieldNamesFromFallbackSource,
  extractFallbackFieldNamesFromSource
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
      "second-field"
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
      "second-field"
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
      "final-field"
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
      "chunk_method"
    ])
  })
})

const derivedGuard =
  'import type { ApiPath } from "@tldw/api-types"; export type ClientPath = ApiPath'
const schema = {
  paths: { "/api/v1/chat/completions": {}, "/api/v1/health": {} }
}
const stubDerivedPaths = (generated: string) => {
  const read = fs.readFileSync
  vi.spyOn(fs, "readFileSync").mockImplementation((file, ...args) => {
    if (String(file).endsWith("/openapi-guard.ts")) return derivedGuard
    if (String(file).endsWith("/api-types/paths.d.ts")) return generated
    return read(file, ...args)
  })
}
afterEach(() => vi.restoreAllMocks())

describe("generated API path verification", () => {
  it("compares the actual generated union instead of certifying zero paths", () => {
    stubDerivedPaths(
      'export type ApiPath = "/api/v1/chat/completions" | "/api/v1/health"'
    )
    expect(extractClientPaths(schema)).toEqual([
      "/api/v1/chat/completions",
      "/api/v1/health"
    ])
  })

  it("accepts the generated literal for a one-route backend", () => {
    stubDerivedPaths('export type ApiPath = "/api/v1/health"')
    expect(extractClientPaths({ paths: { "/api/v1/health": {} } })).toEqual([
      "/api/v1/health"
    ])
  })

  it("refuses a union that omits a current backend route", () => {
    stubDerivedPaths('export type ApiPath = "/api/v1/health" | "/api/v1/old"')
    expect(() => extractClientPaths(schema)).toThrow("/api/v1/chat/completions")
  })

  it.each([
    "export type ApiPath = never",
    "export type AnotherPath = string",
    'export type ApiPath = "/api/v1/chat/completions" | "/api/v1/health" | string'
  ])("refuses an unreadable or empty generated union: %s", (generated) => {
    stubDerivedPaths(generated)
    expect(() => extractClientPaths(schema)).toThrow(
      "Generated API path union is empty or unreadable"
    )
  })
})
