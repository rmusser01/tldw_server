import { describe, expect, it } from "vitest"

import { isNotesServerUnavailableError } from "../notes-list-error"

// NL-16: the "Notes server unavailable" banner is only for a server that is
// down or out of reach, not for every list failure.

const httpError = (message: string, status: number) => Object.assign(new Error(message), { status })

describe("isNotesServerUnavailableError", () => {
  it("is true when the request never reached the server", () => {
    expect(isNotesServerUnavailableError(new Error("Failed to fetch (GET /api/v1/notes/?limit=20)"))).toBe(true)
    expect(isNotesServerUnavailableError(Object.assign(new Error("Network error"), { status: 0 }))).toBe(true)
    expect(isNotesServerUnavailableError(new Error("Request timed out"))).toBe(true)
  })

  it("is true when the server answers with a 5xx error", () => {
    expect(isNotesServerUnavailableError(httpError("Service Unavailable", 503))).toBe(true)
    expect(isNotesServerUnavailableError(httpError("Bad Gateway", 502))).toBe(true)
  })

  it("is false for a rate limit or a client-side failure", () => {
    expect(isNotesServerUnavailableError(httpError("Too Many Requests", 429))).toBe(false)
    expect(isNotesServerUnavailableError(new TypeError("Cannot read properties of undefined (reading 'map')"))).toBe(false)
    expect(isNotesServerUnavailableError(new SyntaxError("Unexpected token < in JSON at position 0"))).toBe(false)
  })

  it("is false for permission errors and aborted requests", () => {
    expect(isNotesServerUnavailableError(httpError("Forbidden", 403))).toBe(false)
    expect(isNotesServerUnavailableError(Object.assign(new Error("The operation was aborted"), { name: "AbortError" }))).toBe(false)
    expect(isNotesServerUnavailableError(null)).toBe(false)
  })
})
