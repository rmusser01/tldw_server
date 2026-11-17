import { describe, expect, it } from "vitest"

import { ApiError, TldwApiError } from "../api-error"

/**
 * TASK-13526 stage 2 / ADR-065: one canonical error model across both
 * clients. The web shell re-exports this same ApiError class, so these
 * invariants hold for every consumer in both shells.
 */
describe("canonical ApiError model", () => {
  it("keeps the web-compatible field surface (statusCode alias included)", () => {
    const error = new ApiError("Provider unavailable", {
      status: 503,
      detail: "upstream down",
      errorCode: "provider_unavailable",
      retryAfter: 30
    })
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe("ApiError")
    expect(error.status).toBe(503)
    expect(error.statusCode).toBe(503)
    expect(error.detail).toBe("upstream down")
    expect(error.errorCode).toBe("provider_unavailable")
    expect(error.retryAfter).toBe(30)
  })

  it("allows network-style errors with no status", () => {
    const error = new ApiError("Network Error", { detail: "Network Error" })
    expect(error.status).toBeUndefined()
    expect(error.statusCode).toBeUndefined()
  })

  it("TldwApiError remains a strict subclass with a required status", () => {
    const error = new TldwApiError("Request failed", 422, { code: "invalid" })
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe("TldwApiError")
    expect(error.status).toBe(422)
    expect(error.detail).toEqual({ code: "invalid" })
  })

  it("cross-name instanceof works in both directions of construction", () => {
    const webStyle = new ApiError("boom", { status: 500 })
    expect(webStyle).not.toBeInstanceOf(TldwApiError)
    const sharedStyle: ApiError = new TldwApiError("boom", 500, null)
    expect(sharedStyle).toBeInstanceOf(TldwApiError)
  })
})
