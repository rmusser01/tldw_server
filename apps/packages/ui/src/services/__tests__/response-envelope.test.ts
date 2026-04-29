import { describe, expect, it } from "vitest"

import {
  isApiResponseEnvelope,
  unwrapApiResponseData,
  unwrapApiResponseEnvelope
} from "@/services/response-envelope"

describe("response envelope helpers", () => {
  it("unwraps canonical success envelopes", () => {
    const payload = {
      success: true,
      data: { id: 7, name: "Demo" },
      metadata: { request_id: "req-1" }
    }

    expect(isApiResponseEnvelope(payload)).toBe(true)
    expect(unwrapApiResponseEnvelope(payload)).toEqual({ id: 7, name: "Demo" })
  })

  it("returns null for canonical error envelopes without data", () => {
    const payload = {
      success: false,
      error: "Failed to load",
      error_code: "LOAD_FAILED"
    }

    expect(isApiResponseEnvelope(payload)).toBe(true)
    expect(unwrapApiResponseEnvelope(payload)).toBeNull()
  })

  it("detects metadata-only canonical envelopes", () => {
    const payload = {
      success: true,
      metadata: { request_id: "req-2" }
    }

    expect(isApiResponseEnvelope(payload)).toBe(true)
    expect(unwrapApiResponseEnvelope(payload)).toBeNull()
  })

  it("leaves legacy success-shaped endpoint payloads unchanged", () => {
    const payload = {
      success: true,
      file_id: "generated-file",
      hard_delete: false
    }

    expect(isApiResponseEnvelope(payload)).toBe(false)
    expect(unwrapApiResponseEnvelope(payload)).toBe(payload)
  })

  it("unwraps transitional data wrappers without requiring endpoint payload changes", () => {
    const payload = {
      data: { id: 9, name: "Existing response" }
    }

    expect(isApiResponseEnvelope(payload)).toBe(false)
    expect(unwrapApiResponseEnvelope(payload)).toBe(payload)
    expect(unwrapApiResponseData(payload)).toEqual({ id: 9, name: "Existing response" })
  })

  it("returns null for transitional error-only wrappers", () => {
    const payload = {
      error: "Failed to load"
    }

    expect(isApiResponseEnvelope(payload)).toBe(false)
    expect(unwrapApiResponseData<number>(payload)).toBeNull()
  })

  it("does not unwrap domain objects that happen to contain data", () => {
    const payload = {
      id: "domain-object",
      data: "domain field"
    }

    expect(isApiResponseEnvelope(payload)).toBe(false)
    expect(unwrapApiResponseData(payload)).toBe(payload)
  })
})
