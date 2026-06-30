import { describe, expect, expectTypeOf, it } from "vitest"

import {
  isApiResponseEnvelope,
  unwrapApiResponseData,
  unwrapApiResponseEnvelope,
  type ApiPaginatedPayload,
  type ApiPaginationMeta,
  type ApiResponseEnvelope,
  type CursorPaginationMeta,
  type OffsetPaginationMeta,
  type PagePaginationMeta
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

  it("types union inputs without leaking the envelope shape", () => {
    type Payload = { id: number }
    const payload: ApiResponseEnvelope<Payload> | Payload | null | undefined =
      Math.random() > 0.5 ? { success: true, data: { id: 3 } } : { id: 4 }
    const legacyPayload = { success: true, file_id: "generated-file" }

    const unwrapped = unwrapApiResponseEnvelope(payload)
    const unwrappedLegacy = unwrapApiResponseEnvelope(legacyPayload)

    expectTypeOf(unwrapped).toEqualTypeOf<Payload | null | undefined>()
    expectTypeOf(unwrappedLegacy).toEqualTypeOf<typeof legacyPayload>()
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

  it("types canonical pagination metadata without changing payload shapes", () => {
    type TableListPayload = ApiPaginatedPayload<{
      tables: { id: string }[]
      total: number
    }, OffsetPaginationMeta>

    const payload: TableListPayload = {
      tables: [{ id: "table-1" }],
      total: 1,
      pagination: {
        mode: "offset",
        limit: 25,
        offset: 0,
        total: 1,
        has_more: false,
        next_offset: null
      }
    }
    const envelope: ApiResponseEnvelope<TableListPayload> = {
      success: true,
      data: payload
    }

    const unwrapped = unwrapApiResponseEnvelope(envelope)

    expect(unwrapped?.pagination).toEqual(payload.pagination)
    expectTypeOf(payload.pagination).toEqualTypeOf<OffsetPaginationMeta>()
    expectTypeOf(unwrapped).toEqualTypeOf<TableListPayload | null>()
    expectTypeOf<ApiPaginationMeta>().toEqualTypeOf<
      OffsetPaginationMeta | PagePaginationMeta | CursorPaginationMeta
    >()
  })
})
