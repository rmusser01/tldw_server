import { describe, expect, it } from "vitest"
import { isDefinitiveWriteRejection, isNotesProvenancePolicyUnavailable, TldwApiError } from "../api-error"

const policy = { error_code: "notes_provenance_encryption_unsupported", message: "Knowledge provenance could not be saved; refresh its state and retry." }

describe("pending write outcomes", () => {
  it.each([
    { status: 409, details: { detail: policy } },
    new TldwApiError(policy.message, 409, policy),
  ])("retains identity when replay is blocked by the Notes read policy", error => {
    expect(isNotesProvenancePolicyUnavailable(error)).toBe(true)
    expect(isDefinitiveWriteRejection(error)).toBe(false)
  })
  it.each([400, 401, 403, 404, 409, 413, 422, 429])("releases a definitively rejected HTTP %s request", status => {
    expect(isDefinitiveWriteRejection({ status })).toBe(true)
  })
  it.each([{}, new Error("Lost response"), { status: 408 }, { status: 500 }, { status: 503 }])("retains identity for uncertain request outcomes", error => {
    expect(isDefinitiveWriteRejection(error)).toBe(false)
  })
})
