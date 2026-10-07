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
  it.each([400, 409, 422])("releases a definitively rejected HTTP %s request", status => {
    expect(isDefinitiveWriteRejection({ status })).toBe(true)
  })
  it.each([401, 403, 404, 405, 408, 410, 413, 429, 500, 502, 503])("retains uncertain identity through pre-receipt HTTP %s failures", status => {
    expect(isDefinitiveWriteRejection({ status })).toBe(false)
  })
  it.each([
    "notes_organization_sync_not_ready",
    "notes_organization_sync_domains_incomplete",
    "notes_provenance_owner_mismatch",
    "notes_provenance_bootstrap_source_invalid",
    "unknown_future_availability_error",
  ])("retains uncertain identity for structured availability code %s", error_code => {
    expect(isDefinitiveWriteRejection({ status: 409, details: { detail: { error_code } } })).toBe(false)
  })
  it("retains uncertain identity when a server capability is temporarily unavailable", () => {
    expect(isDefinitiveWriteRejection(new TldwApiError("Unsupported", 400, { error_code: "sync_v2_keywords_not_supported" }))).toBe(false)
  })
  it.each([
    "notes_provenance_version_conflict", "notes_note_version_conflict", "notes_organization_version_conflict",
    "notes_provenance_expected_version_invalid", "notes_note_expected_version_invalid",
    "notes_provenance_idempotency_conflict", "sync_server_origin_idempotency_conflict",
    "sync_server_origin_batch_idempotency_conflict", "sync_server_origin_restore_conflict",
  ])("releases known input/version rejection %s", error_code => {
    expect(isDefinitiveWriteRejection(new TldwApiError("Conflict", 409, { error_code }))).toBe(true)
  })
  it.each([{}, new Error("Lost response"), { status: 408 }, { status: 500 }, { status: 503 }])("retains identity for uncertain request outcomes", error => {
    expect(isDefinitiveWriteRejection(error)).toBe(false)
  })
})
