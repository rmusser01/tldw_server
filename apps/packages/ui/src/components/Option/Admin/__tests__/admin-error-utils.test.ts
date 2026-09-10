import { describe, expect, it } from "vitest"
import {
  deriveAdminGuardFromError,
  isServiceUnavailableError,
  sanitizeAdminErrorMessage
} from "../admin-error-utils"

describe("admin error utilities", () => {
  it("derives guard state for forbidden and missing admin APIs", () => {
    expect(deriveAdminGuardFromError(new Error("Request failed: 403"))).toBe(
      "forbidden"
    )
    expect(deriveAdminGuardFromError(new Error("Request failed: 404"))).toBe(
      "notFound"
    )
    expect(
      deriveAdminGuardFromError({
        status: 404,
        message: "Not Found (GET /api/v1/admin/billing/overview)"
      })
    ).toBe("notFound")
    expect(deriveAdminGuardFromError(new Error("network down"))).toBe(null)
  })

  it("does not misdiagnose a 503 service outage as missing admin APIs", () => {
    // Regression: a 503 from e.g. the llama.cpp runtime used to render the
    // "Admin APIs are not available on this server" wall with a wrong remedy.
    expect(deriveAdminGuardFromError(new Error("Request failed: 503"))).toBe(
      null
    )
    expect(isServiceUnavailableError(new Error("Request failed: 503"))).toBe(
      true
    )
    expect(isServiceUnavailableError({ status: 502, message: "bad gateway" })).toBe(
      true
    )
    expect(isServiceUnavailableError(new Error("Request failed: 404"))).toBe(
      false
    )
    expect(isServiceUnavailableError(new Error("network down"))).toBe(false)
  })

  it("redacts endpoints and filesystem paths from user-facing errors", () => {
    const message = sanitizeAdminErrorMessage(
      new Error(
        "Request failed: 503 (GET /api/v1/admin/llamacpp/status) config=/Users/dev/.config/tldw/config.txt"
      ),
      "fallback message"
    )

    expect(message).toContain("[redacted-path]")
    expect(message).not.toContain("/api/v1/admin/llamacpp/status")
    expect(message).not.toContain("/Users/dev/.config/tldw/config.txt")
    // The endpoint parenthetical carries no information once redacted, so it
    // is dropped entirely rather than shown as "(GET [admin-endpoint])".
    expect(message).not.toContain("[admin-endpoint]")
    expect(message).toContain("Request failed: 503")
  })

  it("keeps redacting endpoints that appear outside parentheticals", () => {
    const message = sanitizeAdminErrorMessage(
      new Error("Upstream rejected GET /api/v1/admin/users with 500"),
      "fallback message"
    )

    expect(message).toContain("[admin-endpoint]")
    expect(message).not.toContain("/api/v1/admin/users")
  })

  it("returns fallback when message is missing", () => {
    expect(sanitizeAdminErrorMessage(null, "fallback message")).toBe(
      "fallback message"
    )
  })
})
