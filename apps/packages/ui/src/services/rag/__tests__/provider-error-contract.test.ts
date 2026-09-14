import { describe, expect, it } from "vitest"
import {
  PUBLIC_RAG_PROVIDER_ERROR_MESSAGES,
  asPublicRagProviderErrorCode,
  asValidatedHttpStatus,
  getValidatedHttpStatus,
  sanitizeRagProviderFailure,
} from "../provider-error-contract"

const sentinel = "sk-provider-secret-/Users/private/provider.log"

describe("RAG provider error contract", () => {
  it("accepts only exact public provider error codes", () => {
    expect(asPublicRagProviderErrorCode("provider_disabled")).toBe(
      "provider_disabled"
    )

    for (const value of [
      "Provider_Disabled",
      " provider_disabled ",
      "__proto__",
      "unknown_provider_failure",
      503,
      null,
    ]) {
      expect(asPublicRagProviderErrorCode(value)).toBeNull()
    }
  })

  it("accepts only integer HTTP statuses in the protocol range", () => {
    expect(asValidatedHttpStatus(100)).toBe(100)
    expect(asValidatedHttpStatus(599)).toBe(599)

    for (const value of [99, 600, 503.5, "503", Number.NaN, Infinity, null]) {
      expect(asValidatedHttpStatus(value)).toBeUndefined()
    }
  })

  it("uses the first valid status in direct, response, statusCode order", () => {
    expect(
      getValidatedHttpStatus({
        status: 429,
        response: { status: 503 },
        statusCode: 401,
      })
    ).toBe(429)
    expect(
      getValidatedHttpStatus({
        status: "429",
        response: { status: 503 },
        statusCode: 401,
      })
    ).toBe(503)
    expect(
      getValidatedHttpStatus({
        status: 700,
        response: { status: 99 },
        statusCode: 422,
      })
    ).toBe(422)
    expect(
      getValidatedHttpStatus({
        status: "503",
        response: { status: 503.5 },
        statusCode: "422",
      })
    ).toBeUndefined()
  })

  it.each([
    ["detail", { detail: { error_code: "provider_disabled", message: sentinel } }],
    [
      "details.detail",
      {
        details: {
          detail: { error_code: "provider_disabled", message: sentinel },
        },
      },
    ],
    [
      "details",
      { details: { error_code: "provider_disabled", message: sentinel } },
    ],
    [
      "data.detail",
      { data: { detail: { error_code: "provider_disabled", message: sentinel } } },
    ],
    [
      "data.details.detail",
      {
        data: {
          details: {
            detail: { error_code: "provider_disabled", message: sentinel },
          },
        },
      },
    ],
    [
      "response.data.details.detail",
      {
        response: {
          status: 403,
          data: {
            details: {
              detail: { error_code: "provider_disabled", message: sentinel },
            },
          },
        },
      },
    ],
  ])("normalizes the %s structured envelope", (_shape, error) => {
    expect(sanitizeRagProviderFailure(error)).toEqual({
      message: PUBLIC_RAG_PROVIDER_ERROR_MESSAGES.provider_disabled,
      ...("response" in error ? { status: 403 } : {}),
      code: "provider_disabled",
      details: {
        detail: {
          error_code: "provider_disabled",
          message: PUBLIC_RAG_PROVIDER_ERROR_MESSAGES.provider_disabled,
        },
      },
    })
  })

  it.each([
    [
      "direct detail over data and response data",
      {
        detail: { error_code: "provider_disabled", message: sentinel },
        data: {
          detail: {
            error_code: "provider_configuration_invalid",
            message: sentinel,
          },
        },
        response: {
          data: {
            detail: {
              error_code: "credential_store_unavailable",
              message: sentinel,
            },
          },
        },
      },
    ],
    [
      "top-level data over response data",
      {
        data: { detail: { error_code: "provider_disabled", message: sentinel } },
        response: {
          data: {
            detail: {
              error_code: "credential_store_unavailable",
              message: sentinel,
            },
          },
        },
      },
    ],
  ])("prefers %s", (_case, error) => {
    expect(sanitizeRagProviderFailure(error).code).toBe("provider_disabled")
  })

  it("canonicalizes recognized errors and strips secret-bearing unknown fields", () => {
    const sanitized = sanitizeRagProviderFailure({
      status: 503,
      data: {
        details: {
          error_code: "credential_store_unavailable",
          message: sentinel,
          upstream_body: sentinel,
        },
        debug: sentinel,
      },
      request: { authorization: sentinel },
      stack: sentinel,
    })

    expect(sanitized).toEqual({
      message:
        PUBLIC_RAG_PROVIDER_ERROR_MESSAGES.credential_store_unavailable,
      status: 503,
      code: "credential_store_unavailable",
      details: {
        detail: {
          error_code: "credential_store_unavailable",
          message:
            PUBLIC_RAG_PROVIDER_ERROR_MESSAGES.credential_store_unavailable,
        },
      },
    })
    expect(JSON.stringify(sanitized)).not.toContain(sentinel)
  })

  it.each([
    ["blank", "   "],
    ["oversized", `${"x".repeat(241)}${sentinel}`],
  ])("rejects a %s structured message", (_case, message) => {
    const sanitized = sanitizeRagProviderFailure({
      status: 503,
      details: {
        error_code: "provider_unavailable",
        message,
        secret: sentinel,
      },
    })

    expect(sanitized).toEqual({
      message: "RAG search failed due to a server error.",
      status: 503,
    })
    expect(JSON.stringify(sanitized)).not.toContain(sentinel)
  })

  it("accepts a structured message at the length limit without returning it", () => {
    const sanitized = sanitizeRagProviderFailure({
      detail: {
        error_code: "provider_unavailable",
        message: "x".repeat(240),
      },
    })

    expect(sanitized.message).toBe(
      PUBLIC_RAG_PROVIDER_ERROR_MESSAGES.provider_unavailable
    )
    expect(sanitized.message).not.toContain("x".repeat(240))
  })

  it.each([
    [400, "RAG search request is invalid."],
    [401, "RAG search failed. Authentication is required."],
    [403, "RAG search failed. Access was denied."],
    [404, "RAG search endpoint is unavailable."],
    [408, "RAG search timed out. Try again."],
    [422, "RAG search request is invalid."],
    [429, "RAG search is rate limited. Please wait and try again."],
    [500, "RAG search failed due to a server error."],
    [599, "RAG search failed due to a server error."],
    [409, "RAG search failed."],
  ])("maps HTTP %i without returning raw text", (status, message) => {
    const sanitized = sanitizeRagProviderFailure({ status, error: sentinel })

    expect(sanitized).toEqual({ message, status })
    expect(JSON.stringify(sanitized)).not.toContain(sentinel)
  })

  it.each([
    [
      new Error(`Failed to fetch https://provider.invalid?token=${sentinel}`),
      "Cannot reach server. Check your connection and try again.",
    ],
    [new Error(`request ETIMEDOUT ${sentinel}`), "RAG search timed out. Try again."],
    [sentinel, "RAG search failed."],
    [{ message: sentinel, extra: sentinel }, "RAG search failed."],
  ])("classifies transport failures without leaking raw text", (error, message) => {
    const sanitized = sanitizeRagProviderFailure(error)

    expect(sanitized).toEqual({ message })
    expect(JSON.stringify(sanitized)).not.toContain(sentinel)
  })
})
