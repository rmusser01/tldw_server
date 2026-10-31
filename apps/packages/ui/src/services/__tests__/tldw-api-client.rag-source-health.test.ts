import { describe, expect, it, vi } from "vitest"

import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

describe("ragSourceHealth client", () => {
  it("requests the focused source health endpoint", async () => {
    const request = vi.fn().mockResolvedValue({ sources: [] })

    await chatRagMethods.ragSourceHealth.call({ request } as any)

    expect(request).toHaveBeenCalledWith({
      path: "/api/v1/rag/source-health",
      method: "GET",
    })
  })
})

const runRejectedRagSearch = async (error: unknown) => {
  const requestWithCurrentConfig = vi.fn().mockRejectedValue(error)
  const result = await chatRagMethods.ragSearch
    .call(
      {
        normalizeRagQuery: (query: string) => query,
        requestWithCurrentConfig,
      } as any,
      "provider failure"
    )
    .catch((caught) => caught)

  expect(requestWithCurrentConfig).toHaveBeenCalledTimes(1)
  return result as Error & { code?: string; status?: number }
}

describe("ragSearch provider error sanitization", () => {
  const sentinel = "sk-raw-provider-secret-/Users/private/provider.log"

  it.each([
    [
      "direct detail",
      {
        detail: {
          error_code: "provider_authentication_failed",
          message: sentinel,
        },
      },
      502,
      "provider_authentication_failed",
      "The selected provider credentials could not be authenticated.",
    ],
    [
      "nested details.detail",
      {
        details: {
          detail: {
            error_code: "credential_store_unavailable",
            message: sentinel,
          },
        },
      },
      503,
      "credential_store_unavailable",
      "Provider credential storage is temporarily unavailable.",
    ],
    [
      "direct details",
      {
        details: {
          error_code: "credential_scope_revoked",
          message: sentinel,
        },
      },
      503,
      "credential_scope_revoked",
      "The selected provider credential scope is no longer available.",
    ],
  ])(
    "uses client-owned copy for %s",
    async (_shape, detailShape, status, code, expectedMessage) => {
      const error = Object.assign(new Error(sentinel), detailShape, { status })

      const sanitized = await runRejectedRagSearch(error)

      expect(sanitized).toMatchObject({
        message: expectedMessage,
        code,
        status,
      })
      expect(JSON.stringify(sanitized)).not.toContain(sentinel)
      expect(sanitized.message).not.toContain(sentinel)
    }
  )

  it.each([
    [
      "unknown code",
      502,
      {
        detail: {
          error_code: "unknown_provider_failure",
          message: sentinel,
        },
      },
    ],
    [
      "missing message",
      503,
      { detail: { error_code: "provider_unavailable" } },
    ],
    [
      "non-string message",
      502,
      {
        details: {
          detail: {
            error_code: "provider_unavailable",
            message: { sentinel },
          },
        },
      },
    ],
    [
      "overlong message",
      503,
      {
        details: {
          error_code: "provider_unavailable",
          message: `${"x".repeat(241)}${sentinel}`,
        },
      },
    ],
    ["raw 502", 502, {}],
    ["raw 503", 503, { details: sentinel }],
  ])("uses a status-safe generic for %s", async (_case, status, detailShape) => {
    const error = Object.assign(new Error(sentinel), detailShape, { status })

    const sanitized = await runRejectedRagSearch(error)

    expect(sanitized).toMatchObject({
      message: "RAG search failed due to a server error.",
      status,
    })
    expect(sanitized.code).toBeUndefined()
    expect(sanitized.message).not.toContain(sentinel)
  })

  it("does not preserve an invalid HTTP status", async () => {
    const sanitized = await runRejectedRagSearch(
      Object.assign(new Error(sentinel), { status: 700 })
    )

    expect(sanitized.status).toBeUndefined()
    expect(sanitized.message).toBe("RAG search failed.")
  })
})
