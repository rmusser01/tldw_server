import { describe, expect, it } from "vitest"
import {
  mapKnowledgeQaExportErrorMessage,
  mapKnowledgeQaSearchErrorMessage,
} from "../errorMessages"

describe("Knowledge QA error message mapping", () => {
  it("maps search timeout and connection failures to actionable copy", () => {
    expect(mapKnowledgeQaSearchErrorMessage(new Error("request timed out"))).toBe(
      "Search timed out. Try the Fast preset or reduce sources."
    )
    expect(mapKnowledgeQaSearchErrorMessage(new Error("network unreachable"))).toBe(
      "Cannot reach server. Check your connection and try again."
    )
  })

  it("uses client-owned provider copy and ignores the error message", () => {
    const sentinel = "sk-provider-secret-/Users/private/provider.log"
    const error = Object.assign(new Error(sentinel), {
      code: "invalid_provider_credentials",
      status: 503,
    })

    const message = mapKnowledgeQaSearchErrorMessage(error)

    expect(message).toBe("The selected provider credentials are invalid.")
    expect(message).not.toContain(sentinel)
  })

  it.each([
    [
      "provider_disabled",
      "The selected provider is disabled by administrator policy.",
    ],
    [
      "model_not_allowed",
      "The selected model is not allowed for this provider.",
    ],
  ])(
    "preserves policy code %s for stream and non-stream failures",
    (code, expected) => {
      const sentinel = "sk-policy-secret-/Users/private/provider.log"
      const terminalMessage = mapKnowledgeQaSearchErrorMessage({
        event: {
          type: "error",
          code,
          status_code: 403,
          message: sentinel,
        },
      })
      const nonStreamMessage = mapKnowledgeQaSearchErrorMessage(
        Object.assign(new Error(sentinel), { code, status: 403 })
      )

      expect(terminalMessage).toBe(expected)
      expect(nonStreamMessage).toBe(expected)
      expect(terminalMessage).not.toContain(sentinel)
      expect(nonStreamMessage).not.toContain(sentinel)
    }
  )

  it.each([
    [502, "RAG search failed due to a server error."],
    [503, "RAG search failed due to a server error."],
    [429, "RAG search is rate limited. Please wait and try again."],
  ])("uses status-safe copy for HTTP %i", (status, expected) => {
    const sentinel = "sk-provider-secret-/Users/private/provider.log"
    const error = Object.assign(new Error(sentinel), { status })

    const message = mapKnowledgeQaSearchErrorMessage(error)

    expect(message).toBe(expected)
    expect(message).not.toContain(sentinel)
  })

  it("does not return unknown raw error text", () => {
    const sentinel = "sk-provider-secret-/Users/private/provider.log"

    const message = mapKnowledgeQaSearchErrorMessage(
      new Error(sentinel),
      "Search failed"
    )

    expect(message).toBe("Search failed")
    expect(message).not.toContain(sentinel)
  })

  it("maps export failures to chatbook-specific messaging", () => {
    expect(mapKnowledgeQaExportErrorMessage(new Error("404 not found"))).toBe(
      "Chatbook export failed. Thread was not found."
    )
    expect(
      mapKnowledgeQaExportErrorMessage(new Error("HTTP 401 unauthorized"))
    ).toBe("Chatbook export failed. You are not authorized to export this thread.")
    expect(
      mapKnowledgeQaExportErrorMessage(new Error("HTTP 403 forbidden"))
    ).toBe("Chatbook export failed. You do not have permission to export this thread.")
    expect(
      mapKnowledgeQaExportErrorMessage(
        new Error("HTTP 422: validation failed: content_selections is required")
      )
    ).toBe(
      "Chatbook export failed. Export request is invalid. Check the selected thread and try again."
    )
    expect(
      mapKnowledgeQaExportErrorMessage(new Error("Failed to fetch"))
    ).toBe("Chatbook export failed. Cannot reach server.")
    expect(
      mapKnowledgeQaExportErrorMessage(new Error("HTTP 500: internal server error"))
    ).toBe("Chatbook export failed due to a server error. Please try again.")
    expect(
      mapKnowledgeQaExportErrorMessage(new Error("HTTP 429: too many requests"))
    ).toBe(
      "Chatbook export failed. Too many export requests. Please wait and try again."
    )
  })
})
