import { beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import {
  ServicePromptApiError,
  servicePromptMethods
} from "../service-prompts"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

const detail = {
  id: "chat.rag.answer",
  label: "RAG answer",
  description: "Description",
  parts: [
    {
      key: "template",
      label: "Template",
      mode: "template" as const,
      required_variables: ["context", "question"]
    }
  ],
  affected_workflows: [{ id: "chat.main.rag", label: "Main chat RAG" }],
  default_parts: { template: "Default {context} {question}" },
  saved_parts: null,
  effective_parts: { template: "Default {context} {question}" },
  source: "packaged" as const,
  revision: null
}

describe("Service Prompt API methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("lists the forward-compatible catalog with capability 404 expected", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce([detail])
    const controller = new AbortController()

    await expect(
      servicePromptMethods.listServicePrompts({ signal: controller.signal })
    ).resolves.toEqual([detail])

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/service-prompts",
      method: "GET",
      expectedStatuses: [404],
      abortSignal: controller.signal
    })
  })

  it("gets one encoded definition path and preserves actionable errors", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ...detail,
      id: "chat.rag.answer/unsafe"
    })
    const controller = new AbortController()

    await servicePromptMethods.getServicePrompt("chat.rag.answer/unsafe", {
      signal: controller.signal
    })

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/service-prompts/chat.rag.answer%2Funsafe",
      method: "GET",
      expectedStatuses: [404, 500],
      abortSignal: controller.signal
    })
  })

  it("puts the complete parts and compare-and-swap revision", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(detail)
    const controller = new AbortController()
    const request = {
      parts: { template: "Use {context}; answer {question}" },
      expected_revision: "revision-1"
    }

    await servicePromptMethods.saveServicePrompt(
      "chat.rag.answer",
      request,
      { signal: controller.signal }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/service-prompts/chat.rag.answer",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: request,
      expectedStatuses: [404, 409, 422, 500],
      abortSignal: controller.signal
    })
  })

  it("deletes conditionally and encodes the revision query value", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ...detail,
      id: "chat.rag.answer/unsafe"
    })
    const controller = new AbortController()

    await servicePromptMethods.resetServicePrompt(
      "chat.rag.answer/unsafe",
      "revision/value",
      { signal: controller.signal }
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path:
        "/api/v1/service-prompts/chat.rag.answer%2Funsafe?expected_revision=revision%2Fvalue",
      method: "DELETE",
      expectedStatuses: [404, 409, 422, 500],
      abortSignal: controller.signal
    })
  })

  it("omits the reset query when no revision exists", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(detail)

    await servicePromptMethods.resetServicePrompt(
      "chat.rag.answer",
      null
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/service-prompts/chat.rag.answer",
      method: "DELETE",
      expectedStatuses: [404, 409, 422, 500],
      abortSignal: undefined
    })
  })

  it.each([
    {
      name: "direct",
      error: {
        status: 409,
        detail: {
          code: "service_prompt_revision_conflict",
          message: "Changed since load.",
          current_revision: "revision-2"
        }
      }
    },
    {
      name: "extension proxy",
      error: {
        status: 409,
        details: {
          detail: {
            code: "service_prompt_revision_conflict",
            message: "Changed since load.",
            current_revision: "revision-2"
          }
        }
      }
    }
  ])("normalizes $name errors into one typed shape", async ({ error }) => {
    vi.mocked(bgRequest).mockRejectedValueOnce(error)

    const rejection = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context} {question}" },
        expected_revision: "revision-1"
      })
      .catch((caught) => caught)

    expect(rejection).toBeInstanceOf(ServicePromptApiError)
    expect(rejection).toMatchObject({
      message: "Changed since load.",
      status: 409,
      code: "service_prompt_revision_conflict",
      currentRevision: "revision-2"
    })
  })

  it("normalizes validation and corrupt-reset metadata without authored text", async () => {
    vi.mocked(bgRequest)
      .mockRejectedValueOnce({
        status: 422,
        details: {
          detail: {
            code: "service_prompt_validation_failed",
            message: "Validation failed.",
            field_errors: { template: "Template is invalid." }
          }
        }
      })
      .mockRejectedValueOnce({
        status: 500,
        detail: {
          code: "service_prompt_corrupt_override",
          message: "Saved override is corrupt.",
          revision: "revision-corrupt",
          can_reset: true
        }
      })

    const invalid = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context}" },
        expected_revision: null
      })
      .catch((caught) => caught)
    const corrupt = await servicePromptMethods
      .getServicePrompt("chat.rag.answer")
      .catch((caught) => caught)

    expect(invalid).toMatchObject({
      status: 422,
      code: "service_prompt_validation_failed",
      fieldErrors: { template: "Template is invalid." }
    })
    expect(corrupt).toMatchObject({
      status: 500,
      code: "service_prompt_corrupt_override",
      revision: "revision-corrupt",
      canReset: true
    })
  })

  it("preserves AbortError cancellation instead of converting it to an API error", async () => {
    const aborted = new DOMException("Aborted", "AbortError")
    vi.mocked(bgRequest).mockRejectedValueOnce(aborted)

    await expect(
      servicePromptMethods.listServicePrompts()
    ).rejects.toBe(aborted)
  })

  it.each([
    ["null", null],
    ["object instead of array", {}],
    ["null item", [null]],
    ["malformed item", [{ ...detail, parts: null }]]
  ])("rejects malformed successful catalog payloads: %s", async (_name, payload) => {
    vi.mocked(bgRequest).mockResolvedValueOnce(payload)

    const rejection = await servicePromptMethods
      .listServicePrompts()
      .catch((error) => error)

    expect(rejection).toBeInstanceOf(ServicePromptApiError)
    expect(rejection).toMatchObject({
      status: 0,
      code: "service_prompt_protocol_error",
      message: "Service Prompt server response was invalid."
    })
  })

  it("accepts a structurally valid unknown catalog id", async () => {
    const future = { ...detail, id: "chat.future.definition" }
    vi.mocked(bgRequest).mockResolvedValueOnce([future])

    await expect(servicePromptMethods.listServicePrompts()).resolves.toEqual([
      future
    ])
  })

  it("rejects malformed and mismatched successful detail responses", async () => {
    const authoredSentinel = "PROMPT_BODY_MUST_NOT_APPEAR"
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({
        ...detail,
        id: "chat.rag.question_rewrite"
      })
      .mockResolvedValueOnce({
        ...detail,
        effective_parts: { template: authoredSentinel },
        revision: 7
      })

    const mismatched = await servicePromptMethods
      .getServicePrompt("chat.rag.answer")
      .catch((error) => error)
    const malformed = await servicePromptMethods
      .getServicePrompt("chat.rag.answer")
      .catch((error) => error)

    for (const rejection of [mismatched, malformed]) {
      expect(rejection).toMatchObject({
        status: 0,
        code: "service_prompt_protocol_error",
        message: "Service Prompt server response was invalid."
      })
      expect(String(rejection)).not.toContain(authoredSentinel)
    }
  })

  it.each([
    ["default parts", { default_parts: { template: 5 } }],
    ["saved parts", { saved_parts: [] }],
    ["effective parts", { effective_parts: null }],
    ["source", { source: "unexpected" }],
    ["revision", { revision: 7 }]
  ])("rejects malformed successful detail %s shape", async (_name, change) => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ ...detail, ...change })

    await expect(
      servicePromptMethods.getServicePrompt("chat.rag.answer")
    ).rejects.toMatchObject({
      status: 0,
      code: "service_prompt_protocol_error"
    })
  })

  it("rejects mismatched PUT and malformed reset results", async () => {
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({
        ...detail,
        id: "chat.web_search.answer"
      })
      .mockResolvedValueOnce({
        ...detail,
        affected_workflows: [{ id: 5, label: "invalid" }]
      })

    const saved = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context} {question}" },
        expected_revision: null
      })
      .catch((error) => error)
    const reset = await servicePromptMethods
      .resetServicePrompt("chat.rag.answer", null)
      .catch((error) => error)

    expect(saved).toMatchObject({
      status: 0,
      code: "service_prompt_protocol_error"
    })
    expect(reset).toMatchObject({
      status: 0,
      code: "service_prompt_protocol_error"
    })
  })
})
