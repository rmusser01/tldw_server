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

const VALID_REVISION = "123e4567-e89b-42d3-a456-426614174000"

const userDetail = (
  parts: Record<string, string>,
  id = "chat.rag.answer"
) => ({
  ...detail,
  id,
  saved_parts: parts,
  effective_parts: parts,
  source: "user" as const,
  revision: VALID_REVISION
})

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
    const controller = new AbortController()
    const request = {
      parts: { template: "Use {context}; answer {question}" },
      expected_revision: "revision-1"
    }
    vi.mocked(bgRequest).mockResolvedValueOnce(userDetail(request.parts))

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
      name: "top-level detail compatibility input",
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
      name: "bgRequest details transport input",
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

  it.each([
    {
      name: "top-level detail compatibility input",
      error: {
        status: 422,
        detail: [{ type: "missing", loc: ["body", "parts"] }]
      }
    },
    {
      name: "bgRequest details transport input",
      error: {
        status: 422,
        details: {
          detail: [{ type: "missing", loc: ["body", "parts"] }]
        }
      }
    }
  ])("keeps $name structural 422 errors generic", async ({ error }) => {
    vi.mocked(bgRequest).mockRejectedValueOnce(error)

    const rejection = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context} {question}" },
        expected_revision: null
      })
      .catch((caught) => caught) as ServicePromptApiError

    expect(rejection).toBeInstanceOf(ServicePromptApiError)
    expect(rejection.status).toBe(422)
    expect(rejection.code).toBeUndefined()
    expect(rejection.fieldErrors).toBeUndefined()
  })

  it.each([
    {
      name: "wrong-code",
      detail: {
        code: "other_validation_failed",
        field_errors: { template: "Rejected." }
      }
    },
    {
      name: "blank-message",
      detail: {
        code: "service_prompt_validation_failed",
        field_errors: { template: "   " }
      }
    },
    {
      name: "non-string-message",
      detail: {
        code: "service_prompt_validation_failed",
        field_errors: { template: 42 }
      }
    }
  ])("does not expose $name field errors as semantic", async ({ detail }) => {
    vi.mocked(bgRequest).mockRejectedValueOnce({
      status: 422,
      details: { detail }
    })

    const rejection = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context} {question}" },
        expected_revision: null
      })
      .catch((caught) => caught) as ServicePromptApiError

    expect(rejection).toBeInstanceOf(ServicePromptApiError)
    expect(rejection.status).toBe(422)
    expect(rejection.fieldErrors).toBeUndefined()
  })

  it("leaves well-formed field keys for the active prompt schema to validate", async () => {
    vi.mocked(bgRequest).mockRejectedValueOnce({
      status: 422,
      details: {
        detail: {
          code: "service_prompt_validation_failed",
          field_errors: { stale_part: "Rejected." }
        }
      }
    })

    const rejection = await servicePromptMethods
      .saveServicePrompt("chat.rag.answer", {
        parts: { template: "{context} {question}" },
        expected_revision: null
      })
      .catch((caught) => caught) as ServicePromptApiError

    expect(rejection).toMatchObject({
      status: 422,
      code: "service_prompt_validation_failed",
      fieldErrors: { stale_part: "Rejected." }
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

  it.each([
    ["empty", [{ ...detail.parts[0], key: "" }]],
    ["duplicate", [detail.parts[0], { ...detail.parts[0] }]]
  ])("rejects %s declared part keys", async (_name, parts) => {
    vi.mocked(bgRequest).mockResolvedValueOnce([{ ...detail, parts }])

    await expect(servicePromptMethods.listServicePrompts()).rejects.toMatchObject({
      status: 0,
      code: "service_prompt_protocol_error"
    })
  })

  it.each([
    ["default", { default_parts: {} }],
    ["effective", { effective_parts: { template: detail.effective_parts.template, extra: "x" } }],
    ["saved", {
      ...userDetail({ template: "Saved {context} {question}" }),
      saved_parts: {}
    }]
  ])("requires %s records to have exactly the declared keys", async (_name, change) => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ ...detail, ...change })

    await expect(
      servicePromptMethods.getServicePrompt("chat.rag.answer")
    ).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
  })

  it.each([
    ["saved parts", { saved_parts: { template: detail.default_parts.template } }],
    ["revision", { revision: VALID_REVISION }],
    ["different effective parts", {
      effective_parts: { template: "Different {context} {question}" }
    }]
  ])("rejects noncanonical packaged detail with %s", async (_name, change) => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ ...detail, ...change })

    await expect(
      servicePromptMethods.getServicePrompt("chat.rag.answer")
    ).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
  })

  it.each([
    ["null saved parts", { saved_parts: null }],
    ["null revision", { revision: null }],
    ["invalid revision", { revision: "not-a-uuid" }],
    ["different effective parts", {
      effective_parts: { template: "Different {context} {question}" }
    }]
  ])("rejects noncanonical user detail with %s", async (_name, change) => {
    const saved = { template: "Saved {context} {question}" }
    vi.mocked(bgRequest).mockResolvedValueOnce({
      ...userDetail(saved),
      ...change
    })

    await expect(
      servicePromptMethods.getServicePrompt("chat.rag.answer")
    ).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
  })

  it.each([
    ["packaged response", detail],
    ["different user parts", userDetail({
      template: "Server changed {context} {question}"
    })]
  ])("rejects PUT success with %s", async (_name, response) => {
    const parts = { template: "Submitted {context} {question}" }
    vi.mocked(bgRequest).mockResolvedValueOnce(response)

    await expect(servicePromptMethods.saveServicePrompt(
      "chat.rag.answer",
      { parts, expected_revision: null }
    )).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
  })

  it("rejects a DELETE success that is not canonical packaged state", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(userDetail({
      template: "Still saved {context} {question}"
    }))

    await expect(
      servicePromptMethods.resetServicePrompt("chat.rag.answer", VALID_REVISION)
    ).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
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
