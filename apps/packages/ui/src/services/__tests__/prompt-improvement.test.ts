import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  apiSend: vi.fn()
}))

vi.mock("@/services/api-send", () => ({
  apiSend: (...args: unknown[]) => mocks.apiSend(...args)
}))

import {
  PromptImprovementApiError,
  collectProtectedTokens,
  improvePrompt,
  type PromptImproveRequest,
  type PromptImproveResponse
} from "@/services/prompt-improvement"
import {
  fetchPromptCapabilities,
  type PromptImprovementLimits
} from "@/services/prompts-api"

const limits: PromptImprovementLimits = {
  max_request_bytes: 64_000,
  max_draft_chars: 24_000,
  max_candidate_chars: 24_000,
  max_raw_output_chars: 32_000,
  max_findings: 5,
  max_finding_text_chars: 500,
  max_provider_chars: 100,
  max_model_chars: 500,
  max_meta_prompt_version_chars: 100,
  max_warning_chars: 100,
  max_warnings: 16,
  max_protected_tokens: 64,
  max_protected_token_kind_chars: 50,
  max_protected_token_chars: 500,
  max_protected_token_occurrences: 100,
  max_protected_token_total_chars: 4_000
}

const improvedResponse = (operationId = "11111111-1111-4111-8111-111111111111") =>
  ({
    schema_version: 1,
    operation_id: operationId,
    status: "improved",
    improved_text: "Write a concise summary.",
    findings: [
      {
        category: "concision",
        issue: "The request repeated itself.",
        change: "Removed the repetition."
      }
    ],
    review_required: false,
    warnings: [],
    resolved_model: {
      provider: "openai",
      model: "gpt-5-mini",
      display_name: "GPT-5 mini"
    },
    meta_prompt_version: "prompt-improvement-v1"
  }) satisfies PromptImproveResponse

const request = (): PromptImproveRequest => ({
  operation_id: "11111111-1111-4111-8111-111111111111",
  target: "system",
  text: "Be concise.",
  model_selection: { selected_model: "auto" },
  protected_tokens: []
})

describe("prompt improvement service", () => {
  beforeEach(() => {
    mocks.apiSend.mockReset()
  })

  it("sends the isolated target and route only in a static POST body", async () => {
    const response = improvedResponse()
    mocks.apiSend.mockResolvedValue({ ok: true, status: 200, data: response })

    await expect(
      improvePrompt({
        operation_id: response.operation_id,
        target: "user_message",
        text: "Summarize {{topic}} for @alice using /brief.",
        model_selection: {
          selected_model: "openai/gpt-5-mini",
          provider_hint: "openai"
        },
        protected_tokens: [
          { kind: "template_variable", value: "{{topic}}", occurrences: 1 },
          { kind: "mention", value: "@alice", occurrences: 1 },
          { kind: "slash_command", value: "/brief", occurrences: 1 }
        ]
      })
    ).resolves.toEqual(response)

    expect(mocks.apiSend).toHaveBeenCalledTimes(1)
    expect(mocks.apiSend).toHaveBeenCalledWith({
      path: "/api/v1/prompts/improve",
      method: "POST",
      body: {
        operation_id: response.operation_id,
        target: "user_message",
        text: "Summarize {{topic}} for @alice using /brief.",
        model_selection: {
          selected_model: "openai/gpt-5-mini",
          provider_hint: "openai"
        },
        protected_tokens: [
          { kind: "template_variable", value: "{{topic}}", occurrences: 1 },
          { kind: "mention", value: "@alice", occurrences: 1 },
          { kind: "slash_command", value: "/brief", occurrences: 1 }
        ]
      }
    })
    const request = mocks.apiSend.mock.calls[0][0]
    expect(request.path).not.toContain("Summarize")
    expect(request.path).not.toContain("gpt-5-mini")
    expect(request.path).not.toContain("?")
  })

  it("projects exact request, route, and protected-token DTOs", async () => {
    const response = improvedResponse()
    mocks.apiSend.mockResolvedValue({ ok: true, status: 200, data: response })
    const privateRequest: PromptImproveRequest & { conversation_history: string } = {
      ...request(),
      conversation_history: "private history",
      model_selection: {
        selected_model: "auto",
        provider_hint: null,
        api_key: "private key",
        rag_context: "private RAG"
      } as PromptImproveRequest["model_selection"],
      protected_tokens: [
        {
          kind: "mention",
          value: "@alice",
          occurrences: 1,
          attachment_body: "private attachment",
          tool_state: "private tool state"
        } as PromptImproveRequest["protected_tokens"][number]
      ]
    }

    await improvePrompt(privateRequest)

    expect(mocks.apiSend.mock.calls[0][0].body).toEqual({
      operation_id: privateRequest.operation_id,
      target: "system",
      text: "Be concise.",
      model_selection: { selected_model: "auto", provider_hint: null },
      protected_tokens: [{ kind: "mention", value: "@alice", occurrences: 1 }]
    })
  })

  it("collects only bounded recognized tokens visible in this target", () => {
    const text = "Use {{topic}} twice: {{topic}}. Ask @alice and run /brief."
    const tokens = collectProtectedTokens(
      text,
      [
        { kind: "saved_variable", value: "{{topic}}" },
        { kind: "mention", value: "@alice" },
        { kind: "mention", value: "@alice" },
        { kind: "slash_command", value: "/brief" },
        { kind: "attachment_reference", value: "hidden attachment body" },
        { kind: "mention", value: "x".repeat(501) }
      ],
      limits
    )

    expect(tokens).toEqual([
      { kind: "template_variable", value: "{{topic}}", occurrences: 2 },
      { kind: "saved_variable", value: "{{topic}}", occurrences: 2 },
      { kind: "mention", value: "@alice", occurrences: 1 },
      { kind: "slash_command", value: "/brief", occurrences: 1 }
    ])
  })

  it("projects recognized token fields before returning preservation hints", () => {
    const recognized = {
      kind: "mention",
      value: "@alice",
      attachment_body: "private attachment",
      saved_prompt_metadata: "private saved metadata"
    }

    expect(collectProtectedTokens("Ask @alice", [recognized], limits)).toEqual([
      { kind: "mention", value: "@alice", occurrences: 1 }
    ])
  })

  it("uses Python-compatible code-point limits and non-overlapping counts", () => {
    const unicodeLimits = {
      ...limits,
      max_protected_token_kind_chars: 3,
      max_protected_token_chars: 3,
      max_protected_token_total_chars: 8
    }

    expect(
      collectProtectedTokens(
        "😀😀😀 aaa b\0c",
        [
          { kind: "😀", value: "😀😀" },
          { kind: "k", value: "aa" },
          { kind: "a\0b", value: "c" },
          { kind: "a", value: "b\0c" }
        ],
        unicodeLimits
      )
    ).toEqual([
      { kind: "😀", value: "😀😀", occurrences: 1 },
      { kind: "k", value: "aa", occurrences: 1 },
      { kind: "a\0b", value: "c", occurrences: 1 },
      { kind: "a", value: "b\0c", occurrences: 1 }
    ])
  })

  it("rejects a token one code point over the exact boundary", () => {
    expect(
      collectProtectedTokens(
        "😀😀",
        [{ kind: "k", value: "😀😀" }],
        { ...limits, max_protected_token_chars: 1 }
      )
    ).toEqual([])
  })

  it("normalizes stable server errors without retrying", async () => {
    mocks.apiSend.mockResolvedValue({
      ok: false,
      status: 429,
      data: {
        code: "provider_rate_limited",
        message: "The active provider is temporarily rate limited.",
        retryable: true,
        retry_after_seconds: 20,
        request_id: "req-1"
      }
    })

    const pending = improvePrompt({
      operation_id: "11111111-1111-4111-8111-111111111111",
      target: "system",
      text: "Be concise.",
      model_selection: { selected_model: "auto" },
      protected_tokens: []
    })

    await expect(pending).rejects.toMatchObject({
      name: "PromptImprovementApiError",
      code: "provider_rate_limited",
      retryable: true,
      retryAfterSeconds: 20,
      requestId: "req-1",
      status: 429
    })
    expect(mocks.apiSend).toHaveBeenCalledTimes(1)
  })

  it("maps a valid bounded error envelope to local public copy", async () => {
    mocks.apiSend.mockResolvedValue({
      ok: false,
      status: 429,
      data: {
        code: "provider_rate_limited",
        message: "private draft echo from a mixed server",
        retryable: true,
        retry_after_seconds: 20,
        request_id: "req-1"
      }
    })

    await expect(improvePrompt(request())).rejects.toMatchObject({
      code: "provider_rate_limited",
      message: "The active provider is temporarily rate limited.",
      retryable: true,
      retryAfterSeconds: 20,
      requestId: "req-1"
    })
  })

  it("accepts Python-bounded astral text in error and success envelopes", async () => {
    mocks.apiSend
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        data: {
          code: "invalid_input",
          message: "😀".repeat(300),
          retryable: false,
          request_id: "req-astral"
        }
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        data: { ...improvedResponse(), improved_text: "😀" }
      })

    await expect(improvePrompt(request())).rejects.toMatchObject({
      code: "invalid_input",
      requestId: "req-astral"
    })
    await expect(
      improvePrompt(request(), { ...limits, max_candidate_chars: 1 })
    ).resolves.toMatchObject({ improved_text: "😀" })
  })

  it.each([
    ["missing message", { code: "invalid_input", retryable: false, request_id: "req-1" }],
    [
      "oversized message",
      {
        code: "invalid_input",
        message: "x".repeat(301),
        retryable: false,
        request_id: "req-1"
      }
    ],
    [
      "non-boolean retryable",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: "false",
        request_id: "req-1"
      }
    ],
    [
      "negative retry delay",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: false,
        retry_after_seconds: -1,
        request_id: "req-1"
      }
    ],
    [
      "oversized retry delay",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: false,
        retry_after_seconds: 86_401,
        request_id: "req-1"
      }
    ],
    [
      "fractional retry delay",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: false,
        retry_after_seconds: 1.5,
        request_id: "req-1"
      }
    ],
    [
      "oversized request ID",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: false,
        request_id: "r".repeat(129)
      }
    ],
    [
      "extra field",
      {
        code: "invalid_input",
        message: "Invalid input.",
        retryable: false,
        request_id: "req-1",
        draft_echo: "private draft"
      }
    ]
  ])("rejects a malformed error envelope with %s", async (_label, data) => {
    mocks.apiSend.mockResolvedValue({ ok: false, status: 400, data })

    await expect(improvePrompt(request())).rejects.toMatchObject({
      code: "internal_error",
      message: "Prompt improvement failed.",
      retryable: false,
      retryAfterSeconds: null,
      requestId: null
    })
  })

  it.each([
    [
      "candidate",
      { ...improvedResponse(), improved_text: "xxxx" },
      { ...limits, max_candidate_chars: 3 }
    ],
    [
      "finding count",
      {
        ...improvedResponse(),
        findings: [
          improvedResponse().findings[0],
          improvedResponse().findings[0]
        ]
      },
      { ...limits, max_findings: 1 }
    ],
    [
      "finding issue",
      {
        ...improvedResponse(),
        findings: [
          { ...improvedResponse().findings[0], issue: "xxxx" }
        ]
      },
      { ...limits, max_finding_text_chars: 3 }
    ],
    [
      "finding change",
      {
        ...improvedResponse(),
        findings: [
          { ...improvedResponse().findings[0], change: "xxxx" }
        ]
      },
      { ...limits, max_finding_text_chars: 3 }
    ],
    [
      "warning count",
      { ...improvedResponse(), warnings: ["one", "two"] },
      { ...limits, max_warnings: 1 }
    ],
    [
      "warning text",
      { ...improvedResponse(), warnings: ["xxxx"] },
      { ...limits, max_warning_chars: 3 }
    ],
    [
      "provider",
      {
        ...improvedResponse(),
        resolved_model: { ...improvedResponse().resolved_model, provider: "xxxx" }
      },
      { ...limits, max_provider_chars: 3 }
    ],
    [
      "model",
      {
        ...improvedResponse(),
        resolved_model: { ...improvedResponse().resolved_model, model: "xxxx" }
      },
      { ...limits, max_model_chars: 3 }
    ],
    [
      "display name",
      {
        ...improvedResponse(),
        resolved_model: {
          ...improvedResponse().resolved_model,
          display_name: "xxxx"
        }
      },
      { ...limits, max_model_chars: 3 }
    ],
    [
      "meta prompt version",
      { ...improvedResponse(), meta_prompt_version: "xxxx" },
      { ...limits, max_meta_prompt_version_chars: 3 }
    ]
  ])("rejects a success response over the advertised %s bound", async (_label, data, responseLimits) => {
    mocks.apiSend.mockResolvedValue({ ok: true, status: 200, data })

    await expect(improvePrompt(request(), responseLimits)).rejects.toMatchObject({
      code: "invalid_model_output"
    })
  })

  it("rejects unknown fields in a schema-version-1 success response", async () => {
    mocks.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ...improvedResponse(), hidden_reasoning: "private" }
    })

    await expect(improvePrompt(request(), limits)).rejects.toMatchObject({
      code: "invalid_model_output"
    })
  })

  it("fails closed for malformed success and unknown error payloads", async () => {
    mocks.apiSend
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        data: { ...improvedResponse(), schema_version: 2 }
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        data: {
          code: "new_server_error",
          message: "Be concise. private draft echo",
          retryable: true,
          request_id: "req-2"
        }
      })

    const request = {
      operation_id: "11111111-1111-4111-8111-111111111111",
      target: "system" as const,
      text: "Be concise.",
      model_selection: { selected_model: "auto" },
      protected_tokens: []
    }

    await expect(improvePrompt(request)).rejects.toMatchObject({
      code: "invalid_model_output"
    })
    try {
      await improvePrompt(request)
      throw new Error("expected failure")
    } catch (error) {
      expect(error).toBeInstanceOf(PromptImprovementApiError)
      expect(error).toMatchObject({ code: "internal_error", retryable: false })
      expect((error as Error).message).not.toContain("private draft echo")
    }
  })

  it("normalizes thrown transport failures without exposing exception text", async () => {
    mocks.apiSend.mockRejectedValue(
      new Error("Extension failed while handling private draft contents")
    )

    try {
      await improvePrompt({
        operation_id: "11111111-1111-4111-8111-111111111111",
        target: "system",
        text: "private draft contents",
        model_selection: { selected_model: "auto" },
        protected_tokens: []
      })
      throw new Error("expected failure")
    } catch (error) {
      expect(error).toMatchObject({
        name: "PromptImprovementApiError",
        code: "provider_unavailable",
        retryable: true,
        status: 0
      })
      expect((error as Error).message).not.toContain("private draft contents")
    }
  })
})

describe("prompt capability discovery", () => {
  beforeEach(() => {
    mocks.apiSend.mockReset()
  })

  it("keeps both advertised unsupported capabilities disabled", async () => {
    mocks.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        prompt_improvement_v1: { supported: false, limits },
        single_text_recipe_v2: { supported: false }
      }
    })

    await expect(fetchPromptCapabilities()).resolves.toEqual({
      availability: "unavailable",
      prompt_improvement_v1: { supported: false, limits },
      single_text_recipe_v2: { supported: false }
    })
  })

  it("ignores additive capability and limit fields after validating known fields", async () => {
    mocks.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        prompt_improvement_v1: {
          supported: true,
          limits: { ...limits, future_limit: 123 },
          future_metadata: { safe: true }
        },
        single_text_recipe_v2: { supported: false, future_metadata: true },
        prompt_improvement_v2: { supported: false }
      }
    })

    await expect(fetchPromptCapabilities()).resolves.toEqual({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits },
      single_text_recipe_v2: { supported: false }
    })
  })

  it.each([
    ["missing known limit", Object.fromEntries(Object.entries(limits).slice(1))],
    ["zero known limit", { ...limits, max_candidate_chars: 0 }],
    ["fractional known limit", { ...limits, max_candidate_chars: 1.5 }],
    ["string known limit", { ...limits, max_candidate_chars: "24" }]
  ])("fails closed for a capability with %s", async (_label, invalidLimits) => {
    mocks.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        prompt_improvement_v1: { supported: true, limits: invalidLimits },
        single_text_recipe_v2: { supported: false }
      }
    })

    await expect(fetchPromptCapabilities()).resolves.toEqual({
      availability: "unavailable",
      prompt_improvement_v1: { supported: false, limits: null },
      single_text_recipe_v2: { supported: false }
    })
  })

  it.each([
    ["404", { ok: false, status: 404, error: "Not found" }],
    ["offline", { ok: false, status: 0, error: "private network detail" }],
    ["malformed", { ok: true, status: 200, data: { prompt_improvement_v1: {} } }],
    ["unknown", { ok: true, status: 200, data: { future_capability: true } }]
  ])("fails closed when capability discovery is %s", async (_label, response) => {
    mocks.apiSend.mockResolvedValue(response)

    await expect(fetchPromptCapabilities()).resolves.toEqual({
      availability: "unavailable",
      prompt_improvement_v1: { supported: false, limits: null },
      single_text_recipe_v2: { supported: false }
    })
  })
})
