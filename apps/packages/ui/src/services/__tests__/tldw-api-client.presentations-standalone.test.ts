import { beforeEach, describe, expect, it, vi } from "vitest"

import { tldwRequest } from "@/services/tldw/request-core"
import { presentationsMethods, type TldwApiClientCore } from "@/services/tldw/domains/presentations"
import type { StandaloneHtmlPresentationStudioRecord } from "@/services/tldw/TldwApiClient"

const ACCEPTED_CONTENT_KINDS = "structured_slides,standalone_html"
const ACCEPT_HEADER = "X-Slides-Accept-Content-Kinds"
const FIXED_DISPOSITION = 'attachment; filename="presentation.html"'

const structuredDetail = {
  id: "structured-1",
  content_kind: "structured_slides",
  title: "Structured deck",
  description: null,
  theme: "black",
  marp_theme: null,
  template_id: null,
  visual_style_id: null,
  visual_style_scope: null,
  visual_style_name: null,
  visual_style_version: null,
  visual_style_snapshot: null,
  settings: null,
  studio_data: null,
  slides: [
    {
      order: 0,
      layout: "title",
      title: "Opening",
      content: "Hello",
      speaker_notes: null,
      metadata: {}
    }
  ],
  custom_css: null,
  source_type: "manual",
  source_ref: null,
  source_query: null,
  created_at: "2026-07-15T00:00:00Z",
  last_modified: "2026-07-15T00:00:01Z",
  deleted: false,
  client_id: "1",
  version: 3
}

const standaloneDetail = {
  id: "html-1",
  content_kind: "standalone_html",
  title: "Standalone deck",
  description: null,
  theme: "black",
  source_type: "prompt",
  source_ref: null,
  source_query: null,
  created_at: "2026-07-15T00:00:00Z",
  last_modified: "2026-07-15T00:00:01Z",
  deleted: false,
  client_id: "1",
  version: 7,
  html_document: "<!doctype html><title>Standalone deck</title>",
  html_sha256: "a".repeat(64),
  html_bytes: 50,
  html_slide_count: 1,
  generation_provenance: {
    schema_version: 1,
    source_kind: "prompt",
    source_ref: null,
    source_snapshot_hmac_sha256: "b".repeat(64),
    digest_key_id: "slides-generation-v1",
    source_bytes: 8,
    provider: "openai",
    model: "test-model",
    adapter_id: "openai",
    endpoint_identity: "https://api.openai.com/v1",
    prompt_sha256: "c".repeat(64)
  }
}

const capabilities = {
  schema_version: 1,
  content_kind_request_header: ACCEPT_HEADER,
  content_kinds: {
    structured_slides: { read: true, edit: true },
    standalone_html: {
      read: true,
      edit: true,
      export_attachment: true,
      draft_attachment: true,
      reason: null,
      limits: {
        max_document_bytes: 1_048_576,
        max_source_write_bytes: 1_048_576,
        max_draft_attachment_bytes: 1_048_576,
        max_slides: 30,
        max_nesting_depth: 128
      }
    }
  },
  generation_modes: {
    structured_slides: {
      enabled: true,
      transport: "existing_source_endpoints"
    },
    standalone_html: {
      enabled: true,
      reason: null,
      transport: "slides_generation_job",
      source_kinds: ["prompt", "chat", "media", "notes", "rag"],
      provider: "openai",
      model: "test-model",
      adapter_id: "openai",
      endpoint_identity: "https://api.openai.com/v1",
      generation_config_revision: `sha256:${"d".repeat(64)}`,
      input_limits: {
        max_request_bytes: 4_194_304,
        max_source_chars: 200_000,
        max_source_tokens: 50_000,
        max_audience_chars: 500,
        max_source_identifier_bytes: 256,
        max_note_ids: 100,
        max_rag_query_chars: 20_000,
        max_rag_top_k: 100
      },
      output_limits: {
        max_provider_response_bytes: 8_388_608,
        max_document_bytes: 1_048_576
      }
    }
  }
}

const generationRequest = {
  generation_mode: "standalone_html" as const,
  generation_config_revision: `sha256:${"d".repeat(64)}`,
  source: { kind: "prompt" as const, prompt: "Explain the system" },
  html_options: {
    presentation_type: "tech-sharing" as const,
    audience: "backend engineers",
    slide_count: 10,
    visual_direction: "dark-technical" as const,
    delivery_style: "speaker-led" as const
  }
}

const pendingReceipt = {
  generation_id: "018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  status: "queued",
  status_url: "/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  presentation_id: null
}

const responseEnvelope = (
  data: unknown,
  options?: { status?: number; headers?: Record<string, string>; ok?: boolean }
) => ({
  ok: options?.ok ?? true,
  status: options?.status ?? 200,
  data,
  headers: options?.headers ?? { "content-type": "application/json" }
})

const createCore = (requestImpl: (...args: any[]) => any): TldwApiClientCore => ({
  ensureConfigForRequest: vi.fn(async () => ({})),
  request: vi.fn(requestImpl) as unknown as TldwApiClientCore["request"],
  resolveApiPath: vi.fn(async (_key: string, candidates: string[]) => candidates[0]),
  fillPathParams: vi.fn((template: string, values: string | string[]) => {
    const replacements = Array.isArray(values) ? values : [values]
    let index = 0
    return template.replace(/\{[^}]+\}/g, () => replacements[index++] ?? "")
  })
})

const requestHeaders = (request: unknown): Record<string, string> =>
  (request as { headers?: Record<string, string> }).headers ?? {}

const mutateCapabilities = (mutate: (value: any) => void): any => {
  const value = JSON.parse(JSON.stringify(capabilities))
  mutate(value)
  return value
}

describe("standalone presentation client contracts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("normalizes structured detail and preserves the response weak ETag", async () => {
    const client = createCore(async () =>
      responseEnvelope(structuredDetail, { headers: { etag: 'W/"v3"' } })
    )

    const result = await presentationsMethods.getPresentation.call(client, "structured-1")

    expect(result).toEqual({
      record: expect.objectContaining({
        id: "structured-1",
        content_kind: "structured_slides",
        slides: structuredDetail.slides
      }),
      etag: 'W/"v3"'
    })
  })

  it("normalizes standalone detail without adding structured slide fields", async () => {
    const client = createCore(async () =>
      responseEnvelope(standaloneDetail, { headers: { ETag: '"v7"' } })
    )

    const result = await presentationsMethods.getPresentation.call(client, "html-1")

    expect(result).toEqual({
      record: expect.objectContaining({
        id: "html-1",
        content_kind: "standalone_html",
        html_document: standaloneDetail.html_document,
        html_sha256: "a".repeat(64),
        html_bytes: 50,
        html_slide_count: 1
      }),
      etag: '"v7"'
    })
    expect(result.record).not.toHaveProperty("slides")
  })

  it.each([
    ["missing slides", { ...structuredDetail, slides: undefined }],
    ["non-array slides", { ...structuredDetail, slides: "not-an-array" }]
  ])("rejects explicit structured detail with %s", async (_case, detail) => {
    const client = createCore(async () => responseEnvelope(detail))

    const error = await presentationsMethods.getPresentation
      .call(client, "structured-1")
      .then(
        () => null,
        (failure: unknown) => failure
      )

    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).toBe("Invalid presentation detail response")
  })

  it.each([
    ["missing HTML document", { ...standaloneDetail, html_document: undefined }],
    ["non-string HTML document", { ...standaloneDetail, html_document: 42 }],
    ["missing digest", { ...standaloneDetail, html_sha256: undefined }],
    ["malformed digest", { ...standaloneDetail, html_sha256: "private-source-digest" }],
    ["missing byte count", { ...standaloneDetail, html_bytes: undefined }],
    ["non-integer byte count", { ...standaloneDetail, html_bytes: 1.5 }],
    ["missing slide count", { ...standaloneDetail, html_slide_count: undefined }],
    ["non-integer slide count", { ...standaloneDetail, html_slide_count: 1.5 }],
    ["missing creation timestamp", { ...standaloneDetail, created_at: undefined }],
    ["non-string modification timestamp", { ...standaloneDetail, last_modified: 7 }],
    ["missing provenance", { ...standaloneDetail, generation_provenance: undefined }],
    ["non-object provenance", { ...standaloneDetail, generation_provenance: [] }],
    ["missing id", { ...standaloneDetail, id: undefined }],
    ["missing title", { ...standaloneDetail, title: undefined }],
    ["missing theme", { ...standaloneDetail, theme: undefined }],
    ["missing client identity", { ...standaloneDetail, client_id: undefined }],
    ["non-integer version", { ...standaloneDetail, version: "7" }],
    ["missing deleted flag", { ...standaloneDetail, deleted: undefined }]
  ])("rejects standalone detail with %s using a bounded error", async (_case, detail) => {
    const client = createCore(async () => responseEnvelope(detail))

    const error = await presentationsMethods.getPresentation.call(client, "html-1").then(
      () => null,
      (failure: unknown) => failure
    )

    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).toBe("Invalid presentation detail response")
    expect((error as Error).message).not.toContain("private-source-digest")
  })

  it("preserves unknown and missing-HTML discriminators as source-free read-only records", async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(
        responseEnvelope({
          ...standaloneDetail,
          content_kind: "future_canvas",
          future_payload: "must-not-enter-state"
        })
      )
      .mockResolvedValueOnce(
        responseEnvelope({
          ...standaloneDetail,
          content_kind: undefined
        })
      )
    const client = createCore(request)

    const unknown = await presentationsMethods.getPresentation.call(client, "future-1")
    const missing = await presentationsMethods.getPresentation.call(client, "missing-1")

    expect(unknown.record).toEqual(
      expect.objectContaining({
        content_kind: "unsupported",
        unsupported_content_kind: "future_canvas",
        read_only: true
      })
    )
    expect(missing.record).toEqual(
      expect.objectContaining({
        content_kind: "unsupported",
        unsupported_content_kind: null,
        read_only: true
      })
    )
    for (const result of [unknown, missing]) {
      expect(result.record).not.toHaveProperty("slides")
      expect(result.record).not.toHaveProperty("html_document")
      expect(result.record).not.toHaveProperty("future_payload")
    }
  })

  it("lists discriminated source-free summaries and preserves unknown kinds", async () => {
    const client = createCore(async () => ({
      presentations: [
        {
          id: "structured-1",
          content_kind: "structured_slides",
          title: "Structured",
          description: null,
          theme: "black",
          created_at: "2026-07-15T00:00:00Z",
          last_modified: "2026-07-15T00:00:01Z",
          deleted: false,
          version: 2,
          provenance: { source_kind: null, provider: null, model: null },
          slide_count: 3,
          html_document: "must be dropped"
        },
        {
          id: "html-1",
          content_kind: "standalone_html",
          title: "HTML",
          description: null,
          theme: "black",
          created_at: "2026-07-15T00:00:00Z",
          last_modified: "2026-07-15T00:00:01Z",
          deleted: false,
          version: 7,
          provenance: {
            source_kind: "prompt",
            provider: "openai",
            model: "test-model"
          },
          html_slide_count: 4,
          html_bytes: 1234,
          html_document: "must be dropped"
        },
        {
          id: "future-1",
          content_kind: "future_canvas",
          title: "Future",
          theme: "black",
          created_at: "2026-07-15T00:00:00Z",
          last_modified: "2026-07-15T00:00:01Z",
          deleted: false,
          version: 1,
          provenance: { source_kind: null, provider: null, model: null },
          future_payload: "must be dropped"
        }
      ],
      total: 3,
      limit: 50,
      offset: 0,
      pagination: {
        mode: "offset",
        limit: 50,
        offset: 0,
        total: 3,
        has_more: false,
        next_offset: null
      },
      has_more: false,
      next_offset: null
    }))

    const result = await (presentationsMethods as any).listPresentations.call(client, {
      limit: 50,
      offset: 0
    })

    expect(
      result.presentations.map((record: { content_kind: string }) => record.content_kind)
    ).toEqual(["structured_slides", "standalone_html", "unsupported"])
    expect(result.presentations[2]).toEqual(
      expect.objectContaining({
        unsupported_content_kind: "future_canvas",
        read_only: true
      })
    )
    for (const record of result.presentations) {
      expect(record).not.toHaveProperty("html_document")
      expect(record).not.toHaveProperty("future_payload")
      expect(record).not.toHaveProperty("slides")
    }
  })

  it("returns source-free presentation metadata and its response ETag", async () => {
    const metadata = {
      id: "html-1",
      content_kind: "standalone_html",
      title: "HTML",
      description: null,
      theme: "black",
      created_at: "2026-07-15T00:00:00Z",
      last_modified: "2026-07-15T00:00:01Z",
      deleted: false,
      version: 7,
      provenance: {
        source_kind: "prompt",
        provider: "openai",
        model: "test-model"
      },
      html_slide_count: 4,
      html_bytes: 1234,
      html_document: "must be dropped"
    }
    const client = createCore(async () => responseEnvelope(metadata, { headers: { etag: '"v7"' } }))

    const result = await (presentationsMethods as any).getPresentationMetadata.call(
      client,
      "html-1"
    )

    expect(result).toEqual({
      record: expect.objectContaining({
        content_kind: "standalone_html",
        html_slide_count: 4,
        html_bytes: 1234
      }),
      etag: '"v7"'
    })
    expect(result.record).not.toHaveProperty("html_document")
    expect(requestHeaders((client.request as any).mock.calls[0][0])).not.toHaveProperty(
      ACCEPT_HEADER
    )
  })

  it("accepts the exact capabilities shape without adding negotiation", async () => {
    const client = createCore(async () => capabilities)

    const result = await (presentationsMethods as any).getSlidesCapabilities.call(client)

    expect(result).toEqual(capabilities)
    expect((client.request as any).mock.calls[0][0]).toEqual({
      path: "/api/v1/slides/capabilities",
      method: "GET"
    })
  })

  it.each([
    [
      "available validator with editing disabled",
      (value: any) => {
        value.content_kinds.standalone_html.edit = false
      }
    ],
    [
      "available validator with export disabled",
      (value: any) => {
        value.content_kinds.standalone_html.export_attachment = false
      }
    ],
    [
      "unavailable validator with editing enabled",
      (value: any) => {
        Object.assign(value.content_kinds.standalone_html, {
          edit: true,
          export_attachment: false,
          reason: "validator_unavailable"
        })
      }
    ],
    [
      "unavailable validator with export enabled",
      (value: any) => {
        Object.assign(value.content_kinds.standalone_html, {
          edit: false,
          export_attachment: true,
          reason: "validator_unavailable"
        })
      }
    ],
    [
      "standalone reads disabled",
      (value: any) => {
        value.content_kinds.standalone_html.read = false
      }
    ],
    [
      "draft recovery disabled",
      (value: any) => {
        value.content_kinds.standalone_html.draft_attachment = false
      }
    ]
  ])("rejects impossible content capability state: %s", async (_case, mutate) => {
    const client = createCore(async () =>
      mutateCapabilities(mutate as (value: any) => void)
    )

    await expect((presentationsMethods as any).getSlidesCapabilities.call(client)).rejects.toThrow(
      "Invalid Slides capabilities response"
    )
  })

  it.each([
    [
      "enabled with a reason",
      (value: any) => {
        value.generation_modes.standalone_html.reason = "feature_disabled"
      }
    ],
    [
      "enabled without a provider",
      (value: any) => {
        value.generation_modes.standalone_html.provider = null
      }
    ],
    [
      "enabled with a blank model",
      (value: any) => {
        value.generation_modes.standalone_html.model = " "
      }
    ],
    [
      "enabled with a blank adapter",
      (value: any) => {
        value.generation_modes.standalone_html.adapter_id = ""
      }
    ],
    [
      "enabled with a blank endpoint",
      (value: any) => {
        value.generation_modes.standalone_html.endpoint_identity = "\t"
      }
    ],
    [
      "enabled with a malformed revision",
      (value: any) => {
        value.generation_modes.standalone_html.generation_config_revision = "sha256:ABC"
      }
    ],
    [
      "disabled without a reason",
      (value: any) => {
        Object.assign(value.generation_modes.standalone_html, {
          enabled: false,
          reason: null,
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: null
        })
      }
    ],
    ...["provider", "model", "adapter_id", "endpoint_identity"].map((field) => [
      `disabled with ${field}`,
      (value: any) => {
        Object.assign(value.generation_modes.standalone_html, {
          enabled: false,
          reason: "feature_disabled",
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: null,
          [field]: "must-be-null"
        })
      }
    ]),
    [
      "disabled with a revision",
      (value: any) => {
        Object.assign(value.generation_modes.standalone_html, {
          enabled: false,
          reason: "feature_disabled",
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: `sha256:${"e".repeat(64)}`
        })
      }
    ],
    [
      "unknown safe reason",
      (value: any) => {
        Object.assign(value.generation_modes.standalone_html, {
          enabled: false,
          reason: "provider-secret-outage",
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: null
        })
      }
    ],
    [
      "missing source kind",
      (value: any) => {
        value.generation_modes.standalone_html.source_kinds = ["prompt", "chat", "media", "notes"]
      }
    ],
    [
      "duplicate source kind",
      (value: any) => {
        value.generation_modes.standalone_html.source_kinds = [
          "prompt",
          "chat",
          "media",
          "notes",
          "notes"
        ]
      }
    ],
    [
      "source kinds out of order",
      (value: any) => {
        value.generation_modes.standalone_html.source_kinds = [
          "chat",
          "prompt",
          "media",
          "notes",
          "rag"
        ]
      }
    ],
    [
      "extra response field",
      (value: any) => {
        value.generation_modes.standalone_html.unexpected = true
      }
    ]
  ])("rejects impossible generation capability state: %s", async (_case, mutate) => {
    const client = createCore(async () =>
      mutateCapabilities(mutate as (value: any) => void)
    )

    await expect((presentationsMethods as any).getSlidesCapabilities.call(client)).rejects.toThrow(
      "Invalid Slides capabilities response"
    )
  })

  it.each([
    [
      "zero content limit",
      (value: any) => {
        value.content_kinds.standalone_html.limits.max_document_bytes = 0
      }
    ],
    [
      "negative content limit",
      (value: any) => {
        value.content_kinds.standalone_html.limits.max_nesting_depth = -1
      }
    ],
    [
      "fractional input limit",
      (value: any) => {
        value.generation_modes.standalone_html.input_limits.max_source_chars = 1.5
      }
    ],
    [
      "zero output limit",
      (value: any) => {
        value.generation_modes.standalone_html.output_limits.max_document_bytes = 0
      }
    ]
  ])("rejects invalid effective capability limit: %s", async (_case, mutate) => {
    const client = createCore(async () =>
      mutateCapabilities(mutate as (value: any) => void)
    )

    await expect((presentationsMethods as any).getSlidesCapabilities.call(client)).rejects.toThrow(
      "Invalid Slides capabilities response"
    )
  })

  it.each([
    ["feature disabled with validator available", false],
    ["feature disabled with validator unavailable", true]
  ])("accepts Task 11 disabled capabilities: %s", async (_case, validatorUnavailable) => {
    const disabled = mutateCapabilities((value) => {
      Object.assign(value.generation_modes.standalone_html, {
        enabled: false,
        reason: "feature_disabled",
        provider: null,
        model: null,
        adapter_id: null,
        endpoint_identity: null,
        generation_config_revision: null
      })
      if (validatorUnavailable) {
        Object.assign(value.content_kinds.standalone_html, {
          edit: false,
          export_attachment: false,
          reason: "validator_unavailable"
        })
      }
    })
    const client = createCore(async () => disabled)

    await expect(
      (presentationsMethods as any).getSlidesCapabilities.call(client)
    ).resolves.toEqual(disabled)
  })

  it("rejects generation enabled when the validator is unavailable", async () => {
    const invalid = mutateCapabilities((value) => {
      Object.assign(value.content_kinds.standalone_html, {
        edit: false,
        export_attachment: false,
        reason: "validator_unavailable"
      })
    })
    const client = createCore(async () => invalid)

    await expect((presentationsMethods as any).getSlidesCapabilities.call(client)).rejects.toThrow(
      "Invalid Slides capabilities response"
    )
  })

  it("submits the closed generation request with idempotency but no kind negotiation", async () => {
    const client = createCore(async () => pendingReceipt)

    const result = await (presentationsMethods as any).submitPresentationGeneration.call(
      client,
      generationRequest,
      {
        idempotencyKey: "task13-generation-key"
      }
    )

    expect(result).toEqual(pendingReceipt)
    expect((client.request as any).mock.calls[0][0]).toEqual({
      path: "/api/v1/slides/generations",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": "task13-generation-key"
      },
      body: generationRequest
    })
  })

  it.each([
    {
      ...pendingReceipt,
      status: "running",
      progress_text: "Resolving source"
    },
    {
      ...pendingReceipt,
      status: "completed",
      presentation_id: "html-1",
      content_kind: "standalone_html"
    },
    {
      ...pendingReceipt,
      status: "failed",
      error_code: "provider_timeout",
      error_message: "The provider timed out."
    },
    {
      ...pendingReceipt,
      status: "cancelled",
      error_code: "generation_cancelled"
    }
  ])("accepts the closed $status generation status variant", async (receipt) => {
    const client = createCore(async () => receipt)

    const result = await (presentationsMethods as any).getPresentationGeneration.call(
      client,
      receipt.generation_id
    )

    expect(result).toEqual(receipt)
    expect(requestHeaders((client.request as any).mock.calls[0][0])).not.toHaveProperty(
      ACCEPT_HEADER
    )
  })

  it("returns a validated receipt with bounded Retry-After metadata without changing the receipt-only method", async () => {
    const client = createCore(async (request) => {
      if (request.returnResponse) {
        return {
          ...responseEnvelope(pendingReceipt),
          retryAfterMs: 90_000
        }
      }
      return pendingReceipt
    })

    await expect(
      (presentationsMethods as any).getPresentationGenerationStatus.call(
        client,
        pendingReceipt.generation_id
      )
    ).resolves.toEqual({
      receipt: pendingReceipt,
      retryAfterMs: 60_000
    })
    expect((client.request as any).mock.calls[0][0]).toEqual({
      path: `/api/v1/slides/generations/${pendingReceipt.generation_id}`,
      method: "GET",
      returnResponse: true
    })

    await expect(
      (presentationsMethods as any).getPresentationGeneration.call(
        client,
        pendingReceipt.generation_id
      )
    ).resolves.toEqual(pendingReceipt)
  })

  it("fails closed for unknown or widened generation receipt shapes", async () => {
    const client = createCore(async () => ({
      ...pendingReceipt,
      status: "paused",
      html_document: "must not be accepted"
    }))

    await expect(
      (presentationsMethods as any).getPresentationGeneration.call(
        client,
        pendingReceipt.generation_id
      )
    ).rejects.toThrow("Invalid Slides generation response")
  })

  it("adds content-kind negotiation to every applicable existing presentation request", async () => {
    const client = createCore(async (request) => {
      if (request.returnResponse) {
        return responseEnvelope(
          request.responseType === "arrayBuffer" ? new Uint8Array([1]).buffer : structuredDetail,
          {
            headers:
              request.responseType === "arrayBuffer"
                ? {
                    "content-type": "application/octet-stream",
                    "content-disposition": FIXED_DISPOSITION
                  }
                : { etag: 'W/"v3"' }
          }
        )
      }
      if (request.path.includes("render-jobs")) {
        return { job_id: 1, status: "queued", job_type: "presentation.render" }
      }
      if (request.path.includes("render-artifacts")) {
        return { presentation_id: "structured-1", artifacts: [] }
      }
      return structuredDetail
    })

    await (presentationsMethods as any).listPresentations.call(client)
    await presentationsMethods.getPresentation.call(client, "structured-1")
    await presentationsMethods.createPresentation.call(client, {
      title: "Deck",
      slides: []
    })
    await presentationsMethods.patchPresentation.call(client, "structured-1", {
      title: "Deck"
    })
    await presentationsMethods.submitPresentationRenderJob.call(
      client,
      "structured-1",
      { format: "mp4" },
      { ifMatch: 'W/"v3"' }
    )
    await presentationsMethods.getPresentationRenderJob.call(client, 1)
    await presentationsMethods.listPresentationRenderArtifacts.call(client, "structured-1")
    await presentationsMethods.exportPresentation.call(client, "structured-1", "pdf")

    for (const [request] of (client.request as any).mock.calls) {
      expect(requestHeaders(request)[ACCEPT_HEADER]).toBe(ACCEPTED_CONTENT_KINDS)
    }
  })

  it("saves validated standalone source as an exact raw string and adopts only the response ETag", async () => {
    const source = "<!doctype html>\n<title>Café 😀</title>"
    const client = createCore(async () =>
      responseEnvelope(
        { ...standaloneDetail, html_document: source, version: 8 },
        { headers: { etag: '"opaque-strong-tag"' } }
      )
    )

    const result = await (presentationsMethods as any).saveStandaloneHtmlSource.call(
      client,
      "html-1",
      source,
      {
        ifMatch: '"v7"'
      }
    )

    expect(result.etag).toBe('"opaque-strong-tag"')
    expect(result.record.html_document).toBe(source)
    expect((client.request as any).mock.calls[0][0]).toEqual({
      path: "/api/v1/slides/presentations/html-1/html-source",
      method: "PUT",
      headers: {
        "Content-Type": "application/octet-stream",
        "If-Match": '"v7"',
        [ACCEPT_HEADER]: ACCEPTED_CONTENT_KINDS
      },
      body: source,
      returnResponse: true
    })
  })

  it("rejects a structured raw-save response and keeps the return type standalone-specific", async () => {
    const invalidClient = createCore(async () =>
      responseEnvelope(structuredDetail, { headers: { etag: 'W/"v3"' } })
    )

    await expect(
      presentationsMethods.saveStandaloneHtmlSource.call(
        invalidClient,
        "html-1",
        "<!doctype html><title>Private source</title>",
        { ifMatch: '"v7"' }
      )
    ).rejects.toThrow("Standalone presentation required")

    const validClient = createCore(async () =>
      responseEnvelope(standaloneDetail, { headers: { etag: '"v7"' } })
    )
    const result = await presentationsMethods.saveStandaloneHtmlSource.call(
      validClient,
      "html-1",
      standaloneDetail.html_document,
      { ifMatch: '"v7"' }
    )

    const standaloneRecord: StandaloneHtmlPresentationStudioRecord = result.record
    expect(standaloneRecord.content_kind).toBe("standalone_html")
  })

  it.each([
    ["NUL", "<!doctype html>\u0000<title>bad</title>"],
    ["lone high surrogate", "<!doctype html><title>\ud800</title>"],
    ["lone low surrogate", "<!doctype html><title>\udc00</title>"],
    ["more than 1 MiB of UTF-8", "😀".repeat(262_145)]
  ])("rejects %s standalone source before dispatch", async (_case, source) => {
    const client = createCore(async () => responseEnvelope(standaloneDetail))

    await expect(
      (presentationsMethods as any).saveStandaloneHtmlSource.call(client, "html-1", source, {
        ifMatch: '"v7"'
      })
    ).rejects.toThrow(/^Standalone HTML source /)
    expect(client.request).not.toHaveBeenCalled()
  })

  it("returns exact saved and draft attachment bytes after strict response validation", async () => {
    const savedBytes = Uint8Array.from([0, 255, 10, 42])
    const draftBytes = Uint8Array.from([9, 8, 7, 6])
    const request = vi
      .fn()
      .mockResolvedValueOnce(
        responseEnvelope(savedBytes.buffer, {
          headers: {
            "content-type": "application/octet-stream",
            "content-disposition": FIXED_DISPOSITION
          }
        })
      )
      .mockResolvedValueOnce(
        responseEnvelope(draftBytes, {
          headers: {
            "content-type": "application/octet-stream",
            "content-disposition": FIXED_DISPOSITION
          }
        })
      )
    const client = createCore(request)

    const saved = await (presentationsMethods as any).downloadStandaloneHtmlPresentation.call(
      client,
      "html-1"
    )
    const draft = await (presentationsMethods as any).downloadStandaloneHtmlDraft.call(
      client,
      "html-1",
      "<!doctype html><title>Draft</title>"
    )

    expect(saved).toEqual(savedBytes)
    expect(draft).toEqual(draftBytes)
    expect((client.request as any).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        path: "/api/v1/slides/presentations/html-1/export?format=html",
        method: "GET",
        headers: {
          Accept: "application/octet-stream",
          [ACCEPT_HEADER]: ACCEPTED_CONTENT_KINDS
        },
        responseType: "arrayBuffer",
        returnResponse: true
      })
    )
    expect((client.request as any).mock.calls[1][0]).toEqual(
      expect.objectContaining({
        path: "/api/v1/slides/presentations/html-1/draft-attachment",
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          Accept: "application/octet-stream",
          [ACCEPT_HEADER]: ACCEPTED_CONTENT_KINDS
        },
        body: "<!doctype html><title>Draft</title>"
      })
    )
  })

  it.each([
    ["wrong status", responseEnvelope(new ArrayBuffer(0), { status: 202 })],
    [
      "wrong MIME",
      responseEnvelope(new ArrayBuffer(0), {
        headers: {
          "content-type": "text/html",
          "content-disposition": FIXED_DISPOSITION
        }
      })
    ],
    [
      "wrong disposition",
      responseEnvelope(new ArrayBuffer(0), {
        headers: {
          "content-type": "application/octet-stream",
          "content-disposition": 'attachment; filename="model-title.html"'
        }
      })
    ],
    [
      "non-byte body",
      responseEnvelope("<!doctype html>", {
        headers: {
          "content-type": "application/octet-stream",
          "content-disposition": FIXED_DISPOSITION
        }
      })
    ]
  ])("rejects a saved attachment with %s", async (_case, response) => {
    const client = createCore(async () => response)

    await expect(
      (presentationsMethods as any).downloadStandaloneHtmlPresentation.call(client, "html-1")
    ).rejects.toThrow("Invalid standalone HTML attachment response")
  })
})

describe("raw request retry semantics", () => {
  it("replays the exact standalone string and negotiation headers after token refresh", async () => {
    const source = "<!doctype html>\n<title>Café 😀</title>"
    let config = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "old-token",
      refreshToken: "refresh-token"
    }
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "expired" }), {
          status: 401,
          headers: { "content-type": "application/json" }
        })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      )

    const response = await tldwRequest(
      {
        path: "/api/v1/slides/presentations/html-1/html-source",
        method: "PUT",
        headers: {
          "Content-Type": "application/octet-stream",
          "If-Match": '"v7"',
          [ACCEPT_HEADER]: ACCEPTED_CONTENT_KINDS
        },
        body: source
      },
      {
        getConfig: async () => config,
        refreshAuth: async () => {
          config = { ...config, accessToken: "new-token" }
        },
        fetchFn
      }
    )

    expect(response.ok).toBe(true)
    expect(fetchFn).toHaveBeenCalledTimes(2)
    for (const [, init] of fetchFn.mock.calls) {
      expect(init.body).toBe(source)
      expect((init.headers as Record<string, string>)[ACCEPT_HEADER]).toBe(ACCEPTED_CONTENT_KINDS)
      expect((init.headers as Record<string, string>)["If-Match"]).toBe('"v7"')
    }
    expect((fetchFn.mock.calls[1][1].headers as Record<string, string>).Authorization).toBe(
      "Bearer new-token"
    )
  })

  it("continues to serialize an existing JSON request identically on refresh replay", async () => {
    const body = { title: "Deck", slides: [] }
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      )

    await tldwRequest(
      {
        path: "/api/v1/slides/presentations",
        method: "POST",
        body
      },
      {
        getConfig: async () => ({
          serverUrl: "http://127.0.0.1:8000",
          authMode: "multi-user",
          accessToken: "token",
          refreshToken: "refresh-token"
        }),
        refreshAuth: async () => undefined,
        fetchFn
      }
    )

    expect(fetchFn.mock.calls[0][1].body).toBe('{"title":"Deck","slides":[]}')
    expect(fetchFn.mock.calls[1][1].body).toBe('{"title":"Deck","slides":[]}')
  })
})
