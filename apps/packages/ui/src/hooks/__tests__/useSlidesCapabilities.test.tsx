import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getSlidesCapabilities: vi.fn(),
  online: true
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getSlidesCapabilities: (...args: unknown[]) => mocks.getSlidesCapabilities(...args)
  }
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.online
}))

const enabledCapabilities = {
  schema_version: 1,
  content_kind_request_header: "X-Slides-Accept-Content-Kinds",
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
    structured_slides: { enabled: true, transport: "existing_source_endpoints" },
    standalone_html: {
      enabled: true,
      reason: null,
      transport: "slides_generation_job",
      source_kinds: ["prompt", "chat", "media", "notes", "rag"],
      provider: "openai",
      model: "gpt-5-mini",
      adapter_id: "openai-responses",
      endpoint_identity: "https://api.openai.com/v1",
      generation_config_revision: `sha256:${"a".repeat(64)}`,
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
} as const

const loadSubject = () =>
  vi.importActual<typeof import("../useSlidesCapabilities")>(
    ["..", "useSlidesCapabilities"].join("/")
  )

describe("useSlidesCapabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.online = true
  })

  it("fetches the authoritative Slides capability on entry and explicit Retry", async () => {
    mocks.getSlidesCapabilities.mockResolvedValue(enabledCapabilities)
    const { useSlidesCapabilities } = await loadSubject()
    const { result } = renderHook(() => useSlidesCapabilities())

    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(1)
    expect(result.current.canGenerate).toBe(true)
    expect(result.current.capabilities).toEqual(enabledCapabilities)

    await act(async () => result.current.retry())
    expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(2)
  })

  it("keeps read and draft access available when generation is disabled", async () => {
    mocks.getSlidesCapabilities.mockResolvedValue({
      ...enabledCapabilities,
      generation_modes: {
        ...enabledCapabilities.generation_modes,
        standalone_html: {
          ...enabledCapabilities.generation_modes.standalone_html,
          enabled: false,
          reason: "feature_disabled",
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: null
        }
      }
    })
    const { useSlidesCapabilities } = await loadSubject()
    const { result } = renderHook(() => useSlidesCapabilities())

    await waitFor(() => expect(result.current.status).toBe("generation_disabled"))
    expect(result.current.reason).toBe("feature_disabled")
    expect(result.current.canGenerate).toBe(false)
    expect(result.current.canReadStandalone).toBe(true)
    expect(result.current.canDraftStandalone).toBe(true)
  })

  it("distinguishes validator-unavailable read/draft-only support", async () => {
    mocks.getSlidesCapabilities.mockResolvedValue({
      ...enabledCapabilities,
      content_kinds: {
        ...enabledCapabilities.content_kinds,
        standalone_html: {
          ...enabledCapabilities.content_kinds.standalone_html,
          edit: false,
          export_attachment: false,
          reason: "validator_unavailable"
        }
      },
      generation_modes: {
        ...enabledCapabilities.generation_modes,
        standalone_html: {
          ...enabledCapabilities.generation_modes.standalone_html,
          enabled: false,
          reason: "validator_unavailable",
          provider: null,
          model: null,
          adapter_id: null,
          endpoint_identity: null,
          generation_config_revision: null
        }
      }
    })
    const { useSlidesCapabilities } = await loadSubject()
    const { result } = renderHook(() => useSlidesCapabilities())

    await waitFor(() => expect(result.current.status).toBe("validator_unavailable"))
    expect(result.current.canReadStandalone).toBe(true)
    expect(result.current.canDraftStandalone).toBe(true)
    expect(result.current.canEditStandalone).toBe(false)
  })

  it("fails closed on malformed or unknown capability responses and recovers on Retry", async () => {
    mocks.getSlidesCapabilities
      .mockRejectedValueOnce(new Error("Invalid Slides capabilities response"))
      .mockResolvedValueOnce(enabledCapabilities)
    const { useSlidesCapabilities } = await loadSubject()
    const { result } = renderHook(() => useSlidesCapabilities())

    await waitFor(() => expect(result.current.status).toBe("error"))
    expect(result.current.canGenerate).toBe(false)
    expect(result.current.capabilities).toBeNull()

    await act(async () => result.current.retry())
    expect(result.current.status).toBe("ready")
    expect(result.current.canGenerate).toBe(true)
  })

  it("does not infer support while offline and fetches when connectivity returns", async () => {
    mocks.online = false
    mocks.getSlidesCapabilities.mockResolvedValue(enabledCapabilities)
    const { useSlidesCapabilities } = await loadSubject()
    const { result, rerender } = renderHook(() => useSlidesCapabilities())

    expect(result.current.status).toBe("offline")
    expect(result.current.canGenerate).toBe(false)
    expect(mocks.getSlidesCapabilities).not.toHaveBeenCalled()

    mocks.online = true
    rerender()
    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(1)
  })
})
