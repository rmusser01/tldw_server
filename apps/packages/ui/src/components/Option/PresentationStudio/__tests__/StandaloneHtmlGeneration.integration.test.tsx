import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { presentationsMethods, type TldwApiClientCore } from "@/services/tldw/domains/presentations"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  submit: vi.fn(),
  status: vi.fn(),
  statusEnvelope: vi.fn()
}))

const statusCore: TldwApiClientCore = {
  ensureConfigForRequest: vi.fn(async () => ({})),
  request: vi.fn((...args: unknown[]) => mocks.statusEnvelope(...args)) as TldwApiClientCore["request"],
  resolveApiPath: vi.fn(async (_key: string, candidates: string[]) => candidates[0]),
  fillPathParams: vi.fn((template: string) => template)
}

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    submitPresentationGeneration: (...args: unknown[]) => mocks.submit(...args),
    getPresentationGenerationStatus: (...args: unknown[]) => mocks.status(...args)
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: (...args: unknown[]) => mocks.getCurrentUser(...args) }
}))

const revision = `sha256:${"b".repeat(64)}`
const capabilities = {
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
        max_slides: 5,
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
      provider: "canonical-provider",
      model: "allowed-model",
      adapter_id: "built-in-adapter",
      endpoint_identity: "https://provider.example/v1",
      generation_config_revision: revision,
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
      output_limits: { max_provider_response_bytes: 8_388_608, max_document_bytes: 1_048_576 }
    }
  }
} as const

const pendingReceipt = {
  generation_id: "generation-1",
  status: "queued",
  status_url: "/api/v1/slides/generations/generation-1",
  presentation_id: null
}

const loadSubject = () =>
  vi.importActual<typeof import("../StandaloneHtmlGenerationForm")>(
    ["..", "StandaloneHtmlGenerationForm"].join("/")
  )

const renderReadyForm = async () => {
  const { StandaloneHtmlGenerationForm } = await loadSubject()
  render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)
  await waitFor(() => expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeEnabled())
  fireEvent.change(screen.getByLabelText("Subject and material"), { target: { value: "Committed source" } })
  fireEvent.change(screen.getByLabelText("Audience"), { target: { value: "Engineers" } })
}

describe("Standalone HTML form and client integration", () => {
  beforeEach(() => {
    mocks.getConfig.mockReset()
    mocks.getCurrentUser.mockReset()
    mocks.submit.mockReset()
    mocks.status.mockReset()
    mocks.statusEnvelope.mockReset()
    sessionStorage.clear()
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://tldw.example/base" })
    mocks.getCurrentUser.mockResolvedValue({ id: 42 })
    mocks.status.mockImplementation((generationId: string) =>
      presentationsMethods.getPresentationGenerationStatus.call(statusCore, generationId)
    )
  })

  it("commits the immutable snapshot and disabled controls to the DOM before POST starts", async () => {
    let committedAtInvocation = false
    mocks.submit.mockImplementation(async () => {
      committedAtInvocation = Boolean(
        screen.queryByRole("heading", { name: "Submitted request" }) &&
        (screen.getByLabelText("Subject and material") as HTMLTextAreaElement).disabled &&
        (screen.getByLabelText("Audience") as HTMLInputElement).disabled
      )
      return {
        ...pendingReceipt,
        status: "failed",
        error_code: "provider_failed",
        error_message: "Provider failed"
      }
    })
    await renderReadyForm()

    screen.getByRole("button", { name: "Generate standalone presentation" })
      .closest("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }))
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    expect(committedAtInvocation).toBe(true)
  })

  it.each([
    [401, "Sign in required"],
    [404, "Generation not found"],
    [429, "Status checks paused"],
    [503, "Status unavailable"]
  ])("preserves real client-envelope HTTP %s semantics in the form", async (status, label) => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.statusEnvelope.mockResolvedValue({
      ok: false,
      status,
      data: { detail: { error_code: "generation_status_unavailable", source: "PRIVATE SOURCE" } },
      headers: { "retry-after": "90" },
      retryAfterMs: 90_000
    })
    await renderReadyForm()

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    expect(await screen.findByText(label)).toBeVisible()
    expect(screen.queryByText(/PRIVATE SOURCE/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Resume" })).toBeVisible()
  })

  it("handles a real normalized completed receipt without a binding", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.statusEnvelope.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        ...pendingReceipt,
        status: "completed",
        presentation_id: null,
        content_kind: "standalone_html"
      },
      headers: {},
      retryAfterMs: null
    })
    await renderReadyForm()

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    expect(await screen.findByText("generation_completed_without_presentation")).toBeVisible()
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible()
  })

  it("clears stale progress when a terminal receipt omits progress", async () => {
    mocks.submit.mockResolvedValue({
      ...pendingReceipt,
      status: "running",
      progress_text: "Validating generated document"
    })
    mocks.statusEnvelope.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        ...pendingReceipt,
        status: "failed",
        error_code: "provider_failed",
        error_message: "Provider failed"
      },
      headers: {},
      retryAfterMs: null
    })
    await renderReadyForm()

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    expect(await screen.findByText("Failed")).toBeVisible()
    expect(screen.queryByText("Validating generated document")).not.toBeInTheDocument()
  })

  it("uses the effective five-slide cap in the rendered control", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.statusEnvelope.mockReturnValue(new Promise(() => undefined))
    await renderReadyForm()

    const slideCount = screen.getByLabelText("Approximate slide count")
    expect(slideCount).toHaveAttribute("max", "5")
    fireEvent.change(slideCount, { target: { value: "6" } })
    expect(await screen.findByText("Slide count must be an integer from 1 to 5.")).toBeVisible()
  })
})
