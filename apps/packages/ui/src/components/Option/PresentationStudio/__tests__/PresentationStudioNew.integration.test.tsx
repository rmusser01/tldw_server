import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  getSlidesCapabilities: vi.fn(),
  submit: vi.fn(),
  status: vi.fn(),
  navigate: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return { ...actual, useNavigate: () => mocks.navigate }
})

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    getSlidesCapabilities: (...args: unknown[]) => mocks.getSlidesCapabilities(...args),
    submitPresentationGeneration: (...args: unknown[]) => mocks.submit(...args),
    getPresentationGenerationStatus: (...args: unknown[]) => mocks.status(...args)
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: (...args: unknown[]) => mocks.getCurrentUser(...args) }
}))

vi.mock("../PresentationStudioPage", () => ({
  PresentationStudioPage: () => <h2>Structured presentation setup</h2>
}))

const makeCapabilities = (revisionCharacter: string, provider: string, model: string) => ({
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
        max_slides: 12,
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
      provider,
      model,
      adapter_id: "openai-responses",
      endpoint_identity: "https://provider.example/v1",
      generation_config_revision: `sha256:${revisionCharacter.repeat(64)}`,
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
})

const firstCapabilities = makeCapabilities("a", "old-provider", "old-model")
const refreshedCapabilities = makeCapabilities("d", "new-provider", "new-model")
const disabledCapabilities = {
  ...firstCapabilities,
  generation_modes: {
    ...firstCapabilities.generation_modes,
    standalone_html: {
      ...firstCapabilities.generation_modes.standalone_html,
      enabled: false,
      reason: "feature_disabled",
      provider: null,
      model: null,
      adapter_id: null,
      endpoint_identity: null,
      generation_config_revision: null
    }
  }
}

const oldDraftKey = "tldw:presentation-studio:html:draft:v1:https%3A%2F%2Ftldw.example:42"

const storeDraft = (source: string) => {
  sessionStorage.setItem(oldDraftKey, JSON.stringify({
    schemaVersion: 1,
    timestamp: Date.now(),
    values: {
      source,
      presentationType: "tech-sharing",
      audience: "Engineers",
      slideCount: 8,
      visualDirection: "dark-technical",
      deliveryStyle: "speaker-led"
    },
    generationConfigRevision: `sha256:${"a".repeat(64)}`
  }))
}

const loadSubject = () =>
  vi.importActual<typeof import("../PresentationStudioNew")>(
    ["..", "PresentationStudioNew"].join("/")
  )

describe("PresentationStudioNew authority refresh integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://tldw.example/base" })
    mocks.getCurrentUser.mockResolvedValue({ id: 42 })
    mocks.status.mockReturnValue(new Promise(() => undefined))
  })

  afterEach(() => vi.restoreAllMocks())

  it("keeps the quota-failed form mounted across 409 refresh and requires the fresh revision before resubmit", async () => {
    let resolveRefresh: ((value: unknown) => void) | undefined
    mocks.getSlidesCapabilities
      .mockResolvedValueOnce(firstCapabilities)
      .mockReturnValueOnce(new Promise((resolve) => { resolveRefresh = resolve }))
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("changed"), {
        status: 409,
        details: { error_code: "generation_configuration_changed" }
      }))
      .mockResolvedValueOnce({
        generation_id: "generation-new-revision",
        status: "failed",
        status_url: "/api/v1/slides/generations/generation-new-revision",
        presentation_id: null,
        error_code: "provider_failed",
        error_message: "Provider failed"
      })
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    const htmlOption = await screen.findByRole("radio", { name: /Standalone HTML/ })
    await waitFor(() => expect(htmlOption).toBeEnabled())
    fireEvent.click(htmlOption)
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeEnabled())

    const setSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "setItem")
      .mockImplementation(() => { throw new DOMException("quota", "QuotaExceededError") })
    fireEvent.change(screen.getByLabelText("Subject and material"), {
      target: { value: "Keep this source in mounted form memory" }
    })
    fireEvent.change(screen.getByLabelText("Audience"), { target: { value: "Engineers" } })
    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))

    expect(await screen.findByText("generation_configuration_changed")).toBeVisible()
    await waitFor(() => expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(2))
    expect(screen.getByDisplayValue("Keep this source in mounted form memory")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeDisabled()
    expect(screen.getByText(/Refreshing generation capabilities/i)).toBeVisible()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    await act(async () => resolveRefresh?.(refreshedCapabilities))
    expect(await screen.findByText("new-provider")).toBeVisible()
    expect(screen.getByText("new-model")).toBeVisible()
    expect(screen.getByText(`sha256:${"d".repeat(64)}`)).toBeVisible()
    expect(screen.getByDisplayValue("Keep this source in mounted form memory")).toBeEnabled()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(2))
    expect(mocks.submit.mock.calls[1][0]).toEqual(expect.objectContaining({
      generation_config_revision: `sha256:${"d".repeat(64)}`,
      source: { kind: "prompt", prompt: "Keep this source in mounted form memory" }
    }))
    setSpy.mockRestore()
  })

  it("keeps quota-only source mounted through a failed 409 network refresh and later Retry", async () => {
    mocks.getSlidesCapabilities
      .mockResolvedValueOnce(firstCapabilities)
      .mockRejectedValueOnce(new Error("refresh network failure"))
      .mockResolvedValue(refreshedCapabilities)
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("changed"), {
        status: 409,
        details: { error_code: "generation_configuration_changed" }
      }))
      .mockResolvedValueOnce({
        generation_id: "generation-after-network-retry",
        status: "failed",
        status_url: "/api/v1/slides/generations/generation-after-network-retry",
        presentation_id: null,
        error_code: "provider_failed",
        error_message: "Provider failed"
      })
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    const htmlOption = await screen.findByRole("radio", { name: /Standalone HTML/ })
    await waitFor(() => expect(htmlOption).toBeEnabled())
    fireEvent.click(htmlOption)
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeEnabled())
    const setSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "setItem")
      .mockImplementation(() => { throw new DOMException("quota", "QuotaExceededError") })
    fireEvent.change(screen.getByLabelText("Subject and material"), { target: { value: "Quota-only network source" } })
    fireEvent.change(screen.getByLabelText("Audience"), { target: { value: "Engineers" } })
    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))

    expect(await screen.findByText("Generation capabilities could not refresh")).toBeVisible()
    expect(screen.getByDisplayValue("Quota-only network source")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeDisabled()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("new-provider")).toBeVisible()
    expect(screen.getByDisplayValue("Quota-only network source")).toBeEnabled()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(2))
    expect(mocks.submit.mock.calls[1][0]).toEqual(expect.objectContaining({
      generation_config_revision: `sha256:${"d".repeat(64)}`,
      source: { kind: "prompt", prompt: "Quota-only network source" }
    }))
    setSpy.mockRestore()
  })

  it("keeps quota-only source mounted when 409 refresh cannot confirm its scope", async () => {
    let resolveUnconfirmedRefresh: ((value: unknown) => void) | undefined
    mocks.getSlidesCapabilities
      .mockResolvedValueOnce(firstCapabilities)
      .mockReturnValueOnce(new Promise((resolve) => { resolveUnconfirmedRefresh = resolve }))
      .mockResolvedValue(refreshedCapabilities)
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("changed"), {
        status: 409,
        details: { error_code: "generation_configuration_changed" }
      }))
      .mockResolvedValueOnce({
        generation_id: "generation-after-scope-retry",
        status: "failed",
        status_url: "/api/v1/slides/generations/generation-after-scope-retry",
        presentation_id: null,
        error_code: "provider_failed",
        error_message: "Provider failed"
      })
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    const htmlOption = await screen.findByRole("radio", { name: /Standalone HTML/ })
    await waitFor(() => expect(htmlOption).toBeEnabled())
    fireEvent.click(htmlOption)
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeEnabled())
    const setSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "setItem")
      .mockImplementation(() => { throw new DOMException("quota", "QuotaExceededError") })
    fireEvent.change(screen.getByLabelText("Subject and material"), { target: { value: "Quota-only unconfirmed source" } })
    fireEvent.change(screen.getByLabelText("Audience"), { target: { value: "Engineers" } })
    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    await waitFor(() => expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(2))

    mocks.getConfig.mockRejectedValueOnce(new Error("confirmation unavailable"))
    await act(async () => resolveUnconfirmedRefresh?.(refreshedCapabilities))
    expect(await screen.findByText("Generation capabilities could not refresh")).toBeVisible()
    expect(screen.getByDisplayValue("Quota-only unconfirmed source")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Generate standalone presentation" })).toBeDisabled()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("new-provider")).toBeVisible()
    expect(screen.getByDisplayValue("Quota-only unconfirmed source")).toBeEnabled()
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(2))
    expect(mocks.submit.mock.calls[1][0]).toEqual(expect.objectContaining({
      generation_config_revision: `sha256:${"d".repeat(64)}`,
      source: { kind: "prompt", prompt: "Quota-only unconfirmed source" }
    }))
    setSpy.mockRestore()
  })

  it("scrubs hydrated old-scope source when deferred capability confirmation proves a mismatch", async () => {
    storeDraft("Hydrated old-scope source")
    let resolveOldCapability: ((value: unknown) => void) | undefined
    mocks.getSlidesCapabilities.mockReturnValue(new Promise((resolve) => { resolveOldCapability = resolve }))
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    expect(await screen.findByDisplayValue("Hydrated old-scope source")).toBeDisabled()
    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://other.example/base" })
    mocks.getCurrentUser.mockResolvedValueOnce({ id: 77 })
    await act(async () => resolveOldCapability?.(firstCapabilities))

    await waitFor(() => expect(sessionStorage.getItem(oldDraftKey)).toBeNull())
    expect(screen.queryByDisplayValue("Hydrated old-scope source")).not.toBeInTheDocument()
    expect(screen.queryByText("Hydrated old-scope source")).not.toBeInTheDocument()
    expect(screen.getByRole("radio", { name: /Standalone HTML/ })).toBeDisabled()
    expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(1)
  })

  it("offers combined Retry for recovery-present capability failure", async () => {
    storeDraft("Recovery source during capability error")
    mocks.getSlidesCapabilities
      .mockRejectedValueOnce(new Error("capability unavailable"))
      .mockResolvedValue(refreshedCapabilities)
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    expect(await screen.findByDisplayValue("Recovery source during capability error")).toBeDisabled()
    expect(screen.getByText("Generation capabilities could not load")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    expect(await screen.findByText("new-provider")).toBeVisible()
    expect(screen.getByDisplayValue("Recovery source during capability error")).toBeEnabled()
  })

  it("rechecks source-free recovery when generation is disabled and the first probe failed", async () => {
    storeDraft("Draft hidden by first probe outage")
    const keySpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "key")
      .mockImplementation(() => { throw new DOMException("storage unavailable", "SecurityError") })
    mocks.getSlidesCapabilities.mockResolvedValue(disabledCapabilities)
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    const htmlOption = await screen.findByRole("radio", { name: /Standalone HTML/ })
    await waitFor(() => expect(htmlOption).toBeEnabled())
    fireEvent.click(htmlOption)
    expect(await screen.findByText("Standalone generation is disabled")).toBeVisible()
    keySpy.mockRestore()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => expect(mocks.getSlidesCapabilities).toHaveBeenCalledTimes(2))
    expect(await screen.findByText("Standalone generation is disabled")).toBeVisible()
    expect(screen.getByDisplayValue("Draft hidden by first probe outage")).toBeDisabled()
    expect(screen.getByText("Preserved draft")).toBeVisible()
  })
})
