import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  slides: null as any,
  recovery: null as any,
  retry: vi.fn(),
  retryRecovery: vi.fn(),
  page: vi.fn(),
  form: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return { ...actual, useNavigate: () => mocks.navigate }
})

vi.mock("@/hooks/useSlidesCapabilities", () => ({
  useSlidesCapabilities: () => mocks.slides
}))

vi.mock("@/hooks/useStandaloneHtmlGeneration", () => ({
  useStandaloneHtmlRecoveryProbe: () => mocks.recovery
}))

vi.mock("../PresentationStudioPage", () => ({
  PresentationStudioPage: (props: any) => {
    mocks.page(props)
    return props.embedded ? <h2>Structured presentation setup</h2> : <h1>Presentation Studio</h1>
  }
}))

vi.mock("../StandaloneHtmlGenerationForm", () => ({
  StandaloneHtmlGenerationForm: (props: any) => {
    mocks.form(props)
    return <button type="button">Resume existing generation</button>
  }
}))

const enabledCapabilities = {
  content_kinds: {
    standalone_html: { read: true, edit: true, draft_attachment: true, limits: { max_slides: 30 } }
  },
  generation_modes: {
    standalone_html: {
      enabled: true,
      provider: "provider",
      model: "model",
      adapter_id: "adapter",
      endpoint_identity: "https://provider.example",
      generation_config_revision: `sha256:${"a".repeat(64)}`
    }
  }
}

const slidesState = (overrides: Record<string, unknown> = {}) => ({
  status: "ready",
  reason: null,
  canGenerate: true,
  capabilities: enabledCapabilities,
  retry: mocks.retry,
  ...overrides
})

const loadSubject = () =>
  vi.importActual<typeof import("../PresentationStudioNew")>(
    ["..", "PresentationStudioNew"].join("/")
  )

describe("PresentationStudioNew", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.slides = slidesState()
    mocks.recovery = { status: "none", retry: mocks.retryRecovery }
  })

  it("gates the HTML option until capability confirmation and exposes recovery actions for errors", async () => {
    mocks.slides = slidesState({ status: "loading", canGenerate: false, capabilities: null })
    const { PresentationStudioNew } = await loadSubject()
    const { rerender } = render(<PresentationStudioNew />)
    expect(screen.getByRole("radio", { name: /Standalone HTML/ })).toBeDisabled()

    mocks.slides = slidesState({ status: "error", canGenerate: false, capabilities: null })
    rerender(<PresentationStudioNew />)
    expect(screen.getByRole("radio", { name: /Standalone HTML/ })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Retry generation capabilities" }))
    expect(mocks.retry).toHaveBeenCalledTimes(1)
    expect(mocks.retryRecovery).toHaveBeenCalledTimes(1)
  })

  it.each([
    ["generation_disabled", "feature_disabled", "Standalone generation is disabled"],
    ["validator_unavailable", "validator_unavailable", "Standalone validation is unavailable"]
  ])("shows exact %s state with Retry after authoritative confirmation", async (status, reason, title) => {
    mocks.slides = slidesState({
      status,
      reason,
      canGenerate: false,
      capabilities: {
        ...enabledCapabilities,
        generation_modes: {
          standalone_html: {
            enabled: false,
            reason,
            provider: null,
            model: null,
            adapter_id: null,
            endpoint_identity: null,
            generation_config_revision: null
          }
        }
      }
    })
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    fireEvent.click(screen.getByRole("radio", { name: /Standalone HTML/ }))
    expect(screen.getByText(title)).toBeVisible()
    expect(screen.getByText(reason)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(mocks.retry).toHaveBeenCalledTimes(1)
    expect(mocks.retryRecovery).toHaveBeenCalledTimes(1)
  })

  it("auto-offers source-free recovery even when current capability is unavailable", async () => {
    mocks.slides = slidesState({ status: "error", canGenerate: false, capabilities: null })
    mocks.recovery = { status: "available", retry: mocks.retryRecovery }
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    expect(screen.getByRole("radio", { name: /Standalone HTML/ })).toBeChecked()
    expect(screen.getByRole("button", { name: "Resume existing generation" })).toBeVisible()
    expect(mocks.form).toHaveBeenCalledWith(expect.objectContaining({
      capabilities: null,
      recoveryOnly: true
    }))
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(mocks.retry).toHaveBeenCalledTimes(1)
    expect(mocks.retryRecovery).toHaveBeenCalledTimes(1)
  })

  it("uses one top-level heading and accurately describes downloaded execution", async () => {
    const { PresentationStudioNew } = await loadSubject()
    render(<PresentationStudioNew />)

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
    expect(mocks.page).toHaveBeenCalledWith(expect.objectContaining({ mode: "new", embedded: true }))
    expect(screen.getByText(/can run only after you download and open it outside tldw/i)).toBeVisible()
    expect(screen.queryByText(/non-executing file/i)).not.toBeInTheDocument()
  })
})
