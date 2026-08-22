import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  hook: null as any
}))

vi.mock("@/hooks/useStandaloneHtmlGeneration", () => ({
  useStandaloneHtmlGeneration: () => mocks.hook
}))

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
      limits: { max_document_bytes: 1_048_576, max_source_write_bytes: 1_048_576, max_draft_attachment_bytes: 1_048_576, max_slides: 30, max_nesting_depth: 128 }
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
      generation_config_revision: `sha256:${"b".repeat(64)}`,
      input_limits: { max_request_bytes: 4_194_304, max_source_chars: 200_000, max_source_tokens: 50_000, max_audience_chars: 500, max_source_identifier_bytes: 256, max_note_ids: 100, max_rag_query_chars: 20_000, max_rag_top_k: 100 },
      output_limits: { max_provider_response_bytes: 8_388_608, max_document_bytes: 1_048_576 }
    }
  }
} as const

const request = {
  generation_mode: "standalone_html",
  generation_config_revision: `sha256:${"b".repeat(64)}`,
  source: { kind: "prompt", prompt: "Explain the trust boundary" },
  html_options: {
    presentation_type: "tech-sharing",
    audience: "Security engineers",
    slide_count: 8,
    visual_direction: "dark-technical",
    delivery_style: "speaker-led"
  }
}

const baseHook = () => ({
  scopeReady: true,
  draft: {
    source: "",
    presentationType: "tech-sharing",
    audience: "",
    slideCount: 10,
    visualDirection: "auto",
    deliveryStyle: "speaker-led"
  },
  fieldErrors: {},
  editError: null,
  phase: "idle",
  locked: false,
  snapshot: null,
  backendStatus: null,
  progressText: null,
  safeError: null,
  recoveryAvailable: false,
  draftRecoveryAvailable: false,
  storageWarning: null,
  updateField: vi.fn(),
  submit: vi.fn(),
  resume: vi.fn(),
  stopWaiting: vi.fn(),
  forget: vi.fn(),
  startDifferent: vi.fn(),
  tryAgain: vi.fn()
})

const loadSubject = () =>
  vi.importActual<typeof import("../StandaloneHtmlGenerationForm")>(
    ["..", "StandaloneHtmlGenerationForm"].join("/")
  )

describe("StandaloneHtmlGenerationForm", () => {
  beforeEach(() => {
    mocks.hook = baseHook()
  })

  it("renders the closed direct-material form, security copy, and configured target metadata", async () => {
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    const source = screen.getByLabelText("Subject and material")
    const audience = screen.getByLabelText("Audience")
    for (const control of [source, audience]) {
      expect(control).toHaveAttribute("spellcheck", "false")
      expect(control).toHaveAttribute("autocorrect", "off")
      expect(control).toHaveAttribute("autocapitalize", "off")
      expect(control).toHaveAttribute("autocomplete", "off")
      expect(control).toHaveAttribute("data-1p-ignore", "true")
      expect(control).not.toHaveAttribute("name")
    }

    expect(screen.getByText(/text-only outline/)).toBeVisible()
    expect(screen.getByText(/runs only if you download and open the file outside tldw/)).toBeVisible()
    expect(screen.getByText("canonical-provider")).toBeVisible()
    expect(screen.getByText("allowed-model")).toBeVisible()
    expect(screen.getByText("built-in-adapter")).toBeVisible()
    expect(screen.getByText("https://provider.example/v1")).toBeVisible()
    expect(screen.getByText(`sha256:${"b".repeat(64)}`)).toBeVisible()
    expect(screen.queryByRole("combobox", { name: /provider|model/i })).not.toBeInTheDocument()

    const presentationType = screen.getByLabelText("Presentation type")
    expect(within(presentationType).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual([
      "pitch-deck", "tech-sharing", "product-launch", "weekly-report", "course-module", "keynote", "data-report", "training", "social-media", "case-study", "comparison", "roadmap"
    ])
    const visualDirection = screen.getByLabelText("Visual direction")
    expect(within(visualDirection).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual([
      "auto", "dark-technical", "minimal-light", "editorial", "corporate", "soft-pastel", "bold-creative", "neo-brutalist"
    ])
    expect(screen.getByLabelText("Speaker-led")).toBeVisible()
    expect(screen.getByLabelText("Self-guided")).toBeVisible()
    expect(screen.getByText(/concise speaker notes/)).toBeVisible()
    expect(screen.getByText(/does not autoplay or auto-advance/)).toBeVisible()
  })

  it("maps edits and submission to the generation hook and exposes local validation", async () => {
    mocks.hook.fieldErrors = {
      source: "Subject and material is required.",
      audience: "Audience must not contain NUL characters.",
      slideCount: "Slide count must be an integer from 1 to 30."
    }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    fireEvent.change(screen.getByLabelText("Subject and material"), { target: { value: "new material" } })
    expect(mocks.hook.updateField).toHaveBeenCalledWith("source", "new material")
    expect(screen.getByText("Subject and material is required.")).toHaveAttribute("role", "alert")
    expect(screen.getByText("Audience must not contain NUL characters.")).toBeVisible()
    expect(screen.getByText("Slide count must be an integer from 1 to 30.")).toBeVisible()

    fireEvent.click(screen.getByRole("button", { name: "Generate standalone presentation" }))
    expect(mocks.hook.submit).toHaveBeenCalledTimes(1)
  })

  it("renders the immutable request before POST and locks every field", async () => {
    mocks.hook = { ...baseHook(), phase: "submitting", locked: true, snapshot: request }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    expect(screen.getByRole("heading", { name: "Submitted request" })).toBeVisible()
    expect(screen.getByText("Explain the trust boundary")).toBeVisible()
    expect(screen.getByText("Security engineers")).toBeVisible()
    const submittedRequest = screen.getByRole("heading", { name: "Submitted request" }).closest("section")!
    expect(within(submittedRequest).getByText(`sha256:${"b".repeat(64)}`)).toBeVisible()
    expect(screen.queryByText(/idempotency/i)).not.toBeInTheDocument()
    expect(screen.getByLabelText("Subject and material")).toBeDisabled()
    expect(screen.getByLabelText("Audience")).toBeDisabled()
    expect(screen.getByLabelText("Presentation type")).toBeDisabled()
    expect(screen.getByLabelText("Approximate slide count")).toBeDisabled()
    expect(screen.getByLabelText("Visual direction")).toBeDisabled()
    expect(screen.getByLabelText("Speaker-led")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Submitting request" })).toBeDisabled()
  })

  it("shows real backend state and recovery actions without inventing percentage progress", async () => {
    mocks.hook = { ...baseHook(), phase: "polling", locked: true, snapshot: request, backendStatus: "running", progressText: "Validating generated document" }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    expect(screen.getByText("Running")).toBeVisible()
    expect(screen.getByText("Validating generated document")).toBeVisible()
    expect(screen.queryByText(/%/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Stop waiting" }))
    expect(mocks.hook.stopWaiting).toHaveBeenCalledTimes(1)
  })

  it("requires inline confirmation before starting a different request and exposes terminal recovery", async () => {
    mocks.hook = { ...baseHook(), phase: "ambiguous", locked: true, snapshot: request, recoveryAvailable: true }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    const { rerender } = render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    fireEvent.click(screen.getByRole("button", { name: "Start a different request" }))
    expect(screen.getByText(/original request may still complete/)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Confirm different request" }))
    expect(mocks.hook.startDifferent).toHaveBeenCalledTimes(1)

    mocks.hook = { ...baseHook(), phase: "failed", locked: false, snapshot: request, safeError: "generation_quarantined" }
    rerender(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)
    expect(screen.getByText("Failed")).toBeVisible()
    expect(screen.getByText("generation_quarantined")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(mocks.hook.tryAgain).toHaveBeenCalledTimes(1)
  })

  it("warns that Forget is local only and reports unavailable recovery storage", async () => {
    mocks.hook = { ...baseHook(), phase: "stopped", locked: true, snapshot: request, recoveryAvailable: true, storageWarning: "Reload recovery is unavailable." }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={capabilities as any} />)

    expect(screen.getByText("Reload recovery is unavailable.")).toBeVisible()
    expect(screen.getByText(/does not cancel generation or delete a presentation/)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Resume" }))
    fireEvent.click(screen.getByRole("button", { name: "Forget this job; generation continues" }))
    expect(mocks.hook.resume).toHaveBeenCalledTimes(1)
    expect(mocks.hook.forget).toHaveBeenCalledTimes(1)
  })

  it("exposes preserved draft recovery and Forget without a generation receipt", async () => {
    mocks.hook = {
      ...baseHook(),
      draft: { ...baseHook().draft, source: "Preserved direct material", audience: "Engineers" },
      draftRecoveryAvailable: true
    }
    const { StandaloneHtmlGenerationForm } = await loadSubject()
    render(<StandaloneHtmlGenerationForm capabilities={null} recoveryOnly />)

    expect(screen.getByText("Preserved draft")).toBeVisible()
    expect(screen.getByDisplayValue("Preserved direct material")).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Forget preserved draft" }))
    expect(mocks.hook.forget).toHaveBeenCalledTimes(1)
  })
})
