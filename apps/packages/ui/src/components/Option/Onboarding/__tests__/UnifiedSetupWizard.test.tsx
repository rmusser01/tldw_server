// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const setupHookMocks = vi.hoisted(() => ({
  saveStep: vi.fn(),
  skip: vi.fn(),
  loadProviderCatalog: vi.fn(),
  loadAudioRecommendations: vi.fn(),
  saveProvider: vi.fn(),
  saveIngestDefaults: vi.fn(),
  saveAudioDefaults: vi.fn(),
  saveOptionalAdvanced: vi.fn(),
  verifyFirstChat: vi.fn(),
  complete: vi.fn(),
  refresh: vi.fn()
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://127.0.0.1:3000",
        api_origin: "http://127.0.0.1:8000",
        browser_access: "local"
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    providerCatalog: [
      {
        provider_key: "openai",
        label: "OpenAI",
        provider_type: "hosted_api_key",
        supports_preflight: true,
        recommended_for_first_chat: true
      }
    ],
    audioRecommendations: [],
    loading: false,
    error: null,
    refresh: setupHookMocks.refresh,
    loadProviderCatalog: setupHookMocks.loadProviderCatalog,
    loadAudioRecommendations: setupHookMocks.loadAudioRecommendations,
    saveStep: setupHookMocks.saveStep,
    skip: setupHookMocks.skip,
    saveProvider: setupHookMocks.saveProvider,
    saveIngestDefaults: setupHookMocks.saveIngestDefaults,
    saveAudioDefaults: setupHookMocks.saveAudioDefaults,
    saveOptionalAdvanced: setupHookMocks.saveOptionalAdvanced,
    verifyFirstChat: setupHookMocks.verifyFirstChat,
    complete: setupHookMocks.complete
  })
}))

describe("UnifiedSetupWizard", () => {
  beforeEach(() => {
    setupHookMocks.saveStep.mockReset()
    setupHookMocks.skip.mockReset()
    setupHookMocks.loadProviderCatalog.mockReset().mockResolvedValue([])
    setupHookMocks.loadAudioRecommendations.mockReset().mockResolvedValue([])
    setupHookMocks.saveProvider.mockReset().mockResolvedValue({
      provider_key: "openai",
      status: "saved"
    })
    setupHookMocks.saveIngestDefaults.mockReset().mockResolvedValue({
      status: "saved",
      step: "ingest_defaults",
      requires_restart: false
    })
    setupHookMocks.saveAudioDefaults.mockReset().mockResolvedValue({
      status: "saved",
      step: "audio_defaults",
      requires_restart: false
    })
    setupHookMocks.saveOptionalAdvanced.mockReset().mockResolvedValue({
      status: "saved",
      step: "optional_advanced",
      requires_restart: false
    })
    setupHookMocks.verifyFirstChat.mockReset().mockResolvedValue({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello."
    })
    setupHookMocks.complete.mockReset().mockResolvedValue({
      success: true,
      message: "completed",
      requires_restart: false,
      install_plan_submitted: false
    })
    setupHookMocks.refresh.mockReset().mockResolvedValue({
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    })
    setupHookMocks.saveStep.mockResolvedValue({
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    })
  })

  it("renders a focused first-run heading and setup path choices", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)

    expect(
      screen.getByRole("heading", { name: /first-time setup/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /solo, docker/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /solo, local/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /multi-user/i })).toBeInTheDocument()
  })

  it("shows multi-user exit guidance instead of continuing solo wizard", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)
    fireEvent.click(screen.getByRole("button", { name: /multi-user/i }))

    expect(screen.getByText(/multi-user setup guide/i)).toBeInTheDocument()
  })

  it("requires privacy and security acknowledgement before provider setup", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }))

    expect(
      await screen.findByRole("heading", { name: /privacy and security/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled()

    fireEvent.click(screen.getByLabelText(/i understand/i))
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled()
  })

  it("does not advance past setup path if progress cannot be saved", async () => {
    setupHookMocks.saveStep.mockRejectedValueOnce(new Error("save failed"))
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }))

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not be saved/i)
    })
    expect(
      screen.getByRole("heading", { name: /choose your setup path/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: /privacy and security/i })
    ).not.toBeInTheDocument()
  })

  it("does not advance to provider setup if privacy acknowledgement cannot be saved", async () => {
    setupHookMocks.saveStep
      .mockResolvedValueOnce({
        status: "in_progress",
        completed_steps: ["setup_path"],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: ["setup_path"],
        first_chat: { completed: false }
      })
      .mockRejectedValueOnce(new Error("save failed"))
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }))

    await screen.findByRole("heading", { name: /privacy and security/i })
    fireEvent.click(screen.getByLabelText(/i understand/i))
    fireEvent.click(screen.getByRole("button", { name: /continue/i }))

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not be saved/i)
    })
    expect(
      screen.getByRole("heading", { name: /privacy and security/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: /chat provider/i })
    ).not.toBeInTheDocument()
  })

  it("reports skipped state to the parent route resolver", async () => {
    setupHookMocks.skip.mockResolvedValueOnce({
      status: "skipped",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false },
      skip_reason: "user_skip"
    })
    const onStateChange = vi.fn()
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard onStateChange={onStateChange} />)
    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }))

    await waitFor(() => {
      expect(onStateChange).toHaveBeenCalledWith(
        expect.objectContaining({ status: "skipped" })
      )
    })
  })

  it("resumes at first chat when backend state includes the saved provider model", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: [
            "setup_path",
            "privacy_security",
            "providers",
            "ingest_defaults",
            "audio_defaults",
            "optional_advanced"
          ],
          skipped_steps: [],
          step_data: {
            providers: {
              acknowledged: true,
              default_provider: "openai",
              default_model: "gpt-4.1-mini"
            }
          },
          acknowledged_steps: [],
          first_chat: { completed: false }
        }}
      />
    )

    expect(
      screen.getByRole("heading", { name: /first chat/i })
    ).toBeInTheDocument()
    expect(screen.getByText(/openai \/ gpt-4.1-mini/i)).toBeInTheDocument()
  })
})
