// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const setupHookMocks = vi.hoisted(() => ({
  saveStep: vi.fn(),
  skip: vi.fn()
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
    loading: false,
    error: null,
    saveStep: setupHookMocks.saveStep,
    skip: setupHookMocks.skip
  })
}))

describe("UnifiedSetupWizard", () => {
  beforeEach(() => {
    setupHookMocks.saveStep.mockReset()
    setupHookMocks.skip.mockReset()
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
})
