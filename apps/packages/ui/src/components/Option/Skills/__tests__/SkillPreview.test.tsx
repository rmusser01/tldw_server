import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Modal } from "antd"
import { SkillPreview } from "../SkillPreview"
import type React from "react"

const tldwClientMock = vi.hoisted(() => ({
  executeSkill: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

const renderPreview = (props: Partial<React.ComponentProps<typeof SkillPreview>> = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <SkillPreview
        skillName={props.skillName ?? "summarize"}
        runtime={props.runtime}
        onClose={props.onClose ?? vi.fn()}
      />
    </QueryClientProvider>
  )
}

describe("SkillPreview test-run semantics", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        configurable: true,
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    if (typeof globalThis.ResizeObserver === "undefined") {
      globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      } as unknown as typeof ResizeObserver
    }

    tldwClientMock.executeSkill.mockResolvedValue({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: ["search"],
      model_override: null,
      execution_mode: "fork",
      fork_output: "Summary output",
      dry_run: false
    })
  })

  afterEach(() => {
    cleanup()
    Modal.destroyAll()
  })

  it("discloses execution risk before running a skill", () => {
    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    expect(within(dialog).getByRole("status")).toHaveTextContent("")
    expect(
      within(dialog).getByText(
        "This renders the skill with your arguments. Fork-mode skills may call the configured model and allowed tools."
      )
    ).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Render prompt only" })).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Run test" })).toBeInTheDocument()
    expect(within(dialog).queryByRole("button", { name: "Preview" })).not.toBeInTheDocument()
  })

  it("shows selected runtime impact before running a skill", () => {
    renderPreview({
      runtime: {
        execution_mode: "fork",
        test_run_may_call_model: true,
        declares_tools: true,
        declared_tool_count: 2,
        model_override: "gpt-4o",
        auto_invocation_enabled: false
      }
    })

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    expect(within(dialog).getByText("Runtime impact")).toBeInTheDocument()
    expect(within(dialog).getByText("Fork")).toBeInTheDocument()
    expect(within(dialog).getByText("Test may call model")).toBeInTheDocument()
    expect(within(dialog).getByText("2 tools declared")).toBeInTheDocument()
    expect(within(dialog).getByText("Model override")).toBeInTheDocument()
    expect(within(dialog).getByText("Auto invocation off")).toBeInTheDocument()
    expect(
      within(dialog).getByText(
        "Render prompt only does not invoke fork, model, or tool execution."
      )
    ).toBeInTheDocument()
  })

  it("keeps fork execution disclosure separate from model-call allowance", () => {
    renderPreview({
      runtime: {
        execution_mode: "fork",
        test_run_may_call_model: false,
        declares_tools: false,
        declared_tool_count: 0,
        model_override: null,
        auto_invocation_enabled: true
      }
    })

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    expect(within(dialog).getByText("Fork")).toBeInTheDocument()
    expect(within(dialog).getByText("Prompt only by default")).toBeInTheDocument()
    expect(
      within(dialog).getByText(
        "Run test uses fork execution for this skill; model calls are disabled."
      )
    ).toBeInTheDocument()
    expect(
      within(dialog).queryByText("Run test uses inline prompt execution for this skill.")
    ).not.toBeInTheDocument()
  })

  it("runs the skill with supplied arguments from the explicit test action", async () => {
    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    fireEvent.change(within(dialog).getByPlaceholderText("Enter test arguments..."), {
      target: { value: "chapter 1" }
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Run test" }))

    await waitFor(() => {
      expect(tldwClientMock.executeSkill).toHaveBeenCalledWith(
        "summarize",
        "chapter 1",
        { dryRun: false }
      )
    })
  })

  it("renders the prompt only without executing fork output", async () => {
    tldwClientMock.executeSkill.mockResolvedValueOnce({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: ["search"],
      model_override: null,
      execution_mode: "fork",
      fork_output: null,
      dry_run: true
    })

    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    fireEvent.change(within(dialog).getByPlaceholderText("Enter test arguments..."), {
      target: { value: "chapter 1" }
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Render prompt only" }))

    await waitFor(() => {
      expect(tldwClientMock.executeSkill).toHaveBeenCalledWith(
        "summarize",
        "chapter 1",
        { dryRun: true }
      )
    })

    expect(await within(dialog).findByText("Dry render")).toBeInTheDocument()
    expect(within(dialog).queryByText("Fork Output")).not.toBeInTheDocument()
  })

  it("prevents duplicate skill executions while a test run is pending", async () => {
    let resolveExecution: (value: {
      skill_name: string
      rendered_prompt: string
      allowed_tools: string[]
      model_override: null
      execution_mode: "fork"
      fork_output: string
      dry_run: false
    }) => void = () => {}
    tldwClientMock.executeSkill.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveExecution = resolve
        })
    )

    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    const argsInput = within(dialog).getByPlaceholderText("Enter test arguments...")
    fireEvent.click(within(dialog).getByRole("button", { name: "Run test" }))

    await waitFor(() => {
      expect(tldwClientMock.executeSkill).toHaveBeenCalledTimes(1)
    })

    expect(argsInput).toBeDisabled()
    fireEvent.keyDown(argsInput, { key: "Enter", code: "Enter", charCode: 13 })
    expect(tldwClientMock.executeSkill).toHaveBeenCalledTimes(1)

    resolveExecution({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: ["search"],
      model_override: null,
      execution_mode: "fork",
      fork_output: "Summary output",
      dry_run: false
    })
  })

  it("announces pending and completed test-run state", async () => {
    let resolveExecution: (value: {
      skill_name: string
      rendered_prompt: string
      allowed_tools: string[]
      model_override: null
      execution_mode: "inline"
      fork_output: null
      dry_run: false
    }) => void = () => {}
    tldwClientMock.executeSkill.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveExecution = resolve
        })
    )

    renderPreview()

    fireEvent.click(screen.getByRole("button", { name: "Run test" }))

    const pendingStatus = await screen.findByText("Running test for summarize")
    expect(pendingStatus.closest('[role="status"]')).toBeInTheDocument()

    resolveExecution({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: [],
      model_override: null,
      execution_mode: "inline",
      fork_output: null,
      dry_run: false
    })

    const completedStatus = await screen.findByText("Test result ready for summarize")
    expect(completedStatus.closest('[role="status"]')).toBeInTheDocument()
  })

  it("announces dry render completion separately from executed test completion", async () => {
    tldwClientMock.executeSkill.mockResolvedValueOnce({
      skill_name: "summarize",
      rendered_prompt: "Summarize chapter 1",
      allowed_tools: [],
      model_override: null,
      execution_mode: "inline",
      fork_output: null,
      dry_run: true
    })

    renderPreview()

    fireEvent.click(screen.getByRole("button", { name: "Render prompt only" }))

    const completedStatus = await screen.findByText("Rendered prompt ready for summarize")
    expect(completedStatus.closest('[role="status"]')).toBeInTheDocument()
  })

  it("renders execution failures as alerts", async () => {
    tldwClientMock.executeSkill.mockRejectedValueOnce(new Error("Model unavailable"))
    renderPreview()

    fireEvent.click(screen.getByRole("button", { name: "Run test" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Model unavailable")
  })
})
