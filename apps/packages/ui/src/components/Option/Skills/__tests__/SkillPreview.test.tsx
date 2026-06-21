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
      fork_output: "Summary output"
    })
  })

  afterEach(() => {
    cleanup()
    Modal.destroyAll()
  })

  it("discloses execution risk before running a skill", () => {
    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    expect(
      within(dialog).getByText(
        "This renders the skill with your arguments. Fork-mode skills may call the configured model and allowed tools."
      )
    ).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Run test" })).toBeInTheDocument()
    expect(within(dialog).queryByRole("button", { name: "Preview" })).not.toBeInTheDocument()
  })

  it("runs the skill with supplied arguments from the explicit test action", async () => {
    renderPreview()

    const dialog = screen.getByRole("dialog", { name: "Test run" })
    fireEvent.change(within(dialog).getByPlaceholderText("Enter test arguments..."), {
      target: { value: "chapter 1" }
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Run test" }))

    await waitFor(() => {
      expect(tldwClientMock.executeSkill).toHaveBeenCalledWith("summarize", "chapter 1")
    })
  })

  it("prevents duplicate skill executions while a test run is pending", async () => {
    let resolveExecution: (value: {
      skill_name: string
      rendered_prompt: string
      allowed_tools: string[]
      model_override: null
      execution_mode: "fork"
      fork_output: string
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
      fork_output: "Summary output"
    })
  })

  it("renders execution failures as alerts", async () => {
    tldwClientMock.executeSkill.mockRejectedValueOnce(new Error("Model unavailable"))
    renderPreview()

    fireEvent.click(screen.getByRole("button", { name: "Run test" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Model unavailable")
  })
})
