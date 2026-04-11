import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { PresetSelector } from "../SettingsPanel/PresetSelector"
import { SettingsPanel } from "../SettingsPanel"
import { ExpertSettings } from "../SettingsPanel/ExpertSettings"

const state = {
  preset: "balanced",
  setPreset: vi.fn((preset: string) => {
    state.preset = preset
  }),
  expertMode: false,
  toggleExpertMode: vi.fn(() => {
    state.expertMode = !state.expertMode
  }),
  resetSettings: vi.fn(),
  settings: DEFAULT_RAG_SETTINGS,
  updateSetting: vi.fn(),
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => state,
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasWebSearch: true },
    loading: false,
  }),
}))

describe("SettingsPanel behavior and copy guardrails", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.preset = "balanced"
    state.expertMode = false
    state.settings = DEFAULT_RAG_SETTINGS
  })

  it("uses plain-language preset descriptions and keyboard radiogroup navigation", () => {
    render(<PresetSelector />)

    expect(screen.getByText("Fastest - text matching only")).toBeInTheDocument()
    expect(
      screen.getByText("Recommended - combines text and meaning")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Most thorough - includes fact-checking")
    ).toBeInTheDocument()

    const fastRadio = screen.getByRole("radio", { name: /Fast/i })
    fastRadio.focus()
    fireEvent.keyDown(fastRadio, { key: "ArrowRight" })

    const balancedRadio = screen.getByRole("radio", { name: /Balanced/i })
    expect(balancedRadio).toHaveFocus()
    fireEvent.keyDown(balancedRadio, { key: "Enter" })
    expect(state.setPreset).toHaveBeenCalledWith("balanced")
  })

  it("keeps expert all-options sources labels aligned with basic-mode wording", () => {
    render(<ExpertSettings />)

    fireEvent.click(screen.getByRole("button", { name: /All Options/i }))
    fireEvent.change(screen.getByLabelText(/Filter option keys/i), {
      target: { value: "sources" },
    })

    expect(screen.getByText("Documents & Media")).toBeInTheDocument()
    expect(screen.getByText("Story Characters")).toBeInTheDocument()
    expect(screen.getByText("Conversations")).toBeInTheDocument()
    expect(screen.getByText("Task Boards")).toBeInTheDocument()
  })

  it("shows scope note, balanced-reset copy, and preserves drawer focus trap/backdrop behavior", () => {
    const onClose = vi.fn()
    const { container, rerender } = render(<SettingsPanel open onClose={onClose} />)

    expect(
      screen.getByText("Changes apply to your next search. Previous answers are not affected.")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Reset to Balanced Defaults" })
    ).toBeInTheDocument()

    const dialog = screen.getByRole("dialog", { name: "RAG Settings" })
    expect(dialog.className).toContain("w-96")
    expect(dialog.className).toContain("max-w-[calc(100vw-2rem)]")
    expect(dialog.className).toContain("animate-in")
    expect(dialog.className).toContain("slide-in-from-right")
    expect(dialog.className).toContain("duration-200")

    const focusable = dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
    )
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    expect(first).toBeTruthy()
    expect(last).toBeTruthy()

    last.focus()
    fireEvent.keyDown(document, { key: "Tab" })
    expect(first).toHaveFocus()

    first.focus()
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true })
    expect(last).toHaveFocus()

    const backdrop = container.querySelector('div[aria-hidden="true"]')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop!)
    expect(onClose).toHaveBeenCalledTimes(1)

    rerender(<SettingsPanel open={false} onClose={onClose} />)
    rerender(<SettingsPanel open onClose={onClose} />)
    expect(
      screen.getByText("Changes apply to your next search. Previous answers are not affected.")
    ).toBeInTheDocument()
  })

  it("does not crash when expert-mode onboarding storage is blocked", () => {
    const getItemSpy = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })

    render(<SettingsPanel open onClose={vi.fn()} />)

    expect(() => {
      fireEvent.click(screen.getByRole("switch", { name: "Basic Mode" }))
    }).not.toThrow()

    expect(state.toggleExpertMode).toHaveBeenCalledTimes(1)

    getItemSpy.mockRestore()
    setItemSpy.mockRestore()
  })

  it("hides the expert-mode onboarding hint when toggling back to basic mode", () => {
    localStorage.removeItem("knowledgeqa-expert-mode-seen")

    const { rerender } = render(<SettingsPanel open onClose={vi.fn()} />)

    fireEvent.click(screen.getByRole("switch", { name: "Basic Mode" }))
    rerender(<SettingsPanel open onClose={vi.fn()} />)
    expect(screen.getByText("Welcome to Expert Mode")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("switch", { name: "Expert Mode" }))
    rerender(<SettingsPanel open onClose={vi.fn()} />)

    expect(screen.queryByText("Welcome to Expert Mode")).not.toBeInTheDocument()
  })
})
