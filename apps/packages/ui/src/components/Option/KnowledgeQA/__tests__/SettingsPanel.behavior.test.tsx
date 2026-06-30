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
      screen.getByText("Deep search - more sources and verification")
    ).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: /Deep/i })).toBeInTheDocument()
    expect(screen.queryByRole("radio", { name: /Thorough/i })).not.toBeInTheDocument()

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
    expect(screen.getByText("Characters")).toBeInTheDocument()
    expect(screen.getByText("Chats")).toBeInTheDocument()
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

  it("resets settings, closes on Escape, and restores focus to the opener", () => {
    const onClose = vi.fn()

    function Harness() {
      const [open, setOpen] = React.useState(false)
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open settings
          </button>
          <SettingsPanel
            open={open}
            onClose={() => {
              onClose()
              setOpen(false)
            }}
          />
        </>
      )
    }

    render(<Harness />)

    const opener = screen.getByRole("button", { name: "Open settings" })
    opener.focus()
    fireEvent.click(opener)

    expect(screen.getByRole("dialog", { name: "RAG Settings" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Close settings panel" })).toHaveFocus()

    fireEvent.click(screen.getByRole("button", { name: "Reset to Balanced Defaults" }))
    expect(state.resetSettings).toHaveBeenCalledTimes(1)

    fireEvent.keyDown(document, { key: "Escape" })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog", { name: "RAG Settings" })).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
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

  it("closes on Escape before nested controls can trap the event", () => {
    const onClose = vi.fn()
    render(<SettingsPanel open onClose={onClose} />)

    const dialog = screen.getByRole("dialog", { name: "RAG Settings" })
    dialog.addEventListener("keydown", (event) => event.stopPropagation())
    fireEvent.keyDown(dialog, { key: "Escape" })

    expect(onClose).toHaveBeenCalledOnce()
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
