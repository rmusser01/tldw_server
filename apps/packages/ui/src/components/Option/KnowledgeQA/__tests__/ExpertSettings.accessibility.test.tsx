import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ExpertSettings } from "../SettingsPanel/ExpertSettings"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"

const state = {
  settings: DEFAULT_RAG_SETTINGS,
  updateSetting: vi.fn(),
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    settings: state.settings,
    updateSetting: state.updateSetting,
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasWebSearch: true },
    loading: false,
  })
}))

describe("ExpertSettings accessibility", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.settings = DEFAULT_RAG_SETTINGS
  })

  it("announces accordion open and closed state with aria-expanded", () => {
    render(<ExpertSettings />)

    const searchToggle = document.querySelector<HTMLButtonElement>(
      'button[aria-controls="section-content-search"]'
    )

    expect(searchToggle).not.toBeNull()
    expect(searchToggle).toHaveAttribute("aria-expanded", "true")
    expect(document.getElementById("section-content-search")).toBeInTheDocument()

    fireEvent.click(searchToggle!)
    expect(searchToggle).toHaveAttribute("aria-expanded", "false")
    expect(document.getElementById("section-content-search")).not.toBeInTheDocument()
  })

  it("keeps aria-controls relationships valid for all accordion sections", () => {
    render(<ExpertSettings />)

    const sectionToggles = document.querySelectorAll<HTMLButtonElement>(
      'button[aria-controls^="section-content-"]'
    )
    expect(sectionToggles.length).toBeGreaterThan(0)

    sectionToggles.forEach((toggle) => {
      const contentId = toggle.getAttribute("aria-controls")
      expect(contentId).toBeTruthy()

      const isExpanded = toggle.getAttribute("aria-expanded") === "true"
      const panel = contentId ? document.getElementById(contentId) : null
      if (isExpanded) {
        expect(panel).toBeInTheDocument()
      } else {
        expect(panel).not.toBeInTheDocument()
      }
    })
  })

  it("supports keyboard toggle on accordion section buttons", async () => {
    const user = userEvent.setup()
    render(<ExpertSettings />)

    const queryEnhancementToggle = screen.getByRole("button", {
      name: /Query Enhancement/i,
    })
    expect(queryEnhancementToggle).toHaveAttribute("aria-expanded", "false")

    queryEnhancementToggle.focus()
    await user.keyboard("{Enter}")
    expect(queryEnhancementToggle).toHaveAttribute("aria-expanded", "true")

    await user.keyboard(" ")
    expect(queryEnhancementToggle).toHaveAttribute("aria-expanded", "false")
  })

  it("filters All Options and reports when no keys match", () => {
    render(<ExpertSettings />)

    fireEvent.click(screen.getByRole("button", { name: /All Options/i }))
    fireEvent.change(screen.getByLabelText(/Filter option keys/i), {
      target: { value: "definitely_no_matching_rag_key" },
    })

    expect(screen.getByText("No option keys match this filter.")).toBeInTheDocument()
  })

  it("updates boolean controls from the All Options editor", () => {
    render(<ExpertSettings />)

    fireEvent.click(screen.getByRole("button", { name: /All Options/i }))
    fireEvent.change(screen.getByLabelText(/Filter option keys/i), {
      target: { value: "enable_generation" },
    })
    fireEvent.click(screen.getByRole("switch", { name: "enable_generation" }))

    expect(state.updateSetting).toHaveBeenCalledWith(
      "enable_generation",
      !DEFAULT_RAG_SETTINGS.enable_generation
    )
  })
})
