import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeReadyState } from "../empty/KnowledgeReadyState"

const defaultProps = {
  suggestedPrompts: ["What changed?"],
  onPromptClick: vi.fn(),
  onContinueRecent: vi.fn(),
  onSelectSources: vi.fn(),
  onAddSources: vi.fn(),
  hasSources: true,
  hasRecentSession: false,
  webFallbackEnabled: false,
}

function renderReadyState(overrides: Partial<typeof defaultProps> = {}) {
  return render(
    <MemoryRouter>
      <KnowledgeReadyState {...defaultProps} {...overrides} />
    </MemoryRouter>
  )
}

describe("KnowledgeReadyState activation", () => {
  it("frames /knowledge as QA over existing sources and exposes Quick Ingest as the add-source path", () => {
    const onAddSources = vi.fn()
    renderReadyState({ hasSources: false, onAddSources })

    expect(screen.getByText("Ask Your Library")).toBeInTheDocument()
    expect(
      screen.getByText(/This page answers questions over searchable sources/i)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Add sources" }))
    expect(onAddSources).toHaveBeenCalledOnce()
  })

  it("distinguishes no history from a resumable history state", () => {
    const { rerender } = renderReadyState({ hasRecentSession: false })

    expect(screen.getByText("No previous QA sessions yet.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Continue recent session/i })).toBeDisabled()

    rerender(
      <MemoryRouter>
        <KnowledgeReadyState {...defaultProps} hasRecentSession />
      </MemoryRouter>
    )

    expect(screen.getByText("Recent QA session available.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Continue recent session/i })).not.toBeDisabled()
  })

  it("explains web fallback privacy and default-provider behavior in the empty source state", () => {
    renderReadyState({ hasSources: false, webFallbackEnabled: true })

    expect(
      screen.getByText(/Web fallback uses your configured server default provider/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Queries stay on your tldw server unless web fallback is enabled/i)
    ).toBeInTheDocument()
  })
})
