import { render, screen, fireEvent } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { LowQualityRecoveryBanner } from "../panels/LowQualityRecoveryBanner"

describe("LowQualityRecoveryBanner", () => {
  const defaultProps = {
    onRefine: vi.fn(),
    onEnableWeb: vi.fn(),
    onSelectSources: vi.fn(),
    onDismiss: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the recovery message", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    expect(
      screen.getByText(/This answer has limited evidence/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/checking source status, or enabling web fallback/i)
    ).toBeInTheDocument()
  })

  it("renders through the shared RecoveryCallout primitive", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    expect(
      screen
        .getByText(/limited evidence/i)
        .closest("[data-ds-component]")
    ).toHaveAttribute("data-ds-component", "RecoveryCallout")
  })

  it("announces the conditionally mounted recovery guidance as a polite status", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    const status = screen.getByRole("status")

    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveAttribute("aria-atomic", "true")
    expect(status).toHaveTextContent(/limited evidence/i)
  })

  it("calls onEnableWeb when web button clicked", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: /include web/i }))
    expect(defaultProps.onEnableWeb).toHaveBeenCalled()
  })

  it("calls onDismiss from the contextual dismiss action", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery suggestions" }))
    expect(defaultProps.onDismiss).toHaveBeenCalled()
    expect(screen.getByText("Dismiss")).toBeInTheDocument()
  })

  it("calls onSelectSources when select sources clicked", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: /select different/i }))
    expect(defaultProps.onSelectSources).toHaveBeenCalled()
  })

  it("calls onRefine when refine button clicked", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: /more specific/i }))
    expect(defaultProps.onRefine).toHaveBeenCalled()
  })

  it("summarizes source diagnostics when low-confidence recovery has source status", () => {
    render(
      <LowQualityRecoveryBanner
        {...defaultProps}
        sourceStatus={{
          media_db: { status: "searched", count: 2 },
          notes: { status: "empty", count: 0, reason: "no_matching_entries" },
          world_books: {
            status: "unavailable",
            count: 0,
            reason: "no_retriever_configured",
          },
        }}
      />
    )

    expect(
      screen.getByText("Source diagnostics: 1 searched, 1 empty, 1 unavailable.")
    ).toBeInTheDocument()
  })

  it("surfaces selected source-health caveats without implying automatic web fallback", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} sourceHealthCaveatCount={2} />)

    expect(
      screen.getByText("2 selected sources need attention before search.")
    ).toBeInTheDocument()
    expect(screen.queryByText(/will enable web fallback/i)).not.toBeInTheDocument()
  })
})
