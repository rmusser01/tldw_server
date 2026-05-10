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
      screen.getByText(/sources may not closely match/i)
    ).toBeInTheDocument()
  })

  it("renders through the shared RecoveryCallout primitive", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    expect(
      screen
        .getByText(/sources may not closely match/i)
        .closest("[data-ds-component]")
    ).toHaveAttribute("data-ds-component", "RecoveryCallout")
  })

  it("announces the conditionally mounted recovery guidance as a polite status", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    const status = screen.getByRole("status")

    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveAttribute("aria-atomic", "true")
    expect(status).toHaveTextContent(/sources may not closely match/i)
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
})
