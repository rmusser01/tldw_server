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

  it("calls onEnableWeb when web button clicked", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: /include web/i }))
    expect(defaultProps.onEnableWeb).toHaveBeenCalled()
  })

  it("calls onDismiss when close button clicked", () => {
    render(<LowQualityRecoveryBanner {...defaultProps} />)
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }))
    expect(defaultProps.onDismiss).toHaveBeenCalled()
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
