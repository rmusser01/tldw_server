import React from "react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultOrOpts?: unknown) => {
      if (typeof defaultOrOpts === "string") return defaultOrOpts
      const options = defaultOrOpts as Record<string, string>
      return (options?.defaultValue ?? key).replace(/\{\{(\w+)\}\}/g, (_match, token) => String(options[token] ?? ""))
    },
  }),
}))

// Mock lucide-react icons as simple spans with data-testid
vi.mock("lucide-react", () => ({
  Check: (props: Record<string, unknown>) => (
    <span data-testid="icon-check" {...props} />
  ),
  Circle: (props: Record<string, unknown>) => (
    <span data-testid="icon-circle" {...props} />
  ),
  Loader2: (props: Record<string, unknown>) => (
    <span data-testid="icon-loader" {...props} />
  ),
}))

// ---------------------------------------------------------------------------
// Mock the IngestWizardContext hook so we can control state
// ---------------------------------------------------------------------------

const mockGoToStep = vi.fn()

let mockState = {
  currentStep: 1 as 1 | 2 | 3 | 4 | 5,
  highestStep: 1 as 1 | 2 | 3 | 4 | 5,
  queueItems: [] as Array<{ id: string }>,
  selectedPreset: "standard" as string,
  processingState: {
    status: "idle" as string,
    perItemProgress: [] as Array<{ progressPercent: number; status: string }>,
    elapsed: 0,
    estimatedRemaining: 0,
  },
}

vi.mock("../IngestWizardContext", () => ({
  useIngestWizard: () => ({
    state: mockState,
    goToStep: mockGoToStep,
  }),
}))

// Import AFTER mocks are registered
import { IngestWizardStepper } from "../IngestWizardStepper"

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("IngestWizardStepper", () => {
  beforeEach(() => {
    mockGoToStep.mockClear()
    mockState = {
      currentStep: 1,
      highestStep: 1,
      queueItems: [],
      selectedPreset: "standard",
      processingState: {
        status: "idle",
        perItemProgress: [],
        elapsed: 0,
        estimatedRemaining: 0,
      },
    }
  })

  it("summarizes confirmed finished items while another item is still processing", () => {
    mockState.currentStep = 4
    mockState.highestStep = 4
    mockState.processingState.status = "running"
    mockState.processingState.perItemProgress = [
      { status: "complete", progressPercent: 100 },
      { status: "processing", progressPercent: 42 },
    ]
    render(<IngestWizardStepper />)
    expect(screen.getByText("1/2 finished")).toBeInTheDocument()
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument()
  })

  it("renders 5 step labels", () => {
    render(<IngestWizardStepper />)
    // Each step renders a button with an aria-label containing "Step N:"
    const buttons = screen.getAllByRole("button")
    expect(buttons).toHaveLength(5)
  })

  it("marks the current step with aria-current='step'", () => {
    mockState.currentStep = 2
    mockState.highestStep = 2
    render(<IngestWizardStepper />)

    const buttons = screen.getAllByRole("button")
    // Step 2 (index 1) should be current
    expect(buttons[1]).toHaveAttribute("aria-current", "step")
    // Other steps should not
    expect(buttons[0]).not.toHaveAttribute("aria-current")
    expect(buttons[2]).not.toHaveAttribute("aria-current")
  })

  it("completed steps show the check icon", () => {
    mockState.currentStep = 3
    mockState.highestStep = 3
    render(<IngestWizardStepper />)

    // Steps 1 and 2 are completed (< currentStep 3)
    const checkIcons = screen.getAllByTestId("icon-check")
    expect(checkIcons.length).toBe(2)
  })

  it("clicking a completed step calls goToStep", async () => {
    mockState.currentStep = 3
    mockState.highestStep = 3
    render(<IngestWizardStepper />)

    const buttons = screen.getAllByRole("button")
    // Step 1 (index 0) is completed and clickable
    await userEvent.click(buttons[0])
    expect(mockGoToStep).toHaveBeenCalledWith(1)
  })

  it("cannot click future steps (they are disabled)", async () => {
    mockState.currentStep = 2
    mockState.highestStep = 2
    render(<IngestWizardStepper />)

    const buttons = screen.getAllByRole("button")
    // Steps 3, 4, 5 (indices 2, 3, 4) are future and should be disabled
    expect(buttons[2]).toBeDisabled()
    expect(buttons[3]).toBeDisabled()
    expect(buttons[4]).toBeDisabled()

    // Clicking a disabled/future step should NOT call goToStep
    await userEvent.click(buttons[3])
    expect(mockGoToStep).not.toHaveBeenCalled()
  })

  it("current step button is not clickable (disabled)", () => {
    mockState.currentStep = 2
    mockState.highestStep = 2
    render(<IngestWizardStepper />)

    const buttons = screen.getAllByRole("button")
    // The current step (index 1) is not completed, so not clickable
    expect(buttons[1]).toBeDisabled()
  })

  it("renders step labels including Add, Configure, Review, Processing, Results", () => {
    render(<IngestWizardStepper />)
    // Steps where defaultLabel === shortLabel render the same text twice
    // (one in "hidden sm:inline", one in "inline sm:hidden"), so use getAllByText.
    expect(screen.getAllByText("Add").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("Config")).toBeInTheDocument()
    expect(screen.getAllByText("Review").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("Proc.")).toBeInTheDocument()
    expect(screen.getAllByText("Results").length).toBeGreaterThanOrEqual(1)
  })
})
