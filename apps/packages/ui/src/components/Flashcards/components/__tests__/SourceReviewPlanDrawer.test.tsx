import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SourceReviewPlanDrawer } from "../SourceReviewPlanDrawer"

const createPlanMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof options === "string") return options
      if (options?.defaultValue) {
        return options.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) => String(options[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("../../hooks/useSourceReviewQueries", () => ({
  useCreateSourceReviewPlanMutation: () => ({
    mutateAsync: createPlanMock,
    isPending: false
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn()
  })
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

const addRequiredPlanDetails = () => {
  fireEvent.change(screen.getByLabelText("Plan title"), {
    target: { value: "Cardiac physiology" }
  })
  fireEvent.change(screen.getByLabelText("Source ID"), {
    target: { value: "note-42" }
  })
  fireEvent.change(screen.getByLabelText("Source label"), {
    target: { value: "Cardiac physiology notes" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Add source" }))
}

describe("SourceReviewPlanDrawer", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    createPlanMock.mockResolvedValue({ id: 7, title: "Cardiac physiology" })
    vi.spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions").mockReturnValue({
      timeZone: "America/Los_Angeles"
    } as Intl.ResolvedDateTimeFormatOptions)
  })

  it("seeds the requested review schedule presets", () => {
    render(<SourceReviewPlanDrawer open onClose={vi.fn()} />)

    for (const label of [
      "Day 1",
      "Day 3",
      "Day 7",
      "Day 14",
      "Day 28",
      "3 months",
      "6 months"
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  })

  it("shows invalid-row errors and blocks creation", () => {
    render(<SourceReviewPlanDrawer open onClose={vi.fn()} />)
    addRequiredPlanDetails()

    fireEvent.change(screen.getByLabelText("Offset 1"), {
      target: { value: "0" }
    })

    expect(
      screen.getByText("Enter a whole number between 1 and 3650.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create plan" })).toBeDisabled()
  })

  it("blocks invalid timezones and oversized source excerpts", () => {
    render(<SourceReviewPlanDrawer open onClose={vi.fn()} />)
    fireEvent.change(screen.getByLabelText("Plan title"), {
      target: { value: "Cardiac physiology" }
    })
    fireEvent.change(screen.getByLabelText("Timezone"), {
      target: { value: "Mars/Olympus" }
    })
    fireEvent.change(screen.getByLabelText("Source ID"), {
      target: { value: "note-42" }
    })
    fireEvent.change(screen.getByLabelText("Source excerpt"), {
      target: { value: "x".repeat(20_001) }
    })

    expect(screen.getByText("Enter a valid IANA timezone.")).toBeInTheDocument()
    expect(
      screen.getByText("Use 20,000 characters or fewer.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add source" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Create plan" })).toBeDisabled()
  })

  it("marks both exact duplicate schedule rows", () => {
    render(<SourceReviewPlanDrawer open onClose={vi.fn()} />)
    addRequiredPlanDetails()

    fireEvent.change(screen.getByLabelText("Offset 2"), {
      target: { value: "1" }
    })

    expect(
      screen.getAllByText("Duplicate interval and activity.")
    ).toHaveLength(2)
    expect(screen.getByRole("button", { name: "Create plan" })).toBeDisabled()
  })

  it("submits the timezone and every valid schedule row", async () => {
    const onCreated = vi.fn()
    const onClose = vi.fn()
    render(
      <SourceReviewPlanDrawer open onClose={onClose} onCreated={onCreated} />
    )
    addRequiredPlanDetails()
    fireEvent.change(screen.getByLabelText("Start date"), {
      target: { value: "2026-07-09" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Create plan" }))

    await waitFor(() => {
      expect(createPlanMock).toHaveBeenCalledWith({
        title: "Cardiac physiology",
        starts_on: "2026-07-09",
        timezone: "America/Los_Angeles",
        source_items: [
          {
            source_type: "note",
            source_id: "note-42",
            label: "Cardiac physiology notes"
          }
        ],
        schedule: [
          { offset_value: 1, offset_unit: "day", activity_type: "reread" },
          { offset_value: 3, offset_unit: "day", activity_type: "reread" },
          { offset_value: 7, offset_unit: "day", activity_type: "reread" },
          { offset_value: 14, offset_unit: "day", activity_type: "reread" },
          { offset_value: 28, offset_unit: "day", activity_type: "reread" },
          { offset_value: 3, offset_unit: "month", activity_type: "reread" },
          { offset_value: 6, offset_unit: "month", activity_type: "reread" }
        ]
      })
    })
    expect(onCreated).toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })

  it("keeps backend validation feedback visible in the drawer", async () => {
    createPlanMock.mockRejectedValue({
      response: {
        data: { detail: "Two rows resolve to the same review time." }
      }
    })
    render(<SourceReviewPlanDrawer open onClose={vi.fn()} />)
    addRequiredPlanDetails()

    fireEvent.click(screen.getByRole("button", { name: "Create plan" }))

    expect(
      await screen.findByText("Two rows resolve to the same review time.")
    ).toBeInTheDocument()
  })
})
