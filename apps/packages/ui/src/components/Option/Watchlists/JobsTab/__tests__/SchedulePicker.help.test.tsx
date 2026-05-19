import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SchedulePicker } from "../SchedulePicker"

const telemetryMock = vi.hoisted(() => ({
  trackWatchlistsPreventionTelemetry: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) =>
      typeof defaultValue === "string" ? defaultValue : _key
  })
}))

vi.mock("@/utils/watchlists-prevention-telemetry", () => ({
  trackWatchlistsPreventionTelemetry: (...args: any[]) =>
    telemetryMock.trackWatchlistsPreventionTelemetry(...args)
}))

describe("SchedulePicker contextual help", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    telemetryMock.trackWatchlistsPreventionTelemetry.mockResolvedValue(undefined)
    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
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
  })

  it("shows a cron help trigger in the advanced schedule section", () => {
    render(<SchedulePicker value={null} onChange={vi.fn()} />)

    expect(
      screen.getAllByText("Most users should use presets. Turn on cron only for uncommon timing.")
        .length
    ).toBeGreaterThan(0)
    expect(screen.getByTestId("watchlists-help-cron")).toBeInTheDocument()
  })

  it("shows localized frequency guidance and emits prevention telemetry for too-frequent cron", async () => {
    const onChange = vi.fn()
    render(<SchedulePicker value={null} onChange={onChange} />)

    fireEvent.click(screen.getByRole("switch"))
    fireEvent.change(
      screen.getByPlaceholderText(
        "Cron expression (advanced), e.g., 0 9 * * MON"
      ),
      { target: { value: "* * * * *" } }
    )
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(screen.getByText(/Schedule is too frequent/i)).toBeInTheDocument()
    })
    expect(onChange).not.toHaveBeenCalled()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_validation_blocked",
        surface: "schedule_picker",
        rule: "schedule_too_frequent"
      })
    )
  })

  it("shows invalid-format guidance before apply when cron fields are incomplete", () => {
    const onChange = vi.fn()
    render(<SchedulePicker value={null} onChange={onChange} />)

    fireEvent.click(screen.getByRole("switch"))
    fireEvent.change(
      screen.getByPlaceholderText(
        "Cron expression (advanced), e.g., 0 9 * * MON"
      ),
      { target: { value: "0 9 * *" } }
    )

    expect(
      screen.getByText("Use exactly 5 cron fields: minute hour day-of-month month day-of-week.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled()
    expect(onChange).not.toHaveBeenCalled()
  })

  it("offers quick examples for advanced cron entry", () => {
    const onChange = vi.fn()
    render(<SchedulePicker value={null} onChange={onChange} />)

    fireEvent.click(screen.getByRole("switch"))
    expect(
      screen.getByText(
        "If cron is new, start with a quick example below and edit one field at a time."
      )
    ).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("schedule-example-daily0900"))
    expect(
      screen.getByPlaceholderText("Cron expression (advanced), e.g., 0 9 * * MON")
    ).toHaveValue("0 9 * * *")

    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(onChange).toHaveBeenCalledWith("0 9 * * *")
  })

  it("lets users configure every-N-hour cadence without custom cron", async () => {
    const onChange = vi.fn()
    render(<SchedulePicker value={null} onChange={onChange} />)

    fireEvent.mouseDown(screen.getByRole("combobox"))
    fireEvent.click(await screen.findByText("Every N hours/minutes"))
    expect(screen.getByLabelText("Interval unit")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Interval value"), { target: { value: "5" } })

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith("0 */5 * * *")
    })
  })
})
