// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Modal, message } from "antd"
import { JobFormModal } from "../JobFormModal"

const servicesMock = vi.hoisted(() => ({
  createWatchlistJob: vi.fn(),
  updateWatchlistJob: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  fetchWatchlistGroups: vi.fn(),
  fetchJobOutputTemplates: vi.fn(),
  fetchWatchlistTemplates: vi.fn(),
  previewWatchlistJob: vi.fn(),
  testWatchlistAudioSettings: vi.fn()
}))

const telemetryMock = vi.hoisted(() => ({
  trackWatchlistsPreventionTelemetry: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

const translationMock = vi.hoisted(() => ({
  catalog: {
    "watchlists:jobs.form.scheduleTooFrequent":
      "Schedule is too frequent. Minimum interval is every {{minutes}} minutes.",
    "watchlists:schedule.tooFrequentRemediation":
      "Increase schedule interval to meet the minimum cadence."
  } as Record<string, string>,
  t: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string },
    maybeOptions?: Record<string, unknown>
  ) => {
    const templateFromCatalog = translationMock.catalog[key]
    if (typeof fallbackOrOptions === "string") {
      return interpolate(templateFromCatalog || fallbackOrOptions, maybeOptions)
    }
    if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
      const maybeDefault = fallbackOrOptions.defaultValue
      if (typeof maybeDefault === "string") {
        return interpolate(templateFromCatalog || maybeDefault, maybeOptions)
      }
    }
    return templateFromCatalog || key
  }
}))

const mockMessageHandle = (): ReturnType<typeof message.error> =>
  undefined as unknown as ReturnType<typeof message.error>

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMock.t
  })
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistJob: (...args: unknown[]) => servicesMock.createWatchlistJob(...args),
  updateWatchlistJob: (...args: unknown[]) => servicesMock.updateWatchlistJob(...args),
  fetchWatchlistSources: (...args: unknown[]) => servicesMock.fetchWatchlistSources(...args),
  fetchWatchlistGroups: (...args: unknown[]) => servicesMock.fetchWatchlistGroups(...args),
  fetchJobOutputTemplates: (...args: unknown[]) => servicesMock.fetchJobOutputTemplates(...args),
  fetchWatchlistTemplates: (...args: unknown[]) => servicesMock.fetchWatchlistTemplates(...args),
  previewWatchlistJob: (...args: unknown[]) => servicesMock.previewWatchlistJob(...args),
  testWatchlistAudioSettings: (...args: unknown[]) => servicesMock.testWatchlistAudioSettings(...args)
}))

vi.mock("@/utils/watchlists-prevention-telemetry", () => ({
  trackWatchlistsPreventionTelemetry: (...args: unknown[]) =>
    telemetryMock.trackWatchlistsPreventionTelemetry(...args)
}))

vi.mock("../ScopeSelector", () => ({
  ScopeSelector: ({ onChange }: { onChange: (value: unknown) => void }) => (
    <div>
      <button
        type="button"
        data-testid="scope-setter"
        onClick={() => onChange({ sources: [1], tags: ["tech"] })}
      >
        Set scope
      </button>
      <button
        type="button"
        data-testid="scope-setter-10"
        onClick={() =>
          onChange({
            sources: Array.from({ length: 10 }, (_value, index) => index + 1),
            tags: ["tech"]
          })}
      >
        Set scope 10
      </button>
      <button
        type="button"
        data-testid="scope-setter-50"
        onClick={() =>
          onChange({
            sources: Array.from({ length: 50 }, (_value, index) => index + 1),
            tags: ["tech"]
          })}
      >
        Set scope 50
      </button>
    </div>
  )
}))

vi.mock("../SchedulePicker", () => ({
  SchedulePicker: ({ onChange }: { onChange: (value: string) => void }) => (
    <div>
      <button
        type="button"
        data-testid="schedule-setter"
        onClick={() => onChange("0 9 * * *")}
      >
        Set schedule
      </button>
      <button
        type="button"
        data-testid="schedule-setter-too-frequent"
        onClick={() => onChange("* * * * *")}
      >
        Set too-frequent schedule
      </button>
    </div>
  )
}))

vi.mock("../FilterBuilder", () => ({
  FilterBuilder: ({ onChange }: { onChange: (value: unknown) => void }) => (
    <button
      type="button"
      data-testid="filters-setter"
      onClick={() =>
        onChange([
          {
            type: "keyword",
            action: "include",
            value: { keywords: ["ai"], match: "any" },
            is_active: true
          }
        ])
      }
    >
      Set filters
    </button>
  )
}))

describe("JobFormModal live summary", () => {
  const createObjectUrlMock = vi.fn(() => "blob:job-form-audio-preview")
  const revokeObjectUrlMock = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    telemetryMock.trackWatchlistsPreventionTelemetry.mockResolvedValue(undefined)
    vi.spyOn(Modal, "confirm").mockImplementation((config: any) => {
      config?.onOk?.()
      return {
        destroy: vi.fn(),
        update: vi.fn()
      } as any
    })
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

    servicesMock.fetchWatchlistSources.mockResolvedValue({
      items: [
        {
          id: 1,
          name: "Source One",
          url: "https://example.com/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["tech"],
          created_at: "2026-01-15T00:00:00Z",
          updated_at: "2026-01-15T00:00:00Z",
          last_scraped_at: null,
          status: "healthy"
        }
      ],
      total: 1,
      page: 1,
      size: 500,
      has_more: false
    })
    servicesMock.fetchWatchlistGroups.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 500,
      has_more: false
    })
    servicesMock.fetchJobOutputTemplates.mockResolvedValue({
      items: [],
      total: 0
    })
    servicesMock.fetchWatchlistTemplates.mockResolvedValue({
      items: []
    })
    servicesMock.previewWatchlistJob.mockResolvedValue({
      items: [],
      total: 0,
      ingestable: 0,
      filtered: 0
    })
    servicesMock.testWatchlistAudioSettings.mockResolvedValue(
      new Uint8Array([1, 2, 3, 4]).buffer
    )

    ;(URL as unknown as { createObjectURL?: (blob: Blob) => string }).createObjectURL =
      createObjectUrlMock
    ;(URL as unknown as { revokeObjectURL?: (url: string) => void }).revokeObjectURL =
      revokeObjectUrlMock
  })

  it("updates the live summary as authoring fields change", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
      expect(servicesMock.fetchJobOutputTemplates).toHaveBeenCalled()
    })

    expect(screen.getByTestId("job-form-summary-name")).toHaveTextContent("Untitled monitor")
    expect(screen.getByTestId("job-form-summary-scope")).toHaveTextContent("No feeds selected")
    expect(screen.getByTestId("job-form-summary-schedule")).toHaveTextContent("Not scheduled")
    expect(screen.getByTestId("job-form-summary-filters")).toHaveTextContent("No filters configured")
    expect(screen.getByTestId("job-form-summary-preview")).toHaveTextContent(
      "Save this monitor once to load sample candidates for live filter preview."
    )

    fireEvent.change(
      screen.getByPlaceholderText("e.g., Daily Tech News"),
      { target: { value: "Morning Brief" } }
    )
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    const collapseHeaders = Array.from(document.querySelectorAll(".ant-collapse-header"))
    fireEvent.click(collapseHeaders[1] as Element)
    fireEvent.click(screen.getByTestId("schedule-setter"))
    fireEvent.click(collapseHeaders[2] as Element)
    fireEvent.click(screen.getByTestId("filters-setter"))

    await waitFor(() => {
      expect(screen.getByTestId("job-form-summary-name")).toHaveTextContent("Morning Brief")
      expect(screen.getByTestId("job-form-summary-scope")).toHaveTextContent("1 feed, 1 tag")
      expect(screen.getByTestId("job-form-summary-schedule")).toHaveTextContent("Daily at 09:00")
      expect(screen.getByTestId("job-form-summary-filters")).toHaveTextContent("1 filters configured")
      expect(screen.getByTestId("job-form-summary-scope-lines")).toHaveTextContent("Source One")
      expect(screen.getByTestId("job-form-summary-scope-lines")).toHaveTextContent("tech")
    })
  })

  it("shows localized remediation for structured watchlists validation errors", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())
    servicesMock.createWatchlistJob.mockRejectedValueOnce(
      Object.assign(new Error("Request failed"), {
        status: 422,
        details: {
          detail: {
            code: "watchlists_validation_error",
            rule: "schedule_too_frequent",
            message_key: "watchlists:jobs.form.scheduleTooFrequent",
            message: "Schedule is too frequent. Minimum interval is every 5 minutes.",
            remediation_key: "watchlists:schedule.tooFrequentRemediation",
            remediation: "Increase schedule interval to meet the minimum cadence.",
            meta: {
              minimum_minutes: 7
            }
          }
        }
      })
    )

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Morning Brief" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(servicesMock.createWatchlistJob).toHaveBeenCalledTimes(1)
      expect(messageErrorSpy).toHaveBeenCalled()
    })

    const renderedError = String(messageErrorSpy.mock.calls.at(-1)?.[0] || "")
    expect(renderedError).toContain("every 7 minutes")
    expect(renderedError).toContain("Increase schedule interval to meet the minimum cadence.")
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_validation_blocked",
        surface: "job_form",
        rule: "schedule_too_frequent"
      })
    )

    messageErrorSpy.mockRestore()
  })

  it("shows mapped remediation copy for non-validation save failures", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    servicesMock.createWatchlistJob.mockRejectedValueOnce(
      Object.assign(new Error("upstream unavailable"), {
        status: 503
      })
    )

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Morning Brief" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(servicesMock.createWatchlistJob).toHaveBeenCalledTimes(1)
      expect(messageErrorSpy).toHaveBeenCalled()
    })

    const renderedError = String(messageErrorSpy.mock.calls.at(-1)?.[0] || "")
    expect(renderedError).toContain("Could not save monitor.")
    expect(renderedError).toContain("Retry in a moment")

    messageErrorSpy.mockRestore()
  })

  it("includes audio defaults in create payload when audio briefing is enabled", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Audio Morning Brief" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(servicesMock.createWatchlistJob).toHaveBeenCalledTimes(1)
    })

    expect(servicesMock.createWatchlistJob).toHaveBeenCalledWith(
      expect.objectContaining({
        output_prefs: expect.objectContaining({
          generate_audio: true,
          audio_voice: "alloy",
          audio_speed: 1,
          target_audio_minutes: 8
        })
      })
    )
  })

  it("shows practical audio setup guidance in monitor form", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))

    expect(screen.getByTestId("job-form-audio-practical-hint")).toHaveTextContent(
      "Practical default: Alloy voice at 1.0 speed and 8-minute target, then tune after your first run."
    )
  })

  it("tests audio settings and renders a playable sample when generation succeeds", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByTestId("job-form-audio-test-button"))

    await waitFor(() => {
      expect(servicesMock.testWatchlistAudioSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          voice: "alloy",
          speed: 1,
          response_format: "mp3"
        })
      )
    })

    expect(await screen.findByTestId("job-form-audio-test-player")).toBeInTheDocument()
    expect(screen.getByTestId("job-form-audio-test-success")).toHaveTextContent(
      "Sample ready. Listen before saving."
    )
  })

  it("shows loading feedback while an audio sample is being generated", async () => {
    let resolveAudioSample: ((value: ArrayBuffer) => void) | undefined
    servicesMock.testWatchlistAudioSettings.mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => {
        resolveAudioSample = resolve
      })
    )

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))

    const testButton = screen.getByTestId("job-form-audio-test-button")
    fireEvent.click(testButton)

    expect(testButton).toBeDisabled()
    expect(screen.getByTestId("job-form-audio-test-loading")).toHaveTextContent(
      "Generating sample audio..."
    )

    resolveAudioSample?.(new Uint8Array([9, 8, 7, 6]).buffer)

    await waitFor(() => {
      expect(screen.getByTestId("job-form-audio-test-player")).toBeInTheDocument()
    })
  })

  it("shows an inline error when audio sample generation fails", async () => {
    servicesMock.testWatchlistAudioSettings.mockRejectedValueOnce(
      Object.assign(new Error("audio backend unavailable"), { status: 503 })
    )

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByTestId("job-form-audio-test-button"))

    await waitFor(() => {
      expect(screen.getByTestId("job-form-audio-test-error")).toHaveTextContent(
        "Could not generate audio sample"
      )
    })
    expect(screen.queryByTestId("job-form-audio-test-player")).not.toBeInTheDocument()
  })

  it("blocks audio sample tests when advanced background URI is invalid", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Show advanced audio options" }))
    fireEvent.change(screen.getByPlaceholderText("file:///path/to/bed.mp3"), {
      target: { value: "not-a-valid-uri" }
    })
    fireEvent.click(screen.getByTestId("job-form-audio-test-button"))

    await waitFor(() => {
      expect(messageErrorSpy).toHaveBeenCalledWith(
        "Background track must start with https://, http://, or file://."
      )
    })
    expect(servicesMock.testWatchlistAudioSettings).not.toHaveBeenCalled()

    messageErrorSpy.mockRestore()
  })

  it("guides basic mode through scope and schedule before review step", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    expect(screen.getByTestId("job-form-basic-stepper")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("job-form-basic-next"))

    expect(messageErrorSpy).toHaveBeenCalledWith(
      "Please select at least one feed, group, or tag"
    )

    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-basic-next"))
    expect(screen.getByTestId("job-form-basic-step-schedule")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("job-form-basic-next"))
    expect(messageErrorSpy).toHaveBeenCalledWith(
      "Set a schedule before continuing to review."
    )

    fireEvent.click(screen.getByTestId("schedule-setter"))
    fireEvent.click(screen.getByTestId("job-form-basic-next"))
    fireEvent.click(screen.getByTestId("job-form-basic-next"))

    expect(screen.getByTestId("job-form-basic-review")).toBeInTheDocument()
    expect(screen.getByTestId("job-form-basic-review")).toHaveTextContent("1 feed, 1 tag")

    messageErrorSpy.mockRestore()
  })

  it("blocks submit with actionable guidance when schedule is too frequent", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Too Frequent Monitor" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-basic-next"))
    fireEvent.click(screen.getByTestId("schedule-setter-too-frequent"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(messageErrorSpy).toHaveBeenCalledWith(
        "Schedule is too frequent. Minimum interval is every 5 minutes."
      )
    })
    expect(servicesMock.createWatchlistJob).not.toHaveBeenCalled()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_validation_blocked",
        surface: "job_form",
        rule: "schedule_too_frequent"
      })
    )

    messageErrorSpy.mockRestore()
  })

  it("blocks submit when editing with invalid email recipients", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(
      <JobFormModal
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        initialValues={{
          id: 88,
          name: "Invalid Email Monitor",
          description: "existing",
          scope: { sources: [1] },
          schedule_expr: "0 9 * * *",
          timezone: "UTC",
          active: true,
          output_prefs: {
            deliveries: {
              email: {
                enabled: true,
                recipients: ["invalid-email"]
              }
            }
          },
          job_filters: null,
          created_at: "2026-01-15T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(messageErrorSpy).toHaveBeenCalledWith(
        "Fix invalid email recipients before saving."
      )
    })
    expect(servicesMock.updateWatchlistJob).not.toHaveBeenCalled()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_validation_blocked",
        surface: "job_form",
        rule: "invalid_email_recipients"
      })
    )

    messageErrorSpy.mockRestore()
  })

  it("hydrates audio preferences for edits and clears audio fields when disabled", async () => {
    servicesMock.updateWatchlistJob.mockResolvedValueOnce({
      id: 77
    })

    render(
      <JobFormModal
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        initialValues={{
          id: 77,
          name: "Existing audio monitor",
          description: "existing",
          scope: { sources: [1] },
          schedule_expr: "0 9 * * *",
          timezone: "UTC",
          active: true,
          output_prefs: {
            template: { default_name: "briefing_markdown" },
            generate_audio: true,
            audio_voice: "nova",
            audio_speed: 1.25,
            target_audio_minutes: 12
          },
          job_filters: null,
          created_at: "2026-01-15T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
      expect(servicesMock.previewWatchlistJob).toHaveBeenCalledWith(77, {
        limit: 60,
        per_source: 12
      })
    })

    fireEvent.click(screen.getByText("Output & Delivery"))

    const audioSwitch = screen.getByTestId("job-form-audio-enabled-switch")
    expect(audioSwitch).toHaveAttribute("aria-checked", "true")
    expect(screen.getByDisplayValue("1.25")).toBeInTheDocument()
    expect(screen.getByDisplayValue("12")).toBeInTheDocument()
    fireEvent.click(audioSwitch)

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(servicesMock.updateWatchlistJob).toHaveBeenCalledTimes(1)
    })

    const [, payload] = servicesMock.updateWatchlistJob.mock.calls[0]
    expect(payload.output_prefs).toEqual(
      expect.objectContaining({
        generate_audio: false,
        template: { default_name: "briefing_markdown" }
      })
    )
    expect(payload.output_prefs).not.toHaveProperty("audio_voice")
    expect(payload.output_prefs).not.toHaveProperty("audio_speed")
    expect(payload.output_prefs).not.toHaveProperty("target_audio_minutes")
  })

  it("persists advanced audio settings when provided", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Advanced Audio Brief" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Show advanced audio options" }))
    fireEvent.change(screen.getByPlaceholderText("file:///path/to/bed.mp3"), {
      target: { value: "file:///tmp/news-bed.mp3" }
    })
    fireEvent.change(
      screen.getByPlaceholderText('{ "HOST": "af_heart", "REPORTER": "am_adam" }'),
      { target: { value: '{ "HOST": "af_heart", "REPORTER": "am_adam" }' } }
    )

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(servicesMock.createWatchlistJob).toHaveBeenCalledTimes(1)
    })

    expect(servicesMock.createWatchlistJob).toHaveBeenCalledWith(
      expect.objectContaining({
        output_prefs: expect.objectContaining({
          generate_audio: true,
          background_audio_uri: "file:///tmp/news-bed.mp3",
          voice_map: {
            HOST: "af_heart",
            REPORTER: "am_adam"
          }
        })
      })
    )
  }, 10000)

  it("blocks save when advanced audio voice map JSON is invalid", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Invalid Audio Map" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Show advanced audio options" }))
    fireEvent.change(
      screen.getByPlaceholderText('{ "HOST": "af_heart", "REPORTER": "am_adam" }'),
      { target: { value: "{invalid json}" } }
    )

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(messageErrorSpy).toHaveBeenCalledWith(
        "Voice map must be valid JSON with marker-to-voice string pairs."
      )
    })
    expect(servicesMock.createWatchlistJob).not.toHaveBeenCalled()

    messageErrorSpy.mockRestore()
  }, 10000)

  it("blocks save when advanced audio background URI is invalid", async () => {
    const messageErrorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Invalid Audio URI" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Show advanced audio options" }))
    fireEvent.change(screen.getByPlaceholderText("file:///path/to/bed.mp3"), {
      target: { value: "background-track" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(messageErrorSpy).toHaveBeenCalledWith(
        "Background track must start with https://, http://, or file://."
      )
    })
    expect(servicesMock.createWatchlistJob).not.toHaveBeenCalled()

    messageErrorSpy.mockRestore()
  }, 10000)

  it("preserves advanced settings when switching back to basic mode", async () => {
    const messageInfoSpy = vi
      .spyOn(message, "info")
      .mockImplementation(() => mockMessageHandle())

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Mode Preservation Monitor" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.change(screen.getByPlaceholderText("Defaults to output title"), {
      target: { value: "Ops Digest" }
    })
    fireEvent.click(screen.getByTestId("job-form-mode-basic"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(servicesMock.createWatchlistJob).toHaveBeenCalledTimes(1)
    })
    expect(messageInfoSpy).toHaveBeenCalledWith(
      "Advanced settings are preserved and will still apply, but they are hidden in Basic mode."
    )
    expect(servicesMock.createWatchlistJob).toHaveBeenCalledWith(
      expect.objectContaining({
        output_prefs: expect.objectContaining({
          deliveries: expect.objectContaining({
            email: expect.objectContaining({
              subject: "Ops Digest"
            })
          })
        })
      })
    )

    messageInfoSpy.mockRestore()
  })

  it("shows confidence risks until required setup is complete", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    expect(screen.getByTestId("job-form-confidence-panel")).toBeInTheDocument()
    expect(screen.getByTestId("job-form-confidence-status")).toHaveTextContent("Needs attention")
    expect(screen.getByTestId("job-form-confidence-risk-scope")).toHaveTextContent(
      "Select at least one feed, group, or tag."
    )

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Confidence Monitor" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))

    await waitFor(() => {
      expect(screen.getByTestId("job-form-confidence-status")).toHaveTextContent("Ready to save")
    })
    expect(screen.queryByTestId("job-form-confidence-risk-scope")).not.toBeInTheDocument()
  })

  it("flags preserved hidden advanced settings as confidence risk in basic mode", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Hidden Advanced Confidence Monitor" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.change(screen.getByPlaceholderText("Defaults to output title"), {
      target: { value: "Ops Digest" }
    })
    fireEvent.click(screen.getByTestId("job-form-mode-basic"))

    expect(screen.getByTestId("job-form-confidence-risk-hidden-advanced")).toHaveTextContent(
      "Advanced settings are preserved and hidden in Basic mode."
    )
  })

  it("surfaces delivery and hidden-advanced consequences in live summary", async () => {
    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    expect(screen.getByTestId("job-form-summary-delivery")).toHaveTextContent("No automatic delivery")
    expect(screen.getByTestId("job-form-summary-audio")).toHaveTextContent("Disabled")

    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.change(screen.getByPlaceholderText("Defaults to output title"), {
      target: { value: "Ops Digest" }
    })

    expect(screen.getByTestId("job-form-summary-audio")).toHaveTextContent(
      "Enabled (alloy, 8 min target)"
    )
    fireEvent.click(screen.getByTestId("job-form-mode-basic"))

    expect(screen.getByTestId("job-form-summary-hidden-advanced")).toHaveTextContent(
      "Advanced settings are active and hidden in Basic mode."
    )
  })

  it("requires explicit confirmation for recurring delivery settings", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce((config: any) => {
      config?.onCancel?.()
      return {
        destroy: vi.fn(),
        update: vi.fn()
      } as any
    })

    render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech News"), {
      target: { value: "Recurring Delivery Confirmation" }
    })
    fireEvent.click(screen.getByTestId("scope-setter"))
    fireEvent.click(screen.getByTestId("job-form-mode-advanced"))
    fireEvent.click(screen.getByText("Output & Delivery"))
    fireEvent.click(screen.getByTestId("job-form-audio-enabled-switch"))
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalledTimes(1)
    })
    const confirmConfig = confirmSpy.mock.calls[0][0]
    expect(confirmConfig.title).toBe("Confirm recurring delivery settings")
    expect(servicesMock.createWatchlistJob).not.toHaveBeenCalled()
  })

  it("restores focus to the launch control after modal close", async () => {
    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open monitor form"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(<JobFormModal open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalled()
      expect(servicesMock.fetchWatchlistGroups).toHaveBeenCalled()
    })

    const monitorNameInput = screen.getByPlaceholderText("e.g., Daily Tech News")
    monitorNameInput.focus()
    expect(monitorNameInput).toHaveFocus()

    rerender(<JobFormModal open={false} onClose={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })
})
