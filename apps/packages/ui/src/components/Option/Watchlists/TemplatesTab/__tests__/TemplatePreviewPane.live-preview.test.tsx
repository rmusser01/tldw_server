// @vitest-environment jsdom

import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { TemplatePreviewPane } from "../TemplatePreviewPane"

const serviceMocks = vi.hoisted(() => ({
  previewWatchlistTemplate: vi.fn(),
  flowCheckWatchlistTemplateSections: vi.fn()
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
  t: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string },
    maybeOptions?: Record<string, unknown>
  ) => {
    if (typeof fallbackOrOptions === "string") {
      return interpolate(fallbackOrOptions, maybeOptions)
    }
    if (
      fallbackOrOptions &&
      typeof fallbackOrOptions === "object" &&
      typeof fallbackOrOptions.defaultValue === "string"
    ) {
      return interpolate(fallbackOrOptions.defaultValue, maybeOptions)
    }
    return key
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMock.t
  })
}))

vi.mock("antd", async () => {
  const ReactModule = await import("react")

  const Alert = ({ type, title, description }: any) => (
    <div data-testid={`alert-${type}`}>
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
    </div>
  )

  const RadioGroup = ({ value, onChange, children }: any) => (
    <div role="radiogroup">
      {ReactModule.Children.map(children, (child: any) =>
        ReactModule.cloneElement(child, {
          __groupValue: value,
          __onGroupChange: onChange
        })
      )}
    </div>
  )

  const RadioButton = ({ value, children, __groupValue, __onGroupChange }: any) => (
    <button
      type="button"
      role="radio"
      aria-checked={__groupValue === value}
      onClick={() => __onGroupChange?.({ target: { value } })}
    >
      {children}
    </button>
  )

  const Select = ({ value, onChange, options = [], placeholder }: any) => (
    <select
      data-testid="template-preview-run-select"
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const next = event.target.value
        onChange?.(next ? Number(next) : undefined)
      }}
    >
      <option value="">{placeholder}</option>
      {options.map((option: any) => (
        <option key={option.value} value={String(option.value)}>
          {option.label}
        </option>
      ))}
    </select>
  )

  const Spin = () => <div data-testid="template-preview-loading">Loading</div>
  const Button = ({ children, onClick, loading: _loading, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  return {
    Alert,
    Button,
    Radio: {
      Group: RadioGroup,
      Button: RadioButton
    },
    Select,
    Spin
  }
})

vi.mock("@/services/watchlists", () => ({
  previewWatchlistTemplate: (...args: unknown[]) => serviceMocks.previewWatchlistTemplate(...args),
  flowCheckWatchlistTemplateSections: (...args: unknown[]) =>
    serviceMocks.flowCheckWatchlistTemplateSections(...args)
}))

vi.mock("@/utils/watchlists-prevention-telemetry", () => ({
  trackWatchlistsPreventionTelemetry: (...args: unknown[]) =>
    telemetryMock.trackWatchlistsPreventionTelemetry(...args)
}))

describe("TemplatePreviewPane preview clarity", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    serviceMocks.previewWatchlistTemplate.mockResolvedValue({
      rendered: "# Preview",
      context_keys: ["items"],
      warnings: []
    })
    serviceMocks.flowCheckWatchlistTemplateSections.mockResolvedValue({
      issues: [],
      diff: "",
      sections: [],
      mode: "suggest_only"
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("shows clear static vs live mode guidance", () => {
    render(
      <TemplatePreviewPane
        content="# Heading"
        format="md"
        runs={[{ id: 11, label: "Run #11" }]}
      />
    )

    expect(screen.getByTestId("template-preview-mode-note")).toHaveTextContent(
      "Static preview renders markdown/html locally and does not evaluate Jinja2 control flow."
    )

    fireEvent.click(screen.getByRole("radio", { name: "Live (render with run data)" }))

    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_template_preview_mode_changed",
        surface: "template_editor",
        mode: "live"
      })
    )
    expect(screen.getByTestId("template-preview-mode-note")).toHaveTextContent(
      "Live preview renders with data from a completed run to validate loops, variables, and conditionals."
    )
    expect(screen.getByText("Select a run to preview the template with real data.")).toBeInTheDocument()
    expect(
      screen
        .getByText("Select a run to preview the template with real data.")
        .closest('[data-ds-component="Alert"]')
    ).not.toBeNull()
  })

  it("renders live preview warnings when server render returns warning metadata", async () => {
    serviceMocks.previewWatchlistTemplate.mockResolvedValueOnce({
      rendered: "# Live output",
      context_keys: ["items", "groups"],
      warnings: ["Missing item.summary", "Unknown group key"]
    })

    render(
      <TemplatePreviewPane
        content="# Heading"
        format="md"
        runs={[{ id: 22, label: "Run #22" }]}
      />
    )

    fireEvent.click(screen.getByRole("radio", { name: "Live (render with run data)" }))
    fireEvent.change(screen.getByTestId("template-preview-run-select"), {
      target: { value: "22" }
    })

    await waitFor(() => {
      expect(serviceMocks.previewWatchlistTemplate).toHaveBeenCalledWith(
        "# Heading",
        22,
        "md",
        expect.any(AbortSignal)
      )
    }, { timeout: 2000 })

    expect(screen.getByText("Render warnings")).toBeInTheDocument()
    expect(screen.getByText("Render warnings").closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(screen.getByText("Missing item.summary")).toBeInTheDocument()
    expect(screen.getByText("Unknown group key")).toBeInTheDocument()
    expect(screen.getByText("Live output")).toBeInTheDocument()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_template_preview_rendered",
        surface: "template_editor",
        mode: "live",
        status: "success",
        warning_count: 2,
        run_id: 22
      })
    )
  })

  it("shows no-runs guidance when live mode is selected without completed runs", () => {
    render(<TemplatePreviewPane content="# Heading" format="md" runs={[]} />)

    fireEvent.click(screen.getByRole("radio", { name: "Live (render with run data)" }))

    expect(screen.getByText("No completed runs available for live preview.")).toBeInTheDocument()
    expect(
      screen
        .getByText("No completed runs available for live preview.")
        .closest('[data-ds-component="Alert"]')
    ).not.toBeNull()
    expect(
      screen.getByText(
        "Run a monitor once from Activity, then return here to preview templates with real data."
      )
    ).toBeInTheDocument()
  })

  it("shows actionable error fallback when live preview fails", async () => {
    serviceMocks.previewWatchlistTemplate.mockRejectedValueOnce(new Error("Template renderer unavailable"))

    render(
      <TemplatePreviewPane
        content="# Heading"
        format="md"
        runs={[{ id: 33, label: "Run #33" }]}
      />
    )

    fireEvent.click(screen.getByRole("radio", { name: "Live (render with run data)" }))
    fireEvent.change(screen.getByTestId("template-preview-run-select"), {
      target: { value: "33" }
    })

    await waitFor(() => {
      expect(serviceMocks.previewWatchlistTemplate).toHaveBeenCalledWith(
        "# Heading",
        33,
        "md",
        expect.any(AbortSignal)
      )
    }, { timeout: 2000 })

    expect(screen.getByText("Live preview failed")).toBeInTheDocument()
    expect(screen.getByText("Live preview failed").closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(
      screen.getByText("Check template syntax or choose another run, then try live preview again.")
    ).toBeInTheDocument()
    expect(screen.getByText("Template renderer unavailable")).toBeInTheDocument()
    expect(
      screen.getByText("No preview content yet. The template will render after a short delay.")
    ).toBeInTheDocument()
    expect(telemetryMock.trackWatchlistsPreventionTelemetry).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "watchlists_template_preview_rendered",
        surface: "template_editor",
        mode: "live",
        status: "error",
        warning_count: 0,
        run_id: 33
      })
    )
  })

  it("renders flow-check failures through the design-system Alert primitive", async () => {
    serviceMocks.flowCheckWatchlistTemplateSections.mockRejectedValueOnce(
      new Error("Flow-check service unavailable")
    )

    render(
      <TemplatePreviewPane
        content="# Heading"
        format="md"
        runs={[{ id: 44, label: "Run #44" }]}
      />
    )

    fireEvent.click(screen.getByRole("radio", { name: "Live (render with run data)" }))
    fireEvent.change(screen.getByTestId("template-preview-run-select"), {
      target: { value: "44" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run flow-check" }))

    await waitFor(() => {
      expect(screen.getByText("Flow-check service unavailable")).toBeInTheDocument()
    })

    expect(
      screen
        .getByText("Flow-check service unavailable")
        .closest('[data-ds-component="Alert"]')
    ).not.toBeNull()
  })
})
