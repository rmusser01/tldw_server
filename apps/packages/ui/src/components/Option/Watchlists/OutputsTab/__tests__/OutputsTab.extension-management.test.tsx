// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OutputsTab } from "../OutputsTab"

const mocks = vi.hoisted(() => ({
  createWatchlistOutputMock: vi.fn(),
  downloadWatchlistOutputBinaryMock: vi.fn(),
  downloadWatchlistOutputMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> },
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn((query: string) => ({
      matches: query.includes("max-width") ? width <= 767 : width >= 768,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, icon, onClick, loading, disabled, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
      onClick={(event) => onClick?.(event)}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
  const Select = ({ value, onChange, options = [], placeholder, allowClear, ...rest }: any) => (
    <select
      aria-label={placeholder ?? "select"}
      data-testid={rest["data-testid"]}
      value={value == null ? "" : String(value)}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [] }: any) => (
    <table role="table" aria-label={ariaLabel || "table"}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  return {
    Alert: ({ message, description, action }: any) => (
      <div role="status">
        <div>{message}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    Button,
    Input: ({ value, onChange, ...rest }: any) => (
      <input value={value || ""} onChange={(event) => onChange?.(event)} {...rest} />
    ),
    InputNumber: ({ value, onChange, placeholder, ...rest }: any) => (
      <input
        aria-label={placeholder}
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(event.currentTarget.value ? Number(event.currentTarget.value) : null)}
        {...rest}
      />
    ),
    Modal: ({ open, children }: any) => (open ? <div data-testid="outputs-regenerate-modal">{children}</div> : null),
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistOutput: (...args: unknown[]) => mocks.createWatchlistOutputMock(...args),
  downloadWatchlistOutput: (...args: unknown[]) => mocks.downloadWatchlistOutputMock(...args),
  downloadWatchlistOutputBinary: (...args: unknown[]) => mocks.downloadWatchlistOutputBinaryMock(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistOutputs: (...args: unknown[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistTemplates: (...args: unknown[]) => mocks.fetchWatchlistTemplatesMock(...args),
  recordWatchlistsOnboardingTelemetry: vi.fn(() => Promise.resolve())
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: () => null
}))

vi.mock("../ReportBuilderDrawer", () => ({
  ReportBuilderDrawer: ({ open, selectedWatchlist }: any) =>
    open ? <div data-testid="report-builder-open">Builder for {selectedWatchlist?.name}</div> : null
}))

const buildOutput = (overrides: Record<string, unknown> = {}) => ({
  id: 18,
  job_id: 4,
  run_id: 91,
  title: "CTI Evidence Report",
  format: "md",
  type: "briefing",
  metadata: {
    report_snapshot_path: "watchlists/report-18.evidence.json",
    report_readiness: {
      state: "ready",
      score: 96,
      warnings: []
    },
    source_count: 2,
    alert_count: 3,
    weak_evidence_warning_count: 0,
    deliveries: {
      email: {
        channel: "email",
        status: "failed",
        detail: "smtp timeout"
      }
    }
  },
  created_at: "2026-05-15T08:00:00Z",
  expires_at: null,
  expired: false,
  version: 1,
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  watchlists: [
    {
      id: 42,
      name: "Healthcare ransomware",
      domain: "cti_osint",
      status: "active",
      priority: "critical",
      tags: [],
      created_at: "2026-05-15T00:00:00Z"
    }
  ],
  selectedWatchlistId: 42,
  outputs: [buildOutput()],
  outputsLoading: false,
  outputsTotal: 1,
  outputsPage: 1,
  outputsPageSize: 20,
  outputsJobFilter: null,
  outputsRunFilter: null,
  outputPreviewOpen: false,
  selectedOutputId: null,
  setOutputs: vi.fn(),
  setOutputsLoading: vi.fn(),
  setOutputsPage: vi.fn(),
  setOutputsPageSize: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setActiveTab: vi.fn(),
  openRunDetail: vi.fn(),
  openJobForm: vi.fn(),
  openOutputPreview: vi.fn(),
  closeOutputPreview: vi.fn(),
  ...overrides
})

describe("OutputsTab constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.storeStateRef.current = baseState()
    mocks.createWatchlistOutputMock.mockResolvedValue({ id: 999 })
    mocks.downloadWatchlistOutputMock.mockResolvedValue("")
    mocks.downloadWatchlistOutputBinaryMock.mockResolvedValue(new ArrayBuffer(0))
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [{ id: 4, name: "Morning Monitor" }],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [buildOutput()], total: 1, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({ items: [], total: 0, has_more: false })
  })

  it("replaces the Reports table with report cards and preserves report actions at extension width", async () => {
    setViewport(420)
    render(<OutputsTab />)

    expect(await screen.findByTestId("watchlists-outputs-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Reports table" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show advanced filters" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create report" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show failed only" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open failed runs" })).toBeInTheDocument()

    const card = screen.getByTestId("watchlists-output-card-18")
    expect(within(card).getByText("CTI Evidence Report")).toBeInTheDocument()
    expect(within(card).getByText("Evidence snapshot")).toBeInTheDocument()
    expect(within(card).getByText("Ready")).toBeInTheDocument()
    expect(within(card).getByText("2 sources")).toBeInTheDocument()
    expect(within(card).getByText("3 alerts")).toBeInTheDocument()
    expect(within(card).getByText(/email Failed/)).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Morning Monitor" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "#91" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Preview" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Download" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Regenerate" })).toBeInTheDocument()

    fireEvent.click(within(card).getByRole("button", { name: "Morning Monitor" }))
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenCalledWith("jobs")
    expect(mocks.storeStateRef.current.openJobForm).toHaveBeenCalledWith(4)

    fireEvent.click(within(card).getByRole("button", { name: "#91" }))
    expect(mocks.storeStateRef.current.setRunsJobFilter).toHaveBeenCalledWith(4)
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenLastCalledWith("runs")
    expect(mocks.storeStateRef.current.openRunDetail).toHaveBeenCalledWith(91)
  })

  it("preserves the desktop Reports table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<OutputsTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Reports table" })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-outputs-constrained-list")).not.toBeInTheDocument()
  })
})
