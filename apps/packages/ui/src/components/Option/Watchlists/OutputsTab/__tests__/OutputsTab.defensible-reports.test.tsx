// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OutputsTab } from "../OutputsTab"

const mocks = vi.hoisted(() => ({
  createWatchlistOutputMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  downloadWatchlistOutputMock: vi.fn(),
  downloadWatchlistOutputBinaryMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, loading, disabled, "aria-label": ariaLabel, ...rest }: any) => (
    <button
      type="button"
      aria-label={ariaLabel}
      disabled={Boolean(loading || disabled)}
      onClick={() => onClick?.()}
      {...rest}
    >
      {children || ariaLabel}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="outputs-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.id ?? rowIndex} data-testid={`outputs-row-${record.id}`}>
          {columns.map((column: any, columnIndex: number) => {
            const key = String(column.key ?? column.dataIndex ?? columnIndex)
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            const content = column.render ? column.render(value, record, rowIndex) : value
            return <div key={key}>{content}</div>
          })}
        </div>
      ))}
    </div>
  )

  return {
    Alert: ({ title, message, description, action }: any) => (
      <div role="status">
        <div>{title ?? message}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    Button,
    Input: ({ value, onChange, ...rest }: any) => (
      <input value={value || ""} onChange={(event) => onChange?.(event)} {...rest} />
    ),
    InputNumber: ({ value, onChange, ...rest }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(event.currentTarget.value ? Number(event.currentTarget.value) : null)}
        {...rest}
      />
    ),
    Modal: ({ open, children }: any) => (open ? <div data-testid="antd-modal">{children}</div> : null),
    Select: ({ value, onChange, options = [], allowClear, ...rest }: any) => (
      <select
        data-testid={rest["data-testid"] || "antd-select"}
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
    ),
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
  createWatchlistOutput: (...args: any[]) => mocks.createWatchlistOutputMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistOutputs: (...args: any[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistTemplates: (...args: any[]) => mocks.fetchWatchlistTemplatesMock(...args),
  downloadWatchlistOutput: (...args: any[]) => mocks.downloadWatchlistOutputMock(...args),
  downloadWatchlistOutputBinary: (...args: any[]) => mocks.downloadWatchlistOutputBinaryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../ReportBuilderDrawer", () => ({
  ReportBuilderDrawer: ({ open, selectedWatchlist }: any) =>
    open ? <div data-testid="report-builder-open">Builder for {selectedWatchlist?.name}</div> : null
}))

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: ({ output, open }: any) =>
    open ? <div data-testid="output-preview-open">Previewing {output?.title}</div> : null
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
        status: "sent"
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
  outputs: [],
  outputsLoading: false,
  outputsTotal: 0,
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

describe("OutputsTab defensible reports UI", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput()],
      outputsTotal: 1
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.createWatchlistOutputMock.mockResolvedValue({ id: 999 })
  })

  it("shows readiness, evidence availability, source count, and alert count in Reports rows", async () => {
    render(<OutputsTab />)

    expect(screen.getByText("CTI Evidence Report")).toBeInTheDocument()
    expect(screen.getByText("Ready")).toBeInTheDocument()
    expect(screen.getByText("Evidence snapshot")).toBeInTheDocument()
    expect(screen.getByText("2 sources")).toBeInTheDocument()
    expect(screen.getByText("3 alerts")).toBeInTheDocument()
  })

  it("opens the report builder from the Reports toolbar", async () => {
    render(<OutputsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create report" }))

    expect(await screen.findByTestId("report-builder-open")).toHaveTextContent(
      "Builder for Healthcare ransomware"
    )
  })

  it("keeps evidence access tied to the selected output preview", async () => {
    const openOutputPreview = vi.fn()
    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput()],
      outputsTotal: 1,
      openOutputPreview
    })

    render(<OutputsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    expect(openOutputPreview).toHaveBeenCalledWith(18)
  })
})
