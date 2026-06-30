import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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
  const Button = ({ children, onClick, loading, ...rest }: any) => (
    <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <table data-testid="outputs-table">
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const key = String(column.key ?? column.dataIndex ?? columnIndex)
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render
                ? column.render(value, record, rowIndex)
                : value
              return <td key={key}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )

  return {
    Button,
    Input: ({ value, onChange, ...rest }: any) => (
      <input value={value || ""} onChange={(event) => onChange?.(event)} {...rest} />
    ),
    InputNumber: ({ value, onChange, ...rest }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.currentTarget.value))}
        {...rest}
      />
    ),
    Modal: ({ open, children }: any) => (open ? <div>{children}</div> : null),
    Select: ({ value, onChange, options = [], allowClear, ...rest }: any) => (
      <select
        data-testid={rest["data-testid"] || "antd-select"}
        value={value == null ? "" : String(value)}
        onChange={(event) => {
          const next = event.currentTarget.value
          onChange?.(next === "" ? null : next)
        }}
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

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: () => null
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  outputs: [
    {
      id: 44,
      job_id: 7,
      run_id: 101,
      title: "Daily Brief",
      format: "md",
      type: "briefing",
      content: "hello",
      metadata: {},
      created_at: "2026-02-23T08:00:00Z",
      expires_at: null,
      expired: false
    }
  ],
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
  openRunDetail: vi.fn(),
  setActiveTab: vi.fn(),
  openJobForm: vi.fn(),
  openOutputPreview: vi.fn(),
  closeOutputPreview: vi.fn(),
  ...overrides
})

describe("OutputsTab relationship jump actions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [{ id: 7, name: "Morning Monitor" }],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistOutputMock.mockResolvedValue({})
    mocks.downloadWatchlistOutputMock.mockResolvedValue("")
    mocks.downloadWatchlistOutputBinaryMock.mockResolvedValue(new ArrayBuffer(0))
  })

  it("deep-links output rows to monitor and run destinations", () => {
    render(<OutputsTab />)

    fireEvent.click(screen.getByTestId("watchlists-output-open-job-44"))
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenCalledWith("jobs")
    expect(mocks.storeStateRef.current.openJobForm).toHaveBeenCalledWith(7)

    fireEvent.click(screen.getByTestId("watchlists-outputs-advanced-toggle"))
    const statusFilterCallsBefore =
      mocks.storeStateRef.current.setRunsStatusFilter.mock.calls.length
    fireEvent.click(screen.getByTestId("watchlists-output-open-run-44"))
    expect(mocks.storeStateRef.current.setRunsJobFilter).toHaveBeenCalledWith(7)
    expect(mocks.storeStateRef.current.setRunsStatusFilter).toHaveBeenCalledTimes(
      statusFilterCallsBefore + 1
    )
    expect(mocks.storeStateRef.current.setRunsStatusFilter).toHaveBeenLastCalledWith(null)
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenLastCalledWith("runs")
    expect(mocks.storeStateRef.current.openRunDetail).toHaveBeenCalledWith(101)
  })
})
