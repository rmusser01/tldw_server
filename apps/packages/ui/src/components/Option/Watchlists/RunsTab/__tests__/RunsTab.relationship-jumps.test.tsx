import React from "react"
import { fireEvent, screen } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

const mocks = vi.hoisted(() => ({
  fetchJobRunsMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
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
  const Select = ({ value, onChange, options = [], allowClear, ...rest }: any) => (
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
  )

  const Button = ({ children, onClick, loading, ...rest }: any) => (
    <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <table data-testid="runs-table">
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
    Select,
    Button,
    Dropdown: ({ children }: any) => <>{children}</>,
    Table,
    Progress: () => <div />,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    Space: ({ children }: any) => <>{children}</>,
    Alert: ({ title, description, action }: any) => (
      <div>
        <div>{title}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    message: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  fetchJobRuns: (...args: any[]) => mocks.fetchJobRunsMock(...args),
  exportRunsCsv: (...args: any[]) => mocks.exportRunsCsvMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args),
  cancelWatchlistRun: vi.fn()
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../RunDetailDrawer", () => ({
  RunDetailDrawer: () => null
}))

const runs = [
    {
      id: 33,
      job_id: 5,
      status: "completed",
      started_at: "2026-02-18T00:00:00Z",
      finished_at: "2026-02-18T00:05:00Z",
      stats: {
        items_found: 12,
        items_ingested: 8,
        items_filtered: 3,
        items_errored: 0
      }
    }
  ]

const baseState = (overrides: Record<string, unknown> = {}) => ({
  runsPage: 1,
  runsPageSize: 20,
  runsJobFilter: null,
  runsStatusFilter: null,
  runDetailOpen: false,
  selectedRunId: null,
  setRunsPage: vi.fn(),
  setRunsPageSize: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setPollingActive: vi.fn(),
  setActiveTab: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  ...overrides
})

describe("RunsTab relationship jump actions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: runs, total: 1, has_more: false })
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.exportRunsCsvMock.mockResolvedValue("")
  })

  it("opens reports destination with run and monitor filters", async () => {
    render(<RunsTab />)

    fireEvent.click(await screen.findByTestId("watchlists-run-open-outputs-33"))

    expect(mocks.storeStateRef.current.setOutputsJobFilter).toHaveBeenCalledWith(5)
    expect(mocks.storeStateRef.current.setOutputsRunFilter).toHaveBeenCalledWith(33)
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenCalledWith("outputs")
  })
})
