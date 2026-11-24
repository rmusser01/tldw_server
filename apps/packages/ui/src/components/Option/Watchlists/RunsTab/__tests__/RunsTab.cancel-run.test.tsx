import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

const mocks = vi.hoisted(() => ({
  fetchJobRunsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  cancelWatchlistRunMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) => {
      if (typeof defaultValue === "string") return defaultValue
      return _key
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
        <option key={String(option.value)} value={String(option.value)} disabled={Boolean(option.disabled)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Button = ({ children, onClick, loading, icon, ...rest }: any) => (
    <button
      type="button"
      data-testid={rest["data-testid"]}
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
    >
      {icon}
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="runs-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.id ?? rowIndex} data-testid={`row-${record.id ?? rowIndex}`}>
          {columns.map((column: any, columnIndex: number) => {
            const key = String(column.key ?? column.dataIndex ?? columnIndex)
            if (key !== "actions") return null
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            return (
              <div key={key}>
                {column.render ? column.render(value, record, rowIndex) : null}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )

  return {
    Select,
    Dropdown: ({ children }: any) => <>{children}</>,
    Button,
    Table,
    Progress: () => <div />,
    Tooltip: ({ children }: any) => <>{children}</>,
    Space: ({ children }: any) => <>{children}</>,
    message: {
      success: mocks.messageSuccessMock,
      error: mocks.messageErrorMock,
      warning: vi.fn(),
    },
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  fetchJobRuns: (...args: any[]) => mocks.fetchJobRunsMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args),
  cancelWatchlistRun: (...args: any[]) => mocks.cancelWatchlistRunMock(...args),
  exportRunsCsv: (...args: any[]) => mocks.exportRunsCsvMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../RunDetailDrawer", () => ({
  RunDetailDrawer: () => null
}))

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
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setActiveTab: vi.fn(),
  setPollingActive: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  ...overrides,
})

describe("RunsTab cancellation controls", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.exportRunsCsvMock.mockResolvedValue("")
  })

  it("shows cancel action for running runs and updates status on successful cancel", async () => {
    const runningRun = {
      id: 101,
      job_id: 10,
      status: "running",
      started_at: new Date().toISOString(),
      finished_at: null,
      stats: {}
    }
    mocks.fetchWatchlistRunsMock.mockResolvedValue({
      items: [runningRun],
      total: 1,
      has_more: false
    })
    mocks.cancelWatchlistRunMock.mockImplementation(async () => {
      mocks.fetchWatchlistRunsMock.mockResolvedValue({
        items: [{ ...runningRun, status: "cancelled" }],
        total: 1,
        has_more: false
      })
      return {
        run_id: 101,
        status: "cancelled",
        cancelled: true
      }
    })

    render(<RunsTab />)

    const cancelButton = await screen.findByTestId("watchlists-run-cancel-101")
    fireEvent.click(cancelButton)

    await waitFor(() => {
      expect(mocks.cancelWatchlistRunMock).toHaveBeenCalledWith(101)
      expect(screen.queryByTestId("watchlists-run-cancel-101")).not.toBeInTheDocument()
      expect(mocks.messageSuccessMock).toHaveBeenCalledWith("Run cancelled")
    })
  })

  it("does not show cancel action for completed runs", async () => {
    const completedRun = {
      id: 202,
      job_id: 10,
      status: "completed",
      started_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
      stats: {}
    }
    mocks.fetchWatchlistRunsMock.mockResolvedValue({
      items: [completedRun],
      total: 1,
      has_more: false
    })

    render(<RunsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-run-open-outputs-202")).toBeInTheDocument()
      expect(screen.queryByTestId("watchlists-run-cancel-202")).not.toBeInTheDocument()
    })
  })

  it("opens Reports filtered to a run from Activity row action", async () => {
    const setOutputsJobFilter = vi.fn()
    const setOutputsRunFilter = vi.fn()
    const setActiveTab = vi.fn()
    const completedRun = {
      id: 303,
      job_id: 44,
      status: "completed",
      started_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
      stats: {
        items_ingested: 12
      }
    }
    mocks.storeStateRef.current = baseState({
      setOutputsJobFilter,
      setOutputsRunFilter,
      setActiveTab
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({
      items: [completedRun],
      total: 1,
      has_more: false
    })

    render(<RunsTab />)

    fireEvent.click(await screen.findByTestId("watchlists-run-open-outputs-303"))

    expect(setOutputsJobFilter).toHaveBeenCalledWith(44)
    expect(setOutputsRunFilter).toHaveBeenCalledWith(303)
    expect(setActiveTab).toHaveBeenCalledWith("outputs")
  })
})
