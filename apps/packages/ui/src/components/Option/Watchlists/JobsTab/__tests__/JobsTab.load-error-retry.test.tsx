import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { JobsTab } from "../JobsTab"

const mocks = vi.hoisted(() => ({
  fetchWatchlistJobsMock: vi.fn(),
  getWatchlistJobMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  createWatchlistJobMock: vi.fn(),
  updateWatchlistJobMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  restoreWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> },
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, loading, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
      {...rest}
    >
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [], "aria-label": ariaLabel }: any) => (
    <table data-testid="jobs-table" role="table" aria-label={ariaLabel}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const key = String(column.key ?? column.dataIndex ?? columnIndex)
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={key}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  const Popconfirm = ({ children }: any) => <>{children}</>
  const Space = ({ children }: any) => <>{children}</>
  const Switch = ({ "aria-label": ariaLabel }: any) => (
    <button type="button" aria-label={ariaLabel}>
      switch
    </button>
  )
  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Alert = ({ title, description, action }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      {action}
    </div>
  )

  return {
    Button,
    Table,
    Popconfirm,
    Space,
    Switch,
    Tag,
    Tooltip,
    Alert,
    message: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistJob: (...args: any[]) => mocks.createWatchlistJobMock(...args),
  deleteWatchlistJob: (...args: any[]) => mocks.deleteWatchlistJobMock(...args),
  fetchWatchlistGroups: (...args: any[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  getWatchlistJob: (...args: any[]) => mocks.getWatchlistJobMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  restoreWatchlistJob: (...args: any[]) => mocks.restoreWatchlistJobMock(...args),
  triggerWatchlistRun: (...args: any[]) => mocks.triggerWatchlistRunMock(...args),
  updateWatchlistJob: (...args: any[]) => mocks.updateWatchlistJobMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../JobFormModal", () => ({
  JobFormModal: ({ open, initialValues }: { open: boolean; initialValues?: { name: string } }) =>
    open ? <div data-testid="job-form">{initialValues?.name ?? "Create monitor"}</div> : null
}))

vi.mock("../JobPreviewModal", () => ({
  JobPreviewModal: () => null
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  jobs: [],
  jobsLoading: false,
  jobsTotal: 0,
  jobsPage: 1,
  jobsPageSize: 20,
  jobFormOpen: false,
  jobFormEditId: null,
  setJobs: vi.fn(),
  setJobsLoading: vi.fn(),
  setJobsPage: vi.fn(),
  setJobsPageSize: vi.fn(),
  openJobForm: vi.fn(),
  closeJobForm: vi.fn(),
  addJob: vi.fn(),
  updateJobInList: vi.fn(),
  removeJob: vi.fn(),
  addRun: vi.fn(),
  ...overrides
})

const buildJob = () => ({
  id: 5,
  name: "Morning monitor",
  description: "Daily run",
  active: true,
  scope: { sources: [1] },
  job_filters: { filters: [] },
  schedule_expr: "0 9 * * *",
  timezone: "UTC",
  output_prefs: {},
  created_at: "2026-02-18T00:00:00Z",
  updated_at: "2026-02-18T00:00:00Z",
  last_run_at: null,
  next_run_at: null
})

describe("JobsTab load-error retry", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.createWatchlistJobMock.mockResolvedValue({})
    mocks.updateWatchlistJobMock.mockResolvedValue({})
    mocks.deleteWatchlistJobMock.mockResolvedValue({})
    mocks.restoreWatchlistJobMock.mockResolvedValue({})
    mocks.triggerWatchlistRunMock.mockResolvedValue({})
  })

  it("renders contextual load error and retries monitor load", async () => {
    mocks.fetchWatchlistJobsMock
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce({ items: [], total: 0, has_more: false })

    render(<JobsTab />)

    expect(screen.getByRole("table", { name: "Monitors table" })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText("Could not load Monitors.")).toBeInTheDocument()
      expect(screen.getByText("Check server connection and try again. Details: Failed to fetch")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(mocks.fetchWatchlistJobsMock).toHaveBeenCalledTimes(2)
    })
  })

  it("shares a monitor request between readers of the same watchlist page", async () => {
    mocks.storeStateRef.current = baseState({ selectedWatchlistId: 7 })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })

    render(<><JobsTab /><JobsTab /></>)

    await waitFor(() => {
      expect(mocks.fetchWatchlistJobsMock).toHaveBeenCalledTimes(1)
      expect(mocks.fetchWatchlistJobsMock).toHaveBeenCalledWith(
        { watchlist_id: 7, page: 1, size: 20 },
        expect.any(AbortSignal)
      )
    })
  })

  it("shows explicit active-state text next to the monitor toggle", async () => {
    const job = buildJob()
    mocks.storeStateRef.current = baseState({
      jobs: [job],
      jobsTotal: 1
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [job],
      total: 1,
      page: 1,
      size: 20,
      has_more: false
    })

    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByText("Enabled")).toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: "Toggle active for Morning monitor" })).toBeInTheDocument()
  })

  it("opens the existing monitor when a relationship points outside the current page", async () => {
    mocks.storeStateRef.current = baseState({ jobFormOpen: true, jobFormEditId: 5 })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getWatchlistJobMock.mockResolvedValue(buildJob())

    render(<JobsTab />)

    expect(screen.queryByText("Create monitor")).not.toBeInTheDocument()
    expect(await screen.findByText("Morning monitor")).toBeInTheDocument()
    expect(mocks.getWatchlistJobMock).toHaveBeenCalledWith(5, expect.any(AbortSignal))
  })
})
