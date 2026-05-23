import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

const mocks = vi.hoisted(() => ({
  fetchJobRunsMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
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

  const Button = ({ children, onClick, loading, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
      {...rest}
    >
      {children}
    </button>
  )

  const Dropdown = ({ children }: any) => <>{children}</>
  const Table = () => <div data-testid="runs-table" />
  const Progress = () => <div />
  const Tooltip = ({ children }: any) => <>{children}</>
  const Space = ({ children }: any) => <>{children}</>
  const Alert = ({ title, description, action }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      {action}
    </div>
  )

  return {
    Select,
    Button,
    Dropdown,
    Table,
    Progress,
    Tooltip,
    Space,
    Alert,
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
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../RunDetailDrawer", () => ({
  RunDetailDrawer: () => null
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  runs: [],
  runsLoading: false,
  runsTotal: 0,
  runsPage: 1,
  runsPageSize: 20,
  runsJobFilter: null,
  runsStatusFilter: null,
  pollingActive: false,
  runDetailOpen: false,
  selectedRunId: null,
  setRuns: vi.fn(),
  setRunsLoading: vi.fn(),
  setRunsPage: vi.fn(),
  setRunsPageSize: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setPollingActive: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  updateRunInList: vi.fn(),
  ...overrides
})

describe("RunsTab load-error retry", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.exportRunsCsvMock.mockResolvedValue("")
  })

  it("shows contextual load error with retry and reloads on retry click", async () => {
    mocks.fetchWatchlistRunsMock
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce({ items: [], total: 0, has_more: false })

    render(<RunsTab />)

    await waitFor(() => {
      const loadErrorTitle = screen.getByText("Could not load Activity.")
      expect(loadErrorTitle).toBeInTheDocument()
      expect(loadErrorTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
      expect(screen.getByText("Check server connection and try again. Details: Failed to fetch")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(mocks.fetchWatchlistRunsMock).toHaveBeenCalledTimes(2)
    })
  })

  it("shows persistent reliability attention with deep-link actions for failed and stalled runs", async () => {
    const openRunDetail = vi.fn()
    const setRunsStatusFilter = vi.fn()
    mocks.storeStateRef.current = baseState({
      openRunDetail,
      setRunsStatusFilter,
      runs: [
        {
          id: 71,
          job_id: 3,
          status: "failed",
          started_at: "2026-02-18T08:00:00Z",
          finished_at: "2026-02-18T08:05:00Z",
          error_msg: "timeout while fetching",
          stats: {}
        },
        {
          id: 72,
          job_id: 4,
          status: "running",
          started_at: "2026-02-18T06:00:00Z",
          finished_at: null,
          error_msg: null,
          stats: {}
        }
      ]
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })

    render(<RunsTab />)

    await waitFor(() => {
      const attentionTitle = screen.getByText("Reliability attention required")
      expect(attentionTitle).toBeInTheDocument()
      expect(attentionTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
      expect(
        screen.getByText(
          "1 failed run and 1 stalled run need review. The source request timed out. Retry, or lower concurrency for this source."
        )
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "View newest failed run" }))
    expect(openRunDetail).toHaveBeenCalledWith(71)

    fireEvent.click(screen.getByRole("button", { name: "Show failed runs" }))
    expect(setRunsStatusFilter).toHaveBeenCalledWith("failed")
  })
})
