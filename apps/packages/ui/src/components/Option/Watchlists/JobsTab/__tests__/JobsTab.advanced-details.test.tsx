// @vitest-environment jsdom

import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { JobsTab } from "../JobsTab"

const ADVANCED_COLUMNS_STORAGE_KEY = "watchlists:jobs:advanced-columns:v1"

const mocks = vi.hoisted(() => ({
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  createWatchlistJobMock: vi.fn(),
  updateWatchlistJobMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  restoreWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
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
  const Button = ({ children, onClick, loading, danger: _danger, ...rest }: any) => (
    <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  const Popconfirm = ({ children }: any) => <>{children}</>
  const Space = ({ children }: any) => <>{children}</>
  const Switch = () => null
  const Tag = ({ children, ...rest }: any) => <span {...rest}>{children}</span>
  const Tooltip = ({ title, children }: any) => (
    <div>
      {children}
      {title ? <div>{title}</div> : null}
    </div>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <table data-testid="jobs-table">
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

  return {
    Button,
    Popconfirm,
    Space,
    Switch,
    Table,
    Tag,
    Tooltip,
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

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistJob: (...args: any[]) => mocks.createWatchlistJobMock(...args),
  deleteWatchlistJob: (...args: any[]) => mocks.deleteWatchlistJobMock(...args),
  fetchWatchlistGroups: (...args: any[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  restoreWatchlistJob: (...args: any[]) => mocks.restoreWatchlistJobMock(...args),
  triggerWatchlistRun: (...args: any[]) => mocks.triggerWatchlistRunMock(...args),
  updateWatchlistJob: (...args: any[]) => mocks.updateWatchlistJobMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("../JobFormModal", () => ({
  JobFormModal: () => null
}))

vi.mock("../JobPreviewModal", () => ({
  JobPreviewModal: () => null
}))

vi.mock("../../shared", () => ({
  CronDisplay: () => <span>Cron</span>
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

const buildJob = (id: number) => ({
  id,
  name: `Monitor ${id}`,
  description: "Daily scan",
  active: true,
  scope: {
    sources: [1, 2],
    groups: [7],
    tags: ["tech"]
  },
  job_filters: {
    filters: [
      {
        type: "keyword",
        action: "include",
        value: { keywords: ["ai"] }
      }
    ]
  },
  schedule_expr: "0 9 * * *",
  timezone: "UTC",
  output_prefs: {},
  created_at: "2026-02-18T00:00:00Z",
  updated_at: "2026-02-18T00:00:00Z",
  last_run_at: null,
  next_run_at: null
})

describe("JobsTab advanced details disclosure", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem(ADVANCED_COLUMNS_STORAGE_KEY)

    const job = buildJob(77)

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
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.createWatchlistJobMock.mockResolvedValue({})
    mocks.updateWatchlistJobMock.mockResolvedValue({})
    mocks.deleteWatchlistJobMock.mockResolvedValue({})
    mocks.restoreWatchlistJobMock.mockResolvedValue({})
    mocks.triggerWatchlistRunMock.mockResolvedValue({})
  })

  afterEach(() => {
    localStorage.removeItem(ADVANCED_COLUMNS_STORAGE_KEY)
  })

  it("starts with compact summaries and expands advanced columns on demand", async () => {
    render(<JobsTab />)

    expect(await screen.findByTestId("job-compact-summary-77")).toHaveTextContent("2 feeds, 1 group, 1 tag • 1 filters")
    expect(screen.queryByText("2 feeds, 1 group, 1 tag")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-jobs-advanced-toggle"))

    await waitFor(() => {
      expect(screen.getByText("2 feeds, 1 group, 1 tag")).toBeInTheDocument()
    })
    expect(localStorage.getItem(ADVANCED_COLUMNS_STORAGE_KEY)).toBe("1")
  })

  it("shows an audio chip when monitor output preferences enable audio briefing", async () => {
    const audioJob = {
      id: 88,
      name: "Audio Monitor",
      description: "Audio digest",
      active: true,
      scope: { sources: [1] },
      job_filters: { filters: [] },
      schedule_expr: "0 9 * * *",
      timezone: "UTC",
      output_prefs: { generate_audio: true },
      created_at: "2026-02-18T00:00:00Z",
      updated_at: "2026-02-18T00:00:00Z",
      last_run_at: null,
      next_run_at: null
    }

    mocks.storeStateRef.current = baseState({
      jobs: [audioJob],
      jobsTotal: 1
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [audioJob],
      total: 1,
      page: 1,
      size: 20,
      has_more: false
    })

    render(<JobsTab />)

    expect(await screen.findByTestId("job-audio-enabled-chip-88")).toHaveTextContent("Audio on")
  })

  it.each([1, 10, 50])(
    "keeps monitor table summaries actionable for %i monitors in compact and advanced density modes",
    async (monitorCount) => {
      const jobs = Array.from({ length: monitorCount }, (_unused, index) => buildJob(1000 + index))
      mocks.storeStateRef.current = baseState({
        jobs,
        jobsTotal: monitorCount
      })
      mocks.fetchWatchlistJobsMock.mockResolvedValue({
        items: jobs,
        total: monitorCount,
        page: 1,
        size: monitorCount,
        has_more: false
      })

      render(<JobsTab />)

      await waitFor(() => {
        expect(screen.getByTestId("job-compact-summary-1000")).toBeInTheDocument()
      })
      expect(document.querySelectorAll("[data-testid^='job-compact-summary-']")).toHaveLength(monitorCount)

      fireEvent.click(screen.getByTestId("watchlists-jobs-advanced-toggle"))

      await waitFor(() => {
        expect(screen.getByTestId("job-scope-summary-1000")).toHaveTextContent("2 feeds, 1 group, 1 tag")
      })
      expect(document.querySelectorAll("[data-testid^='job-scope-summary-']")).toHaveLength(monitorCount)
      expect(localStorage.getItem(ADVANCED_COLUMNS_STORAGE_KEY)).toBe("1")
    }
  )
})
