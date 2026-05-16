// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { JobsTab } from "../JobsTab"

const ADVANCED_COLUMNS_STORAGE_KEY = "watchlists:jobs:advanced-columns:v1"

const mocks = vi.hoisted(() => ({
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  createWatchlistJobMock: vi.fn(),
  restoreWatchlistJobMock: vi.fn(),
  updateWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  showUndoNotificationMock: vi.fn(),
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
  const Button = ({ children }: any) => <button type="button">{children}</button>
  const Popconfirm = ({ children }: any) => <>{children}</>
  const Switch = () => null
  const Space = ({ children }: any) => <>{children}</>
  const Tag = ({ children }: any) => <span>{children}</span>
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
    Popconfirm,
    Space,
    Switch,
    Table,
    Tag,
    Tooltip,
    message: {
      success: mocks.messageSuccessMock,
      error: mocks.messageErrorMock,
      warning: vi.fn()
    }
  }
})

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: mocks.showUndoNotificationMock
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

describe("JobsTab scope/filter summaries", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem(ADVANCED_COLUMNS_STORAGE_KEY, "1")

    const job = {
      id: 77,
      name: "Morning Monitor",
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
          },
          {
            type: "author",
            action: "exclude",
            value: { authors: ["spam-bot"] }
          }
        ]
      },
      schedule_expr: "0 9 * * *",
      timezone: "UTC",
      output_prefs: {
        template: { default_name: "briefing_markdown" },
        generate_audio: true
      },
      created_at: "2026-02-18T00:00:00Z",
      updated_at: "2026-02-18T00:00:00Z",
      last_run_at: null,
      next_run_at: null
    }

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

    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [
        { id: 1, name: "TechCrunch" },
        { id: 2, name: "Ars Technica" }
      ],
      total: 2,
      page: 1,
      size: 200,
      has_more: false
    })

    mocks.fetchWatchlistGroupsMock.mockResolvedValue({
      items: [{ id: 7, name: "Daily News" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
  })

  afterEach(() => {
    localStorage.removeItem(ADVANCED_COLUMNS_STORAGE_KEY)
  })

  it("renders compact scope and filter summaries and exposes names in tooltip content", async () => {
    render(<JobsTab />)

    expect(await screen.findByText("2 feeds, 1 group, 1 tag")).toBeInTheDocument()
    expect(screen.getByText("Include keyword: ai (1 more)")).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText("Feeds: TechCrunch, Ars Technica")).toBeInTheDocument()
      expect(screen.getByText("Groups: Daily News")).toBeInTheDocument()
      expect(screen.getByText("Tags: tech")).toBeInTheDocument()
      expect(screen.getByText("Exclude author: spam-bot")).toBeInTheDocument()
      expect(screen.getByTestId("job-output-linkage-77")).toHaveTextContent(
        "Output linkage: Template: briefing_markdown • Audio: enabled"
      )
    })
  })
})
