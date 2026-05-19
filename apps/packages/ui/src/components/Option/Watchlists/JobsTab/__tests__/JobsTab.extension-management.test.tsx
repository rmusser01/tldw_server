// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { JobsTab } from "../JobsTab"

const mocks = vi.hoisted(() => ({
  deleteWatchlistJobMock: vi.fn(),
  fetchJobRunsMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  restoreWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  updateWatchlistJobMock: vi.fn(),
  modalConfirmMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
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
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
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
  const Switch = ({ checked, onChange, "aria-label": ariaLabel }: any) => (
    <button type="button" aria-label={ariaLabel} aria-pressed={Boolean(checked)} onClick={() => onChange?.(!checked)} />
  )
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [] }: any) => (
    <table role="table" aria-label={ariaLabel}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? columnIndex)}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  return {
    Alert: ({ title, description, action }: any) => (
      <div>
        <div>{title}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    Button,
    Modal: { confirm: (...args: unknown[]) => mocks.modalConfirmMock(...args) },
    Space: ({ children }: any) => <>{children}</>,
    Switch,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children, title }: any) => (
      <>
        {children}
        {title ? <span>{title}</span> : null}
      </>
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

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  deleteWatchlistJob: (...args: unknown[]) => mocks.deleteWatchlistJobMock(...args),
  fetchJobRuns: (...args: unknown[]) => mocks.fetchJobRunsMock(...args),
  fetchWatchlistGroups: (...args: unknown[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchWatchlistSourcesMock(...args),
  restoreWatchlistJob: (...args: unknown[]) => mocks.restoreWatchlistJobMock(...args),
  triggerWatchlistRun: (...args: unknown[]) => mocks.triggerWatchlistRunMock(...args),
  updateWatchlistJob: (...args: unknown[]) => mocks.updateWatchlistJobMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../JobFormModal", () => ({
  JobFormModal: () => null
}))

vi.mock("../JobPreviewModal", () => ({
  JobPreviewModal: () => null
}))

vi.mock("../../shared", () => ({
  CronDisplay: ({ expression }: { expression?: string | null }) => <span>{expression || "No schedule"}</span>
}))

const buildJob = (overrides: Record<string, unknown> = {}) => ({
  id: 41,
  name: "Healthcare ransomware monitor",
  description: "Daily healthcare ransomware scan",
  active: true,
  scope: {
    sources: [101, 202],
    groups: [7],
    tags: ["ransomware"]
  },
  job_filters: {
    filters: [
      {
        type: "keyword",
        action: "include",
        value: { keywords: ["ransomware"] }
      }
    ]
  },
  schedule_expr: "0 9 * * *",
  timezone: "UTC",
  output_prefs: {
    template: { default_name: "CTI Brief" },
    deliveries: { chatbook: { path: "/briefings/cti.chatbook" } }
  },
  last_run_at: "2026-05-15T14:00:00Z",
  next_run_at: "2026-05-16T14:00:00Z",
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z",
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  jobs: [buildJob()],
  jobsLoading: false,
  jobsTotal: 1,
  jobsPage: 1,
  jobsPageSize: 20,
  jobFormOpen: false,
  jobFormEditId: null,
  selectedWatchlistId: 42,
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

describe("JobsTab constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    const job = buildJob()
    mocks.storeStateRef.current = baseState({ jobs: [job], jobsTotal: 1 })
    mocks.deleteWatchlistJobMock.mockResolvedValue({})
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({
      items: [{ id: 7, name: "Healthcare", parent_group_id: null }],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [job], total: 1, has_more: false })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [
        { id: 101, name: "CISA KEV RSS" },
        { id: 202, name: "Health ISAC" }
      ],
      total: 2,
      has_more: false
    })
    mocks.restoreWatchlistJobMock.mockResolvedValue({})
    mocks.triggerWatchlistRunMock.mockResolvedValue({ id: 500, started_at: "2026-05-15T15:00:00Z" })
    mocks.updateWatchlistJobMock.mockResolvedValue(job)
  })

  it("replaces the monitors table with full monitor management cards at extension width", async () => {
    setViewport(420)
    render(<JobsTab />)

    expect(await screen.findByTestId("watchlists-jobs-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Monitors table" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show advanced details" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add Monitor" })).toBeInTheDocument()

    const card = screen.getByTestId("watchlists-job-card-41")
    expect(within(card).getByText("Healthcare ransomware monitor")).toBeInTheDocument()
    expect(within(card).getByText("Daily healthcare ransomware scan")).toBeInTheDocument()
    expect(within(card).getByText("0 9 * * *")).toBeInTheDocument()
    expect(within(card).getByTestId("job-scope-summary-41")).toHaveTextContent("2 feeds, 1 group, 1 tag")
    expect(within(card).getByTestId("job-filters-summary-41")).toHaveTextContent("1")
    expect(within(card).getByText("Output linkage")).toBeInTheDocument()
    expect(within(card).getByTestId("job-output-linkage-41")).toHaveTextContent("Template: CTI Brief")
    expect(within(card).getByText("Last run")).toBeInTheDocument()
    expect(within(card).getByText("Next run")).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Toggle active for Healthcare ransomware monitor" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Run Now" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Preview Healthcare ransomware monitor" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Edit Healthcare ransomware monitor" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Delete Healthcare ransomware monitor" })).toBeInTheDocument()
  })

  it("preserves the desktop monitors table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Monitors table" })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-jobs-constrained-list")).not.toBeInTheDocument()
  })
})
