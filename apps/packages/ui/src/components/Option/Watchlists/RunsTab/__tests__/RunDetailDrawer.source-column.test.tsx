// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunDetailDrawer } from "../RunDetailDrawer"

const mocks = vi.hoisted(() => ({
  getRunDetailsMock: vi.fn(),
  fetchScrapedItemsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  updateScrapedItemMock: vi.fn(),
  exportRunTalliesCsvMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  cancelWatchlistRunMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  updateRunInListMock: vi.fn(),
  addRunMock: vi.fn(),
  setActiveTabMock: vi.fn(),
  setOutputsJobFilterMock: vi.fn(),
  setOutputsRunFilterMock: vi.fn(),
  openJobFormMock: vi.fn(),
  translateMock: vi.fn(
    (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      const template =
        typeof defaultValue === "string" ? defaultValue : key
      return template.replace(/\{\{\s*([.\w-]+)\s*\}\}/g, (_match, token) => {
        const value = options?.[token]
        return value == null ? "" : String(value)
      })
    }
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) =>
      mocks.translateMock(key, defaultValue, options)
  })
}))

vi.mock("antd", () => {
  const Drawer = ({ open, children, title, extra }: any) =>
    open ? (
      <div>
        <div>{title}</div>
        {extra}
        {children}
      </div>
    ) : null

  const Tabs = ({ items = [] }: any) => (
    <div>
      {items.map((item: any) => (
        <section key={item.key}>
          <h2>{item.label}</h2>
          {item.children}
        </section>
      ))}
    </div>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <table>
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

  const DescriptionsComponent = ({ children }: any) => <div>{children}</div>
  ;(DescriptionsComponent as any).Item = ({ children }: any) => <div>{children}</div>

  const Switch = () => null
  const Tag = ({ children }: any) => <span>{children}</span>
  const Button = ({ children, onClick }: any) => (
    <button type="button" onClick={() => onClick?.()}>
      {children}
    </button>
  )
  const Tooltip = ({ title, children }: any) => (
    <span>
      {children}
      {title ? <span>{title}</span> : null}
    </span>
  )

  return {
    Alert: ({ title, message, description, action, children }: any) => (
      <div>
        {title ? <div>{title}</div> : null}
        {message ? <div>{message}</div> : null}
        {description ? <div>{description}</div> : null}
        {action}
        {children}
      </div>
    ),
    Button,
    Descriptions: DescriptionsComponent,
    Drawer,
    Empty: ({ description }: any) => <div>{description}</div>,
    Spin: ({ children }: any) => <>{children}</>,
    Switch,
    Table,
    Tabs,
    Tag,
    Tooltip,
    message: {
      error: mocks.messageErrorMock,
      success: mocks.messageSuccessMock,
      warning: vi.fn()
    }
  }
})

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      updateRunInList: mocks.updateRunInListMock,
      addRun: mocks.addRunMock,
      setActiveTab: mocks.setActiveTabMock,
      setOutputsJobFilter: mocks.setOutputsJobFilterMock,
      setOutputsRunFilter: mocks.setOutputsRunFilterMock,
      openJobForm: mocks.openJobFormMock
    })
}))

vi.mock("@/services/watchlists", () => ({
  cancelWatchlistRun: (...args: any[]) => mocks.cancelWatchlistRunMock(...args),
  exportRunTalliesCsv: (...args: any[]) => mocks.exportRunTalliesCsvMock(...args),
  fetchScrapedItems: (...args: any[]) => mocks.fetchScrapedItemsMock(...args),
  fetchWatchlistOutputs: (...args: any[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  getRunDetails: (...args: any[]) => mocks.getRunDetailsMock(...args),
  triggerWatchlistRun: (...args: any[]) => mocks.triggerWatchlistRunMock(...args),
  updateScrapedItem: (...args: any[]) => mocks.updateScrapedItemMock(...args)
}))

vi.mock("@/services/watchlists-stream", () => ({
  buildWatchlistsRunWebSocketUrl: () => "ws://example.test",
  parseWatchlistsRunStreamPayload: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(async () => null)
  }
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

const baseRunDetails = {
  id: 10,
  job_id: 1,
  status: "completed",
  started_at: "2026-02-18T10:00:00Z",
  finished_at: "2026-02-18T10:01:00Z",
  stats: {
    items_found: 1,
    items_ingested: 1,
    items_filtered: 0,
    items_errored: 0
  },
  filter_tallies: {},
  error_msg: null,
  log_text: null,
  log_path: null,
  truncated: false,
  filtered_sample: null
}

const baseItemsResponse = {
  items: [
    {
      id: 501,
      run_id: 10,
      job_id: 1,
      source_id: 5,
      url: "https://example.com/article",
      title: "Test item",
      summary: "Summary",
      content: "Body",
      published_at: "2026-02-18T09:59:00Z",
      tags: [],
      status: "ingested",
      reviewed: false,
      created_at: "2026-02-18T10:00:30Z"
    }
  ],
  total: 1,
  page: 1,
  size: 20,
  has_more: false
}

describe("RunDetailDrawer source column", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.translateMock.mockImplementation(
      (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
        const template =
          typeof defaultValue === "string" ? defaultValue : key
        return template.replace(/\{\{\s*([.\w-]+)\s*\}\}/g, (_match, token) => {
          const value = options?.[token]
          return value == null ? "" : String(value)
        })
      }
    )
    mocks.getRunDetailsMock.mockResolvedValue(baseRunDetails)
    mocks.fetchScrapedItemsMock.mockResolvedValue(baseItemsResponse)
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [{ id: 9001, run_id: 10, job_id: 1 }],
      total: 1,
      page: 1,
      size: 1,
      has_more: false
    })
    mocks.updateScrapedItemMock.mockResolvedValue({ reviewed: true })
    mocks.exportRunTalliesCsvMock.mockResolvedValue("")
    mocks.triggerWatchlistRunMock.mockResolvedValue({
      id: 99,
      job_id: 1,
      status: "pending"
    })
    mocks.cancelWatchlistRunMock.mockResolvedValue({ cancelled: true })
  })

  it("shows source name when source lookup data is available", async () => {
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("TechCrunch")).toBeInTheDocument()
      expect(screen.getByText("Included in briefing")).toBeInTheDocument()
      expect(screen.getByText("Monitor #1 produced 1 report for this run.")).toBeInTheDocument()
      expect(screen.getByText("Run linkage").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    })
  })

  it("shows mapped load failure with retry action", async () => {
    mocks.getRunDetailsMock
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce(baseRunDetails)
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Could not load run details.")).toBeInTheDocument()
      expect(screen.getByText("Check server connection and try again. Details: Failed to fetch")).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    })

    screen.getByRole("button", { name: "Retry" }).click()

    await waitFor(() => {
      expect(mocks.getRunDetailsMock).toHaveBeenCalledTimes(2)
    })
  })

  it("renders warning-level load failures as degraded recovery states", async () => {
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockRejectedValue({
      status: 400,
      message: "invalid input"
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      const callout = screen
        .getByText("Could not load run details.")
        .closest('[data-ds-component="RecoveryCallout"]')
      expect(callout).toBeInTheDocument()
      expect(within(callout as HTMLElement).getByText("Degraded")).toBeInTheDocument()
    })
  })

  it("falls back to source id when source lookup data is unavailable", async () => {
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("#5")).toBeInTheDocument()
    })
  })

  it("opens Reports filtered to this run from the detail summary", async () => {
    const onClose = vi.fn()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<RunDetailDrawer open runId={10} onClose={onClose} />)

    const [openReportsButton] = await screen.findAllByRole("button", {
      name: "Open reports for this run"
    })
    expect(screen.getAllByRole("button", { name: "Open reports for this run" })).toHaveLength(1)
    openReportsButton.click()

    expect(mocks.setOutputsJobFilterMock).toHaveBeenCalledWith(1)
    expect(mocks.setOutputsRunFilterMock).toHaveBeenCalledWith(10)
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("outputs")
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it("uses localized duration copy from i18n keys", async () => {
    mocks.translateMock.mockImplementation(
      (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
        if (key === "watchlists:runs.detail.duration.seconds") {
          return `SEC ${options?.count}`
        }
        const template =
          typeof defaultValue === "string" ? defaultValue : key
        return template.replace(/\{\{\s*([.\w-]+)\s*\}\}/g, (_match, token) => {
          const value = options?.[token]
          return value == null ? "" : String(value)
        })
      }
    )
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      started_at: "2026-02-18T10:00:00Z",
      finished_at: "2026-02-18T10:00:02Z"
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("SEC 2")).toBeInTheDocument()
    })
  })

  it("falls back to localized load error copy when details request has no message", async () => {
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockRejectedValue({ message: "" })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Could not load run details.")).toBeInTheDocument()
      expect(
        screen.getByText("Retry the request. If the problem continues, review server diagnostics. Details: Failed to load details")
      ).toBeInTheDocument()
    })
  })

  it("renders remediation actions for failed runs and opens monitor editor", async () => {
    const onClose = vi.fn()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "failed",
      error_msg: "403 forbidden"
    })

    render(<RunDetailDrawer open runId={10} onClose={onClose} />)

    await waitFor(() => {
      expect(screen.getByText("Suggested recovery steps")).toBeInTheDocument()
      expect(
        screen.getByText("Suggested recovery steps").closest('[data-ds-component="RecoveryCallout"]')
      ).toBeInTheDocument()
      expect(
        within(
          screen.getByText("Suggested recovery steps").closest('[data-ds-component="RecoveryCallout"]') as HTMLElement
        ).getByText("Degraded")
      ).toBeInTheDocument()
      expect(screen.getByText("Common causes").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
      expect(screen.getByText("Edit monitor schedule")).toBeInTheDocument()
      expect(screen.getByText("Review source settings")).toBeInTheDocument()
    })

    screen.getByText("Edit monitor schedule").click()
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("jobs")
    expect(mocks.openJobFormMock).toHaveBeenCalledWith(1)
    expect(onClose).toHaveBeenCalled()
  })

  it("shows filtered sample diagnostics when available", async () => {
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      filter_tallies: { "kw:earnings": 2 },
      truncated: true,
      filtered_sample: [
        {
          id: 7001,
          title: "Filtered article",
          status: "filtered"
        }
      ]
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Filtered item sample")).toBeInTheDocument()
      expect(screen.getByText("Filtered article")).toBeInTheDocument()
      expect(
        screen.getByText("Showing 1 recently filtered item for quick diagnosis.")
      ).toBeInTheDocument()
      expect(
        screen
          .getByText("Showing 1 recently filtered item for quick diagnosis.")
          .closest('[data-ds-component="Alert"]')
      ).toBeInTheDocument()
      expect(screen.getByText("Logs truncated").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    })
  })

  it("triggers retry run action from remediation panel", async () => {
    const onClose = vi.fn()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "failed",
      error_msg: "timeout while fetching"
    })

    render(<RunDetailDrawer open runId={10} onClose={onClose} />)

    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "Retry run" })).toHaveLength(1)
    })
    screen.getByRole("button", { name: "Retry run" }).click()

    await waitFor(() => {
      expect(mocks.triggerWatchlistRunMock).toHaveBeenCalledWith(1)
      expect(mocks.addRunMock).toHaveBeenCalled()
      expect(mocks.setActiveTabMock).toHaveBeenCalledWith("runs")
      expect(onClose).toHaveBeenCalled()
    })
  })

  it("opens run-scoped reports from linkage panel", async () => {
    const onClose = vi.fn()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 5, name: "TechCrunch" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [{ id: 9010, run_id: 10, job_id: 1 }],
      total: 3,
      page: 1,
      size: 1,
      has_more: false
    })

    render(<RunDetailDrawer open runId={10} onClose={onClose} />)

    await waitFor(() => {
      expect(screen.getByText("Monitor #1 produced 3 reports for this run.")).toBeInTheDocument()
    })

    expect(screen.getAllByRole("button", { name: "Open reports for this run" })).toHaveLength(1)
    screen.getByRole("button", { name: "Open reports for this run" }).click()
    expect(mocks.setOutputsJobFilterMock).toHaveBeenCalledWith(1)
    expect(mocks.setOutputsRunFilterMock).toHaveBeenCalledWith(10)
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("outputs")
    expect(onClose).toHaveBeenCalled()
  })
})
