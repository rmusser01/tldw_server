import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

const mockState = vi.hoisted(() => ({
  fetchJobRunsMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageWarningMock: vi.fn(),
  messageErrorMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> },
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) => {
      if (typeof defaultValue === "string") return defaultValue
      if (
        defaultValue &&
        typeof defaultValue === "object" &&
        "defaultValue" in (defaultValue as Record<string, unknown>)
      ) {
        const text = (defaultValue as Record<string, unknown>).defaultValue
        if (typeof text === "string") return text
      }
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
        <option
          key={String(option.value)}
          value={String(option.value)}
          disabled={Boolean(option.disabled)}
        >
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

  const Dropdown = ({ children, menu }: any) => (
    <div>
      {children}
      <div data-testid="mock-runs-export-menu">
        {(menu?.items || []).map((item: any) => (
          <button
            key={String(item.key)}
            type="button"
            disabled={Boolean(item.disabled)}
            onClick={() => menu?.onClick?.({ key: item.key })}
          >
            {String(item.label)}
          </button>
        ))}
      </div>
    </div>
  )

  const Table = () => <div data-testid="runs-table" />
  const Progress = () => <div />
  const Tooltip = ({ children }: any) => <>{children}</>
  const Space = ({ children }: any) => <>{children}</>
  const Tag = ({ children }: any) => <span>{children}</span>

  return {
    Select,
    Dropdown,
    Button,
    Table,
    Progress,
    Tooltip,
    Space,
    Tag,
    message: {
      success: mockState.messageSuccessMock,
      warning: mockState.messageWarningMock,
      error: mockState.messageErrorMock,
    },
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  fetchJobRuns: (...args: any[]) => mockState.fetchJobRunsMock(...args),
  exportRunsCsv: (...args: any[]) => mockState.exportRunsCsvMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mockState.fetchWatchlistJobsMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mockState.fetchWatchlistRunsMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) =>
    selector(mockState.storeStateRef.current)
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
  setPollingActive: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  ...overrides,
})

describe("RunsTab CSV export modes", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockState.storeStateRef.current = baseState()
    mockState.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mockState.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mockState.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mockState.exportRunsCsvMock.mockResolvedValue("filter_key,count\nkw:alpha,2\n")

    if (typeof URL.createObjectURL !== "function") {
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        value: vi.fn(() => "blob:mock"),
      })
    } else {
      vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock")
    }
    if (typeof URL.revokeObjectURL !== "function") {
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        value: vi.fn(),
      })
    } else {
      vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    }
  })

  it("requests global aggregate tallies CSV when aggregate mode is selected", async () => {
    render(<RunsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Global tallies summary" }))

    await waitFor(() => {
      expect(mockState.exportRunsCsvMock).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: "global",
          include_tallies: true,
          tallies_mode: "aggregate"
        })
      )
    })
  })

  it("requests per-run tallies CSV for job scope when per-run mode is selected", async () => {
    mockState.storeStateRef.current = baseState({ runsJobFilter: 42 })
    mockState.exportRunsCsvMock.mockResolvedValue(
      "id,job_id,status,started_at,finished_at,items_found,items_ingested,filters_include,filters_exclude,filters_flag,filter_tallies_json\n"
    )

    render(<RunsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Per-run tallies" }))

    await waitFor(() => {
      expect(mockState.exportRunsCsvMock).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: "job",
          job_id: 42,
          include_tallies: true,
          tallies_mode: "per_run"
        })
      )
    })
  })
})
