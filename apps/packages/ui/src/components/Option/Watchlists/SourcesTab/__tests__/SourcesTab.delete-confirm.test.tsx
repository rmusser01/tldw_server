// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesTab } from "../SourcesTab"

type SourceRecord = {
  id: number
  name: string
  url: string
  source_type: "rss" | "site" | "forum"
  active: boolean
  tags: string[]
  status: string
  group_ids?: number[]
  created_at: string
  updated_at: string
}

const mocks = vi.hoisted(() => ({
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistTagsMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  getSourceSeenStatsMock: vi.fn(),
  exportOpmlMock: vi.fn(),
  checkWatchlistSourcesNowMock: vi.fn(),
  createWatchlistSourceMock: vi.fn(),
  deleteWatchlistSourceMock: vi.fn(),
  restoreWatchlistSourceMock: vi.fn(),
  updateWatchlistSourceMock: vi.fn(),
  showUndoNotificationMock: vi.fn(),
  modalConfirmMock: vi.fn(),
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
  const Button = ({ children, onClick, loading, disabled, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
      onClick={onClick}
      {...rest}
    >
      {children}
    </button>
  )

  const Search = ({ value, onChange, onSearch }: any) => (
    <input
      value={value || ""}
      onChange={(event) => onChange?.(event)}
      onKeyDown={(event) => {
        if (event.key === "Enter") onSearch?.(event.currentTarget.value)
      }}
    />
  )

  const Select = ({
    value,
    onChange,
    options = [],
    placeholder,
    allowClear: _allowClear,
    className: _className,
    size: _size,
    ...rest
  }: any) => (
    <select
      aria-label={placeholder ?? "select"}
      value={value ?? ""}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
      {...rest}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Table = ({ dataSource = [], columns = [], rowSelection }: any) => (
    <div data-testid="sources-table">
      {dataSource.map((record: SourceRecord, rowIndex: number) => {
        const selectedKeys = Array.isArray(rowSelection?.selectedRowKeys)
          ? rowSelection.selectedRowKeys
          : []
        const selectedSet = new Set(selectedKeys.map((value: unknown) => String(value)))
        const isSelected = selectedSet.has(String(record.id))

        return (
          <div key={record.id}>
            <button
              type="button"
              data-testid={`select-source-${record.id}`}
              onClick={() => {
                const nextKeys = isSelected
                  ? selectedKeys.filter((key: unknown) => String(key) !== String(record.id))
                  : [...selectedKeys, record.id]
                const selectedRows = dataSource.filter((row: SourceRecord) =>
                  nextKeys.some((key: unknown) => String(key) === String(row.id))
                )
                rowSelection?.onChange?.(nextKeys, selectedRows)
              }}
            >
              Select {record.name}
            </button>
            {columns.map((column: any, columnIndex: number) => {
              const key = String(column.key ?? column.dataIndex ?? columnIndex)
              const value = column.dataIndex ? record[column.dataIndex as keyof SourceRecord] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <div key={key}>{content}</div>
            })}
          </div>
        )
      })}
    </div>
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
    Empty: Object.assign(
      ({ description, children }: any) => (
        <div>
          <div>{description}</div>
          {children}
        </div>
      ),
      { PRESENTED_IMAGE_SIMPLE: null }
    ),
    Input: { Search },
    Modal: { confirm: (...args: any[]) => mocks.modalConfirmMock(...args) },
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Switch: ({ checked, onChange }: any) => (
      <button type="button" onClick={() => onChange?.(!checked)}>toggle</button>
    ),
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: (...args: any[]) => mocks.showUndoNotificationMock(...args)
  })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  checkWatchlistSourcesNow: (...args: any[]) => mocks.checkWatchlistSourcesNowMock(...args),
  createWatchlistSource: (...args: any[]) => mocks.createWatchlistSourceMock(...args),
  deleteWatchlistSource: (...args: any[]) => mocks.deleteWatchlistSourceMock(...args),
  restoreWatchlistSource: (...args: any[]) => mocks.restoreWatchlistSourceMock(...args),
  exportOpml: (...args: any[]) => mocks.exportOpmlMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  getSourceSeenStats: (...args: any[]) => mocks.getSourceSeenStatsMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  fetchWatchlistGroups: (...args: any[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistTags: (...args: any[]) => mocks.fetchWatchlistTagsMock(...args),
  updateWatchlistSource: (...args: any[]) => mocks.updateWatchlistSourceMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../SourceFormModal", () => ({
  SourceFormModal: () => null
}))

vi.mock("../GroupsTree", () => ({
  GroupsTree: () => null
}))

vi.mock("../SourcesBulkImport", () => ({
  SourcesBulkImport: () => null
}))

vi.mock("../SourceSeenDrawer", () => ({
  SourceSeenDrawer: () => null
}))

const buildSource = (id: number): SourceRecord => ({
  id,
  name: `Feed ${id}`,
  url: `https://example.com/feed-${id}.xml`,
  source_type: "rss",
  active: true,
  tags: ["tech"],
  status: "healthy",
  group_ids: [],
  created_at: "2026-02-18T00:00:00Z",
  updated_at: "2026-02-18T00:00:00Z"
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  sources: [buildSource(101), buildSource(202)],
  sourcesLoading: false,
  sourcesTotal: 2,
  sourcesSearch: "",
  sourcesPage: 1,
  sourcesPageSize: 20,
  tags: [],
  groups: [],
  groupsLoading: false,
  selectedGroupId: null,
  selectedTagName: null,
  sourceFormOpen: false,
  sourceFormEditId: null,
  setSources: vi.fn(),
  setSourcesLoading: vi.fn(),
  setSourcesSearch: vi.fn(),
  setSourcesPage: vi.fn(),
  setSourcesPageSize: vi.fn(),
  setTags: vi.fn(),
  setGroups: vi.fn(),
  setGroupsLoading: vi.fn(),
  setActiveTab: vi.fn(),
  setSelectedGroupId: vi.fn(),
  setSelectedTagName: vi.fn(),
  openSourceForm: vi.fn(),
  closeSourceForm: vi.fn(),
  addSource: vi.fn(),
  updateSourceInList: vi.fn(),
  removeSource: vi.fn(),
  ...overrides
})

describe("SourcesTab delete confirmation copy", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [buildSource(101), buildSource(202)],
      total: 2,
      page: 1,
      size: 20,
      has_more: false
    })
    mocks.fetchWatchlistTagsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getSourceSeenStatsMock.mockResolvedValue({
      source_id: 101,
      defer_until: null,
      consec_not_modified: 0
    })
    mocks.exportOpmlMock.mockResolvedValue("<opml></opml>")
    mocks.checkWatchlistSourcesNowMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistSourceMock.mockResolvedValue({})
    mocks.deleteWatchlistSourceMock.mockResolvedValue({
      restore_window_seconds: 10
    })
    mocks.restoreWatchlistSourceMock.mockResolvedValue({})
    mocks.updateWatchlistSourceMock.mockResolvedValue({})
  })

  it("shows undo-window copy for single-feed delete confirmation", async () => {
    render(<SourcesTab />)

    fireEvent.click(screen.getByLabelText("Delete source: Feed 101"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]
    expect(confirmConfig.title).toBe("Delete this feed?")
    expect(confirmConfig.content).toBe("You can undo this deletion for 10 seconds.")
  })

  it("shows in-use impact details plus undo-window hint before single delete", async () => {
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [
        {
          id: 301,
          name: "Morning Monitor",
          active: true,
          scope: { sources: [101] }
        }
      ],
      total: 1,
      has_more: false
    })

    render(<SourcesTab />)

    fireEvent.click(screen.getByLabelText("Delete source: Feed 101"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]
    expect(confirmConfig.title).toBe("Feed is used by active monitors")

    render(<>{confirmConfig.content}</>)
    expect(
      screen.getByText(
        "This feed is referenced by 1 active monitor. Deleting it may break scheduled runs."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByText("You can undo this deletion for 10 seconds.")
    ).toBeInTheDocument()
    expect(screen.getByText("Morning Monitor")).toBeInTheDocument()
  })

  it("surfaces undo-window timing in bulk delete confirmation", async () => {
    render(<SourcesTab />)

    fireEvent.click(screen.getByTestId("select-source-101"))
    fireEvent.click(screen.getByText("Delete"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]
    render(<>{confirmConfig.content}</>)

    expect(
      screen.getByText(
        "This will delete 1 feeds (1 active, 0 inactive). Undo is available for 10 seconds."
      )
    ).toBeInTheDocument()
  })

  it("keeps partial bulk-undo failure guidance actionable", async () => {
    render(<SourcesTab />)

    fireEvent.click(screen.getByTestId("select-source-101"))
    fireEvent.click(screen.getByTestId("select-source-202"))
    fireEvent.click(screen.getByText("Delete"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]
    await confirmConfig.onOk()

    await waitFor(() => {
      expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })

    mocks.restoreWatchlistSourceMock
      .mockResolvedValueOnce({ id: 101 })
      .mockRejectedValueOnce(new Error("restore failed"))

    const undoConfig = mocks.showUndoNotificationMock.mock.calls[0][0]
    await expect(undoConfig.onUndo()).rejects.toThrow(
      "1 restored, 1 failed to restore. Refresh Feeds and retry while the undo timer is active."
    )
  })
})
