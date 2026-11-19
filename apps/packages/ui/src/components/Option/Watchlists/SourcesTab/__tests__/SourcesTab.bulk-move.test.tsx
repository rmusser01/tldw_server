import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
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
  const Button = ({ children, onClick, loading, disabled, danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
      onClick={() => onClick?.()}
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
    size: _size,
    className: _className,
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

  const Table = ({ dataSource = [], rowSelection }: any) => (
    <div data-testid="sources-table">
      {dataSource.map((record: SourceRecord) => {
        const selectedKeys = Array.isArray(rowSelection?.selectedRowKeys)
          ? rowSelection.selectedRowKeys
          : []
        const selectedSet = new Set(selectedKeys.map((value: unknown) => String(value)))
        const isSelected = selectedSet.has(String(record.id))
        return (
          <button
            key={record.id}
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
            {record.name}
          </button>
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
    Empty: ({ description, children }: any) => (
      <div>
        <div>{description}</div>
        {children}
      </div>
    ),
    Input: { Search },
    Modal: { confirm: (...args: any[]) => mocks.modalConfirmMock(...args) },
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Switch: () => <button type="button">switch</button>,
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
  getWatchlistSettings: vi.fn().mockResolvedValue({ forums_enabled: false }),
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

const buildSource = (id: number, groupIds: number[] = []): SourceRecord => ({
  id,
  name: `Feed ${id}`,
  url: `https://example.com/feed-${id}.xml`,
  source_type: "rss",
  active: true,
  tags: ["tech"],
  status: "healthy",
  group_ids: groupIds,
  created_at: "2026-02-18T00:00:00Z",
  updated_at: "2026-02-18T00:00:00Z"
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  sources: [buildSource(101, [1]), buildSource(202, [])],
  sourcesLoading: false,
  sourcesTotal: 2,
  sourcesSearch: "",
  sourcesPage: 1,
  sourcesPageSize: 20,
  tags: [],
  groups: [
    { id: 1, name: "News", parent_group_id: null },
    { id: 2, name: "Research", parent_group_id: null }
  ],
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

describe("SourcesTab bulk move", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [buildSource(101, [1]), buildSource(202, [])], total: 2, has_more: false
    })
    mocks.fetchWatchlistTagsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({
      items: [
        { id: 1, name: "News", parent_group_id: null },
        { id: 2, name: "Research", parent_group_id: null }
      ],
      total: 2,
      has_more: false
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getSourceSeenStatsMock.mockResolvedValue({
      source_id: 101,
      defer_until: null,
      consec_not_modified: 0
    })
    mocks.exportOpmlMock.mockResolvedValue("<opml></opml>")
    mocks.checkWatchlistSourcesNowMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistSourceMock.mockResolvedValue({})
    mocks.deleteWatchlistSourceMock.mockResolvedValue({})
    mocks.restoreWatchlistSourceMock.mockResolvedValue({})
    mocks.updateWatchlistSourceMock.mockImplementation(async (sourceId: number, payload: Record<string, unknown>) => ({
      ...buildSource(sourceId),
      ...payload
    }))
  })

  it("moves selected feeds to a new group and supports undo restoration", async () => {
    render(<SourcesTab />)

    fireEvent.click(await screen.findByTestId("select-source-101"))

    fireEvent.change(screen.getByLabelText("Move to group"), {
      target: { value: "2" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Move" }))

    expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]

    await confirmConfig.onOk()

    await waitFor(() => {
      expect(mocks.updateWatchlistSourceMock).toHaveBeenCalledWith(101, { group_ids: [2] })
    })
    expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1)

    const undoConfig = mocks.showUndoNotificationMock.mock.calls[0][0]
    await undoConfig.onUndo()

    await waitFor(() => {
      expect(mocks.updateWatchlistSourceMock).toHaveBeenCalledWith(101, { group_ids: [1] })
    })
  })

  it("supports bulk move to no group target", async () => {
    render(<SourcesTab />)

    fireEvent.click(await screen.findByTestId("select-source-101"))

    fireEvent.change(screen.getByLabelText("Move to group"), {
      target: { value: "__none__" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Move" }))

    const confirmConfig = mocks.modalConfirmMock.mock.calls[0][0]
    await confirmConfig.onOk()

    await waitFor(() => {
      expect(mocks.updateWatchlistSourceMock).toHaveBeenCalledWith(101, { group_ids: [] })
    })
  })

  it("renders high-density source tables for bulk operations", async () => {
    const manySources = Array.from({ length: 200 }, (_value, index) =>
      buildSource(index + 1, index % 2 === 0 ? [1] : [])
    )
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: manySources, total: manySources.length, has_more: false
    })

    const { container } = render(<SourcesTab />)

    await waitFor(() => {
      expect(container.querySelectorAll("[data-testid^='select-source-']")).toHaveLength(200)
    })
  })
})
