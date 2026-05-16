// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { TemplatesTab } from "../TemplatesTab"

const mocks = vi.hoisted(() => ({
  deleteWatchlistTemplateMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  modalConfirmMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> },
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
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
    t: mocks.tMock
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
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [], rowKey }: any) => (
    <table role="table" aria-label={ariaLabel || "Templates table"}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={typeof rowKey === "function" ? rowKey(record) : record[rowKey] ?? record.name ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  const Empty = ({ children, description }: any) => (
    <div>
      {description}
      {children}
    </div>
  )
  ;(Empty as any).PRESENTED_IMAGE_SIMPLE = null
  return {
    Button,
    Empty,
    Modal: { confirm: (...args: unknown[]) => mocks.modalConfirmMock(...args) },
    Space: ({ children }: any) => <>{children}</>,
    Spin: () => <div role="status">Loading</div>,
    Table,
    Tooltip: ({ children }: any) => <>{children}</>,
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
  deleteWatchlistTemplate: (...args: unknown[]) => mocks.deleteWatchlistTemplateMock(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistTemplates: (...args: unknown[]) => mocks.fetchWatchlistTemplatesMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../TemplateEditor", () => ({
  TemplateEditor: () => null
}))

const buildTemplate = (overrides: Record<string, unknown> = {}) => ({
  name: "cti-mece-brief",
  description: "Structured CTI briefing",
  format: "html",
  version: 3,
  history_count: 5,
  available_versions: [3, 2, 1],
  updated_at: "2026-05-15T10:00:00Z",
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  templates: [buildTemplate()],
  templatesLoading: false,
  selectedWatchlistId: 42,
  setTemplates: vi.fn(),
  setTemplatesLoading: vi.fn(),
  ...overrides
})

describe("TemplatesTab constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.deleteWatchlistTemplateMock.mockResolvedValue({})
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({
      items: [buildTemplate()],
      total: 1,
      has_more: false
    })
  })

  it("replaces the Templates table with template cards at extension width", async () => {
    setViewport(420)
    render(<TemplatesTab />)

    expect(await screen.findByTestId("watchlists-templates-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Templates table" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Template" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()

    const card = screen.getByTestId("watchlists-template-card-cti-mece-brief-html")
    expect(within(card).getByText("cti-mece-brief")).toBeInTheDocument()
    expect(within(card).getByText("Structured CTI briefing")).toBeInTheDocument()
    expect(within(card).getByText("HTML")).toBeInTheDocument()
    expect(within(card).getByText("v3")).toBeInTheDocument()
    expect(within(card).getByText("5 versions")).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Edit" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Delete" })).toBeInTheDocument()
  })

  it("preserves the desktop Templates table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<TemplatesTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Templates table" })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-templates-constrained-list")).not.toBeInTheDocument()
  })

  it("shows constrained loading feedback before templates arrive", () => {
    setViewport(420)
    mocks.storeStateRef.current = baseState({
      templates: [],
      templatesLoading: true
    })

    render(<TemplatesTab />)

    expect(screen.getByTestId("watchlists-templates-constrained-loading")).toBeInTheDocument()
    expect(screen.getByRole("status")).toBeInTheDocument()
    expect(screen.queryByText("No templates yet")).not.toBeInTheDocument()
  })

  it("keeps duplicate template names distinct by format in constrained cards", async () => {
    setViewport(420)
    const markdownTemplate = buildTemplate({
      name: "reading_digest_suggestions",
      description: "Markdown digest",
      format: "md"
    })
    const htmlTemplate = buildTemplate({
      name: "reading_digest_suggestions",
      description: "HTML digest",
      format: "html"
    })
    mocks.storeStateRef.current = baseState({
      templates: [markdownTemplate, htmlTemplate]
    })

    render(<TemplatesTab />)

    expect(await screen.findByTestId("watchlists-template-card-reading_digest_suggestions-md")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-template-card-reading_digest_suggestions-html")).toBeInTheDocument()
  })
})
