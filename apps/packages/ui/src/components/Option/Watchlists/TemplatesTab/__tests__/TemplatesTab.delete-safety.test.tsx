import React from "react"
import { fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "@/components/Option/Watchlists/shared/__tests__/render-with-watchlists-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { TemplatesTab } from "../TemplatesTab"

const mocks = vi.hoisted(() => ({
  fetchWatchlistTemplatesMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  deleteWatchlistTemplateMock: vi.fn(),
  modalConfirmMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageWarningMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  storeStateRef: {
    current: {
      templates: [],
      templatesLoading: false,
      setTemplates: vi.fn(),
      setTemplatesLoading: vi.fn()
    } as Record<string, any>
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) => {
      if (typeof defaultValue === "string") return defaultValue
      return _key
    }
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, danger = false, loading = false }: any) => (
    <button
      type="button"
      data-testid={danger ? "danger-button" : "button"}
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
    >
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="templates-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.name ?? rowIndex}>
          {columns.map((column: any, columnIndex: number) => {
            const key = String(column.key ?? column.dataIndex ?? columnIndex)
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            return (
              <div key={key}>
                {column.render ? column.render(value, record, rowIndex) : null}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )

  const Tooltip = ({ children }: any) => <>{children}</>
  const Space = ({ children }: any) => <>{children}</>
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
    Modal: { confirm: (...args: any[]) => mocks.modalConfirmMock(...args) },
    Space,
    Table,
    Tooltip,
    message: {
      error: mocks.messageErrorMock,
      warning: mocks.messageWarningMock,
      success: mocks.messageSuccessMock
    }
  }
})

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistTemplates: (...args: any[]) => mocks.fetchWatchlistTemplatesMock(...args),
  deleteWatchlistTemplate: (...args: any[]) => mocks.deleteWatchlistTemplateMock(...args)
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("../TemplateEditor", () => ({
  TemplateEditor: () => null
}))

describe("TemplatesTab delete safety warnings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = {
      templates: [
        {
          name: "daily-brief",
          format: "html",
          description: "Daily",
          updated_at: "2026-02-18T00:00:00Z"
        }
      ],
      templatesLoading: false,
      setTemplates: vi.fn(),
      setTemplatesLoading: vi.fn()
    }

    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({
      items: [
        {
          name: "daily-brief",
          format: "html",
          description: "Daily",
          updated_at: "2026-02-18T00:00:00Z"
        }
      ]
    })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })
  })

  it("warns when deleting a template used by active monitors", async () => {
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [
        {
          id: 10,
          name: "Morning Monitor",
          active: true,
          output_prefs: {
            template_name: "daily-brief"
          }
        }
      ],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<TemplatesTab />)

    await waitFor(() => {
      expect(screen.getByTestId("templates-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const config = mocks.modalConfirmMock.mock.calls[0][0]
    expect(config.title).toBe("Template is used by active monitors")
    expect(config.okText).toBe("Delete anyway")

    const { getByText } = render(<>{config.content}</>)
    expect(getByText("Morning Monitor")).toBeInTheDocument()
  })

  it("shows standard delete confirmation when no active monitor uses the template", async () => {
    render(<TemplatesTab />)

    await waitFor(() => {
      expect(screen.getByTestId("templates-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))

    await waitFor(() => {
      expect(mocks.modalConfirmMock).toHaveBeenCalledTimes(1)
    })

    const config = mocks.modalConfirmMock.mock.calls[0][0]
    expect(config.title).toBe("Delete this template?")
    expect(config.content).toBe("This action cannot be undone.")
    expect(config.okText).toBe("Delete")
  })
})
