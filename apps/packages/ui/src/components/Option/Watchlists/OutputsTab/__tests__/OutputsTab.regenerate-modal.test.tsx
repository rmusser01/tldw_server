// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { OutputsTab } from "../OutputsTab"

const mocks = vi.hoisted(() => ({
  createWatchlistOutputMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistTemplatesMock: vi.fn(),
  downloadWatchlistOutputMock: vi.fn(),
  downloadWatchlistOutputBinaryMock: vi.fn(),
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
  const Select = ({ value, onChange, options = [], allowClear, ...rest }: any) => {
    const normalizedValue = value == null ? "" : String(value)
    return (
      <select
        data-testid={rest["data-testid"] || "antd-select"}
        data-value={normalizedValue}
        value={normalizedValue}
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
  }

  const Button = ({ children, onClick, loading, disabled, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
      onClick={() => onClick?.()}
      {...rest}
    >
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="outputs-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.id ?? rowIndex} data-testid={`outputs-row-${record.id}`}>
          {columns.map((column: any, columnIndex: number) => {
            const key = String(column.key ?? column.dataIndex ?? columnIndex)
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            const content = column.render ? column.render(value, record, rowIndex) : value
            return <div key={key}>{content}</div>
          })}
        </div>
      ))}
    </div>
  )

  const Modal = ({ open, children, onOk, onCancel, okText, cancelText, confirmLoading }: any) => (
    open ? (
      <div data-testid="antd-modal">
        {children}
        <button type="button" onClick={() => onCancel?.()}>
          {cancelText || "Cancel"}
        </button>
        <button type="button" disabled={Boolean(confirmLoading)} onClick={() => onOk?.()}>
          {okText || "OK"}
        </button>
      </div>
    ) : null
  )

  return {
    Button,
    Input: ({ value, onChange, ...rest }: any) => (
      <input
        value={value || ""}
        onChange={(event) => onChange?.(event)}
        {...rest}
      />
    ),
    InputNumber: ({ value, onChange, ...rest }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => {
          const raw = event.currentTarget.value
          onChange?.(raw ? Number(raw) : null)
        }}
        {...rest}
      />
    ),
    Modal,
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  createWatchlistOutput: (...args: any[]) => mocks.createWatchlistOutputMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistOutputs: (...args: any[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistTemplates: (...args: any[]) => mocks.fetchWatchlistTemplatesMock(...args),
  downloadWatchlistOutput: (...args: any[]) => mocks.downloadWatchlistOutputMock(...args),
  downloadWatchlistOutputBinary: (...args: any[]) => mocks.downloadWatchlistOutputBinaryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../OutputPreviewDrawer", () => ({
  OutputPreviewDrawer: () => null
}))

const buildOutput = (overrides: Record<string, unknown> = {}) => ({
  id: 18,
  job_id: 4,
  run_id: 91,
  title: "Daily Brief",
  format: "md",
  type: "briefing",
  content: "hello",
  metadata: {
    template_name: "briefing_markdown",
    template_version: 4,
    deliveries: {
      email: {
        channel: "email",
        status: "sent"
      }
    }
  },
  created_at: "2026-02-23T08:00:00Z",
  expires_at: null,
  expired: false,
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  outputs: [],
  outputsLoading: false,
  outputsTotal: 0,
  outputsPage: 1,
  outputsPageSize: 20,
  outputsJobFilter: null,
  outputsRunFilter: null,
  outputPreviewOpen: false,
  selectedOutputId: null,
  setOutputs: vi.fn(),
  setOutputsLoading: vi.fn(),
  setOutputsPage: vi.fn(),
  setOutputsPageSize: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setActiveTab: vi.fn(),
  openRunDetail: vi.fn(),
  openJobForm: vi.fn(),
  openOutputPreview: vi.fn(),
  closeOutputPreview: vi.fn(),
  ...overrides
})

describe("OutputsTab regenerate modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput()],
      outputsTotal: 1
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistTemplatesMock.mockResolvedValue({
      items: [
        {
          name: "briefing_markdown",
          format: "md",
          description: "Briefing",
          content: "# {{ title }}",
          updated_at: "2026-02-23T00:00:00Z",
          version: 4,
          history_count: 3,
          available_versions: [1, 2, 4]
        }
      ]
    })
    mocks.createWatchlistOutputMock.mockResolvedValue({ id: 999 })
    mocks.downloadWatchlistOutputMock.mockResolvedValue("")
    mocks.downloadWatchlistOutputBinaryMock.mockResolvedValue(new ArrayBuffer(0))
  })

  it("hydrates template defaults in regenerate modal for text outputs", async () => {
    render(<OutputsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Regenerate" }))

    await waitFor(() => {
      expect(mocks.fetchWatchlistTemplatesMock).toHaveBeenCalledTimes(1)
    })

    expect(screen.getByTestId("outputs-regenerate-template")).toHaveAttribute(
      "data-value",
      "briefing_markdown"
    )
    expect(
      screen.getByTestId("outputs-regenerate-template-version-select")
    ).toHaveAttribute("data-value", "4")
  })

  it("hides template overrides and submits constrained payload for audio outputs", async () => {
    mocks.storeStateRef.current = baseState({
      outputs: [
        buildOutput({
          format: "mp3",
          type: "tts_audio",
          title: "Morning Audio",
          metadata: {
            template_name: "newsletter_html",
            template_version: 2
          }
        })
      ],
      outputsTotal: 1
    })

    render(<OutputsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Regenerate" }))

    const modal = screen.getByTestId("antd-modal")
    expect(
      within(modal).getByText(
        "Audio outputs regenerate using run audio settings. Template overrides are unavailable."
      )
    ).toBeInTheDocument()
    expect(within(modal).queryByTestId("outputs-regenerate-template")).not.toBeInTheDocument()

    fireEvent.click(within(modal).getByRole("button", { name: "Regenerate" }))

    await waitFor(() => {
      expect(mocks.createWatchlistOutputMock).toHaveBeenCalledTimes(1)
    })

    expect(mocks.createWatchlistOutputMock).toHaveBeenCalledWith(
      expect.objectContaining({
        run_id: 91,
        type: "tts_audio",
        title: "Morning Audio"
      })
    )
    const payload = mocks.createWatchlistOutputMock.mock.calls[0][0]
    expect(payload).not.toHaveProperty("template_name")
    expect(payload).not.toHaveProperty("template_version")
  })

  it("restores focus to regenerate trigger when modal closes", async () => {
    render(<OutputsTab />)

    const trigger = screen.getByRole("button", { name: "Regenerate" })
    trigger.focus()
    fireEvent.click(trigger)

    const modal = await screen.findByTestId("antd-modal")
    fireEvent.click(within(modal).getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(screen.queryByTestId("antd-modal")).not.toBeInTheDocument()
    })

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })
  })

  it("links output rows to upstream monitor and run activity", async () => {
    const setActiveTab = vi.fn()
    const setRunsJobFilter = vi.fn()
    const setRunsStatusFilter = vi.fn()
    const openRunDetail = vi.fn()
    const openJobForm = vi.fn()
    mocks.storeStateRef.current = baseState({
      outputs: [buildOutput()],
      outputsTotal: 1,
      setActiveTab,
      setRunsJobFilter,
      setRunsStatusFilter,
      openRunDetail,
      openJobForm
    })

    render(<OutputsTab />)

    fireEvent.click(screen.getByTestId("watchlists-output-open-job-18"))
    expect(setActiveTab).toHaveBeenCalledWith("jobs")
    expect(openJobForm).toHaveBeenCalledWith(4)

    fireEvent.click(screen.getByTestId("watchlists-output-open-run-18"))
    expect(setRunsJobFilter).toHaveBeenCalledWith(4)
    expect(setRunsStatusFilter).toHaveBeenCalledWith(null)
    expect(setActiveTab).toHaveBeenCalledWith("runs")
    expect(openRunDetail).toHaveBeenCalledWith(91)
  })

  it("renders delivery state only when output metadata contains delivery statuses", async () => {
    mocks.storeStateRef.current = baseState({
      outputs: [
        buildOutput({
          id: 18,
          metadata: {
            deliveries: {
              email: { channel: "email", status: "sent" }
            }
          }
        }),
        buildOutput({
          id: 19,
          title: "Manual Brief",
          metadata: {}
        })
      ],
      outputsTotal: 2
    })

    render(<OutputsTab />)

    expect(screen.getByTestId("outputs-row-18")).toHaveTextContent("email Sent")
    expect(screen.getByTestId("outputs-row-19")).toHaveTextContent("-")
  })
})
