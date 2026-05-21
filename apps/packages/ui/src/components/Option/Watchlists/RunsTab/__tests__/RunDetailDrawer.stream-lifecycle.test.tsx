// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { RunDetailDrawer } from "../RunDetailDrawer"

const mocks = vi.hoisted(() => ({
  getRunDetailsMock: vi.fn(),
  fetchScrapedItemsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  getWatchlistRunAudioMock: vi.fn(),
  getWatchlistRunDiagnosticsMock: vi.fn(),
  retryWatchlistRunAudioMock: vi.fn(),
  retryWatchlistRunDeliveryMock: vi.fn(),
  updateScrapedItemMock: vi.fn(),
  exportRunTalliesCsvMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  cancelWatchlistRunMock: vi.fn(),
  updateRunInListMock: vi.fn(),
  addRunMock: vi.fn(),
  setActiveTabMock: vi.fn(),
  setOutputsJobFilterMock: vi.fn(),
  setOutputsRunFilterMock: vi.fn(),
  openJobFormMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageWarningMock: vi.fn(),
  translateMock: vi.fn(
    (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{\s*([.\w-]+)\s*\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.translateMock
  })
}))

vi.mock("antd", () => {
  const Drawer = ({ open, children, title, extra, onClose, afterOpenChange }: any) => {
    const closeRef = React.useRef<HTMLButtonElement | null>(null)
    React.useEffect(() => {
      afterOpenChange?.(open)
      if (open) {
        closeRef.current?.focus()
      }
    }, [afterOpenChange, open])

    if (!open) return null
    return (
      <div>
        <div>{title}</div>
        {extra}
        <button type="button" ref={closeRef} onClick={() => onClose?.()}>
          Close drawer
        </button>
        {children}
      </div>
    )
  }

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

  const Button = ({ children, onClick, disabled }: any) => (
    <button type="button" disabled={Boolean(disabled)} onClick={() => onClick?.()}>
      {children}
    </button>
  )
  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children, title }: any) => (
    <span>
      {children}
      {title ? <span>{title}</span> : null}
    </span>
  )
  const Switch = ({ checked, onChange, checkedChildren, unCheckedChildren }: any) => (
    <button type="button" onClick={() => onChange?.(!checked)}>
      {checked ? checkedChildren || "on" : unCheckedChildren || "off"}
    </button>
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
      warning: mocks.messageWarningMock
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
  getWatchlistRunDiagnostics: (...args: any[]) => mocks.getWatchlistRunDiagnosticsMock(...args),
  getWatchlistRunAudio: (...args: any[]) => mocks.getWatchlistRunAudioMock(...args),
  retryWatchlistRunAudio: (...args: any[]) => mocks.retryWatchlistRunAudioMock(...args),
  retryWatchlistRunDelivery: (...args: any[]) => mocks.retryWatchlistRunDeliveryMock(...args),
  getRunDetails: (...args: any[]) => mocks.getRunDetailsMock(...args),
  triggerWatchlistRun: (...args: any[]) => mocks.triggerWatchlistRunMock(...args),
  updateScrapedItem: (...args: any[]) => mocks.updateScrapedItemMock(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(async () => ({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single_user",
      apiKey: "test-key"
    }))
  }
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

type MessageHandler = ((event: { data: string }) => void) | null
type OpenHandler = (() => void) | null
type CloseHandler = (() => void) | null
type ErrorHandler = (() => void) | null

class MockWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: MockWebSocket[] = []

  readyState = MockWebSocket.CONNECTING
  onopen: OpenHandler = null
  onmessage: MessageHandler = null
  onclose: CloseHandler = null
  onerror: ErrorHandler = null
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  })

  constructor(public readonly url: string) {
    MockWebSocket.instances.push(this)
  }

  emitOpen() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  emitClose() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }

  emitError() {
    this.onerror?.()
  }

  static reset() {
    MockWebSocket.instances = []
  }
}

const baseRunDetails = {
  id: 10,
  job_id: 1,
  status: "running",
  started_at: "2026-02-18T10:00:00Z",
  finished_at: null,
  stats: {
    items_found: 0,
    items_ingested: 0,
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

describe("RunDetailDrawer stream lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.reset()
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket)
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)
    mocks.translateMock.mockImplementation(
      (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
        if (typeof defaultValue !== "string") return key
        if (!options) return defaultValue
        return defaultValue.replace(/\{\{\s*([.\w-]+)\s*\}\}/g, (_match, token) => {
          const value = options[token]
          return value == null ? "" : String(value)
        })
      }
    )

    mocks.getRunDetailsMock.mockResolvedValue(baseRunDetails)
    mocks.fetchScrapedItemsMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 20,
      has_more: false
    })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 1,
      has_more: false
    })
    mocks.getWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      task_id: null,
      status: "unknown",
      download_url: null
    })
    mocks.getWatchlistRunDiagnosticsMock.mockResolvedValue({
      run_id: 10,
      generated_at: "2026-05-19T04:00:00Z",
      run: { id: 10, status: "completed" }
    })
    mocks.retryWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      stage: "audio",
      retried: true,
      task_id: "retry-audio-10"
    })
    mocks.retryWatchlistRunDeliveryMock.mockResolvedValue({
      run_id: 10,
      stage: "delivery",
      retried: true,
      output_id: 55,
      delivery_results: [{ channel: "email", status: "sent" }]
    })
    mocks.updateScrapedItemMock.mockResolvedValue({})
    mocks.exportRunTalliesCsvMock.mockResolvedValue("")
    mocks.triggerWatchlistRunMock.mockResolvedValue({ id: 999, status: "pending", job_id: 1 })
    mocks.cancelWatchlistRunMock.mockResolvedValue({ cancelled: true })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  const waitForFirstSocket = async (): Promise<MockWebSocket> => {
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    })
    return MockWebSocket.instances[0]!
  }

  const flushMicrotasks = async (count = 3): Promise<void> => {
    for (let i = 0; i < count; i += 1) {
      await Promise.resolve()
    }
  }

  it("applies stream updates and transitions active run to completed without refetch", async () => {
    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    const ws = await waitForFirstSocket()
    ws.emitOpen()

    ws.emitMessage({
      type: "snapshot",
      run: {
        id: 10,
        job_id: 1,
        status: "running",
        started_at: "2026-02-18T10:00:00Z",
        finished_at: null
      },
      stats: {
        items_found: 1,
        items_ingested: 0
      },
      log_tail: "initial line\n",
      log_truncated: false
    })

    await waitFor(() => {
      expect(screen.getByText(/initial line/)).toBeInTheDocument()
      expect(screen.getByText(/Live stream: Connected/)).toBeInTheDocument()
    })

    ws.emitMessage({
      type: "run_update",
      run: {
        id: 10,
        job_id: 1,
        status: "completed",
        started_at: "2026-02-18T10:00:00Z",
        finished_at: "2026-02-18T10:01:00Z"
      },
      stats: {
        items_found: 2,
        items_ingested: 2
      }
    })

    await waitFor(() => {
      expect(screen.getByText("Completed")).toBeInTheDocument()
    })
    expect(mocks.getRunDetailsMock).toHaveBeenCalledTimes(1)
    expect(mocks.updateRunInListMock).toHaveBeenCalledWith(
      10,
      expect.objectContaining({ status: "completed" })
    )
  })

  it("loads and displays pending audio status when the run has an audio task", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_task_id: "audio-task-10"
      }
    })
    mocks.getWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      task_id: "audio-task-10",
      status: "pending",
      download_url: null
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.getWatchlistRunAudioMock).toHaveBeenCalledWith(10)
    })
    expect(screen.getByText("Audio briefing")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText("Queued")).toBeInTheDocument()
    })
  })

  it("displays completed audio with a final download link", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_task_id: "audio-task-10"
      }
    })
    mocks.getWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      task_id: "audio-task-10",
      status: "completed",
      download_url: "/api/v1/watchlists/runs/10/audio/download",
      final_artifact: {
        title: "Final mix",
        mime_type: "audio/mpeg"
      }
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Final audio available")).toBeInTheDocument()
    })
    const link = screen.getByRole("link", { name: "Download final audio" })
    expect(link).toHaveAttribute("href", "/api/v1/watchlists/runs/10/audio/download")
  })

  it("offers stage-specific retry and diagnostics controls without triggering a full rerun", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_task_id: "audio-task-10"
      }
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [{ id: 55 }],
      total: 1,
      page: 1,
      size: 1,
      has_more: false
    })
    mocks.getWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      task_id: "audio-task-10",
      status: "failed",
      download_url: null,
      error: "tts_failed"
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Retry delivery" })).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry audio" })).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Download diagnostics" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry delivery" }))
    fireEvent.click(screen.getByRole("button", { name: "Retry audio" }))
    fireEvent.click(screen.getByRole("button", { name: "Download diagnostics" }))

    await waitFor(() => {
      expect(mocks.retryWatchlistRunDeliveryMock).toHaveBeenCalledWith(10)
      expect(mocks.retryWatchlistRunAudioMock).toHaveBeenCalledWith(10)
      expect(mocks.getWatchlistRunDiagnosticsMock).toHaveBeenCalledWith(10)
    })
    expect(mocks.triggerWatchlistRunMock).not.toHaveBeenCalled()
  })

  it("does not report success when stage-specific retries are skipped", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_task_id: "audio-task-10"
      }
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({
      items: [{ id: 55 }],
      total: 1,
      page: 1,
      size: 1,
      has_more: false
    })
    mocks.getWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      task_id: "audio-task-10",
      status: "failed",
      download_url: null,
      error: "tts_failed"
    })
    mocks.retryWatchlistRunDeliveryMock.mockResolvedValue({
      run_id: 10,
      stage: "delivery",
      retried: false,
      message: "Delivery retry was skipped."
    })
    mocks.retryWatchlistRunAudioMock.mockResolvedValue({
      run_id: 10,
      stage: "audio",
      retried: false,
      message: "Audio retry was skipped."
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Retry delivery" })).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry audio" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry delivery" }))
    fireEvent.click(screen.getByRole("button", { name: "Retry audio" }))

    await waitFor(() => {
      expect(mocks.retryWatchlistRunDeliveryMock).toHaveBeenCalledWith(10)
      expect(mocks.retryWatchlistRunAudioMock).toHaveBeenCalledWith(10)
    })
    expect(mocks.messageWarningMock).toHaveBeenCalledWith("Delivery retry was skipped.")
    expect(mocks.messageWarningMock).toHaveBeenCalledWith("Audio retry was skipped.")
    expect(mocks.messageSuccessMock).not.toHaveBeenCalled()
    expect(mocks.triggerWatchlistRunMock).not.toHaveBeenCalled()
  })

  it("reconnects after unexpected close when run is non-terminal", async () => {
    vi.useFakeTimers()
    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await flushMicrotasks(5)
    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    const ws = MockWebSocket.instances[0]!
    await act(async () => {
      ws.emitOpen()
      await flushMicrotasks()
    })
    expect(screen.getByText(/Live stream: Connected/)).toBeInTheDocument()

    await act(async () => {
      ws.emitClose()
      await flushMicrotasks()
    })
    expect(screen.getByText(/Live stream: Reconnecting/)).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
      await flushMicrotasks()
    })
    expect(MockWebSocket.instances.length).toBe(2)

    const ws2 = MockWebSocket.instances[1]!
    await act(async () => {
      ws2.emitOpen()
      await flushMicrotasks()
    })
    expect(screen.getByText(/Live stream: Connected/)).toBeInTheDocument()
  })

  it("shows stream errors and handles complete events by disconnecting", async () => {
    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    const ws = await waitForFirstSocket()
    ws.emitOpen()
    ws.emitError()

    await waitFor(() => {
      expect(screen.getByText("Stream error")).toBeInTheDocument()
      expect(screen.getByText("Stream error").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
      expect(screen.getByText("Live stream error")).toBeInTheDocument()
    })

    ws.emitMessage({
      type: "complete",
      status: "completed"
    })

    await waitFor(() => {
      expect(screen.getByText(/Live stream: Disconnected/)).toBeInTheDocument()
    })
    expect(mocks.updateRunInListMock).toHaveBeenCalledWith(
      10,
      expect.objectContaining({ status: "completed" })
    )
  })

  it("supports manual stream disable without scheduling reconnects", async () => {
    vi.useFakeTimers()
    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await flushMicrotasks(5)
    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    const ws = MockWebSocket.instances[0]!
    await act(async () => {
      ws.emitOpen()
      await flushMicrotasks()
    })
    expect(screen.getByText(/Live stream: Connected/)).toBeInTheDocument()

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Live" }))
      await flushMicrotasks()
    })
    expect(ws.close).toHaveBeenCalled()
    expect(screen.getByText(/Live stream: Disconnected/)).toBeInTheDocument()

    await act(async () => {
      ws.emitClose()
      await vi.advanceTimersByTimeAsync(20_000)
      await flushMicrotasks()
    })
    expect(MockWebSocket.instances.length).toBe(1)
  })

  it("appends logs and truncates to bounded buffer size", async () => {
    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    const ws = await waitForFirstSocket()
    ws.emitOpen()
    ws.emitMessage({
      type: "snapshot",
      run: {
        id: 10,
        job_id: 1,
        status: "running",
        started_at: "2026-02-18T10:00:00Z",
        finished_at: null
      },
      stats: {},
      log_tail: "start\n",
      log_truncated: false
    })

    const oversizedLog = "a".repeat(210_000)
    ws.emitMessage({
      type: "log",
      text: oversizedLog
    })

    await waitFor(() => {
      expect(screen.getByText("Logs truncated")).toBeInTheDocument()
      expect(screen.getByText("Logs truncated").closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    })

    const pre = document.querySelector("pre")
    expect(pre).not.toBeNull()
    const logText = pre?.textContent || ""
    expect(logText.length).toBe(200_000)
    expect(logText.endsWith("a")).toBe(true)
  })

  it("restores focus to the launch control after drawer closes", async () => {
    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open run detail"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Close drawer" })).toHaveFocus()
    })

    rerender(<RunDetailDrawer open={false} runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })
})
