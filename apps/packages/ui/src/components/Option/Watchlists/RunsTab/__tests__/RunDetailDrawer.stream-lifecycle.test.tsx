// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useWatchlistsQuery } from "@/hooks/useWatchlistsQuery"
import { tldwRequest, type TldwRequestPayload } from "@/services/tldw/request-core"
import { RunDetailDrawer } from "../RunDetailDrawer"

const mocks = vi.hoisted(() => ({
  getConfigMock: vi.fn(),
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
    getConfig: mocks.getConfigMock
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

const CachedRunStatus = () => {
  const { data } = useWatchlistsQuery("runs", { page: 1, size: 20 }, async () => ({
    items: [{ id: 10, status: "running", job_id: 1 }],
    total: 1
  }))
  return <span data-testid="cached-run-status">{data?.items[0]?.status}</span>
}

describe("RunDetailDrawer stream lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.reset()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "self_host")
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

    mocks.getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    mocks.getRunDetailsMock.mockReset().mockResolvedValue(baseRunDetails)
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
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    vi.doUnmock("@/services/background-proxy")
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

  it("polls hosted run details with a session cookie and no browser access token", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    const hostedConfig = { serverUrl: window.location.origin, authMode: "multi-user" }
    mocks.getConfigMock.mockResolvedValue(hostedConfig)

    const completedRun = {
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      log_text: "Morning Brief generated"
    }
    const snapshots = [
      baseRunDetails,
      { ...baseRunDetails, log_text: "Collected two items" },
      null,
      completedRun
    ]
    const cookies = new Map([[window.location.origin, "tldw_session=test-session"]])
    const fetchRunDetails = vi.fn<typeof fetch>(async (input, init) => {
      // Model the browser attaching the session cookie to a same-origin fetch.
      const origin = new URL(String(input), window.location.origin).origin
      const cookie = init?.credentials === "omit" ? undefined : cookies.get(origin)
      if (!cookie || new Headers(init?.headers).has("Authorization")) {
        return Response.json({ detail: "Not authenticated" }, { status: 401 })
      }
      expect(input).toBe("/api/proxy/watchlists/runs/10/details?include_tallies=true")
      expect(init?.method).toBe("GET")
      const snapshot = snapshots.shift()
      if (snapshot === null) return Response.json({ detail: "Unavailable" }, { status: 503 })
      return Response.json(snapshot ?? completedRun)
    })

    vi.doMock("@/services/background-proxy", () => ({
      bgRequest: async (payload: TldwRequestPayload) => {
        const response = await tldwRequest(payload, {
          getConfig: mocks.getConfigMock,
          fetchFn: fetchRunDetails
        })
        if (!response.ok) throw new Error(response.error)
        return response.data
      }
    }))
    const { getRunDetails } = await vi.importActual<typeof import("@/services/watchlists")>(
      "@/services/watchlists"
    )
    mocks.getRunDetailsMock.mockImplementation(getRunDetails)
    vi.useFakeTimers()

    await act(async () => {
      render(
        <>
          <CachedRunStatus />
          <RunDetailDrawer open runId={10} onClose={vi.fn()} />
        </>
      )
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(screen.getByText("Running")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_001)
    })
    expect(screen.getByText("Collected two items")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_001)
    })
    expect(screen.getByText("Collected two items")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_001)
    })
    expect(screen.getByText("Completed")).toBeInTheDocument()
    expect(screen.getByText("Morning Brief generated")).toBeInTheDocument()
    expect(screen.queryByText("Stream error")).not.toBeInTheDocument()
    expect(MockWebSocket.instances).toHaveLength(0)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(screen.getByTestId("cached-run-status")).toHaveTextContent("completed")
    expect(fetchRunDetails).toHaveBeenCalledTimes(4)
  })

  it.each(["closing the drawer", "turning off Live"])(
    "ignores an in-flight hosted refresh after %s",
    async (action) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
      vi.useFakeTimers()
      let finishRefresh!: (details: typeof baseRunDetails) => void
      const inFlight = new Promise<typeof baseRunDetails>((resolve) => {
        finishRefresh = resolve
      })
      mocks.getRunDetailsMock.mockResolvedValueOnce(baseRunDetails).mockReturnValueOnce(inFlight)
      const onClose = vi.fn()
      const { rerender } = render(<RunDetailDrawer open runId={10} onClose={onClose} />)

      await act(async () => {
        await flushMicrotasks()
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(15_000)
      })
      expect(mocks.getRunDetailsMock).toHaveBeenCalledTimes(2)

      if (action === "closing the drawer") {
        rerender(<RunDetailDrawer open={false} runId={10} onClose={onClose} />)
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Live" }))
      }
      await act(async () => {
        finishRefresh({ ...baseRunDetails, status: "completed" })
        await vi.advanceTimersByTimeAsync(15_000)
      })

      expect(screen.queryByText("Completed")).not.toBeInTheDocument()
      expect(mocks.getRunDetailsMock).toHaveBeenCalledTimes(2)
    }
  )

  it("keeps polling the current hosted run when an earlier run loads late", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    vi.useFakeTimers()
    let finishEarlierRun!: (details: typeof baseRunDetails) => void
    const earlierRun = new Promise<typeof baseRunDetails>((resolve) => {
      finishEarlierRun = resolve
    })
    const currentRun = { ...baseRunDetails, id: 11, log_text: "Current run started" }
    mocks.getRunDetailsMock
      .mockReturnValueOnce(earlierRun)
      .mockResolvedValueOnce(currentRun)
      .mockResolvedValueOnce({
        ...currentRun,
        status: "completed",
        log_text: "Current run finished"
      })
    const onClose = vi.fn()
    const { rerender } = render(<RunDetailDrawer open runId={10} onClose={onClose} />)

    await act(async () => {
      rerender(<RunDetailDrawer open runId={11} onClose={onClose} />)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(screen.getByText("Current run started")).toBeInTheDocument()

    await act(async () => {
      finishEarlierRun(baseRunDetails)
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByText("Current run started")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_001)
    })
    expect(screen.getByText("Current run finished")).toBeInTheDocument()
    expect(mocks.getRunDetailsMock).toHaveBeenLastCalledWith(11, expect.any(AbortSignal))
  })

  it("keeps polling the current hosted run when a previous cancellation finishes", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    vi.useFakeTimers()
    let finishCancellation!: (result: { cancelled: boolean }) => void
    const cancellation = new Promise<{ cancelled: boolean }>((resolve) => {
      finishCancellation = resolve
    })
    mocks.cancelWatchlistRunMock.mockReturnValueOnce(cancellation)
    const currentRun = { ...baseRunDetails, id: 11 }
    let finishCurrentRun!: (details: typeof currentRun) => void
    const currentDetails = new Promise<typeof currentRun>((resolve) => {
      finishCurrentRun = resolve
    })
    mocks.getRunDetailsMock
      .mockResolvedValueOnce(baseRunDetails)
      .mockReturnValueOnce(currentDetails)
      .mockResolvedValueOnce({ ...currentRun, status: "completed" })
    const onClose = vi.fn()
    const { rerender } = render(<RunDetailDrawer open runId={10} onClose={onClose} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    fireEvent.click(screen.getByRole("button", { name: "Cancel run" }))
    expect(mocks.cancelWatchlistRunMock).toHaveBeenCalledWith(10)
    await act(async () => {
      rerender(<RunDetailDrawer open runId={11} onClose={onClose} />)
    })
    expect(screen.queryByRole("button", { name: "Cancel run" })).not.toBeInTheDocument()

    await act(async () => {
      finishCurrentRun(currentRun)
      finishCancellation({ cancelled: true })
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(screen.getByText("Running")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_001)
    })
    expect(screen.getByText("Completed")).toBeInTheDocument()
  })

  it("applies stream updates and transitions active run to completed without refetch", async () => {
    render(
      <>
        <CachedRunStatus />
        <RunDetailDrawer open runId={10} onClose={vi.fn()} />
      </>
    )
    await waitFor(() => expect(screen.getByTestId("cached-run-status")).toHaveTextContent("running"))

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
    await waitFor(() => expect(screen.getByTestId("cached-run-status")).toHaveTextContent("completed"))
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
      expect(mocks.getWatchlistRunAudioMock).toHaveBeenCalledWith(10, expect.any(AbortSignal))
    })
    expect(screen.getByText("Audio briefing")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText("Queued")).toBeInTheDocument()
    })
  })

  it("displays requested audio status from run stats when no task was created", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_requested: true,
        audio_briefing_status: "configuration_required",
        audio_briefing_reason: "tts_defaults_unavailable",
        audio_request_id: "wla_no_task"
      }
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Audio briefing")).toBeInTheDocument()
      expect(screen.getByText("Configuration required")).toBeInTheDocument()
      expect(screen.getByText("Reason: tts_defaults_unavailable")).toBeInTheDocument()
      expect(mocks.getWatchlistRunAudioMock).not.toHaveBeenCalled()
    })
  })

  it("shows a retry-created audio task before persisted run stats refresh", async () => {
    mocks.getRunDetailsMock.mockResolvedValue({
      ...baseRunDetails,
      status: "completed",
      finished_at: "2026-02-18T10:01:00Z",
      stats: {
        ...baseRunDetails.stats,
        audio_briefing_requested: true,
        audio_briefing_status: "queue_unavailable",
        audio_briefing_reason: "audio_queue_unavailable"
      }
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("No audio task created")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry audio" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry audio" }))

    await waitFor(() => {
      expect(mocks.retryWatchlistRunAudioMock).toHaveBeenCalledWith(10)
      expect(screen.getByText("Task retry-audio-10")).toBeInTheDocument()
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
      script_artifact: {
        title: "Briefing script",
        uri: "file:///srv/tldw/watchlists/runs/10/script.md",
        download_url: "/api/v1/watchlists/runs/10/audio/script/download"
      },
      speaker_artifacts: [
        {
          speaker_id: "host",
          label: "Host",
          uri: "file:///srv/tldw/watchlists/runs/10/host.mp3"
        },
        {
          speaker_id: "analyst",
          label: "Analyst",
          uri: "file:///srv/tldw/watchlists/runs/10/analyst.mp3",
          download_url: "/api/v1/watchlists/runs/10/audio/speakers/analyst/download"
        }
      ],
      final_artifact: {
        title: "Final mix",
        uri: "file:///srv/tldw/watchlists/runs/10/final.mp3",
        download_url: "/api/v1/watchlists/runs/10/audio/final/download",
        mime_type: "audio/mpeg"
      }
    })

    render(<RunDetailDrawer open runId={10} onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByText("Final audio available")).toBeInTheDocument()
    })
    const link = screen.getByRole("link", { name: "Download final audio" })
    expect(link).toHaveAttribute("href", "/api/v1/watchlists/runs/10/audio/download")
    expect(screen.getByText("Audio artifacts")).toBeInTheDocument()
    expect(screen.getByText("Briefing script")).toBeInTheDocument()
    expect(screen.getByText("Host")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("Final mix")).toBeInTheDocument()
    expect(screen.getByText("script.md")).toBeInTheDocument()
    expect(screen.getByText("host.mp3")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Briefing script" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/10/audio/script/download"
    )
    expect(screen.getByRole("link", { name: "Open Analyst" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/10/audio/speakers/analyst/download"
    )
    expect(screen.getByRole("link", { name: "Open Final mix" })).toHaveAttribute(
      "href",
      "/api/v1/watchlists/runs/10/audio/final/download"
    )
    expect(screen.queryByText(/file:\/\//)).not.toBeInTheDocument()
    expect(screen.queryByText(/srv\/tldw/)).not.toBeInTheDocument()
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
    render(
      <>
        <CachedRunStatus />
        <RunDetailDrawer open runId={10} onClose={vi.fn()} />
      </>
    )
    await waitFor(() => expect(screen.getByTestId("cached-run-status")).toHaveTextContent("running"))

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
    await waitFor(() => expect(screen.getByTestId("cached-run-status")).toHaveTextContent("completed"))
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
