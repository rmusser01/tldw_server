/* @vitest-environment jsdom */
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useACPSession } from "@/hooks/useACPSession"

const { useStorageMock } = vi.hoisted(() => ({
  useStorageMock: vi.fn(),
}))

const { getConfigMock } = vi.hoisted(() => ({
  getConfigMock: vi.fn(),
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: useStorageMock,
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: getConfigMock,
  },
}))

class MockWebSocket {
  static instances: MockWebSocket[] = []
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readonly url: string
  readyState = MockWebSocket.CONNECTING
  onopen: (() => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  sent: string[] = []

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED
  }

  send(payload: string): void {
    this.sent.push(payload)
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  emitMessage(payload: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
  }

  closeWith(code: number, reason = ""): void {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({ code, reason } as CloseEvent)
  }
}

describe("useACPSession", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket)
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => {
      const overrides: Record<string, unknown> = {
        serverUrl: "http://localhost:8000",
        authMode: "single-user",
        apiKey: "test-key",
        accessToken: "",
      }
      return [overrides[key] ?? defaultValue, vi.fn(), { isLoading: false }] as const
    })
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key",
      accessToken: "",
    })
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it.each([4401, 4404, 4429])("does not reconnect for fatal close code %s", async (code) => {
    const { result } = renderHook(() =>
      useACPSession({
        sessionId: "session-1",
        autoConnect: true,
      })
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(MockWebSocket.instances).toHaveLength(1)

    await act(async () => {
      MockWebSocket.instances[0].closeWith(code)
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(60000)
    })

    expect(MockWebSocket.instances).toHaveLength(1)
    expect(result.current.state).toBe("disconnected")
  })

  it("sends denied permission responses and clears the pending request", async () => {
    const { result } = renderHook(() =>
      useACPSession({
        sessionId: "session-1",
        autoConnect: true,
      })
    )

    await act(async () => {
      await Promise.resolve()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      ws.emitMessage({ type: "connected", agent_capabilities: {} })
      ws.emitMessage({
        type: "permission_request",
        request_id: "req-deny-1",
        session_id: "session-1",
        tool_name: "fs.write",
        tool_arguments: { path: "README.md" },
        tier: "individual",
        timeout_seconds: 300,
      })
      await Promise.resolve()
    })

    expect(result.current.state).toBe("waiting_permission")
    expect(result.current.pendingPermissions).toHaveLength(1)

    await act(async () => {
      result.current.denyPermission("req-deny-1")
      await Promise.resolve()
    })

    expect(JSON.parse(ws.sent.at(-1) ?? "{}")).toEqual({
      type: "permission_response",
      request_id: "req-deny-1",
      approved: false,
    })
    expect(result.current.pendingPermissions).toEqual([])
    expect(result.current.state).toBe("running")
  })

  it("retries transient closes and exposes reconnect progress", async () => {
    const { result } = renderHook(() =>
      useACPSession({
        sessionId: "session-1",
        autoConnect: true,
      })
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(MockWebSocket.instances).toHaveLength(1)

    await act(async () => {
      MockWebSocket.instances[0].open()
      MockWebSocket.instances[0].closeWith(1006)
      await Promise.resolve()
    })

    expect(result.current.reconnectInfo).toEqual({
      isReconnecting: true,
      attempt: 1,
      maxAttempts: 10,
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
    expect(result.current.state).toBe("connecting")
  })
})
