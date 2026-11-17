/* @vitest-environment jsdom */
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useACPSession } from "@/hooks/useACPSession"
import { WS_CONFIG } from "@/services/acp/constants"

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
    vi.unstubAllEnvs()
    vi.clearAllMocks()
  })

  it.each([4401, 4403, 4404, 4429])("does not reconnect for fatal close code %s", async (code) => {
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

  it("uses the page origin without query credentials for cookie sessions", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    getConfigMock.mockResolvedValue({
      serverUrl: "https://remote.example.test",
      authMode: "single-user",
      authSource: "cookie-session",
      apiKey: "stale-key",
      accessToken: "stale-token"
    })

    renderHook(() =>
      useACPSession({
        sessionId: "session-1",
        autoConnect: true
      })
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(MockWebSocket.instances).toHaveLength(1)
    expect(MockWebSocket.instances[0].url).toBe(
      `${window.location.origin.replace(/^http/i, "ws")}/api/v1/acp/sessions/session-1/stream`
    )
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
      await vi.advanceTimersByTimeAsync(WS_CONFIG.RECONNECT_DELAY_MS)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
    expect(result.current.state).toBe("connecting")
  })
  it("shows a running externally started task after the socket connects", async () => {
    const onPromptComplete = vi.fn()
    const { result } = renderHook(() => useACPSession({ sessionId: "session-1", onPromptComplete }))
    await act(async () => { await Promise.resolve() })
    const socket = MockWebSocket.instances[0]
    act(() => { socket.open(); socket.emitMessage({ type: "connected", session_id: "session-1" }) })
    expect(result.current.state).toBe("connected")
    act(() => socket.emitMessage({ type: "update", session_id: "session-1", update_type: "agent_message_chunk", data: { text: "Reading the public README" } }))
    expect(result.current.state).toBe("running")
    expect(MockWebSocket.instances).toHaveLength(1)
    expect(socket.sent).toEqual([])
    act(() => result.current.cancel())
    expect(socket.sent.map((message) => JSON.parse(message))).toEqual([{ type: "cancel", session_id: "session-1" }])
    act(() => socket.emitMessage({ type: "update", session_id: "session-1", update_type: "cancelled", data: { message: "Operation cancelled" } }))
    expect(result.current.state).toBe("running")
    act(() => socket.emitMessage({ type: "prompt_settled", session_id: "foreign-session" }))
    expect(result.current.state).toBe("running")
    act(() => socket.emitMessage({ type: "prompt_settled", session_id: "session-1" }))
    expect(result.current.state).toBe("connected")
    expect(onPromptComplete).not.toHaveBeenCalled()
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it.each([
    ["session_info_update", undefined],
    ["available_commands_update", undefined],
    ["current_mode_update", undefined],
    ["config_option_update", undefined],
    ["tool_call_update", "completed"],
    ["tool_call_update", "failed"],
    ["tool_call_update", "cancelled"],
  ])("does not infer active work from idle/retired %s %s", async (kind, status) => {
    const { result } = renderHook(() => useACPSession({ sessionId: "session-1" }))
    await act(async () => { await Promise.resolve() })
    const socket = MockWebSocket.instances[0]
    act(() => { socket.open(); socket.emitMessage({ type: "connected", session_id: "session-1" }) })
    const update = { type: "update", session_id: "session-1", update_type: kind, data: { update: { status } } }
    act(() => socket.emitMessage(update))
    expect(result.current.state).toBe("connected")
    act(() => socket.emitMessage({ ...update, prompt_active: true }))
    expect(result.current.state).toBe("running")
    act(() => socket.emitMessage({ ...update, prompt_active: false }))
    expect(result.current.state).toBe("running")
    act(() => socket.emitMessage({ type: "prompt_settled", session_id: "session-1" }))
    expect(result.current.state).toBe("connected")
    act(() => socket.emitMessage({ ...update, prompt_active: false }))
    expect(result.current.state).toBe("connected")
    act(() => socket.emitMessage({ ...update, prompt_active: true, session_id: "foreign" }))
    expect(result.current.state).toBe("connected")
  })

  it("does not infer work from an explicitly inactive native tool update", async () => {
    const { result } = renderHook(() => useACPSession({ sessionId: "session-1" }))
    await act(async () => { await Promise.resolve() })
    const socket = MockWebSocket.instances[0]
    act(() => { socket.open(); socket.emitMessage({ type: "connected", session_id: "session-1" }) })
    act(() => socket.emitMessage({ type: "update", session_id: "session-1", update_type: "tool_call_update", prompt_active: false, data: { update: { status: "in_progress" } } }))
    expect(result.current.state).toBe("connected")
    act(() => socket.emitMessage({ type: "update", session_id: "session-1", update_type: "tool_call_update", data: { update: { status: "in_progress" } } }))
    expect(result.current.state).toBe("running")
  })

  it("semantic ACP connection keeps a running prompt on equal config rehydration", async () => {
    let stored = { serverUrl: "http://localhost:8000", authMode: "single-user", apiKey: "test-key" }
    useStorageMock.mockImplementation((key: string, fallback: unknown) => [key === "tldwConfig" ? stored : fallback, vi.fn()])
    getConfigMock.mockImplementation(async () => ({ ...stored }))
    const { result, rerender } = renderHook(() => useACPSession({ sessionId: "session-1" }))
    await act(async () => { await Promise.resolve() })
    const socket = MockWebSocket.instances[0]
    act(() => { socket.open(); socket.emitMessage({ type: "connected", session_id: "session-1" }) })
    act(() => result.current.sendPrompt([{ role: "user", content: "Controlled pending prompt" }]))
    stored = { ...stored }
    await act(async () => { rerender(); await Promise.resolve() })
    expect(MockWebSocket.instances).toHaveLength(1)
    expect(socket.readyState).toBe(MockWebSocket.OPEN)
    expect(result.current.state).toBe("running")
    expect(socket.sent.map((entry) => JSON.parse(entry).type)).toEqual(["prompt"])
  })

  it.each([
    ["serverUrl", "http://localhost:9000"], ["authMode", "multi-user"],
    ["apiKey", "replacement-key"], ["accessToken", "replacement-access"],
    ["refreshToken", "replacement-refresh"], ["orgId", 9], ["authSource", "cookie-session"]
  ])("semantic ACP connection retires changed %s", async (field, value) => {
    let stored: Record<string, unknown> = { serverUrl: "http://localhost:8000", authMode: "single-user", apiKey: "test-key" }
    useStorageMock.mockImplementation((key: string, fallback: unknown) => [key === "tldwConfig" ? stored : fallback, vi.fn()])
    getConfigMock.mockImplementation(async () => ({ ...stored }))
    const { rerender } = renderHook(() => useACPSession({ sessionId: "session-1" }))
    await act(async () => { await Promise.resolve() })
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    stored = { ...stored, [field]: value }
    await act(async () => { rerender(); await Promise.resolve() })
    expect(socket.readyState).toBe(MockWebSocket.CLOSED)
    expect(MockWebSocket.instances).toHaveLength(2)
  })

  it("semantic ACP connection retires a changed session and closes on unmount", async () => {
    const { rerender, unmount } = renderHook(({ sessionId }) => useACPSession({ sessionId }), { initialProps: { sessionId: "session-1" } })
    await act(async () => { await Promise.resolve() })
    const original = MockWebSocket.instances[0]
    act(() => original.open())
    await act(async () => { rerender({ sessionId: "session-2" }); await Promise.resolve() })
    expect(original.readyState).toBe(MockWebSocket.CLOSED)
    expect(MockWebSocket.instances).toHaveLength(2)
    unmount()
    expect(MockWebSocket.instances[1].readyState).toBe(MockWebSocket.CLOSED)
  })

})
