import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ACPRestClient, ACPWebSocketClient } from "@/services/acp/client"
import type { ACPWSPermissionRequestMessage } from "@/services/acp/types"

describe("ACPRestClient", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalWindow = globalThis.window

  beforeEach(() => {
    vi.restoreAllMocks()
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "https://webui.example.test",
          protocol: "https:"
        }
      },
      configurable: true
    })
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "window", {
      value: originalWindow,
      configurable: true
    })
  })

  const createClient = () =>
    new ACPRestClient({
      serverUrl: "http://localhost:8000",
      getAuthHeaders: async () => ({ "X-API-KEY": "test-key" }),
      getAuthParams: async () => ({ api_key: "test-key" }),
    })

  it("lists sessions with query params", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ sessions: [], total: 0 }),
    })
    vi.stubGlobal("fetch", fetchMock)

    const client = createClient()
    await client.listSessions({ status: "active", limit: 25, offset: 10 })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/acp/sessions?status=active&limit=25&offset=10",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          "X-API-KEY": "test-key",
        }),
      })
    )
  })

  it("loads detail and usage endpoints", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          session_id: "sess-1",
          messages: [],
          policy_snapshot_version: "resolved-v1",
          policy_snapshot_fingerprint: "snapshot-123",
          policy_snapshot_refreshed_at: "2026-03-14T12:00:00+00:00",
          policy_summary: { allowed_tool_count: 2, approval_mode: "require_approval" },
          policy_provenance_summary: { source_kinds: ["capability_mapping"] },
          policy_refresh_error: null,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ session_id: "sess-1", usage: { total_tokens: 42 } }),
      })
    vi.stubGlobal("fetch", fetchMock)

    const client = createClient()
    const detail = await client.getSessionDetail("sess-1")
    await client.getSessionUsage("sess-1")

    expect(detail.policy_snapshot_version).toBe("resolved-v1")
    expect(detail.policy_snapshot_fingerprint).toBe("snapshot-123")
    expect(detail.policy_summary?.approval_mode).toBe("require_approval")
    expect(detail.policy_provenance_summary?.source_kinds).toEqual(["capability_mapping"])

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/api/v1/acp/sessions/sess-1/detail",
      expect.any(Object)
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/api/v1/acp/sessions/sess-1/usage",
      expect.any(Object)
    )
  })

  it("forks sessions using message_index payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: "fork-1",
        name: "Forked",
        forked_from: "sess-1",
        message_count: 4,
      }),
    })
    vi.stubGlobal("fetch", fetchMock)

    const client = createClient()
    await client.forkSession("sess-1", { message_index: 3, name: "Forked" })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/acp/sessions/sess-1/fork",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message_index: 3, name: "Forked" }),
      })
    )
  })

  it("uses the shared quickstart http base for rest calls", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ sessions: [], total: 0 }),
    })
    vi.stubGlobal("fetch", fetchMock)

    const client = createClient()
    await client.listSessions({ limit: 10 })

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/acp/sessions?limit=10",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-API-KEY": "test-key",
        }),
      })
    )
  })

  it("keeps tier while exposing permission policy metadata", () => {
    const message: ACPWSPermissionRequestMessage = {
      type: "permission_request",
      request_id: "req-1",
      session_id: "sess-1",
      tool_name: "web.search",
      tool_arguments: { query: "opa" },
      tier: "individual",
      timeout_seconds: 30,
      approval_requirement: "approval_required",
      governance_reason: "policy_approval_required",
      deny_reason: undefined,
      provenance_summary: { source_kinds: ["capability_mapping"] },
      runtime_narrowing_reason: "workspace_trust_required",
      policy_snapshot_fingerprint: "snapshot-123",
    }

    expect(message.tier).toBe("individual")
    expect(message.approval_requirement).toBe("approval_required")
    expect(message.policy_snapshot_fingerprint).toBe("snapshot-123")
    expect(message.provenance_summary?.source_kinds).toEqual(["capability_mapping"])
  })

  it("preserves status and code metadata on request errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => ({
        detail: "Session already exists",
        error_code: "duplicate_session"
      }),
    })
    vi.stubGlobal("fetch", fetchMock)

    const client = createClient()

    await expect(client.getAvailableAgents()).rejects.toMatchObject({
      message: "Session already exists",
      status: 409,
      code: "duplicate_session",
      detail: "Session already exists",
      details: {
        detail: "Session already exists",
        error_code: "duplicate_session"
      }
    })
  })
})

describe("ACPWebSocketClient", () => {
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

    constructor(url: string) {
      this.url = url
      MockWebSocket.instances.push(this)
    }

    close(): void {
      this.readyState = MockWebSocket.CLOSED
    }

    send(): void {}

    open(): void {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.()
    }

    closeWith(code: number, reason = ""): void {
      this.readyState = MockWebSocket.CLOSED
      this.onclose?.({ code, reason } as CloseEvent)
    }
  }

  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalWindow = globalThis.window

  const createClient = () =>
    new ACPWebSocketClient({
      serverUrl: "http://localhost:8000",
      getAuthHeaders: async () => ({ "X-API-KEY": "test-key" }),
      getAuthParams: async () => ({ api_key: "test-key" }),
    })

  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket)
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://127.0.0.1:8080",
          protocol: "http:"
        }
      },
      configurable: true
    })
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "window", {
      value: originalWindow,
      configurable: true
    })
  })

  it("uses the webui origin for quickstart websocket sessions", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    const client = createClient()
    await client.connect("sess-1")

    expect(MockWebSocket.instances).toHaveLength(1)
    expect(MockWebSocket.instances[0].url).toBe(
      "ws://127.0.0.1:8080/api/v1/acp/sessions/sess-1/stream?api_key=test-key"
    )
  })

  it.each([4401, 4404, 4429])("does not reconnect for fatal close code %s", async (code) => {
    const client = createClient()

    await client.connect("sess-1")
    expect(MockWebSocket.instances).toHaveLength(1)

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.closeWith(code)

    await vi.advanceTimersByTimeAsync(60000)

    expect(MockWebSocket.instances).toHaveLength(1)
  })
})
