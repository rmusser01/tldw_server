import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { PersonaLiveSessionSummary } from "@/services/persona-live-control"

const mocks = vi.hoisted(() => ({
  listPersonaLiveSessions: vi.fn(),
  createPersonaLiveSession: vi.fn(),
  focusPersonaLiveSession: vi.fn(),
  stopPersonaLiveSession: vi.fn(),
  buildPersonaWebSocketUrl: vi.fn(() => "ws://persona.test/api/v1/persona/stream"),
  ensureConfigForRequest: vi.fn()
}))

vi.mock("@/services/persona-live-control", () => ({
  listPersonaLiveSessions: () => mocks.listPersonaLiveSessions(),
  createPersonaLiveSession: (input: unknown) =>
    mocks.createPersonaLiveSession(input),
  focusPersonaLiveSession: (sessionId: unknown) =>
    mocks.focusPersonaLiveSession(sessionId),
  stopPersonaLiveSession: (sessionId: unknown) =>
    mocks.stopPersonaLiveSession(sessionId)
}))

vi.mock("@/services/persona-stream", () => ({
  buildPersonaWebSocketUrl: (options: unknown) =>
    (mocks.buildPersonaWebSocketUrl as (options: unknown) => unknown)(options)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    ensureConfigForRequest: (requireAuth?: boolean) =>
      mocks.ensureConfigForRequest(requireAuth)
  }
}))

import { usePersonaLiveControl } from "../usePersonaLiveControl"

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []

  readyState = MockWebSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(public readonly url: string) {
    MockWebSocket.instances.push(this)
  }

  send(payload: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error("socket is not open")
    }
    this.sent.push(payload)
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }

  emitOpen() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  emitError() {
    this.onerror?.()
  }
}

const getSentPayloads = (ws: MockWebSocket) =>
  ws.sent.map((payload) => JSON.parse(payload))

const session = (
  overrides: Partial<PersonaLiveSessionSummary> = {}
): PersonaLiveSessionSummary => ({
  sessionId: "sess-1",
  personaId: "persona-1",
  personaName: "Research Buddy",
  lifecycle: "idle",
  status: "active",
  isFocused: false,
  focusedAt: null,
  focusGeneration: null,
  lastActivityAt: null,
  pendingApprovalCount: 0,
  activeToolName: null,
  errorState: null,
  recoveryHint: null,
  suggestedVisualState: null,
  allowedActions: ["send_text_ws", "focus", "stop"],
  capabilities: {
    text: true,
    voice: false,
    browserMicrophoneRequired: false
  },
  ...overrides
})

describe("usePersonaLiveControl", () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.instances = []
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket)
    mocks.ensureConfigForRequest.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key",
      accessToken: ""
    })
    mocks.listPersonaLiveSessions.mockResolvedValue({
      sessions: [],
      focusedSessionId: null
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.stubGlobal("WebSocket", originalWebSocket)
  })

  it("loads sessions and chooses backend-focused session", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [
        session({ sessionId: "sess-a", isFocused: false }),
        session({ sessionId: "sess-b", personaId: "persona-2", isFocused: true })
      ],
      focusedSessionId: "sess-b"
    })

    const { result } = renderHook(() =>
      usePersonaLiveControl({ surface: "companion.conversation" })
    )

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(mocks.listPersonaLiveSessions).toHaveBeenCalledWith({
      personaId: null,
      surface: "companion.conversation"
    })
    expect(result.current.sessions.map((item) => item.sessionId)).toEqual([
      "sess-a",
      "sess-b"
    ])
    expect(result.current.focusedSession?.sessionId).toBe("sess-b")
  })

  it("focuses a session with optimistic pending state then backend result", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-a" }), session({ sessionId: "sess-b" })],
      focusedSessionId: "sess-a"
    })
    mocks.focusPersonaLiveSession.mockResolvedValueOnce(
      session({ sessionId: "sess-b", isFocused: true, focusGeneration: 7 })
    )

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.loading).toBe(false))

    let focusPromise: Promise<PersonaLiveSessionSummary>
    act(() => {
      focusPromise = result.current.focusSession("sess-b")
    })

    expect(result.current.pendingFocusSessionId).toBe("sess-b")

    await act(async () => {
      await focusPromise
    })

    expect(result.current.pendingFocusSessionId).toBeNull()
    expect(result.current.focusedSession?.sessionId).toBe("sess-b")
    expect(result.current.focusedSession?.focusGeneration).toBe(7)
  })

  it("starts a text session with an idempotency key", async () => {
    const created = session({
      sessionId: "sess-created",
      personaId: "persona-started",
      isFocused: true
    })
    mocks.createPersonaLiveSession.mockResolvedValueOnce(created)

    const { result } = renderHook(() =>
      usePersonaLiveControl({
        autoLoad: false,
        surface: "companion.conversation"
      })
    )

    await act(async () => {
      await result.current.startTextSession("persona-started")
    })

    expect(mocks.createPersonaLiveSession).toHaveBeenCalledWith(
      expect.objectContaining({
        personaId: "persona-started",
        reusePolicy: "resume_compatible",
        surface: "companion.conversation",
        idempotencyKey: expect.stringMatching(/^persona-live:/)
      })
    )
    expect(result.current.focusedSession?.sessionId).toBe("sess-created")
  })

  it("stops the focused session and refreshes summaries", async () => {
    mocks.listPersonaLiveSessions
      .mockResolvedValueOnce({
        sessions: [session({ sessionId: "sess-stop", isFocused: true })],
        focusedSessionId: "sess-stop"
      })
      .mockResolvedValueOnce({
        sessions: [session({ sessionId: "sess-stop", lifecycle: "stopped" })],
        focusedSessionId: null
      })
    mocks.stopPersonaLiveSession.mockResolvedValueOnce(
      session({ sessionId: "sess-stop", lifecycle: "stopped" })
    )

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-stop"))

    await act(async () => {
      await result.current.stopSession("sess-stop")
    })

    expect(mocks.stopPersonaLiveSession).toHaveBeenCalledWith("sess-stop")
    expect(mocks.listPersonaLiveSessions).toHaveBeenCalledTimes(2)
    expect(result.current.focusedSession).toBeNull()
  })

  it("opens a WebSocket and sends text with client_message_id", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-send", isFocused: true })],
      focusedSessionId: "sess-send"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-send"))

    let sendPromise: Promise<{ ok: boolean; clientMessageId: string; error?: string }>
    act(() => {
      sendPromise = result.current.sendText("hello buddy", {
        clientMessageId: "client-msg-1"
      })
    })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    expect(result.current.streamState).toBe("connecting")

    act(() => {
      MockWebSocket.instances[0].emitOpen()
    })

    const resultPayload = await sendPromise

    expect(resultPayload).toEqual({ ok: true, clientMessageId: "client-msg-1" })
    expect(mocks.ensureConfigForRequest).toHaveBeenCalledWith(true)
    expect(mocks.buildPersonaWebSocketUrl).toHaveBeenCalledWith(
      expect.objectContaining({ apiKey: "test-key" })
    )
    expect(getSentPayloads(MockWebSocket.instances[0])).toEqual([
      {
        type: "user_message",
        session_id: "sess-send",
        client_message_id: "client-msg-1",
        text: "hello buddy"
      }
    ])
    expect(result.current.streamState).toBe("open")
  })

  it("creates or resumes before sending when the focused session is stopped", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [
        session({
          sessionId: "sess-stopped",
          lifecycle: "stopped",
          isFocused: true,
          allowedActions: []
        })
      ],
      focusedSessionId: "sess-stopped"
    })
    mocks.createPersonaLiveSession.mockResolvedValueOnce(
      session({ sessionId: "sess-resumed", isFocused: true })
    )

    const { result } = renderHook(() =>
      usePersonaLiveControl({ defaultPersonaId: "persona-1" })
    )
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-stopped"))

    let sendPromise: Promise<{ ok: boolean; clientMessageId: string; error?: string }>
    act(() => {
      sendPromise = result.current.sendText("resume and send", {
        clientMessageId: "client-msg-2"
      })
    })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => {
      MockWebSocket.instances[0].emitOpen()
    })

    await expect(sendPromise).resolves.toEqual({
      ok: true,
      clientMessageId: "client-msg-2"
    })
    expect(mocks.createPersonaLiveSession).toHaveBeenCalledWith(
      expect.objectContaining({
        personaId: "persona-1",
        reusePolicy: "resume_compatible"
      })
    )
    expect(getSentPayloads(MockWebSocket.instances[0])[0].session_id).toBe(
      "sess-resumed"
    )
  })

  it("preserves composer text when WebSocket send fails", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-send", isFocused: true })],
      focusedSessionId: "sess-send"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-send"))

    let sendPromise: Promise<{ ok: boolean; clientMessageId: string; error?: string }>
    act(() => {
      sendPromise = result.current.sendText("keep this draft", {
        clientMessageId: "draft-msg"
      })
    })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => {
      MockWebSocket.instances[0].emitError()
    })

    await expect(sendPromise).resolves.toEqual({
      ok: false,
      clientMessageId: "draft-msg",
      error: "Persona live stream failed to connect"
    })
    await waitFor(() =>
      expect(result.current.lastSendError).toBe(
        "Persona live stream failed to connect"
      )
    )
  })

  it("reuses a caller-provided client_message_id when retrying a failed draft", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-send", isFocused: true })],
      focusedSessionId: "sess-send"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-send"))

    const first = result.current.sendText("draft", {
      clientMessageId: "draft-retry"
    })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => {
      MockWebSocket.instances[0].emitError()
    })
    await first

    const second = result.current.sendText("draft", {
      clientMessageId: "draft-retry"
    })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    act(() => {
      MockWebSocket.instances[1].emitOpen()
    })
    await second

    expect(getSentPayloads(MockWebSocket.instances[1])[0].client_message_id).toBe(
      "draft-retry"
    )
  })

  it("keeps text available when voice capability is false", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [
        session({
          sessionId: "sess-text-only",
          isFocused: true,
          capabilities: {
            text: true,
            voice: false,
            browserMicrophoneRequired: false
          }
        })
      ],
      focusedSessionId: "sess-text-only"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() => expect(result.current.focusedSession?.sessionId).toBe("sess-text-only"))

    expect(result.current.canSendText).toBe(true)
    expect(result.current.voiceAvailable).toBe(false)
  })
})
