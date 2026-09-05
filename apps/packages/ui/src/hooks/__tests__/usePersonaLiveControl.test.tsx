import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type {
  PersonaLiveSessionList,
  PersonaLiveSessionSummary
} from "@/services/persona-live-control"

const mocks = vi.hoisted(() => ({
  listPersonaLiveSessions: vi.fn(),
  createPersonaLiveSession: vi.fn(),
  focusPersonaLiveSession: vi.fn(),
  stopPersonaLiveSession: vi.fn(),
  buildPersonaWebSocketUrl: vi.fn(() => ({
    url: "ws://persona.test/api/v1/persona/stream",
    protocols: ["bearer", "test-key"]
  })),
  ensureConfigForRequest: vi.fn()
}))

vi.mock("@/services/persona-live-control", () => ({
  listPersonaLiveSessions: (input: unknown) =>
    mocks.listPersonaLiveSessions(input),
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

  constructor(
    public readonly url: string,
    public readonly protocols?: string | string[]
  ) {
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

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

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

  it("ignores a discarded StrictMode startup list after starting a session", async () => {
    const discarded = deferred<PersonaLiveSessionList>()
    mocks.listPersonaLiveSessions.mockReturnValueOnce(discarded.promise)
    mocks.createPersonaLiveSession.mockResolvedValueOnce(
      session({ sessionId: "sess-current", isFocused: true })
    )
    const { result } = renderHook(() => usePersonaLiveControl(), {
      wrapper: React.StrictMode
    })
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.startTextSession("persona-1")
      discarded.resolve({ sessions: [], focusedSessionId: null })
    })

    expect(result.current.focusedSession?.sessionId).toBe("sess-current")
    expect(result.current.canSendText).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it.each(["start", "focus"])(
    "preserves a newer %s result when an earlier list finishes",
    async (action) => {
      const pendingList = deferred<PersonaLiveSessionList>()
      mocks.listPersonaLiveSessions.mockReturnValueOnce(pendingList.promise)
      const current = session({ sessionId: "sess-current", isFocused: true })
      if (action === "start") {
        mocks.createPersonaLiveSession.mockResolvedValueOnce(current)
      } else {
        mocks.focusPersonaLiveSession.mockResolvedValueOnce(current)
      }
      const { result } = renderHook(() => usePersonaLiveControl())

      await act(async () => {
        if (action === "start") {
          await result.current.startTextSession("persona-1")
        } else {
          await result.current.focusSession("sess-current")
        }
        pendingList.resolve({ sessions: [], focusedSessionId: null })
      })

      expect(result.current.focusedSession?.sessionId).toBe("sess-current")
      expect(result.current.canSendText).toBe(true)
      expect(result.current.loading).toBe(false)
    }
  )

  it.each(["resolve", "reject"])(
    "keeps the latest reload loading when an older request finishes via %s",
    async (outcome) => {
      const older = deferred<PersonaLiveSessionList>()
      const newer = deferred<PersonaLiveSessionList>()
      mocks.listPersonaLiveSessions
        .mockReturnValueOnce(older.promise)
        .mockReturnValueOnce(newer.promise)
      const { result } = renderHook(() => usePersonaLiveControl())
      let reloadPromise!: Promise<PersonaLiveSessionList>
      act(() => {
        reloadPromise = result.current.reload()
      })
      await act(async () => {
        if (outcome === "resolve") {
          older.resolve({
            sessions: [session({ sessionId: "sess-old", isFocused: true })],
            focusedSessionId: "sess-old"
          })
        } else {
          older.reject(new Error("Old list failed"))
        }
      })
      expect(result.current.loading).toBe(true)
      expect(result.current.focusedSession).toBeNull()
      expect(result.current.error).toBeNull()

      await act(async () => {
        newer.resolve({
          sessions: [session({ sessionId: "sess-new", isFocused: true })],
          focusedSessionId: "sess-new"
        })
        await reloadPromise
      })
      expect(result.current.loading).toBe(false)
      expect(result.current.focusedSession?.sessionId).toBe("sess-new")
    }
  )

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

  it.each([false, true])(
    "opens a WebSocket and sends text (StrictMode: %s)",
    async (strictMode) => {
      mocks.listPersonaLiveSessions.mockResolvedValue({
        sessions: [session({ sessionId: "sess-send", isFocused: true })],
        focusedSessionId: "sess-send"
      })

      const { result, unmount } = renderHook(() => usePersonaLiveControl(), {
        wrapper: strictMode ? React.StrictMode : React.Fragment
      })
      await waitFor(() =>
        expect(result.current.focusedSession?.sessionId).toBe("sess-send")
      )

      let sendPromise: Promise<{
        ok: boolean
        clientMessageId: string
        error?: string
      }>
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

      expect(resultPayload).toEqual({
        ok: true,
        clientMessageId: "client-msg-1"
      })
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
      unmount()
      expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.CLOSED)
      expect(MockWebSocket.instances[0].onopen).toBeNull()
      expect(MockWebSocket.instances[0].onclose).toBeNull()
      expect(MockWebSocket.instances[0].onerror).toBeNull()
    }
  )

  it("cancels a send from the discarded StrictMode mount before its session resolves", async () => {
    let resolveDiscarded!: (value: PersonaLiveSessionSummary) => void
    mocks.createPersonaLiveSession
      .mockImplementationOnce(
        () =>
          new Promise<PersonaLiveSessionSummary>((resolve) => {
            resolveDiscarded = resolve
          })
      )
      .mockResolvedValueOnce(
        session({ sessionId: "sess-current", isFocused: true })
      )
    const sends: ReturnType<
      ReturnType<typeof usePersonaLiveControl>["sendText"]
    >[] = []
    const { result } = renderHook(
      () => {
        const live = usePersonaLiveControl({
          autoLoad: false,
          defaultPersonaId: "persona-1"
        })
        const { sendText } = live
        React.useEffect(() => {
          sends.push(
            sendText("hello", { clientMessageId: `mount-${sends.length}` })
          )
        }, [sendText])
        return live
      },
      { wrapper: React.StrictMode }
    )

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const ws = MockWebSocket.instances[0]
    await act(async () => {
      resolveDiscarded(
        session({ sessionId: "sess-discarded", isFocused: true })
      )
    })
    expect(result.current.focusedSession?.sessionId).toBe("sess-current")
    expect(result.current.lastSendError).toBeNull()
    expect(result.current.streamState).toBe("connecting")

    await act(async () => {
      ws.emitOpen()
      await Promise.all(sends)
    })
    expect(await sends[0]).toMatchObject({
      ok: false,
      clientMessageId: "mount-0"
    })
    expect(await sends[1]).toEqual({ ok: true, clientMessageId: "mount-1" })
    expect(getSentPayloads(ws)).toEqual([
      {
        type: "user_message",
        session_id: "sess-current",
        client_message_id: "mount-1",
        text: "hello"
      }
    ])
  })

  it("does not create a socket when configuration resolves after unmount", async () => {
    let resolveConfig!: (value: { serverUrl: string; apiKey: string }) => void
    mocks.ensureConfigForRequest.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveConfig = resolve
        })
    )
    mocks.listPersonaLiveSessions.mockResolvedValue({
      sessions: [session({ isFocused: true })],
      focusedSessionId: "sess-1"
    })
    const { result, unmount } = renderHook(() => usePersonaLiveControl())
    await waitFor(() =>
      expect(result.current.focusedSession?.sessionId).toBe("sess-1")
    )
    let sendPromise!: ReturnType<typeof result.current.sendText>
    act(() => {
      sendPromise = result.current.sendText("draft", {
        clientMessageId: "cancelled"
      })
    })
    await waitFor(() => expect(result.current.streamState).toBe("connecting"))
    unmount()
    resolveConfig({ serverUrl: "http://persona.test", apiKey: "test-key" })

    await expect(sendPromise).resolves.toMatchObject({
      ok: false,
      clientMessageId: "cancelled"
    })
    expect(MockWebSocket.instances).toHaveLength(0)
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

  it.each([false, true])(
    "cancels pending text on unmount (handshake just opened: %s)",
    async (opened) => {
      mocks.listPersonaLiveSessions.mockResolvedValueOnce({
        sessions: [session({ sessionId: "sess-send", isFocused: true })],
        focusedSessionId: "sess-send"
      })

      const { result, unmount } = renderHook(() => usePersonaLiveControl())
      await waitFor(() =>
        expect(result.current.focusedSession?.sessionId).toBe("sess-send")
      )
      vi.useFakeTimers()

      let sendPromise!: ReturnType<typeof result.current.sendText>
      act(() => {
        sendPromise = result.current.sendText("draft", {
          clientMessageId: "draft-timeout"
        })
      })

      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
        await Promise.resolve()
      })

      expect(MockWebSocket.instances).toHaveLength(1)
      expect(vi.getTimerCount()).toBeGreaterThan(0)

      // Opening and unmounting in the same turn also cancels the send continuation.
      act(() => {
        if (opened) MockWebSocket.instances[0].emitOpen()
        unmount()
      })

      expect(vi.getTimerCount()).toBe(0)
      await expect(sendPromise).resolves.toMatchObject({
        ok: false,
        clientMessageId: "draft-timeout"
      })
      expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.CLOSED)
      expect(getSentPayloads(MockWebSocket.instances[0])).toEqual([])
    }
  )

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

  it("opens the stream with the auth subprotocol and no token in the url", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-proto", isFocused: true })],
      focusedSessionId: "sess-proto"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() =>
      expect(result.current.focusedSession?.sessionId).toBe("sess-proto")
    )

    act(() => {
      void result.current.sendText("hi", { clientMessageId: "proto-msg" })
    })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const ws = MockWebSocket.instances[0]
    expect(ws.url).not.toContain("token")
    expect(ws.url).not.toContain("api_key")
    expect(ws.url).not.toContain("test-key")
    expect(ws.protocols).toEqual(["bearer", "test-key"])

    act(() => {
      ws.emitOpen()
    })
  })

  it("times out the handshake when onopen never fires and reports a connect failure", async () => {
    mocks.listPersonaLiveSessions.mockResolvedValueOnce({
      sessions: [session({ sessionId: "sess-timeout", isFocused: true })],
      focusedSessionId: "sess-timeout"
    })

    const { result } = renderHook(() => usePersonaLiveControl())
    await waitFor(() =>
      expect(result.current.focusedSession?.sessionId).toBe("sess-timeout")
    )

    vi.useFakeTimers()

    let sendPromise!: Promise<{
      ok: boolean
      clientMessageId: string
      error?: string
    }>
    act(() => {
      sendPromise = result.current.sendText("hi", {
        clientMessageId: "timeout-msg"
      })
    })

    // Flush the ensureConfigForRequest microtask so the socket + timer exist.
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(MockWebSocket.instances).toHaveLength(1)
    expect(result.current.streamState).toBe("connecting")

    // Advance past the handshake timeout without ever firing onopen.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    await expect(sendPromise).resolves.toEqual({
      ok: false,
      clientMessageId: "timeout-msg",
      error: "Persona live stream failed to connect"
    })
    expect(result.current.streamState).toBe("error")
    expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.CLOSED)
  })
})
