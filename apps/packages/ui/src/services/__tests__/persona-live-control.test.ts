import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import {
  createPersonaLiveSession,
  focusPersonaLiveSession,
  listPersonaLiveSessions,
  stopPersonaLiveSession
} from "@/services/persona-live-control"

describe("persona live-control service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("normalizes live session summaries with safe defaults", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        sessions: [
          {
            session_id: "sess-1",
            persona_id: "persona-1",
            persona_name: "Research Buddy",
            lifecycle: "connected",
            is_focused: true,
            capabilities: { text: true }
          }
        ],
        focused_session_id: "sess-1"
      })
    })

    const result = await listPersonaLiveSessions()

    expect(result.focusedSessionId).toBe("sess-1")
    expect(result.sessions).toEqual([
      expect.objectContaining({
        sessionId: "sess-1",
        personaId: "persona-1",
        personaName: "Research Buddy",
        lifecycle: "connected",
        isFocused: true,
        focusedAt: null,
        focusGeneration: null,
        lastActivityAt: null,
        pendingApprovalCount: 0,
        suggestedVisualState: null,
        allowedActions: [],
        capabilities: {
          text: true,
          voice: false,
          browserMicrophoneRequired: false
        }
      })
    ])
    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/live/sessions",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("preserves focused session ordering fields", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        sessions: [
          {
            session_id: "sess-focused",
            persona_id: "persona-1",
            persona_name: "Focused",
            lifecycle: "idle",
            is_focused: true,
            focused_at: "2026-05-20T12:00:00Z",
            focus_generation: 42,
            last_activity_at: "2026-05-20T12:01:00Z"
          }
        ],
        focused_session_id: "sess-focused"
      })
    })

    const result = await listPersonaLiveSessions({ limit: 3 })
    const focused = result.sessions[0]

    expect(focused.focusedAt).toBe("2026-05-20T12:00:00Z")
    expect(focused.focusGeneration).toBe(42)
    expect(focused.lastActivityAt).toBe("2026-05-20T12:01:00Z")
    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/live/sessions?limit=3",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("marks text-only capability when voice is absent", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        sessions: [
          {
            session_id: "sess-text",
            persona_id: "persona-1",
            persona_name: "Text Buddy",
            lifecycle: "idle",
            capabilities: { text: true, voice: false }
          }
        ],
        focused_session_id: null
      })
    })

    const result = await listPersonaLiveSessions()

    expect(result.sessions[0].capabilities).toEqual({
      text: true,
      voice: false,
      browserMicrophoneRequired: false
    })
  })

  it("calls create with idempotency_key and resume_compatible policy", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        session: {
          session_id: "sess-created",
          persona_id: "persona-1",
          persona_name: "Research Buddy",
          lifecycle: "idle"
        }
      })
    })

    const session = await createPersonaLiveSession({
      personaId: "persona-1",
      idempotencyKey: "create-key",
      surface: "companion.conversation"
    })

    expect(session.sessionId).toBe("sess-created")
    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/live/sessions",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          persona_id: "persona-1",
          reuse_policy: "resume_compatible",
          idempotency_key: "create-key",
          surface: "companion.conversation"
        })
      })
    )
  })

  it("calls focus and stop endpoints", async () => {
    mocks.fetchWithAuth
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          session: {
            session_id: "sess-focus",
            persona_id: "persona-1",
            persona_name: "Research Buddy",
            lifecycle: "idle"
          }
        })
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          session: {
            session_id: "sess-focus",
            persona_id: "persona-1",
            persona_name: "Research Buddy",
            lifecycle: "stopped"
          }
        })
      })

    await focusPersonaLiveSession("sess/focus")
    await stopPersonaLiveSession("sess/focus")

    expect(mocks.fetchWithAuth).toHaveBeenNthCalledWith(
      1,
      "/api/v1/persona/live/sessions/sess%2Ffocus/focus",
      expect.objectContaining({ method: "POST" })
    )
    expect(mocks.fetchWithAuth).toHaveBeenNthCalledWith(
      2,
      "/api/v1/persona/live/sessions/sess%2Ffocus/stop",
      expect.objectContaining({ method: "POST" })
    )
  })
})
