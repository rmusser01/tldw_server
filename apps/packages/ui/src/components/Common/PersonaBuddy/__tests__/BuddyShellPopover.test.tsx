import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { BuddyShellPopover } from "../BuddyShellPopover"

const buildBuddySummary = () => ({
  has_buddy: true,
  persona_name: "Research Buddy",
  role_summary: "Keeps the route on track",
  visual: {
    species_id: "owl",
    silhouette_id: "perch",
    palette_id: "dawn"
  }
})

const buildLiveSession = (overrides: Record<string, unknown> = {}) => ({
  sessionId: "live-session-1",
  personaId: "persona-1",
  personaName: "Research Buddy",
  lifecycle: "connected",
  status: "active",
  isFocused: true,
  focusedAt: null,
  focusGeneration: 1,
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

const renderPopover = (props: Partial<React.ComponentProps<typeof BuddyShellPopover>> = {}) =>
  render(
    <MemoryRouter>
      <BuddyShellPopover
        buddySummary={buildBuddySummary()}
        personaId="persona-1"
        {...props}
      />
    </MemoryRouter>
  )

describe("BuddyShellPopover", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("opens a compact popover with start stop and composer controls", () => {
    renderPopover({
      liveControl: {
        sessions: [buildLiveSession(), buildLiveSession({ sessionId: "live-session-2" })],
        focusedSession: buildLiveSession(),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.getByTestId("persona-buddy-session-select")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument()
    expect(screen.getByTestId("persona-buddy-text-input")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Full Live View" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=live"
    )
    expect(screen.getByRole("link", { name: "Choose/Change Buddy" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=visuals"
    )
  })

  it("starts a session before sending text when no focused session exists", async () => {
    const startTextSession = vi.fn(async () => buildLiveSession())
    const sendText = vi.fn(async () => ({
      ok: true,
      clientMessageId: "client-1"
    }))

    renderPopover({
      liveControl: {
        sessions: [],
        focusedSession: null,
        focusedSessionId: null,
        streamState: "closed",
        canSendText: false,
        pendingFocusSessionId: null,
        startTextSession,
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText
      }
    })

    fireEvent.change(screen.getByTestId("persona-buddy-text-input"), {
      target: { value: "hello buddy" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      expect(startTextSession).toHaveBeenCalledWith("persona-1")
    })
    await waitFor(() => {
      expect(sendText).toHaveBeenCalledWith(
        "hello buddy",
        expect.objectContaining({
          clientMessageId: expect.stringMatching(/^persona-buddy-draft:/)
        })
      )
    })
    expect(screen.getByTestId("persona-buddy-text-input")).toHaveValue("")
  })

  it("preserves composer text when sendText fails", async () => {
    const sendText = vi.fn(async () => ({
      ok: false,
      clientMessageId: "client-fail",
      error: "socket closed"
    }))

    renderPopover({
      liveControl: {
        sessions: [buildLiveSession()],
        focusedSession: buildLiveSession(),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText
      }
    })

    fireEvent.change(screen.getByTestId("persona-buddy-text-input"), {
      target: { value: "retry this" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await screen.findByText("socket closed")
    expect(screen.getByTestId("persona-buddy-text-input")).toHaveValue("retry this")
  })

  it("routes approval-needed state to full Live view without approve buttons", () => {
    renderPopover({
      liveControl: {
        sessions: [buildLiveSession({ pendingApprovalCount: 2 })],
        focusedSession: buildLiveSession({ pendingApprovalCount: 2 }),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.getByText("Needs approval")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Full Live View" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument()
  })

  it("routes voice-capable sessions to the full Live listening surface", () => {
    renderPopover({
      liveControl: {
        sessions: [
          buildLiveSession({
            capabilities: {
              text: true,
              voice: true,
              browserMicrophoneRequired: true
            }
          })
        ],
        focusedSession: buildLiveSession({
          capabilities: {
            text: true,
            voice: true,
            browserMicrophoneRequired: true
          }
        }),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        voiceAvailable: true,
        voiceState: "idle",
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.getByRole("link", { name: "Listen" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=live"
    )
    expect(screen.queryByRole("link", { name: "Stop listening" })).not.toBeInTheDocument()
  })

  it("routes active voice sessions to the full Live stop-listening surface", () => {
    renderPopover({
      liveControl: {
        sessions: [
          buildLiveSession({
            capabilities: {
              text: true,
              voice: true,
              browserMicrophoneRequired: true
            }
          })
        ],
        focusedSession: buildLiveSession({
          capabilities: {
            text: true,
            voice: true,
            browserMicrophoneRequired: true
          }
        }),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        voiceAvailable: true,
        voiceState: "listening",
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.getByRole("link", { name: "Stop listening" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=live"
    )
    expect(screen.queryByRole("link", { name: "Listen" })).not.toBeInTheDocument()
  })

  it("treats pending voice starts as listening before route state changes", () => {
    renderPopover({
      liveControl: {
        sessions: [
          buildLiveSession({
            capabilities: {
              text: true,
              voice: true,
              browserMicrophoneRequired: true
            }
          })
        ],
        focusedSession: buildLiveSession({
          capabilities: {
            text: true,
            voice: true,
            browserMicrophoneRequired: true
          }
        }),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        voiceAvailable: true,
        voiceIsListening: true,
        voiceState: "idle",
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.getByRole("link", { name: "Stop listening" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=live"
    )
    expect(screen.queryByRole("link", { name: "Listen" })).not.toBeInTheDocument()
  })

  it("hides voice controls when the focused session is not voice-capable", () => {
    renderPopover({
      liveControl: {
        sessions: [buildLiveSession()],
        focusedSession: buildLiveSession(),
        focusedSessionId: "live-session-1",
        streamState: "open",
        canSendText: true,
        voiceAvailable: false,
        voiceState: "idle",
        pendingFocusSessionId: null,
        startTextSession: vi.fn(),
        stopSession: vi.fn(),
        focusSession: vi.fn(),
        sendText: vi.fn()
      }
    })

    expect(screen.queryByRole("link", { name: "Listen" })).not.toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Stop listening" })).not.toBeInTheDocument()
  })
})
