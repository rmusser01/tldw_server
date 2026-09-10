import React from "react"
import { render, screen } from "@testing-library/react"
import { it, expect, vi } from "vitest"
import { BuddyShellDock } from "../BuddyShellDock"
import type { PersonaBuddyLiveControlView } from "@/types/persona-buddy"

it("keeps current-session review visible when Buddy is collapsed", () => {
  const focusedSession = { sessionId: "s1", personaId: "p1", personaName: "Migu", lifecycle: "idle", pendingApprovalCount: 0 }
  const liveControl: PersonaBuddyLiveControlView = {
    focusedSession, sessions: [focusedSession], focusedSessionId: "s1", streamState: "open", canSendText: true,
    pendingFocusSessionId: null, startTextSession: vi.fn(), stopSession: vi.fn(), focusSession: vi.fn(), sendText: vi.fn(),
    feedback: { sessionId: "s1", personaId: "p1", clientMessageId: "m1", status: "review", text: "Review plan" }
  }
  const props = {
    buddySummary: { has_buddy: true, persona_name: "Migu", role_summary: null, visual: null },
    personaId: "p1", isOpen: false, position: { x: 16, y: 16 },
    onToggle: vi.fn(), onDragHandlePointerDown: vi.fn(), dockRef: React.createRef<HTMLDivElement>(), liveControl
  }
  const { rerender } = render(<BuddyShellDock {...props} />)
  expect(screen.queryByTestId("persona-buddy-popover")).not.toBeInTheDocument()
  expect(screen.getByTestId("persona-buddy-urgent-badge")).toHaveTextContent("1")
  expect(screen.getByTestId("persona-buddy-live-status")).toHaveTextContent("Needs approval")
  rerender(<BuddyShellDock {...props} liveControl={{ ...liveControl, feedback: { ...liveControl.feedback!, sessionId: "old" } }} />)
  expect(screen.queryByTestId("persona-buddy-urgent-badge")).not.toBeInTheDocument()
})
