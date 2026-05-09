import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SessionHistoryPanel } from "../SessionHistoryPanel"
import type { SessionMetadata } from "@/services/agent/storage"
import type { AgentStatus } from "@/services/agent/types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

const makeSession = (status: AgentStatus, index: number): SessionMetadata => ({
  id: `session-${status}`,
  workspaceId: "workspace-1",
  task: `Task for ${status}`,
  title: `${status} session`,
  status,
  currentStep: index + 1,
  createdAt: `2026-05-09T15:${String(index).padStart(2, "0")}:00.000Z`,
  updatedAt: `2026-05-09T16:${String(index).padStart(2, "0")}:00.000Z`,
  messageCount: index + 2,
  toolCallCount: index
})

const badgeFor = (label: string) => {
  const badge = screen.getByText(label).closest<HTMLElement>('[data-ds-component="Badge"]')
  if (!badge) {
    throw new Error(`Expected ${label} to render inside a Badge`)
  }
  return badge
}

describe("SessionHistoryPanel status badges", () => {
  it("renders agent status labels through shared Badge variants", () => {
    render(
      <SessionHistoryPanel
        sessions={[
          makeSession("idle", 0),
          makeSession("running", 1),
          makeSession("waiting_approval", 2),
          makeSession("complete", 3),
          makeSession("error", 4),
          makeSession("cancelled", 5)
        ]}
        isLoading={false}
        onRestore={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
      />
    )

    expect(badgeFor("Idle")).toHaveAttribute("data-ds-variant", "secondary")
    expect(badgeFor("Running")).toHaveAttribute("data-ds-variant", "info")
    expect(badgeFor("Paused")).toHaveAttribute("data-ds-variant", "warning")
    expect(badgeFor("Complete")).toHaveAttribute("data-ds-variant", "success")
    expect(badgeFor("Error")).toHaveAttribute("data-ds-variant", "danger")
    expect(badgeFor("Cancelled")).toHaveAttribute("data-ds-variant", "warning")
  })
})
