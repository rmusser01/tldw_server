// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ResearchRunStatusStack } from "../ResearchRunStatusStack"
import { VoiceChatIndicator } from "../VoiceChatIndicator"
import type { ChatLinkedResearchRun } from "@/services/tldw/TldwApiClient"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

const linkedRun = (overrides: Partial<ChatLinkedResearchRun>): ChatLinkedResearchRun =>
  ({
    run_id: "research-run-1",
    query: "Check original sources",
    status: "waiting_human",
    phase: "awaiting_plan_review",
    control_state: "running",
    latest_checkpoint_id: "checkpoint-1",
    updated_at: "2026-05-05T12:00:00Z",
    ...overrides
  }) as ChatLinkedResearchRun

describe("Playground status design-system badges", () => {
  it("renders linked research run statuses through uniquely addressed shared Badges", () => {
    render(
      <ResearchRunStatusStack
        runs={[
          linkedRun({ run_id: "research-run-1" }),
          linkedRun({
            run_id: "research-run-2",
            query: "Summarize the finished bundle",
            status: "completed",
            phase: "completed",
            latest_checkpoint_id: null
          })
        ]}
      />
    )

    const rows = screen.getAllByTestId("research-run-status-row")
    const reviewBadge = within(rows[0]).getByTestId(
      "research-run-status-badge-research-run-1"
    )
    const completedBadge = within(rows[1]).getByTestId(
      "research-run-status-badge-research-run-2"
    )

    expect(reviewBadge).toHaveAttribute("data-ds-component", "Badge")
    expect(reviewBadge).toHaveTextContent("Needs review")
    expect(completedBadge).toHaveAttribute("data-ds-component", "Badge")
    expect(completedBadge).toHaveTextContent("Completed")
    expect(screen.queryAllByTestId("research-run-status-badge")).toHaveLength(0)
    expect(screen.getByText("Plan review needed")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=research-run-1"
    )
  })

  it("renders the voice chat status through the shared Badge while preserving stop control", () => {
    const onStop = vi.fn()

    render(
      <VoiceChatIndicator
        state="listening"
        statusLabel="Listening"
        onStop={onStop}
      />
    )

    const badge = screen.getByTestId("voice-chat-status-badge")

    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(badge).toHaveTextContent("Listening")

    fireEvent.click(screen.getByRole("button", { name: "Stop voice chat" }))

    expect(onStop).toHaveBeenCalledTimes(1)
  })
})
