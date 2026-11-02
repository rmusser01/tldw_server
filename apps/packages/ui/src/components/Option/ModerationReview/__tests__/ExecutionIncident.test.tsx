// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { ModerationReviewItem } from "@/services/moderation"
import { ReviewItemDetail } from "../ReviewItemDetail"
import { ReviewQueueList } from "../ReviewQueueList"

const incident: ModerationReviewItem = {
  id: "incident-1", status: "needs_review", phase: "unspecified",
  source_type: "moderation_engine", category: "moderation_engine_failure",
  created_at: "2026-11-02T12:00:00Z", severity: "high",
  excerpt: "[Moderation execution failed; content withheld]",
  recommended_action: "escalate", matches: [], effective_policy: {},
  context: { reason: "deadline_exceeded", operation: "evaluate", correlation_id: "aabb", unsafe: "SECRET" }
}

describe("operational execution incidents", () => {
  it("renders neutral phase and operational metadata without policy match attribution", () => {
    render(<ReviewItemDetail item={incident} />)
    expect(screen.getByText("Unspecified")).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Moderation execution failure" })).toBeInTheDocument()
    expect(screen.getByText("deadline_exceeded")).toBeInTheDocument()
    expect(screen.getByText("evaluate")).toBeInTheDocument()
    expect(screen.getByText("aabb")).toBeInTheDocument()
    expect(screen.queryByText("Matches")).not.toBeInTheDocument()
    expect(screen.queryByText(/SECRET/)).not.toBeInTheDocument()
    expect(screen.getByText(/does not change enforcement/i)).toBeInTheDocument()
  })

  it("selects the same single incident ID for individual and bulk review", () => {
    const select = vi.fn(), bulk = vi.fn()
    render(<ReviewQueueList items={[incident]} selectedItemId={null} onSelect={select} onToggleSelected={bulk} />)
    expect(screen.getByText("Unspecified")).toBeInTheDocument()
    expect(screen.getByText("Operational failure")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button"))
    fireEvent.click(screen.getByRole("checkbox"))
    expect(select).toHaveBeenCalledWith("incident-1")
    expect(bulk).toHaveBeenCalledWith("incident-1")
  })
})
