// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { AuditTimeline } from "../AuditTimeline"
import { ReviewItemDetail } from "../ReviewItemDetail"
import type { ModerationReviewItem } from "@/services/moderation"

const baseItem: ModerationReviewItem = {
  id: "review-1",
  status: "redacted",
  phase: "input",
  source_type: "chat",
  source_id: "conversation-7",
  user_id: "user-1",
  session_id: "session-9",
  created_at: "2026-05-12T20:05:00Z",
  updated_at: "2026-05-12T20:10:00Z",
  severity: "high",
  category: "pii",
  safe_fields: {
    excerpt: false,
    context: false,
    matches: false,
    effective_policy: true
  },
  excerpt: "[content redacted]",
  context: {
    redacted: "true",
    message: "Context removed from moderation review item"
  },
  effective_policy: {
    input_action: "block"
  },
  matches: [
    {
      rule_id: "pii-rule-1",
      pattern_type: "pii",
      category: "pii",
      action: "block",
      sample: "[content redacted]",
      confidence: 0.86
    }
  ],
  recommended_action: "redact",
  content_redacted_at: "2026-05-12T20:10:00Z",
  decision_history: [
    {
      id: "decision-1",
      action: "redact",
      status: "redacted",
      previous_status: "needs_review",
      actor_id: "principal:reviewer",
      reason: "Source deletion request",
      decided_at: "2026-05-12T20:10:00Z",
      undo_eligible: false,
      undo_expires_at: "2026-05-12T20:25:00Z",
      undone_at: null,
      redaction_state: "redacted"
    },
    {
      id: "decision-0",
      action: "approve",
      status: "approved",
      previous_status: "needs_review",
      actor_id: "principal:reviewer",
      reason: "Initial pass",
      decided_at: "2026-05-12T20:08:00Z",
      undo_eligible: false,
      undo_expires_at: "2026-05-12T20:23:00Z",
      undone_at: "2026-05-12T20:09:00Z",
      redaction_state: "not_redacted"
    }
  ]
}

describe("AuditTimeline", () => {
  it("renders sanitized decision history without raw undo tokens", () => {
    render(<AuditTimeline decisions={baseItem.decision_history || []} />)

    expect(screen.getByRole("heading", { name: /decision history/i })).toBeInTheDocument()
    expect(screen.getAllByText(/redact/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/principal:reviewer/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/source deletion request/i)).toBeInTheDocument()
    expect(screen.getByText(/undo unavailable/i)).toBeInTheDocument()
    expect(screen.getByText(/content redacted/i)).toBeInTheDocument()
    expect(screen.queryByText(/undo-1/i)).not.toBeInTheDocument()
  })

  it("surfaces redacted content state in item detail", () => {
    render(<ReviewItemDetail item={baseItem} />)

    expect(screen.getByText(/review content redacted/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\[content redacted\]/i).length).toBeGreaterThan(0)
    expect(screen.getByRole("heading", { name: /decision history/i })).toBeInTheDocument()
  })
})
