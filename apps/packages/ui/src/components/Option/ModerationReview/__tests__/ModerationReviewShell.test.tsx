// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { MODERATION_RULES_PATH } from "@/routes/route-paths"
import { ModerationReviewShell } from "../ModerationReviewShell"

const mocks = vi.hoisted(() => ({
  listModerationReviewItems: vi.fn(),
  getModerationReviewItem: vi.fn(),
  decideModerationReviewItem: vi.fn(),
  undoModerationReviewDecision: vi.fn()
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: "connected_ok",
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/services/moderation", () => ({
  listModerationReviewItems: (...args: unknown[]) => mocks.listModerationReviewItems(...args),
  getModerationReviewItem: (...args: unknown[]) => mocks.getModerationReviewItem(...args),
  decideModerationReviewItem: (...args: unknown[]) => mocks.decideModerationReviewItem(...args),
  undoModerationReviewDecision: (...args: unknown[]) => mocks.undoModerationReviewDecision(...args)
}))

const reviewItem = {
  id: "review-1",
  status: "needs_review",
  phase: "input",
  source_type: "chat",
  source_id: "conversation-7",
  user_id: "user-1",
  session_id: "session-9",
  created_at: "2026-05-12T20:05:00Z",
  updated_at: null,
  severity: "high",
  category: "pii",
  safe_fields: {
    excerpt: true,
    context: true,
    effective_policy: false,
    matches: true
  },
  excerpt: "hello [REDACTED] from user",
  context: {
    source_type: "chat",
    source_id: "conversation-7"
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
      sample: "hello [REDACTED] from user",
      confidence: 0.86
    }
  ],
  recommended_action: "block"
}

const renderShell = (compact = false) =>
  render(
    <MemoryRouter>
      <ModerationReviewShell compact={compact} />
    </MemoryRouter>
  )

describe("ModerationReviewShell", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listModerationReviewItems.mockResolvedValue({
      items: [reviewItem],
      total: 1,
      next_cursor: null
    })
    mocks.getModerationReviewItem.mockResolvedValue(reviewItem)
    mocks.decideModerationReviewItem.mockResolvedValue({
      item: { ...reviewItem, status: "blocked" },
      decision: {
        id: "decision-1",
        item_id: "review-1",
        action: "block",
        status: "blocked",
        previous_status: "needs_review",
        decided_by: "principal:reviewer",
        reason: "Contains private data",
        decided_at: "2026-05-12T20:10:00Z",
        undo_token: "undo-1"
      },
      undo_token: "undo-1"
    })
    mocks.undoModerationReviewDecision.mockResolvedValue(reviewItem)
  })

  it("renders a loaded review queue item with context and safe-field warnings", async () => {
    renderShell()

    expect(
      screen.getByRole("heading", { name: "Moderation Review" })
    ).toBeInTheDocument()
    expect((await screen.findAllByText(/hello \[REDACTED\] from user/i)).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/chat: conversation-7/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/high/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/recommended: block/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/rule: pii-rule-1/i)).toBeInTheDocument()
    expect(screen.getByText(/some policy fields are unavailable/i)).toBeInTheDocument()
  })

  it("links reviewers to content rules without making it the primary destination", () => {
    renderShell()

    const rulesLink = screen.getByRole("link", { name: /open content rules/i })
    expect(rulesLink).toHaveAttribute("href", MODERATION_RULES_PATH)
  })

  it("updates review query filters and refreshes the queue", async () => {
    const user = userEvent.setup()
    renderShell()
    await screen.findAllByText(/hello \[REDACTED\]/i)
    mocks.listModerationReviewItems.mockClear()

    await user.selectOptions(screen.getByLabelText(/status/i), "escalated")
    await user.selectOptions(screen.getByLabelText(/severity/i), "high")
    await user.type(screen.getByLabelText(/search/i), "private")
    await user.click(screen.getByRole("button", { name: /refresh/i }))

    expect(mocks.listModerationReviewItems).toHaveBeenCalledWith(
      expect.objectContaining({
        status: "escalated",
        severity: "high",
        q: "private"
      })
    )
  })

  it("renders permission denied without redirecting to content rules", async () => {
    mocks.listModerationReviewItems.mockRejectedValue({ status: 403, message: "Forbidden" })

    renderShell()

    expect(await screen.findByText(/permission denied/i)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /open content rules/i })).toHaveAttribute(
      "href",
      MODERATION_RULES_PATH
    )
  })

  it("validates decision reasons and shows undo after a decision", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    const user = userEvent.setup()
    mocks.listModerationReviewItems
      .mockResolvedValueOnce({
        items: [reviewItem],
        total: 1,
        next_cursor: null
      })
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        next_cursor: null
      })
    renderShell()

    await screen.findAllByText(/hello \[REDACTED\]/i)
    await user.click(screen.getByRole("button", { name: /^block$/i }))
    expect(screen.getByText(/reason required/i)).toBeInTheDocument()

    await user.type(screen.getByLabelText(/decision reason/i), "Contains private data")
    await user.click(screen.getByRole("button", { name: /^block$/i }))

    await waitFor(() => {
      expect(mocks.decideModerationReviewItem).toHaveBeenCalledWith("review-1", {
        action: "block",
        reason: "Contains private data"
      })
    })
    expect(await screen.findByRole("button", { name: /undo decision/i })).toBeInTheDocument()
    expect(screen.getAllByText(/blocked/i).length).toBeGreaterThan(0)
    expect(confirmSpy).toHaveBeenCalled()
  })

  it("renders compact mode with a full review action", async () => {
    renderShell(true)

    expect(await screen.findByRole("link", { name: /open full review/i })).toHaveAttribute(
      "href",
      "/moderation"
    )
  })
})
