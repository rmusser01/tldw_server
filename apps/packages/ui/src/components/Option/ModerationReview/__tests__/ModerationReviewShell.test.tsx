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
  undoModerationReviewDecision: vi.fn(),
  bulkDecideModerationReviewItems: vi.fn()
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
  undoModerationReviewDecision: (...args: unknown[]) => mocks.undoModerationReviewDecision(...args),
  bulkDecideModerationReviewItems: (...args: unknown[]) => mocks.bulkDecideModerationReviewItems(...args)
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

const reviewItem2 = {
  ...reviewItem,
  id: "review-2",
  source_id: "conversation-8",
  excerpt: "second [REDACTED] from user",
  created_at: "2026-05-12T20:04:00Z"
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
    mocks.bulkDecideModerationReviewItems.mockResolvedValue({
      ok_count: 1,
      error_count: 0,
      results: [
        {
          item_id: "review-1",
          ok: true,
          item: { ...reviewItem, status: "dismissed" }
        }
      ]
    })
    window.localStorage.clear()
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

  it("supports selecting rows and bulk dismissing with partial failure feedback", async () => {
    const user = userEvent.setup()
    mocks.listModerationReviewItems.mockResolvedValue({
      items: [reviewItem, reviewItem2],
      total: 2,
      next_cursor: null
    })
    mocks.getModerationReviewItem.mockImplementation((itemId: string) =>
      Promise.resolve(itemId === "review-2" ? reviewItem2 : reviewItem)
    )
    mocks.bulkDecideModerationReviewItems.mockResolvedValue({
      ok_count: 1,
      error_count: 1,
      results: [
        {
          item_id: "review-1",
          ok: true,
          item: { ...reviewItem, status: "dismissed" }
        },
        {
          item_id: "review-2",
          ok: false,
          error: "not_found"
        }
      ]
    })

    renderShell()
    await screen.findAllByText(/hello \[REDACTED\]/i)

    await user.click(screen.getByLabelText(/select review item review-1/i))
    await user.click(screen.getByLabelText(/select review item review-2/i))
    expect(screen.getByText(/2 selected/i)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /dismiss selected/i }))

    await waitFor(() => {
      expect(mocks.bulkDecideModerationReviewItems).toHaveBeenCalledWith({
        item_ids: ["review-1", "review-2"],
        action: "dismiss",
        reason: undefined
      })
    })
    expect(await screen.findByText(/1 failed/i)).toBeInTheDocument()
    expect(screen.getByText(/review-2: not_found/i)).toBeInTheDocument()
  })

  it("saves and reapplies local filter presets", async () => {
    const user = userEvent.setup()
    renderShell()
    await screen.findAllByText(/hello \[REDACTED\]/i)
    mocks.listModerationReviewItems.mockClear()

    await user.type(screen.getByLabelText(/category/i), "pii")
    await user.selectOptions(screen.getByLabelText(/severity/i), "high")
    await user.type(screen.getByLabelText(/preset name/i), "High PII")
    await user.click(screen.getByRole("button", { name: /save preset/i }))

    await user.clear(screen.getByLabelText(/category/i))
    await user.selectOptions(screen.getByLabelText(/severity/i), "medium")
    await user.selectOptions(screen.getByLabelText(/saved preset/i), "High PII")
    await user.click(screen.getByRole("button", { name: /apply preset/i }))
    await user.click(screen.getByRole("button", { name: /refresh/i }))

    expect(mocks.listModerationReviewItems).toHaveBeenCalledWith(
      expect.objectContaining({
        category: "pii",
        severity: "high"
      })
    )
  })

  it("appends the next page instead of replacing the current worklist", async () => {
    const user = userEvent.setup()
    mocks.listModerationReviewItems
      .mockResolvedValueOnce({
        items: [reviewItem],
        total: 2,
        next_cursor: "1"
      })
      .mockResolvedValueOnce({
        items: [reviewItem2],
        total: 2,
        next_cursor: null
      })
    mocks.getModerationReviewItem.mockImplementation((itemId: string) =>
      Promise.resolve(itemId === "review-2" ? reviewItem2 : reviewItem)
    )

    renderShell()
    await screen.findAllByText(/hello \[REDACTED\]/i)
    await user.click(screen.getByRole("button", { name: /load next page/i }))

    await waitFor(() => {
      expect(mocks.listModerationReviewItems).toHaveBeenLastCalledWith(
        expect.objectContaining({
          cursor: "1",
          sort: "newest"
        })
      )
    })
    expect(screen.getAllByText(/hello \[REDACTED\]/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/second \[REDACTED\]/i).length).toBeGreaterThan(0)
  })

  it("handles scoped keyboard shortcuts without firing while typing", async () => {
    const user = userEvent.setup()
    mocks.listModerationReviewItems.mockResolvedValue({
      items: [reviewItem, reviewItem2],
      total: 2,
      next_cursor: null
    })
    mocks.getModerationReviewItem.mockImplementation((itemId: string) =>
      Promise.resolve(itemId === "review-2" ? reviewItem2 : reviewItem)
    )

    renderShell()
    await screen.findAllByText(/hello \[REDACTED\]/i)

    const shell = screen.getByTestId("moderation-review-shell")
    shell.focus()
    await user.keyboard("n")
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /second \[REDACTED\]/i })).toHaveAttribute("aria-current", "true")
    })

    await user.click(screen.getByLabelText(/search/i))
    await user.keyboard("d")
    expect(mocks.decideModerationReviewItem).not.toHaveBeenCalled()

    shell.focus()
    await user.keyboard("d")
    await waitFor(() => {
      expect(mocks.decideModerationReviewItem).toHaveBeenCalledWith("review-2", {
        action: "dismiss",
        reason: undefined
      })
    })
  })

  it("renders a review-complete state for an empty needs-review queue", async () => {
    mocks.listModerationReviewItems.mockResolvedValue({
      items: [],
      total: 0,
      next_cursor: null
    })
    mocks.getModerationReviewItem.mockResolvedValue(null)

    renderShell()

    expect(await screen.findByText(/review complete/i)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /review audit/i })).toHaveAttribute(
      "href",
      "#moderation-review-audit"
    )
    expect(screen.getByRole("link", { name: /^content rules$/i })).toHaveAttribute(
      "href",
      MODERATION_RULES_PATH
    )
  })
})
