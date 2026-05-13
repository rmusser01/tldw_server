import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  bulkDecideModerationReviewItems,
  decideModerationReviewItem,
  getModerationReviewItem,
  getUserOverride,
  listModerationReviewAudit,
  listModerationReviewItems,
  setUserOverride,
  undoModerationReviewDecision,
  type ModerationUserOverride
} from "@/services/moderation"

describe("moderation service contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("returns rules from getUserOverride payload", async () => {
    mocks.bgRequest.mockResolvedValue({
      enabled: true,
      rules: [
        {
          id: "r1",
          pattern: "bad",
          is_regex: false,
          action: "block",
          phase: "both"
        }
      ]
    })

    const response = await getUserOverride("alice")

    expect((response as any).rules?.[0]).toMatchObject({
      id: "r1",
      action: "block",
      phase: "both"
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/moderation/users/alice"
      })
    )
  })

  it("sends rules in setUserOverride payload", async () => {
    const body: ModerationUserOverride = {
      enabled: true,
      rules: [
        {
          id: "n1",
          pattern: "heads up",
          is_regex: false,
          action: "warn",
          phase: "both"
        }
      ]
    }
    mocks.bgRequest.mockResolvedValue({ persisted: true })

    await setUserOverride("alice", body)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "PUT",
        path: "/api/v1/moderation/users/alice",
        body: expect.objectContaining({
          rules: [
            expect.objectContaining({
              id: "n1",
              action: "warn",
              phase: "both"
            })
          ]
        })
      })
    )
  })

  it("builds review queue list queries with supported filters", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0, next_cursor: null })

    await listModerationReviewItems({
      status: "needs_review",
      category: "pii",
      severity: "high",
      source_type: "chat",
      source_id: "conversation-1",
      user_id: "user-1",
      q: "redacted",
      limit: 25,
      cursor: "50"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path:
          "/api/v1/moderation/review/items?status=needs_review&category=pii&severity=high&source_type=chat&source_id=conversation-1&user_id=user-1&q=redacted&limit=25&cursor=50"
      })
    )
  })

  it("addresses review item detail decision undo bulk and audit endpoints", async () => {
    mocks.bgRequest.mockResolvedValue({})

    await getModerationReviewItem("item/1")
    await decideModerationReviewItem("item/1", {
      action: "block",
      reason: "Contains private data",
      actor_id: "spoofed"
    })
    await undoModerationReviewDecision("item/1", "undo-1")
    await bulkDecideModerationReviewItems({
      item_ids: ["a", "b"],
      action: "dismiss",
      reason: "Batch cleanup"
    })
    await listModerationReviewAudit({
      item_id: "item/1",
      decision_id: "decision/1",
      actor: "reviewer",
      action: "decision.block",
      date_from: "2026-05-12T00:00:00Z",
      date_to: "2026-05-13T00:00:00Z"
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/moderation/review/items/item%2F1",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/moderation/review/items/item%2F1/decision",
      method: "POST",
      body: {
        action: "block",
        reason: "Contains private data",
        actor_id: "spoofed"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/moderation/review/items/item%2F1/undo",
      method: "POST",
      body: { undo_token: "undo-1" }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(4, {
      path: "/api/v1/moderation/review/bulk-decision",
      method: "POST",
      body: {
        item_ids: ["a", "b"],
        action: "dismiss",
        reason: "Batch cleanup"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(5, {
      path:
        "/api/v1/moderation/review/audit?item_id=item%2F1&decision_id=decision%2F1&actor=reviewer&action=decision.block&date_from=2026-05-12T00%3A00%3A00Z&date_to=2026-05-13T00%3A00%3A00Z",
      method: "GET"
    })
  })
})
