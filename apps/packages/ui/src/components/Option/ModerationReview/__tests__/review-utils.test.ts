import { describe, expect, it } from "vitest"

import {
  decisionActionLabel,
  decisionRequiresReason,
  formatReviewDate,
  getReviewItemSourceLabel,
  isPermissionDeniedError,
  sortReviewItems
} from "../review-utils"

const item = {
  id: "item-1",
  status: "needs_review" as const,
  phase: "input" as const,
  source_type: "chat",
  source_id: "conversation-7",
  user_id: "user-1",
  session_id: "session-3",
  created_at: "2026-05-12T20:05:00Z",
  severity: "high" as const,
  category: "pii",
  safe_fields: { excerpt: true },
  excerpt: "hello [REDACTED]",
  effective_policy: {},
  matches: [],
  recommended_action: "block" as const
}

describe("moderation review utils", () => {
  it("formats source and timestamps for dense queue rows", () => {
    expect(getReviewItemSourceLabel(item)).toBe("chat: conversation-7")
    expect(formatReviewDate(item.created_at)).toContain("2026")
  })

  it("sorts review items by created time", () => {
    const older = { ...item, id: "older", created_at: "2026-05-01T00:00:00Z" }
    const newer = { ...item, id: "newer", created_at: "2026-05-03T00:00:00Z" }

    expect(sortReviewItems([older, newer], "newest").map((entry) => entry.id)).toEqual(["newer", "older"])
    expect(sortReviewItems([older, newer], "oldest").map((entry) => entry.id)).toEqual(["older", "newer"])
  })

  it("identifies high-risk decision actions and permission errors", () => {
    expect(decisionRequiresReason("block")).toBe(true)
    expect(decisionRequiresReason("redact")).toBe(true)
    expect(decisionRequiresReason("escalate")).toBe(true)
    expect(decisionRequiresReason("approve")).toBe(false)
    expect(decisionActionLabel("dismiss")).toBe("Dismiss")
    expect(isPermissionDeniedError({ status: 403 })).toBe(true)
    expect(isPermissionDeniedError(new Error("Forbidden"))).toBe(true)
  })
})
