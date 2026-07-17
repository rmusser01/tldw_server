import { beforeEach, describe, expect, it, vi } from "vitest"

const { bgRequest } = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest,
  bgUpload: vi.fn()
}))

import {
  completeSourceReviewOccurrence,
  createSourceReviewPlan,
  deleteSourceReviewPlan,
  listDueSourceReviewOccurrences,
  listSourceReviewPlans,
  skipSourceReviewOccurrence,
  startSourceReviewOccurrence,
  type SourceReviewPlanCreateRequest
} from "@/services/flashcards"

const request: SourceReviewPlanCreateRequest = {
  title: "Cardiac physiology",
  starts_on: "2026-07-09",
  timezone: "UTC",
  source_items: [
    {
      source_type: "note",
      source_id: "note-42",
      label: "Cardiac physiology"
    }
  ],
  schedule: [
    {
      offset_value: 1,
      offset_unit: "day",
      activity_type: "reread"
    }
  ]
}

describe("source review flashcards service", () => {
  beforeEach(() => {
    bgRequest.mockReset()
    bgRequest.mockResolvedValue({})
  })

  it("uses the Flashcards-owned create/list/due endpoints", async () => {
    await createSourceReviewPlan(request)
    await listSourceReviewPlans({ limit: 20, offset: 5 })
    await listDueSourceReviewOccurrences({ limit: 10, offset: 2 })

    expect(bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/flashcards/source-review-plans",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: request,
      abortSignal: undefined
    })
    expect(bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/flashcards/source-review-plans?limit=20&offset=5",
      method: "GET",
      abortSignal: undefined
    })
    expect(bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/flashcards/source-review-plans/due?limit=10&offset=2",
      method: "GET",
      abortSignal: undefined
    })
  })

  it("uses exact occurrence action and plan delete endpoints", async () => {
    await startSourceReviewOccurrence(31)
    await completeSourceReviewOccurrence(31)
    await skipSourceReviewOccurrence(31)
    await deleteSourceReviewPlan(7)

    expect(bgRequest.mock.calls.map(([call]) => [call.path, call.method])).toEqual([
      [
        "/api/v1/flashcards/source-review-plans/occurrences/31/start",
        "POST"
      ],
      [
        "/api/v1/flashcards/source-review-plans/occurrences/31/complete",
        "POST"
      ],
      [
        "/api/v1/flashcards/source-review-plans/occurrences/31/skip",
        "POST"
      ],
      ["/api/v1/flashcards/source-review-plans/7", "DELETE"]
    ])
  })
})
