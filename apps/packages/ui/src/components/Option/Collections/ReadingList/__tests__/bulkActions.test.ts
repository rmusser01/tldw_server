import { describe, expect, it } from "vitest"
import { getBulkFailureLines, normalizeBulkTags } from "@/components/Option/Collections/ReadingList/bulkActions"

describe("reading bulk action helpers", () => {
  it("normalizes comma-separated tag input, keeping first-seen casing", () => {
    expect(normalizeBulkTags(" AI, research,ai ,  notes ")).toEqual([
      "AI",
      "research",
      "notes"
    ])
  })

  it("returns failed item lines with fallback error text", () => {
    const lines = getBulkFailureLines({
      total: 3,
      succeeded: 1,
      failed: 2,
      results: [
        { item_id: "101", success: true },
        { item_id: "202", success: false, error: "item_not_found" },
        { item_id: "303", success: false, error: null }
      ]
    })

    expect(lines).toEqual([
      "#202: item_not_found",
      "#303: update_failed"
    ])
  })
})
