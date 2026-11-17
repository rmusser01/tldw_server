import { describe, expect, it } from "vitest"
import {
  getAlertCount,
  getOutputReportReadiness,
  getOutputReportSnapshotAvailable,
  getSourceCount
} from "../outputMetadata"

describe("saved Watchlist report source counts", () => {
  it("counts the one source retained by a fifty-item automatic briefing without inventing report metadata", () => {
    const metadata = {
      generation_mode: "auto_output",
      fulfillment_mode: "briefing_occurrence",
      selected_count: 50,
      source_counts: { succeeded: 3, failed: 1, deferred: 2 },
      provenance: Array.from({ length: 50 }, (_, index) => ({
        item_id: index + 1,
        source_id: 1,
        url: `https://arxiv.org/abs/2610.${11410 - index}`
      }))
    }
    const before = JSON.stringify(metadata)

    expect(getSourceCount(metadata)).toBe(1)
    expect(getAlertCount(metadata)).toBe(0)
    expect(getOutputReportSnapshotAvailable(metadata)).toBe(false)
    expect(getOutputReportReadiness(metadata)).toEqual({
      state: "legacy_live_only",
      score: 0,
      warnings: []
    })
    expect(JSON.stringify(metadata)).toBe(before)
  })

  it("counts distinct retained source identities rather than selected items or run statuses", () => {
    expect(getSourceCount({
      selected_count: 3,
      source_counts: { succeeded: 7, failed: 2, deferred: 1 },
      provenance: [
        { item_id: 50, source_id: 1 },
        { item_id: 49, source_id: 1 },
        { item_id: 48, source_id: 2 }
      ]
    })).toBe(2)
  })

  it.each([0, 7, "3"])("preserves explicit valid source_count %s", (source_count) => {
    expect(getSourceCount({
      source_count,
      provenance: [{ source_id: 1 }, { source_id: 2 }]
    })).toBe(Number(source_count))
  })

  it.each([-1, 1.5, "unknown", null])("uses provenance when source_count %s is invalid", (source_count) => {
    expect(getSourceCount({
      source_count,
      provenance: [{ source_id: 1 }, { source_id: 1 }, { source_id: 2 }]
    })).toBe(2)
  })

  it("ignores malformed and noncanonical source identities", () => {
    expect(getSourceCount({
      provenance: [
        { source_id: 1 }, { source_id: 1 }, { source_id: 2 },
        { source_id: 0 }, { source_id: -1 }, { source_id: 1.5 },
        { source_id: "3" }, { source_id: true }, { source_id: Number.NaN },
        { source_id: Number.POSITIVE_INFINITY },
        { source_id: Number.MAX_SAFE_INTEGER + 1 },
        null, [], {}, "source"
      ]
    })).toBe(2)
  })

  it.each([null, {}, "metadata", { provenance: { source_id: 1 } }])(
    "returns zero for missing or malformed provenance %j", (metadata) => {
      expect(getSourceCount(metadata)).toBe(0)
    }
  )

  it("does not treat run source status totals as retained report sources", () => {
    expect(getSourceCount({ source_counts: { succeeded: 5, failed: 1, deferred: 2 } })).toBe(0)
  })
})
