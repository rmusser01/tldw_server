import { describe, expect, it } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import {
  buildSourceFilterSummary,
  filterSources,
  hasActiveSourceFilters,
  SOURCE_REVIEW_FILTER_PRESETS,
  sortSources,
  type SourceListViewState
} from "../SourcesPane/source-list-view"

const sources: WorkspaceSource[] = [
  {
    id: "s1",
    mediaId: 1,
    title: "Bravo Document",
    type: "pdf",
    status: "ready",
    reviewState: "needs_review",
    statusDetails: { lifecycleState: "queryable" },
    addedAt: new Date("2026-03-10T00:00:00.000Z"),
    sourceCreatedAt: new Date("2026-03-01T00:00:00.000Z"),
    pageCount: 8,
    fileSize: 2_048
  },
  {
    id: "s2",
    mediaId: 2,
    title: "Alpha Site",
    type: "website",
    status: "error",
    reviewState: "unset",
    statusDetails: { lifecycleState: "failed" },
    addedAt: new Date("2026-03-12T00:00:00.000Z"),
    url: "https://example.com"
  },
  {
    id: "s3",
    mediaId: 3,
    title: "Alpha Audio",
    type: "audio",
    status: "processing",
    reviewState: "reviewed",
    statusDetails: { lifecycleState: "partially_queryable" },
    addedAt: new Date("2026-03-11T00:00:00.000Z"),
    duration: 90
  }
]

const baseViewState: SourceListViewState = {
  expanded: false,
  typeFilters: [],
  statusFilters: [],
  reviewStateFilters: [],
  lifecycleStateFilters: [],
  dateField: "addedAt",
  dateFrom: null,
  dateTo: null,
  requireUrl: false,
  requireFileSize: false,
  requireDuration: false,
  requirePageCount: false,
  fileSizeMin: null,
  fileSizeMax: null,
  durationMin: null,
  durationMax: null,
  pageCountMin: null,
  pageCountMax: null,
  sort: "manual"
}

describe("source-list-view", () => {
  it("filters by type, status, metadata presence, and numeric range", () => {
    const state: SourceListViewState = {
      ...baseViewState,
      typeFilters: ["pdf"],
      statusFilters: ["ready"],
      requirePageCount: true,
      fileSizeMin: 2_000
    }

    expect(filterSources(sources, state).map((source) => source.id)).toEqual(["s1"])
  })

  it("filters by the selected date field and inclusive date range", () => {
    const state: SourceListViewState = {
      ...baseViewState,
      dateField: "sourceCreatedAt",
      dateFrom: "2026-02-28",
      dateTo: "2026-03-02"
    }

    expect(filterSources(sources, state).map((source) => source.id)).toEqual(["s1"])
  })

  it.each([
    ["needs_review", ["s1"]],
    ["unset", ["s2"]]
  ] as const)("filters by review state %s", (reviewState, expectedIds) => {
    const state: SourceListViewState = {
      ...baseViewState,
      reviewStateFilters: [reviewState]
    }

    expect(filterSources(sources, state).map((source) => source.id)).toEqual(
      expectedIds
    )
  })

  it("filters partially queryable sources by lifecycle without conflating status", () => {
    const state: SourceListViewState = {
      ...baseViewState,
      lifecycleStateFilters: ["partially_queryable"]
    }

    expect(filterSources(sources, state).map((source) => source.id)).toEqual(["s3"])
  })

  it("provides needs-review and unreviewed filter presets", () => {
    expect(SOURCE_REVIEW_FILTER_PRESETS.needsReview).toEqual({
      reviewStateFilters: ["needs_review"]
    })
    expect(SOURCE_REVIEW_FILTER_PRESETS.unreviewed).toEqual({
      reviewStateFilters: ["unset"]
    })
  })

  it("sorts by name and falls back to the existing manual order on ties", () => {
    const sorted = sortSources(
      [
        { ...sources[0], title: "Alpha" },
        sources[1]
      ],
      "name_asc"
    )

    expect(sorted.map((source) => source.id)).toEqual(["s1", "s2"])
  })

  it("pushes missing metadata to the end for numeric sorts", () => {
    const sorted = sortSources(sources, "page_count_desc")

    expect(sorted.map((source) => source.id)).toEqual(["s1", "s2", "s3"])
  })

  it("reports whether any advanced filters are active independently of sort", () => {
    expect(hasActiveSourceFilters(baseViewState)).toBe(false)
    expect(
      hasActiveSourceFilters({
        ...baseViewState,
        statusFilters: ["ready"]
      })
    ).toBe(true)
    expect(
      hasActiveSourceFilters({
        ...baseViewState,
        lifecycleStateFilters: ["partially_queryable"]
      })
    ).toBe(true)
  })

  it("builds a compact summary string for collapsed controls", () => {
    const state: SourceListViewState = {
      ...baseViewState,
      typeFilters: ["pdf"],
      statusFilters: ["ready"],
      reviewStateFilters: ["needs_review"],
      lifecycleStateFilters: ["partially_queryable"],
      sort: "added_desc"
    }

    const summary = buildSourceFilterSummary(state)

    expect(summary).toContain("Type=PDF")
    expect(summary).toContain("Status=Ready")
    expect(summary).toContain("Review=Needs review")
    expect(summary).toContain("Lifecycle=Partially queryable")
    expect(summary).toContain("Sort: Added date")
  })
})
