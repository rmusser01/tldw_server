import { describe, expect, it } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import {
  buildSourceFilterSummary,
  filterSources,
  hasActiveSourceFilters,
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
    addedAt: new Date("2026-03-12T00:00:00.000Z"),
    url: "https://example.com"
  },
  {
    id: "s3",
    mediaId: 3,
    title: "Alpha Audio",
    type: "audio",
    status: "processing",
    addedAt: new Date("2026-03-11T00:00:00.000Z"),
    duration: 90
  }
]

const baseViewState: SourceListViewState = {
  expanded: false,
  typeFilters: [],
  statusFilters: [],
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
  })

  it("builds a compact summary string for collapsed controls", () => {
    const state: SourceListViewState = {
      ...baseViewState,
      typeFilters: ["pdf"],
      statusFilters: ["ready"],
      sort: "added_desc"
    }

    const summary = buildSourceFilterSummary(state)

    expect(summary).toContain("Type=PDF")
    expect(summary).toContain("Status=Ready")
    expect(summary).toContain("Sort: Added date")
  })
})
