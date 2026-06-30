import { describe, expect, it } from "vitest"
import {
  buildCitationUsageAnchors,
  buildSourceContentFacetCounts,
  buildHighlightTerms,
  buildSourceTypeCounts,
  detectSourceContentFacet,
  filterItemsByContentFacet,
  filterItemsByDateRange,
  filterItemsByKeyword,
  filterItemsBySourceType,
  formatChunkPosition,
  formatSourceDate,
  getFreshnessDescriptor,
  getResultChunkId,
  getResultSourceId,
  getRelevanceDescriptor,
  getSourceTypeLabel,
  normalizeSourceType,
  splitTextByHighlights,
  sortSourceItems,
  type SourceListItem,
} from "../sourceListUtils"

const makeItem = (
  id: string,
  originalIndex: number,
  overrides: Partial<SourceListItem["result"]> = {}
): SourceListItem => ({
  originalIndex,
  result: {
    id,
    metadata: {
      title: `Title ${id}`,
      source_type: "media_db",
    },
    ...overrides,
  },
})

describe("sourceListUtils", () => {
  it("normalizes source types and returns friendly labels", () => {
    expect(normalizeSourceType("notes")).toBe("notes")
    expect(normalizeSourceType("unknown-type")).toBe("unknown")
    expect(getSourceTypeLabel("media_db")).toBe("Document")
    expect(getSourceTypeLabel("notes", { plural: true })).toBe("Notes")
  })

  it("builds source type counts and filters by active type", () => {
    const items = [
      makeItem("a", 0, { metadata: { source_type: "media_db" } }),
      makeItem("b", 1, { metadata: { source_type: "notes" } }),
      makeItem("c", 2, { metadata: { source_type: "notes" } }),
    ]

    expect(buildSourceTypeCounts(items.map((item) => item.result))).toEqual({
      media_db: 1,
      notes: 2,
    })
    expect(filterItemsBySourceType(items, "notes").map((item) => item.result.id)).toEqual([
      "b",
      "c",
    ])
  })

  it("detects and filters content-type facets", () => {
    const items = [
      makeItem("pdf", 0, {
        metadata: { source_type: "media_db", title: "Spec", file_type: "pdf" },
      }),
      makeItem("video", 1, {
        metadata: {
          source_type: "media_db",
          title: "Demo",
          mime_type: "video/mp4",
        },
      }),
      makeItem("note", 2, {
        metadata: { source_type: "notes", title: "Research note" },
      }),
    ]

    expect(detectSourceContentFacet(items[0].result)).toBe("pdf")
    expect(detectSourceContentFacet(items[1].result)).toBe("video")
    expect(detectSourceContentFacet(items[2].result)).toBe("note")

    const counts = buildSourceContentFacetCounts(items.map((item) => item.result))
    expect(counts.pdf).toBe(1)
    expect(counts.video).toBe(1)
    expect(counts.note).toBe(1)

    expect(filterItemsByContentFacet(items, "pdf").map((item) => item.result.id)).toEqual([
      "pdf",
    ])
  })

  it("filters source items by keyword terms", () => {
    const items = [
      makeItem("alpha", 0, {
        content: "Neural retrieval baseline report",
        metadata: { source_type: "media_db", title: "Alpha report" },
      }),
      makeItem("beta", 1, {
        content: "Product launch memo",
        metadata: { source_type: "notes", title: "Beta notes" },
      }),
    ]

    const filtered = filterItemsByKeyword(items, "retrieval baseline")
    expect(filtered.map((item) => item.result.id)).toEqual(["alpha"])
  })

  it("filters source items by date range", () => {
    const now = Date.parse("2026-02-20T00:00:00.000Z")
    const items = [
      makeItem("recent", 0, {
        metadata: {
          source_type: "media_db",
          title: "Recent",
          created_at: "2026-02-15T00:00:00.000Z",
        },
      }),
      makeItem("old", 1, {
        metadata: {
          source_type: "media_db",
          title: "Old",
          created_at: "2023-01-01T00:00:00.000Z",
        },
      }),
      makeItem("no-date", 2, {
        metadata: {
          source_type: "media_db",
          title: "No date",
        },
      }),
    ]

    expect(filterItemsByDateRange(items, "last_30d", now).map((item) => item.result.id)).toEqual([
      "recent",
    ])
    expect(filterItemsByDateRange(items, "older_365d", now).map((item) => item.result.id)).toEqual([
      "old",
    ])
  })

  it("formats chunk positions across common chunk id formats", () => {
    expect(formatChunkPosition("3/12")).toBe("Section 3 of 12")
    expect(formatChunkPosition("chunk_4_of_20")).toBe("Section 4 of 20")
    expect(formatChunkPosition("chunk-7")).toBe("Section 7")
    expect(formatChunkPosition("128")).toBe("Section 128")
    expect(formatChunkPosition("doc-aabbcc-uuid")).toBeNull()
  })

  it("resolves stable source and chunk identifiers from top-level and metadata values", () => {
    expect(
      getResultSourceId({
        metadata: { media_id: 42 },
      })
    ).toBe("42")
    expect(
      getResultChunkId({
        metadata: { chunk_id: 7 },
      })
    ).toBe("7")
    expect(
      getResultSourceId({
        sourceId: "note-9",
        metadata: { source_id: "stale-id" },
      })
    ).toBe("note-9")
  })

  it("produces relevance descriptor levels with color semantics", () => {
    expect(getRelevanceDescriptor(0.91)?.level).toBe("high")
    expect(getRelevanceDescriptor(0.91)?.label).toBe("Strong relevance")
    expect(getRelevanceDescriptor(0.67)?.level).toBe("moderate")
    expect(getRelevanceDescriptor(0.67)?.label).toBe("Moderate relevance")
    expect(getRelevanceDescriptor(0.21)?.level).toBe("low")
    expect(getRelevanceDescriptor(0.21)?.label).toBe("Weak relevance")
    expect(getRelevanceDescriptor(undefined)).toBeNull()
  })

  it("sorts items by title, date, and cited-first", () => {
    const items = [
      makeItem("z", 0, {
        metadata: {
          title: "Zulu",
          source_type: "media_db",
          created_at: "2026-01-05T12:00:00.000Z",
        },
      }),
      makeItem("a", 1, {
        metadata: {
          title: "Alpha",
          source_type: "notes",
          created_at: "2026-02-05T12:00:00.000Z",
        },
      }),
      makeItem("m", 2, {
        metadata: {
          title: "Mike",
          source_type: "chats",
        },
      }),
    ]
    const cited = new Set<number>([2])

    expect(sortSourceItems(items, "title", cited).map((item) => item.result.id)).toEqual([
      "a",
      "m",
      "z",
    ])
    expect(sortSourceItems(items, "date", cited).map((item) => item.result.id)).toEqual([
      "a",
      "z",
      "m",
    ])
    expect(sortSourceItems(items, "cited", cited).map((item) => item.result.id)).toEqual([
      "m",
      "z",
      "a",
    ])
  })

  it("formats source date from metadata date keys", () => {
    const dated = makeItem("d", 0, {
      metadata: {
        source_type: "media_db",
        title: "Dated source",
        published_at: "2026-02-01T12:00:00.000Z",
      },
    })
    const formatted = formatSourceDate(dated.result)
    expect(formatted).toContain("2026")
  })

  it("classifies source freshness into recent, aging, and stale bands", () => {
    const now = Date.parse("2026-02-20T00:00:00.000Z")
    const recent = makeItem("recent", 0, {
      metadata: {
        source_type: "media_db",
        title: "Recent source",
        created_at: "2026-02-18T00:00:00.000Z",
      },
    })
    const aging = makeItem("aging", 1, {
      metadata: {
        source_type: "media_db",
        title: "Aging source",
        created_at: "2024-10-01T00:00:00.000Z",
      },
    })
    const stale = makeItem("stale", 2, {
      metadata: {
        source_type: "media_db",
        title: "Stale source",
        created_at: "2021-01-01T00:00:00.000Z",
      },
    })

    expect(getFreshnessDescriptor(recent.result, now)?.label).toBe("Updated 2d ago")
    expect(getFreshnessDescriptor(aging.result, now)?.className).toContain("text-warn")
    expect(getFreshnessDescriptor(stale.result, now)?.label).toBe("From 2021")
    expect(getFreshnessDescriptor(stale.result, now)?.className).toContain("text-danger")
  })

  it("builds highlight terms and splits excerpts into highlight segments", () => {
    const terms = buildHighlightTerms("compare source confidence", [
      "confidence thresholds",
    ])
    expect(terms).toContain("source")
    expect(terms).toContain("confidence")

    const segments = splitTextByHighlights(
      "This source has high confidence according to ranking.",
      terms
    )
    expect(segments.some((segment) => segment.highlight)).toBe(true)
    expect(
      segments.some(
        (segment) => segment.highlight && segment.text.toLowerCase() === "source"
      )
    ).toBe(true)
  })

  it("applies word-boundary highlighting for single tokens", () => {
    const segments = splitTextByHighlights("Storage systems improve search", ["rag"])
    expect(
      segments.some(
        (segment) => segment.highlight && segment.text.toLowerCase() === "rag"
      )
    ).toBe(false)
  })

  it("builds sentence-level citation usage anchors with occurrences", () => {
    const answer =
      "First claim cites [3]. Second claim cites [8]. Third sentence revisits [3] and [3]."
    const usage = buildCitationUsageAnchors(answer)

    expect(usage[3]).toEqual([
      {
        sentenceNumber: 1,
        occurrence: 1,
        sentencePreview: "First claim cites [3].",
      },
      {
        sentenceNumber: 3,
        occurrence: 2,
        sentencePreview: "Third sentence revisits [3] and [3].",
      },
    ])
    expect(usage[8]).toEqual([
      {
        sentenceNumber: 2,
        occurrence: 1,
        sentencePreview: "Second claim cites [8].",
      },
    ])
  })
})
