import { describe, expect, it } from "vitest"
import {
  getMediaLibraryItemKey,
  normalizeMediaLibraryResponse
} from "../SourcesPane/media-library-normalization"

describe("media-library-normalization", () => {
  it.each([
    [
      "media",
      { media: [{ id: 1, title: "Media" }], total_count: 9 },
      [{ id: 1, title: "Media" }],
      9
    ],
    [
      "results",
      { results: [{ id: 2, title: "Result" }], total: 8 },
      [{ id: 2, title: "Result" }],
      8
    ],
    [
      "items",
      { items: [{ id: 3, title: "Item" }], count: 7 },
      [{ id: 3, title: "Item" }],
      7
    ],
    [
      "data",
      { data: [{ id: 4, title: "Data" }], pagination: { total: 6 } },
      [{ id: 4, title: "Data" }],
      6
    ]
  ])("normalizes %s response shape", (_label, response, expectedItems, expectedTotal) => {
    const normalized = normalizeMediaLibraryResponse(response)

    expect(normalized.items).toEqual(expectedItems)
    expect(normalized.totalCount).toBe(expectedTotal)
  })

  it("normalizes nested data.items response shape", () => {
    const normalized = normalizeMediaLibraryResponse({
      data: { items: [{ media_id: 5, title: "Nested" }], total: 5 }
    })

    expect(normalized.items).toEqual([{ media_id: 5, title: "Nested" }])
    expect(normalized.totalCount).toBe(5)
  })

  it("returns stable string keys for numeric and string ids", () => {
    expect(getMediaLibraryItemKey({ media_id: 0 })).toBe("0")
    expect(getMediaLibraryItemKey({ id: "abc" })).toBe("abc")
    expect(getMediaLibraryItemKey({ title: "missing" })).toBeNull()
  })
})
