import { describe, expect, it, vi, beforeEach } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

import {
  getMediaCollectionStatusCounts,
  normalizeMediaCollectionResponse
} from "@/services/tldw/conference-collections"
import { mediaMethods } from "@/services/tldw/domains/media"

describe("conference media collection normalizers", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  it("normalizes collection metadata, items, and status counts", () => {
    const normalized = normalizeMediaCollectionResponse({
      id: 7,
      name: "Conference 2026",
      kind: "conference",
      source_url: "https://www.youtube.com/playlist?list=PLtest",
      metadata: { conference_name: "Conference" },
      default_tags: ["conference"],
      created_at: "2026-05-01T00:00:00Z",
      updated_at: "2026-05-02T00:00:00Z",
      items: [
        {
          id: 11,
          collection_id: 7,
          ordinal: 1,
          source_url: "https://www.youtube.com/watch?v=a",
          normalized_source_id: "youtube:video:a",
          source_kind: "youtube_video",
          title: "Opening Keynote",
          speaker: "Ada Lovelace",
          duplicate_status: "new",
          status: "completed",
          media_id: 101,
          content_item_id: 201,
          retry_count: 0,
          warnings: [],
          metadata: { track: "Main" },
          tags: ["keynote"],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-02T00:00:00Z"
        },
        {
          id: 12,
          collection_id: 7,
          ordinal: 2,
          source_url: "https://www.youtube.com/watch?v=b",
          duplicate_status: "duplicate_in_batch",
          status: "failed",
          retry_count: 1,
          error_summary: "Download failed",
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-02T00:00:00Z"
        }
      ]
    })

    expect(normalized.sourceUrl).toBe("https://www.youtube.com/playlist?list=PLtest")
    expect(normalized.metadata.conference_name).toBe("Conference")
    expect(normalized.defaultTags).toEqual(["conference"])
    expect(normalized.items[0]).toMatchObject({
      id: 11,
      sourceUrl: "https://www.youtube.com/watch?v=a",
      normalizedSourceId: "youtube:video:a",
      contentItemId: 201,
      mediaId: 101,
      status: "completed"
    })
    expect(getMediaCollectionStatusCounts(normalized)).toEqual({
      total: 2,
      planned: 0,
      processing: 0,
      completed: 1,
      skippedExisting: 0,
      submitFailed: 0,
      failed: 1,
      cancelled: 0
    })
  })

  it("uses media collection endpoints through the shared media domain", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        id: 7,
        name: "Conference 2026",
        kind: "conference",
        metadata: {},
        default_tags: [],
        created_at: "2026-05-01T00:00:00Z",
        updated_at: "2026-05-01T00:00:00Z",
        items: []
      })
      .mockResolvedValueOnce({
        id: 11,
        collection_id: 7,
        ordinal: 1,
        source_url: "https://www.youtube.com/watch?v=a",
        duplicate_status: "new",
        status: "planned",
        retry_count: 0,
        warnings: [],
        metadata: {},
        tags: [],
        created_at: "2026-05-01T00:00:00Z",
        updated_at: "2026-05-01T00:00:00Z"
      })

    const collection = await mediaMethods.createMediaCollection({
      name: "Conference 2026",
      kind: "conference"
    })
    const item = await mediaMethods.addMediaCollectionItem(7, {
      source_url: "https://www.youtube.com/watch?v=a",
      status: "planned"
    })

    expect(collection.id).toBe(7)
    expect(item.collectionId).toBe(7)
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/collections",
        method: "POST",
        body: { name: "Conference 2026", kind: "conference" }
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/collections/7/items",
        method: "POST",
        body: {
          source_url: "https://www.youtube.com/watch?v=a",
          status: "planned"
        }
      })
    )
  })
})
