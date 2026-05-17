import { describe, expect, it, vi, beforeEach } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

import {
  buildConferenceRetryRequestItems,
  buildConferenceFailedResultExportText,
  classifyConferenceIngestFailure,
  getMediaCollectionStatusCounts,
  resolveConferenceDuplicatePolicy,
  buildConferenceCollectionItemPayload,
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

  it("exports failed conference result rows with collection and retry context", () => {
    const text = buildConferenceFailedResultExportText([
      {
        id: "submit-1",
        status: "error",
        outcome: "submit_failed",
        type: "video",
        title: "Submit Blocked",
        url: "https://example.com/submit",
        collectionItemId: "13",
        error: "Queue unavailable"
      } as any,
      {
        id: "failed-1",
        status: "error",
        outcome: "failed",
        type: "video",
        title: "Bad Video",
        url: "https://example.com/fail",
        collectionItemId: "14",
        retryAttempt: 2,
        error: "Download failed"
      } as any
    ])

    expect(text).toContain("Title: Submit Blocked")
    expect(text).toContain("URL: https://example.com/submit")
    expect(text).toContain("Collection item: 13")
    expect(text).toContain("Status: submit_failed")
    expect(text).toContain("Error: Queue unavailable")
    expect(text).toContain("Title: Bad Video")
    expect(text).toContain("Collection item: 14")
    expect(text).toContain("Status: failed")
    expect(text).toContain("Retry attempt: 2")
    expect(text).toContain("Error: Download failed")
  })

  it("resolves duplicate policies into planned status and submit behavior", () => {
    expect(
      resolveConferenceDuplicatePolicy("duplicate_existing", "skip")
    ).toMatchObject({
      plannedStatus: "skipped_existing",
      shouldSubmitJob: false
    })
    expect(
      resolveConferenceDuplicatePolicy("duplicate_existing", "overwrite")
    ).toMatchObject({
      plannedStatus: "planned",
      shouldSubmitJob: true,
      forceOverwrite: true
    })
    expect(
      resolveConferenceDuplicatePolicy("duplicate_existing", "update_metadata_only")
    ).toMatchObject({
      plannedStatus: "skipped_existing",
      shouldSubmitJob: false
    })
    expect(
      resolveConferenceDuplicatePolicy("duplicate_existing", "include_existing")
    ).toMatchObject({
      plannedStatus: "skipped_existing",
      shouldSubmitJob: false
    })
    expect(resolveConferenceDuplicatePolicy("new", "skip")).toMatchObject({
      plannedStatus: "planned",
      shouldSubmitJob: true
    })
    expect(resolveConferenceDuplicatePolicy("unknown", "skip")).toMatchObject({
      plannedStatus: "planned",
      shouldSubmitJob: true
    })
  })

  it("adds duplicate policy metadata to planned collection item payloads", () => {
    const payload = buildConferenceCollectionItemPayload(
      {
        collectionName: "Conference 2026",
        sharedTags: ["conference"],
        sourcePlaylistUrl: "https://www.youtube.com/playlist?list=PLtest"
      },
      {
        id: "entry-1",
        url: "https://www.youtube.com/watch?v=a",
        playlist: {
          playlistId: "PLtest",
          ordinal: 1,
          normalizedSourceId: "youtube:video:a",
          duplicateStatus: "duplicate_existing"
        },
        conferenceOverride: {
          selected: true,
          duplicatePolicy: "include_existing",
          title: "Existing Talk"
        }
      }
    )

    expect(payload.status).toBe("skipped_existing")
    expect(payload.duplicate_status).toBe("duplicate_existing")
    expect(payload.metadata).toMatchObject({
      duplicate_policy: "include_existing",
      quick_ingest_item_id: "entry-1"
    })
  })

  it("classifies conservative conference ingest failure categories", () => {
    expect(classifyConferenceIngestFailure("Private video")).toBe("auth_required")
    expect(classifyConferenceIngestFailure("HTTP Error 404")).toBe("unavailable")
    expect(classifyConferenceIngestFailure("timed out")).toBe("timeout")
  })

  it("builds selected retry requests from durable failed collection items only", () => {
    const retryItems = buildConferenceRetryRequestItems([
      {
        id: "ok-1",
        status: "ok",
        outcome: "processed",
        type: "video",
        collectionItemId: "11"
      } as any,
      {
        id: "submit-1",
        status: "error",
        outcome: "submit_failed",
        type: "video",
        collectionItemId: "13"
      } as any,
      {
        id: "failed-1",
        status: "error",
        outcome: "failed",
        type: "video",
        collectionItemId: "14",
        retryAttempt: 2
      } as any,
      {
        id: "cancel-1",
        status: "error",
        outcome: "cancelled",
        type: "video",
        collectionItemId: "15"
      } as any,
      {
        id: "legacy-failed",
        status: "error",
        outcome: "failed",
        type: "video"
      } as any
    ])

    expect(retryItems).toEqual([
      {
        resultId: "submit-1",
        collectionItemId: "13",
        retryAttempt: 1,
        idempotencyKey: "conference-retry-13-1"
      },
      {
        resultId: "failed-1",
        collectionItemId: "14",
        retryAttempt: 3,
        idempotencyKey: "conference-retry-14-3"
      },
      {
        resultId: "cancel-1",
        collectionItemId: "15",
        retryAttempt: 1,
        idempotencyKey: "conference-retry-15-1"
      }
    ])
  })
})
