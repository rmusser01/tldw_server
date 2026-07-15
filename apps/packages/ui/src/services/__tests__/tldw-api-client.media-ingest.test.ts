import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import { mediaMethods } from "@/services/tldw/domains/media"
import {
  parsePlaylistIngestRunStreamLine,
  type PlaylistIngestRunSubmissionRequest
} from "@/services/tldw/playlist-ingest"

describe("TldwApiClient media ingest contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses multipart /media/add fields inferred from URL", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/article")

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/add",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/article"]
        })
      })
    )
  })

  it("infers video media_type for youtube URLs", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        })
      })
    )
  })

  it("keeps explicit media_type overrides", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/file.mp4", {
      media_type: "audio"
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          media_type: "audio",
          urls: ["https://example.com/file.mp4"]
        })
      })
    )
  })

  it("forwards timeout and ingest options with urls list", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/video.mp4", {
      timeoutMs: 45000,
      perform_analysis: false,
      perform_chunking: true,
      overwrite_existing: false
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        timeoutMs: 45000,
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://example.com/video.mp4"],
          perform_analysis: false,
          perform_chunking: true,
          overwrite_existing: false
        })
      })
    )
  })

  it("submits media ingest jobs via multipart fields", async () => {
    mocks.bgUpload.mockResolvedValue({ batch_id: "batch-1", jobs: [] })

    const client = new TldwApiClient()
    await client.submitMediaIngestJobs({
      media_type: "video",
      urls: ["https://example.com/video.mp4"],
      perform_analysis: true,
      timeoutMs: 120000
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        timeoutMs: 120000,
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://example.com/video.mp4"],
          perform_analysis: true
        })
      })
    )
  })

  it("fetches ingest job detail and batch list endpoints", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: 12, status: "queued" })
    mocks.bgRequest.mockResolvedValueOnce({ batch_id: "b1", jobs: [] })

    const client = new TldwApiClient()
    await client.getMediaIngestJob(12)
    await client.listMediaIngestJobs({ batch_id: "b1", limit: 25 })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/12",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs?batch_id=b1&limit=25",
        method: "GET"
      })
    )
  })

  it("preflights playlist URLs through the metadata-only media endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({ playlist_id: "PLtest", items: [] })

    const client = new TldwApiClient()
    await client.preflightPlaylist({
      url: "https://www.youtube.com/playlist?list=PLtest",
      max_items: 34
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/playlists/preflight",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          url: "https://www.youtube.com/playlist?list=PLtest",
          max_items: 34
        }
      })
    )
  })

  it("routes chat document upload preflight and drafts through media endpoints", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({ files: [] })
      .mockResolvedValueOnce({
        draft_id: "draft-1",
        expires_at: "2026-07-09T00:00:00Z"
      })
      .mockResolvedValueOnce({ draft_id: "draft-1", payload: {} })
      .mockResolvedValueOnce(undefined)

    const client = new TldwApiClient()
    await client.preflightDocumentUpload({
      files: [
        {
          client_id: "file-1",
          filename: "notes.md",
          mime_type: "text/markdown",
          size_bytes: 12
        }
      ]
    })
    await client.createDocumentUploadDraft({ files: [{ id: "file-1" }] })
    await client.getDocumentUploadDraft("draft-1")
    await client.deleteDocumentUploadDraft("draft-1")

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/document-upload/preflight",
        method: "POST"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/document-upload/drafts",
        method: "POST"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/media/document-upload/drafts/draft-1",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      4,
      expect.objectContaining({
        path: "/api/v1/media/document-upload/drafts/draft-1",
        method: "DELETE"
      })
    )
  })

  it("sends server-side playlist preflight timeout derived from timeoutMs", async () => {
    mocks.bgRequest.mockResolvedValue({ playlist_id: "PLtest", items: [] })

    const client = new TldwApiClient()
    await client.preflightPlaylist({
      url: "https://www.youtube.com/playlist?list=PLtest",
      maxItems: 34,
      timeoutMs: 120000
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/playlists/preflight",
        method: "POST",
        body: {
          url: "https://www.youtube.com/playlist?list=PLtest",
          max_items: 34,
          timeout_seconds: 60
        },
        timeoutMs: 120000
      })
    )
  })

  it("creates and normalizes a version-2 playlist preflight", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      preflight_id: "preflight-1",
      status: "pending",
      status_url: "/api/v1/media/playlist-preflights/preflight-1",
      items_url: "/api/v1/media/playlist-preflights/preflight-1/items",
      expires_at: "2026-07-13T12:00:00Z",
      limits: {
        max_items: 500,
        global_capacity: 4,
        owner_capacity: 2
      }
    })

    const client = new TldwApiClient()
    const result = await client.createPlaylistPreflight({
      url: "https://www.youtube.com/playlist?list=PLtest",
      maxItems: 500,
      timeoutSeconds: 30
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/media/playlist-preflights",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        url: "https://www.youtube.com/playlist?list=PLtest",
        max_items: 500,
        timeout_seconds: 30
      }
    })
    expect(result).toEqual({
      contractVersion: 2,
      preflightId: "preflight-1",
      status: "pending",
      statusUrl: "/api/v1/media/playlist-preflights/preflight-1",
      itemsUrl: "/api/v1/media/playlist-preflights/preflight-1/items",
      expiresAt: "2026-07-13T12:00:00Z",
      limits: {
        maxItems: 500,
        globalCapacity: 4,
        ownerCapacity: 2
      }
    })
  })

  it("gets a preflight and pages items with an opaque cursor", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        preflight_id: "preflight-1",
        status: "ready",
        source_url: "https://www.youtube.com/playlist?list=PLtest",
        source_kind: "youtube_playlist",
        playlist_id: "PLtest",
        summary: {
          playlist_title: "Test playlist",
          total_count: 2,
          loaded_count: 1,
          ingestible_count: 1,
          unavailable_count: 0,
          duplicate_count: 0,
          selected_count: 1,
          warnings: []
        },
        error: null,
        created_at: "2026-07-13T10:00:00Z",
        updated_at: "2026-07-13T10:01:00Z",
        expires_at: "2026-07-13T12:00:00Z"
      })
      .mockResolvedValueOnce({
        contract_version: 2,
        preflight_id: "preflight-1",
        items: [
          {
            occurrence_id: "occ-1",
            ordinal: 1,
            occurrence_index_for_source: 1,
            source_url: "https://www.youtube.com/watch?v=video-1",
            normalized_source_id: "video-1",
            source_kind: "youtube_video",
            availability: "available",
            duplicate_status: "new",
            duplicate_of_occurrence_id: null,
            selected_by_default: true,
            display_metadata: {
              title: "First video",
              channel_or_uploader: "Channel",
              duration_seconds: 90,
              published_at: "2026-07-01",
              thumbnail_url: null,
              playlist_id: "PLtest",
              playlist_title: "Test playlist"
            }
          }
        ],
        next_cursor: "opaque:+/="
      })

    const client = new TldwApiClient()
    const summary = await client.getPlaylistPreflight("preflight-1")
    const page = await client.listPlaylistPreflightItems("preflight-1", {
      cursor: "opaque:+/=",
      limit: 100
    })

    expect(summary).toMatchObject({
      contractVersion: 2,
      preflightId: "preflight-1",
      status: "ready",
      playlistId: "PLtest",
      summary: {
        playlistTitle: "Test playlist",
        totalCount: 2,
        loadedCount: 1,
        selectedCount: 1
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path:
        "/api/v1/media/playlist-preflights/preflight-1/items?cursor=opaque%3A%2B%2F%3D&limit=100",
      method: "GET"
    })
    expect(page).toEqual({
      contractVersion: 2,
      preflightId: "preflight-1",
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          occurrenceIndexForSource: 1,
          normalizedSourceId: "video-1",
          selectedByDefault: true,
          displayMetadata: expect.objectContaining({
            title: "First video",
            channelOrUploader: "Channel",
            durationSeconds: 90,
            playlistTitle: "Test playlist"
          })
        })
      ],
      nextCursor: "opaque:+/="
    })
  })

  it("preserves an omitted preflight total count as unknown", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      preflight_id: "preflight-1",
      status: "ready",
      source_url: "https://www.youtube.com/playlist?list=PLtest",
      source_kind: "youtube_playlist",
      playlist_id: "PLtest",
      summary: {
        playlist_title: "Test playlist",
        loaded_count: 1,
        ingestible_count: 1,
        unavailable_count: 0,
        duplicate_count: 0,
        selected_count: 1,
        warnings: []
      },
      error: null,
      created_at: "2026-07-13T10:00:00Z",
      updated_at: "2026-07-13T10:01:00Z",
      expires_at: "2026-07-13T12:00:00Z"
    })

    const client = new TldwApiClient()
    const result = await client.getPlaylistPreflight("preflight-1")

    expect(result.summary?.totalCount).toBeNull()
  })

  it("materializes selected playlist occurrence ids", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      materialization_id: "materialization-1",
      preflight_id: "preflight-1",
      status: "ready",
      items: [
        {
          occurrence_id: "occ-1",
          ordinal: 1,
          source_url: "https://www.youtube.com/watch?v=video-1",
          normalized_source_id: "video-1",
          source_kind: "youtube_video",
          display_metadata: { title: "First video" }
        }
      ],
      expires_at: "2026-07-20T12:00:00Z"
    })

    const client = new TldwApiClient()
    const result = await client.materializePlaylistPreflight("preflight-1", [
      "occ-1"
    ])

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path:
        "/api/v1/media/playlist-preflights/preflight-1/materializations",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { occurrence_ids: ["occ-1"] }
    })
    expect(result).toMatchObject({
      contractVersion: 2,
      materializationId: "materialization-1",
      preflightId: "preflight-1",
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          normalizedSourceId: "video-1"
        })
      ]
    })
  })

  it("cancels a playlist preflight and propagates its AbortSignal", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)
    const controller = new AbortController()

    const client = new TldwApiClient()
    await client.cancelPlaylistPreflight("preflight/1", {
      signal: controller.signal
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/media/playlist-preflights/preflight%2F1",
      method: "DELETE",
      abortSignal: controller.signal
    })
  })

  it("creates a version-2 run from normalized camelCase inputs", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-1",
      status: "preparing",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-1",
      items_url: "/api/v1/media/ingest/runs/run-1/items",
      events_url: "/api/v1/media/ingest/runs/run-1/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-1",
          ordinal: 1,
          input_kind: "materialized_playlist_item",
          source_url: "https://www.youtube.com/watch?v=video-1",
          source_kind: "youtube_video",
          display_metadata: { title: "First video" },
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null
        }
      ]
    })

    const client = new TldwApiClient()
    const result = await client.createPlaylistIngestRun({
      clientRequestId: "quick-ingest-session-1",
      inputs: [
        {
          inputKind: "materialized_playlist_item",
          occurrenceId: "occ-1",
          materializationId: "materialization-1"
        }
      ],
      reviewOverrides: {
        "occ-1": {
          duplicatePolicy: "overwrite",
          existingMediaId: 7
        }
      },
      processingOptions: { perform_analysis: true },
      playlistSummaries: [{ playlist_id: "PLtest" }],
      newCollection: {
        name: "Test playlist",
        sourceUrl: "https://www.youtube.com/playlist?list=PLtest",
        defaultTags: ["conference"]
      }
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/media/ingest/runs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        client_request_id: "quick-ingest-session-1",
        inputs: [
          {
            input_kind: "materialized_playlist_item",
            occurrence_id: "occ-1",
            materialization_id: "materialization-1"
          }
        ],
        review_overrides: {
          "occ-1": {
            duplicate_policy: "overwrite",
            existing_media_id: 7
          }
        },
        processing_options: { perform_analysis: true },
        playlist_summaries: [{ playlist_id: "PLtest" }],
        new_collection: {
          name: "Test playlist",
          source_url: "https://www.youtube.com/playlist?list=PLtest",
          default_tags: ["conference"]
        }
      }
    })
    expect(result).toMatchObject({
      contractVersion: 2,
      runId: "run-1",
      processingOccurrences: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          inputKind: "materialized_playlist_item",
          sourceUrl: "https://www.youtube.com/watch?v=video-1",
          displayMetadata: { title: "First video" }
        })
      ]
    })
  })

  it("requires the client request identity at the public client transport", () => {
    expectTypeOf<
      Parameters<TldwApiClient["createPlaylistIngestRun"]>[0]
    >().toEqualTypeOf<PlaylistIngestRunSubmissionRequest>()
    expectTypeOf<
      Parameters<typeof mediaMethods.createPlaylistIngestRun>[0]
    >().toEqualTypeOf<PlaylistIngestRunSubmissionRequest>()
  })

  it("gets, pages, cancels, and retries a version-2 ingest run", async () => {
    const summaryWire = {
      contract_version: 2,
      run_id: "run-1",
      status: "running",
      counts: { total: 1, running: 1 },
      version: 2,
      collection_id: null,
      batch_ids: ["batch-1"],
      created_at: "2026-07-13T10:00:00Z",
      updated_at: "2026-07-13T10:01:00Z",
      expires_at: "2026-07-20T10:00:00Z"
    }
    mocks.bgRequest
      .mockResolvedValueOnce(summaryWire)
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-1",
        version: 2,
        items: [
          {
            occurrence_id: "occ-1",
            ordinal: 1,
            input_kind: "direct_url",
            source_url: "https://example.com/video.mp4",
            normalized_source_id: "url:https://example.com/video.mp4",
            source_kind: "video",
            display_metadata: { title: "Video" },
            action: "ingest",
            state: "running",
            outcome: null,
            progress_percent: 25,
            progress_message: "Downloading",
            job_id: 11,
            batch_id: "batch-1",
            media_id: null,
            planned_collection_item_id: null,
            attempt: 1,
            retryable: false
          }
        ],
        next_cursor: "run-cursor"
      })
      .mockResolvedValueOnce(summaryWire)
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-1",
        version: 3,
        processing_occurrences: []
      })

    const client = new TldwApiClient()
    const summary = await client.getPlaylistIngestRun("run-1")
    const page = await client.listPlaylistIngestRunItems("run-1", {
      cursor: "run-cursor",
      limit: 50
    })
    await client.cancelPlaylistIngestRun("run-1", {
      occurrenceIds: ["occ-1"],
      reason: "User cancelled"
    })
    const retry = await client.retryPlaylistIngestRunItems("run-1", [
      "occ-1"
    ])

    expect(summary).toMatchObject({
      contractVersion: 2,
      runId: "run-1",
      collectionId: null,
      batchIds: ["batch-1"]
    })
    expect(page).toMatchObject({
      runId: "run-1",
      nextCursor: "run-cursor",
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          progressPercent: 25,
          progressMessage: "Downloading",
          jobId: 11
        })
      ]
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path:
        "/api/v1/media/ingest/runs/run-1/items?cursor=run-cursor&limit=50",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/media/ingest/runs/run-1/cancel",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        occurrence_ids: ["occ-1"],
        reason: "User cancelled"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(4, {
      path: "/api/v1/media/ingest/runs/run-1/retry",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { occurrence_ids: ["occ-1"] }
    })
    expect(retry).toEqual({
      contractVersion: 2,
      runId: "run-1",
      version: 3,
      processingOccurrences: []
    })
  })

  it("cancels a whole ingest run without a request body", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-1",
      status: "cancelled",
      counts: { total: 1, cancelled: 1 },
      version: 3,
      collection_id: null,
      batch_ids: [],
      created_at: "2026-07-13T10:00:00Z",
      updated_at: "2026-07-13T10:01:00Z",
      expires_at: "2026-07-20T10:00:00Z"
    })

    const client = new TldwApiClient()
    await client.cancelPlaylistIngestRun("run-1")

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/media/ingest/runs/run-1/cancel",
      method: "POST"
    })
  })

  it("streams normalized run event contracts from a resumable cursor", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({
        contract_version: 2,
        run_id: "run-1",
        status: "running",
        counts: { total: 1, running: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-13T10:00:00Z",
        updated_at: "2026-07-13T10:01:00Z",
        expires_at: "2026-07-20T10:00:00Z"
      })
      yield JSON.stringify({
        event_id: 43,
        run_id: "run-1",
        occurrence_id: "occ-1",
        job_id: 11,
        batch_id: "batch-1",
        event_type: "progress",
        state: "running",
        outcome: null,
        progress_percent: 50,
        progress_message: "Halfway",
        occurred_at: "2026-07-13T10:02:00Z"
      })
      yield JSON.stringify({
        event_id: 44,
        run_id: "run-1",
        occurrence_id: null,
        job_id: null,
        batch_id: null,
        event_type: "run_progress",
        state: null,
        outcome: null,
        progress_percent: 60,
        progress_message: "Run progressing",
        occurred_at: "2026-07-13T10:03:00Z"
      })
      yield JSON.stringify({
        run_id: "run-1",
        code: "run_status_unavailable"
      })
      yield JSON.stringify({
        run_id: "run-1",
        min_event_id: 30,
        latest_event_id: 43
      })
    })

    const client = new TldwApiClient()
    const events = []
    for await (const event of client.streamPlaylistIngestRunEvents("run-1", {
      afterId: 42
    })) {
      events.push(event)
    }

    expect(mocks.bgStream).toHaveBeenCalledWith({
      path: "/api/v1/media/ingest/runs/run-1/events/stream?after_id=42",
      method: "GET",
      headers: {
        Accept: "text/event-stream",
        "Last-Event-ID": "42"
      }
    })
    expect(events).toEqual([
      expect.objectContaining({
        kind: "snapshot",
        summary: expect.objectContaining({ runId: "run-1", version: 2 })
      }),
      expect.objectContaining({
        kind: "occurrence",
        event: expect.objectContaining({
          eventId: 43,
          occurrenceId: "occ-1",
          progressPercent: 50
        })
      }),
      expect.objectContaining({
        kind: "run",
        event: expect.objectContaining({
          eventId: 44,
          occurrenceId: null,
          progressPercent: 60
        })
      }),
      {
        kind: "statusUnavailable",
        runId: "run-1",
        code: "run_status_unavailable"
      },
      {
        kind: "resyncRequired",
        runId: "run-1",
        minEventId: 30,
        latestEventId: 43
      }
    ])
  })

  it("ignores incomplete or invalid playlist run event frames", () => {
    const validFrame = {
      event_id: 43,
      run_id: "run-1",
      occurrence_id: "occ-1",
      job_id: 11,
      batch_id: "batch-1",
      event_type: "progress",
      state: "running",
      outcome: null,
      progress_percent: 50,
      progress_message: "Halfway",
      occurred_at: "2026-07-13T10:02:00Z"
    }
    const without = (key: keyof typeof validFrame) =>
      Object.fromEntries(
        Object.entries(validFrame).filter(([candidate]) => candidate !== key)
      )
    const malformedFrames = [
      { run_id: "run-1", event_id: 43, event_type: "progress" },
      { ...validFrame, event_id: -1 },
      { ...validFrame, event_id: 1.5 },
      { ...validFrame, run_id: " " },
      { ...validFrame, event_type: "" },
      { ...validFrame, occurred_at: "" },
      { ...validFrame, job_id: "11" },
      { ...validFrame, state: "future_state" },
      { ...validFrame, outcome: "future_outcome" },
      { ...validFrame, progress_percent: "50" },
      { ...validFrame, progress_message: 50 },
      ...(
        [
          "occurrence_id",
          "job_id",
          "batch_id",
          "state",
          "outcome",
          "progress_percent",
          "progress_message"
        ] as const
      ).map(without)
    ]

    for (const frame of malformedFrames) {
      expect(parsePlaylistIngestRunStreamLine(JSON.stringify(frame))).toBeNull()
    }
  })

  it("classifies a status-less stream interruption as unavailable run status", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield await Promise.reject(
        new Error("socket closed with raw transport detail")
      )
    })

    const client = new TldwApiClient()
    const consume = async () => {
      for await (const _event of client.streamPlaylistIngestRunEvents("run-1")) {
        // Consume until the stream reports its interruption.
      }
    }

    await expect(consume()).rejects.toMatchObject({
      name: "PlaylistIngestPublicError",
      code: "run_status_unavailable",
      message: "Run status is temporarily unavailable. Reconnect to try again."
    })
    await expect(consume()).rejects.not.toThrow(/socket|transport|raw/i)
  })

  it("preserves stream AbortError identity", async () => {
    const abortError = new Error("cancelled")
    abortError.name = "AbortError"
    mocks.bgStream.mockImplementation(async function* () {
      yield await Promise.reject(abortError)
    })

    const client = new TldwApiClient()
    const consume = async () => {
      for await (const _event of client.streamPlaylistIngestRunEvents("run-1")) {
        // Consume until cancellation interrupts the stream.
      }
    }

    await expect(consume()).rejects.toBe(abortError)
  })

  it("keeps transport status zero classified as server unreachable", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield await Promise.reject(
        Object.assign(new Error("network detail"), { status: 0 })
      )
    })

    const client = new TldwApiClient()
    const consume = async () => {
      for await (const _event of client.streamPlaylistIngestRunEvents("run-1")) {
        // Consume until the stream reports its interruption.
      }
    }

    await expect(consume()).rejects.toMatchObject({
      code: "server_unreachable",
      message: "The server could not be reached. Try again."
    })
  })

  it.each([
    ["invalid_run_cancel_request", "cancel"],
    ["invalid_run_retry_request", "retry"]
  ] as const)("maps %s to a stable invalid run request", async (code, action) => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("raw validation detail"), {
        status: 422,
        details: { detail: code }
      })
    )

    const client = new TldwApiClient()
    const request =
      action === "cancel"
        ? client.cancelPlaylistIngestRun("run-1")
        : client.retryPlaylistIngestRunItems("run-1", ["occ-1"])

    await expect(request).rejects.toMatchObject({
      code: "invalid_run_request",
      message: "The playlist ingest request is no longer valid."
    })
  })

  it("preserves request AbortError identity", async () => {
    const abortError = new Error("cancelled")
    abortError.name = "AbortError"
    mocks.bgRequest.mockRejectedValue(abortError)

    const client = new TldwApiClient()

    await expect(client.getPlaylistPreflight("preflight-1")).rejects.toBe(
      abortError
    )
  })

  it("normalizes allowlisted review-required recovery fields", async () => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("raw duplicate detail"), {
        status: 409,
        details: {
          detail: {
            code: "review_required",
            items: [
              {
                occurrence_id: "occ-library",
                reason: "duplicate_action_required",
                evidence: {
                  kind: "library",
                  existing_media_id: 42,
                  duplicate_of_occurrence_id: null,
                  raw_library_path: "/secret/library"
                },
                allowed_actions: [
                  "skip",
                  "include_existing",
                  "update_metadata_only",
                  "overwrite"
                ],
                raw_title: "secret title"
              },
              {
                occurrence_id: "occ-in-run",
                reason: "in_run_duplicate_requires_processing_or_skip",
                evidence: {
                  kind: "in_run",
                  existing_media_id: null,
                  duplicate_of_occurrence_id: "occ-first"
                },
                allowed_actions: ["skip", "overwrite"]
              }
            ],
            raw_trace: "do not expose"
          }
        }
      })
    )

    const client = new TldwApiClient()
    const error = await client
      .createPlaylistIngestRun({
        clientRequestId: "quick-ingest-review-required",
        inputs: []
      })
      .catch((caught: unknown) => caught)

    expect(error).toMatchObject({
      code: "review_required",
      message: "Review the updated duplicate choices before continuing."
    })
    expect((error as { recovery?: unknown }).recovery).toEqual({
      kind: "reviewRequired",
      items: [
        {
          occurrenceId: "occ-library",
          reason: "duplicate_action_required",
          evidence: {
            kind: "library",
            existingMediaId: 42,
            duplicateOfOccurrenceId: null
          },
          allowedActions: [
            "skip",
            "include_existing",
            "update_metadata_only",
            "overwrite"
          ]
        },
        {
          occurrenceId: "occ-in-run",
          reason: "in_run_duplicate_requires_processing_or_skip",
          evidence: {
            kind: "in_run",
            existingMediaId: null,
            duplicateOfOccurrenceId: "occ-first"
          },
          allowedActions: ["skip", "overwrite"]
        }
      ]
    })
    expect(JSON.stringify((error as { recovery?: unknown }).recovery)).not.toMatch(
      /secret|raw_|trace|title|path/i
    )
  })

  it("normalizes duplicate-action-pending recovery run identity", async () => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("raw pending detail"), {
        status: 409,
        details: {
          detail: {
            code: "duplicate_action_pending",
            run_id: "run-1",
            raw_worker_state: "secret"
          }
        }
      })
    )

    const client = new TldwApiClient()
    const error = await client
      .createPlaylistIngestRun({
        clientRequestId: "quick-ingest-duplicate-pending",
        inputs: []
      })
      .catch((caught: unknown) => caught)

    expect((error as { recovery?: unknown }).recovery).toEqual({
      kind: "duplicateActionPending",
      runId: "run-1"
    })
  })

  it.each([
    {
      code: "review_required",
      items: [
        {
          occurrence_id: " ",
          reason: "future_reason",
          evidence: {
            kind: "future_kind",
            existing_media_id: 0,
            duplicate_of_occurrence_id: ""
          },
          allowed_actions: ["future_action"],
          raw_trace: "secret"
        }
      ]
    },
    {
      code: "duplicate_action_pending",
      run_id: " " ,
      raw_trace: "secret"
    }
  ])("fails closed for malformed recovery details", async (detail) => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("raw malformed detail"), {
        status: 409,
        details: { detail }
      })
    )

    const client = new TldwApiClient()
    const error = await client
      .createPlaylistIngestRun({
        clientRequestId: "quick-ingest-malformed-recovery",
        inputs: []
      })
      .catch((caught: unknown) => caught)

    expect(error).toMatchObject({
      code: detail.code,
      recovery: null
    })
    expect(JSON.stringify(error)).not.toMatch(/secret|raw_|trace/i)
  })

  it("maps unknown backend detail to a safe typed playlist ingest error", async () => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("raw yt-dlp traceback"), {
        status: 500,
        details: {
          detail: {
            code: "extractor_crashed",
            message: "raw yt-dlp traceback /secret/path"
          }
        }
      })
    )

    const client = new TldwApiClient()
    const request = client.createPlaylistPreflight({
      url: "https://www.youtube.com/playlist?list=PLtest"
    })

    await expect(request).rejects.toMatchObject({
      name: "PlaylistIngestPublicError",
      code: "playlist_ingest_failed",
      message: "Playlist ingestion is unavailable. Try again.",
      recovery: null
    })
    await expect(request).rejects.not.toThrow(/yt-dlp|traceback|secret/i)
  })

  it("uploads character imports using binary payloads", async () => {
    mocks.bgUpload.mockResolvedValue({
      id: 123,
      name: "Imported Character",
      message: "Character imported successfully"
    })

    const client = new TldwApiClient()
    ;(client as any).ensureConfigForRequest = vi.fn(async () => ({ ok: true }))
    const rawBuffer = new Uint8Array([0x89, 0x50, 0x4e, 0x47]).buffer
    const file = {
      name: "card.png",
      type: "image/png",
      arrayBuffer: vi.fn(async () => rawBuffer)
    } as unknown as File

    await client.importCharacterFile(file, { allowImageOnly: true })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/import",
        method: "POST",
        fileFieldName: "character_file",
        fields: { allow_image_only: true },
        file: expect.objectContaining({
          name: "card.png",
          type: "image/png"
        })
      })
    )

    const callArg = mocks.bgUpload.mock.calls[0][0] as {
      file?: { data?: unknown }
    }
    expect(callArg.file?.data).toBe(rawBuffer)
    expect(Array.isArray(callArg.file?.data)).toBe(false)
  })

  it("uploads yaml character imports through the same endpoint contract", async () => {
    mocks.bgUpload.mockResolvedValue({
      id: 124,
      name: "Imported YAML Character",
      message: "Character imported successfully"
    })

    const client = new TldwApiClient()
    ;(client as any).ensureConfigForRequest = vi.fn(async () => ({ ok: true }))
    const rawBuffer = new TextEncoder().encode("name: YAML Client Test").buffer
    const file = {
      name: "card.yaml",
      type: "text/yaml",
      arrayBuffer: vi.fn(async () => rawBuffer)
    } as unknown as File

    await client.importCharacterFile(file)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/import",
        method: "POST",
        fileFieldName: "character_file",
        file: expect.objectContaining({
          name: "card.yaml",
          type: "text/yaml"
        })
      })
    )

    const callArg = mocks.bgUpload.mock.calls.at(-1)?.[0] as {
      fields?: Record<string, unknown>
    }
    expect(callArg.fields).toBeUndefined()
  })
})
