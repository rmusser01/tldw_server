import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: undefined as string | undefined,
  manifestVersion: 3,
  sendMessage: vi.fn(),
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return mocks.runtimeId
      },
      getManifest: () => ({
        manifest_version: mocks.manifestVersion
      }),
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
    }
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

import {
  __resetQuickIngestRuntimeHealthForTests,
  cancelQuickIngestSession,
  getQuickIngestAnalysisProviderWarning,
  startQuickIngestSession,
  submitQuickIngestBatch
} from "@/services/tldw/quick-ingest-batch"
import { DUPLICATE_SKIP_MESSAGE } from "@/components/Common/QuickIngest/constants"

describe("submitQuickIngestBatch", () => {
  beforeEach(() => {
    __resetQuickIngestRuntimeHealthForTests()
    vi.useRealTimers()
    mocks.runtimeId = undefined
    mocks.manifestVersion = 3
    mocks.sendMessage.mockReset()
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
  })

  it("warns when analysis is enabled without an analysis provider", () => {
    expect(
      getQuickIngestAnalysisProviderWarning({
        common: { perform_analysis: true },
        advancedValues: {}
      } as any)
    ).toBe("Choose an analysis provider before running ingest analysis.")
    expect(
      getQuickIngestAnalysisProviderWarning({
        common: { perform_analysis: true },
        advancedValues: { api_name: "openai" }
      } as any)
    ).toBeNull()
    expect(
      getQuickIngestAnalysisProviderWarning({
        common: { perform_analysis: true },
        advancedValues: { api_provider: "openai" }
      } as any)
    ).toBe("Choose an analysis provider before running ingest analysis.")
  })

  it("delegates a pending version-2 run request to authoritative run chunks", async () => {
    const onTrackingMetadata = vi.fn()
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-v2-1",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-v2-1",
      items_url: "/api/v1/media/ingest/runs/run-v2-1/items",
      events_url: "/api/v1/media/ingest/runs/run-v2-1/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-v2-1",
          ordinal: 1,
          input_kind: "materialized_playlist_item",
          source_url: "https://www.youtube.com/watch?v=server-authority",
          source_kind: "youtube_video",
          display_metadata: { title: "Server title" },
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null,
        },
      ],
    })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-v2-1",
      jobs: [{ id: 501 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-v2-1",
          status: "accepted",
          accepted: true,
          job_id: 501,
          batch_id: "batch-v2-1",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        },
      ],
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "occ-v2-1",
          url: "https://cached.invalid/never-submit",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false,
      },
      conferenceBatchMetadata: {
        collectionName: "Conference archive",
        sharedTags: ["conference", "research"],
        sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf",
      },
      conferenceItemMetadata: {
        "occ-v2-1": {
          playlist: {
            playlistId: "PL-conf",
            ordinal: 1,
            title: "Server title",
            normalizedSourceId: "youtube:video:server-authority",
          },
          conferenceOverride: {
            speaker: "Grace Hopper",
            tags: [" keynote ", "keynote"],
            selected: true,
          },
        },
      },
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "materialized_playlist_item",
            occurrenceId: "occ-v2-1",
            materializationId: "materialization-v2-1",
          },
        ],
      },
      onTrackingMetadata,
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs",
        method: "POST",
        body: expect.objectContaining({
          processing_options: expect.objectContaining({
            perform_analysis: true,
            overwrite_existing: false,
          }),
          new_collection: {
            name: "Conference archive",
            source_url: "https://youtube.com/playlist?list=PL-conf",
            default_tags: ["conference", "research"],
          },
          playlist_summaries: [
            expect.objectContaining({
              occurrence_id: "occ-v2-1",
              playlist: expect.objectContaining({
                playlist_id: "PL-conf",
                normalized_source_id: "youtube:video:server-authority",
              }),
              conference_override: expect.objectContaining({
                speaker: "Grace Hopper",
                tags: ["keynote"],
                selected: true,
              }),
            }),
          ],
        }),
      }),
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        fields: expect.objectContaining({
          run_id: "run-v2-1",
          occurrence_ids: ["occ-v2-1"],
          attempts: [1],
          urls: ["https://www.youtube.com/watch?v=server-authority"],
        }),
      }),
    )
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "cached.invalid",
    )
    expect(JSON.stringify(mocks.bgRequest.mock.calls)).not.toContain(
      "cached.invalid",
    )
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-v2-1",
        submittedItemIds: ["occ-v2-1"],
      }),
    )
    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/501",
      }),
    )
    expect(result).toMatchObject({
      ok: true,
      accepted: true,
      runId: "run-v2-1",
    })
    expect(result.results).toBeUndefined()
  })

  it("persists submission intent before create and run identity before upload", async () => {
    let resolveCreate!: (value: unknown) => void
    let resolveUpload!: (value: unknown) => void
    const createPromise = new Promise((resolve) => {
      resolveCreate = resolve
    })
    const uploadPromise = new Promise((resolve) => {
      resolveUpload = resolve
    })
    const events: string[] = []
    mocks.bgRequest.mockImplementation(() => {
      events.push("create")
      return createPromise
    })
    mocks.bgUpload.mockImplementation(() => {
      events.push("upload")
      return uploadPromise
    })
    const onTrackingMetadata = vi.fn((tracking: any) => {
      events.push(`track:${tracking.submissionState}`)
    })

    const submitted = submitQuickIngestBatch({
      entries: [
        {
          id: "occ-durable-order",
          url: "https://cached.invalid/never-submit",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-durable-order",
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-durable-order",
            url: "https://client.example/durable-order",
          },
        ],
      },
      onTrackingMetadata,
    } as any)

    await vi.waitFor(() => {
      expect(events.slice(0, 2)).toEqual(["track:creating_run", "create"])
    })
    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        submissionState: "creating_run",
        submissionOccurrenceIds: ["occ-durable-order"],
      })
    )
    expect(onTrackingMetadata.mock.calls[0]?.[0].runId).toBeUndefined()
    expect(onTrackingMetadata.mock.calls[0]?.[0].submittedItemIds).toBeUndefined()
    resolveCreate({
      contract_version: 2,
      run_id: "run-durable-order",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-durable-order",
      items_url: "/api/v1/media/ingest/runs/run-durable-order/items",
      events_url: "/api/v1/media/ingest/runs/run-durable-order/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-durable-order",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: "https://server.example/durable-order",
          source_kind: "video",
          display_metadata: {},
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null,
        },
      ],
    })
    await vi.waitFor(() => {
      expect(events).toContain("upload")
      expect(events.indexOf("track:run_created")).toBeLessThan(
        events.indexOf("upload")
      )
    })
    resolveUpload({
      batch_id: "batch-durable-order",
      jobs: [{ id: 701 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-durable-order",
          status: "accepted",
          accepted: true,
          job_id: 701,
          batch_id: "batch-durable-order",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        },
      ],
    })

    await expect(submitted).resolves.toMatchObject({
      ok: true,
      runId: "run-durable-order",
    })
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        submissionState: "submitting",
        runId: "run-durable-order",
        batchIds: ["batch-durable-order"],
        jobIds: [701],
      })
    )
    expect(events.at(-1)).toBe("track:acknowledged")
  })

  it("tracks a run whose create response has only terminal occurrences", async () => {
    const onTrackingMetadata = vi.fn()
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-v2-terminal",
      status: "completed",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-v2-terminal",
      items_url: "/api/v1/media/ingest/runs/run-v2-terminal/items",
      events_url: "/api/v1/media/ingest/runs/run-v2-terminal/events/stream",
      processing_occurrences: [],
    })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "materialized_playlist_item",
            occurrenceId: "occ-terminal-skip",
            materializationId: "materialization-terminal",
          },
        ],
      },
      onTrackingMetadata,
    } as any)

    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-v2-terminal",
        submittedItemIds: ["occ-terminal-skip"],
        jobIds: [],
      }),
    )
    expect(result).toEqual({
      ok: true,
      accepted: true,
      runId: "run-v2-terminal",
    })
  })

  it("separates mixed run occurrences by effective processing fields", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-v2-mixed",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-v2-mixed",
      items_url: "/api/v1/media/ingest/runs/run-v2-mixed/items",
      events_url: "/api/v1/media/ingest/runs/run-v2-mixed/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-audio",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: "https://server.example/audio.mp3",
          source_kind: "audio",
          display_metadata: {},
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null,
        },
        {
          occurrence_id: "occ-video",
          ordinal: 2,
          input_kind: "direct_url",
          source_url: "https://server.example/video.mp4",
          source_kind: "video",
          display_metadata: {},
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null,
        },
      ],
    })
    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-audio",
        jobs: [{ id: 601 }],
        errors: [],
        submissions: [
          {
            occurrence_id: "occ-audio",
            status: "accepted",
            accepted: true,
            job_id: 601,
            batch_id: "batch-audio",
            error_code: null,
            message: null,
            retryable: false,
            attempt: 1,
          },
        ],
      })
      .mockResolvedValueOnce({
        batch_id: "batch-video",
        jobs: [{ id: 602 }],
        errors: [],
        submissions: [
          {
            occurrence_id: "occ-video",
            status: "accepted",
            accepted: true,
            job_id: 602,
            batch_id: "batch-video",
            error_code: null,
            message: null,
            retryable: false,
            attempt: 1,
          },
        ],
      })

    await submitQuickIngestBatch({
      entries: [
        { id: "occ-audio", url: "cached-audio", type: "audio" },
        { id: "occ-video", url: "cached-video", type: "video" },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-audio",
            url: "https://client.example/ignored-audio.mp3",
          },
          {
            inputKind: "direct_url",
            occurrenceId: "occ-video",
            url: "https://client.example/ignored-video.mp4",
          },
        ],
      },
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledTimes(2)
    expect(mocks.bgUpload.mock.calls.map(([request]) => request.fields)).toEqual([
      expect.objectContaining({
        media_type: "audio",
        urls: ["https://server.example/audio.mp3"],
        occurrence_ids: ["occ-audio"],
      }),
      expect.objectContaining({
        media_type: "video",
        urls: ["https://server.example/video.mp4"],
        occurrence_ids: ["occ-video"],
      }),
    ])
  })

  it("surfaces a run submission rate limit without treating the run as accepted", async () => {
    const occurrences = Array.from({ length: 51 }, (_, index) => ({
      occurrence_id: `occ-rate-${index + 1}`,
      ordinal: index + 1,
      input_kind: "direct_url",
      source_url: `https://server.example/video-${index + 1}.mp4`,
      source_kind: "video",
      display_metadata: {},
      state: "staged",
      outcome: null,
      job_id: null,
      batch_id: null,
      attempt: 1,
      planned_collection_item_id: null,
    }))
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-v2-rate-limited",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-v2-rate-limited",
      items_url: "/api/v1/media/ingest/runs/run-v2-rate-limited/items",
      events_url: "/api/v1/media/ingest/runs/run-v2-rate-limited/events/stream",
      processing_occurrences: occurrences,
    })
    mocks.bgUpload.mockRejectedValue(
      Object.assign(new Error("rate limited"), {
        status: 429,
        retryAfterMs: 3_000,
      }),
    )

    const result = await submitQuickIngestBatch({
      entries: occurrences.map((occurrence) => ({
        id: occurrence.occurrence_id,
        url: "https://cached.invalid/ignored",
        type: "video",
      })),
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: occurrences.map((occurrence) => ({
          inputKind: "direct_url",
          occurrenceId: occurrence.occurrence_id,
          url: "https://client.invalid/ignored",
        })),
      },
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(result).toMatchObject({
      ok: false,
      accepted: false,
      runId: "run-v2-rate-limited",
      retryAfterMs: 3_000,
      unsentOccurrenceIds: expect.arrayContaining([
        "occ-rate-1",
        "occ-rate-51",
      ]),
    })
    expect(result.error).toMatch(/try again in 3 seconds/i)
    expect(result.results).toBeUndefined()
  })

  it("keeps earlier accepted chunks attached while cancelling unsent occurrences", async () => {
    const occurrences = Array.from({ length: 101 }, (_, index) => ({
      occurrence_id: `occ-partial-${index + 1}`,
      ordinal: index + 1,
      input_kind: "direct_url",
      source_url: `https://server.example/video-${index + 1}.mp4`,
      source_kind: "video",
      display_metadata: {},
      state: "staged",
      outcome: null,
      job_id: null,
      batch_id: null,
      attempt: 1,
      planned_collection_item_id: null,
    }))
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-partial",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-v2-partial",
        items_url: "/api/v1/media/ingest/runs/run-v2-partial/items",
        events_url: "/api/v1/media/ingest/runs/run-v2-partial/events/stream",
        processing_occurrences: occurrences,
      })
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-partial",
        status: "running",
        counts: { total: 101, running: 50, cancelled: 51 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-partial-1"],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-partial-1",
        jobs: Array.from({ length: 50 }, (_, index) => ({ id: index + 1 })),
        errors: [],
        submissions: Array.from({ length: 50 }, (_, index) => ({
          occurrence_id: `occ-partial-${index + 1}`,
          status: "accepted",
          accepted: true,
          job_id: index + 1,
          batch_id: "batch-partial-1",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        })),
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("rate limited"), {
          status: 429,
          retryAfterMs: 2_000,
        }),
      )

    const onTrackingMetadata = vi.fn()
    const result = await submitQuickIngestBatch({
      entries: occurrences.map((occurrence) => ({
        id: occurrence.occurrence_id,
        url: "https://cached.invalid/ignored",
        type: "video",
      })),
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: occurrences.map((occurrence) => ({
          inputKind: "direct_url",
          occurrenceId: occurrence.occurrence_id,
          url: "https://client.invalid/ignored",
        })),
      },
      onTrackingMetadata,
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledTimes(2)
    expect(result).toMatchObject({
      ok: false,
      accepted: true,
      submissionBlocked: true,
      runId: "run-v2-partial",
      retryAfterMs: 2_000,
      unsentOccurrenceIds: expect.arrayContaining([
        "occ-partial-51",
        "occ-partial-101",
      ]),
    })
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-v2-partial",
        jobIds: expect.arrayContaining([1, 50]),
      }),
    )
    expect(mocks.bgRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-v2-partial/cancel",
        body: expect.objectContaining({
          occurrence_ids: expect.arrayContaining([
            "occ-partial-51",
            "occ-partial-101",
          ]),
          reason: "submission_stopped",
        }),
      }),
    )
  })

  it("surfaces unsent occurrence cleanup failure after an earlier chunk was accepted", async () => {
    const occurrences = Array.from({ length: 51 }, (_, index) => ({
      occurrence_id: `occ-cleanup-${index + 1}`,
      ordinal: index + 1,
      input_kind: "direct_url",
      source_url: `https://server.example/cleanup-${index + 1}.mp4`,
      source_kind: "video",
      display_metadata: {},
      state: "staged",
      outcome: null,
      job_id: null,
      batch_id: null,
      attempt: 1,
      planned_collection_item_id: null,
    }))
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-cleanup-failure",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-v2-cleanup-failure",
        items_url: "/api/v1/media/ingest/runs/run-v2-cleanup-failure/items",
        events_url: "/api/v1/media/ingest/runs/run-v2-cleanup-failure/events/stream",
        processing_occurrences: occurrences,
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("cleanup unavailable"), { status: 503 }),
      )
    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-cleanup-1",
        jobs: Array.from({ length: 50 }, (_, index) => ({ id: index + 1 })),
        errors: [],
        submissions: Array.from({ length: 50 }, (_, index) => ({
          occurrence_id: `occ-cleanup-${index + 1}`,
          status: "accepted",
          accepted: true,
          job_id: index + 1,
          batch_id: "batch-cleanup-1",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        })),
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("rate limited"), {
          status: 429,
          retryAfterMs: 2_000,
        }),
      )

    const result = await submitQuickIngestBatch({
      entries: occurrences.map((occurrence) => ({
        id: occurrence.occurrence_id,
        url: "https://cached.invalid/ignored",
        type: "video",
      })),
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: occurrences.map((occurrence) => ({
          inputKind: "direct_url",
          occurrenceId: occurrence.occurrence_id,
          url: "https://client.invalid/ignored",
        })),
      },
    } as any)

    expect(result).toMatchObject({
      ok: false,
      accepted: true,
      submissionBlocked: true,
      submissionCleanupFailed: true,
      runId: "run-v2-cleanup-failure",
      unsentOccurrenceIds: ["occ-cleanup-51"],
    })
    expect(result.error).toMatch(/could not cancel the unsent/i)
  })

  it("surfaces cleanup failure even when the first chunk accepted nothing", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-first-cleanup-failure",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-v2-first-cleanup-failure",
        items_url: "/api/v1/media/ingest/runs/run-v2-first-cleanup-failure/items",
        events_url: "/api/v1/media/ingest/runs/run-v2-first-cleanup-failure/events/stream",
        processing_occurrences: [
          {
            occurrence_id: "occ-first-cleanup-failure",
            ordinal: 1,
            input_kind: "direct_url",
            source_url: "https://server.example/first-cleanup-failure",
            source_kind: "video",
            display_metadata: {},
            state: "staged",
            outcome: null,
            job_id: null,
            batch_id: null,
            attempt: 1,
            planned_collection_item_id: null,
          },
        ],
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("cleanup unavailable"), { status: 503 })
      )
    mocks.bgUpload.mockRejectedValue(
      Object.assign(new Error("rate limited"), {
        status: 429,
        retryAfterMs: 2_000,
      })
    )

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "occ-first-cleanup-failure",
          url: "https://cached.invalid/ignored",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-first-cleanup-failure",
            url: "https://client.invalid/ignored",
          },
        ],
      },
    } as any)

    expect(result).toMatchObject({
      ok: false,
      accepted: false,
      submissionBlocked: true,
      submissionCleanupFailed: true,
      runId: "run-v2-first-cleanup-failure",
      unsentOccurrenceIds: ["occ-first-cleanup-failure"],
    })
  })

  it("cancels an omitted file occurrence while preserving an accepted URL", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-omitted-file",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-v2-omitted-file",
        items_url: "/api/v1/media/ingest/runs/run-v2-omitted-file/items",
        events_url: "/api/v1/media/ingest/runs/run-v2-omitted-file/events/stream",
        processing_occurrences: [
          {
            occurrence_id: "occ-url-accepted",
            ordinal: 1,
            input_kind: "direct_url",
            source_url: "https://server.example/accepted",
            source_kind: "video",
            display_metadata: {},
            state: "staged",
            outcome: null,
            job_id: null,
            batch_id: null,
            attempt: 1,
            planned_collection_item_id: null,
          },
          {
            occurrence_id: "occ-file-omitted",
            ordinal: 2,
            input_kind: "file_stub",
            source_url: null,
            source_kind: "file",
            display_metadata: {},
            state: "awaiting_upload",
            outcome: null,
            job_id: null,
            batch_id: null,
            attempt: 1,
            planned_collection_item_id: null,
          },
        ],
      })
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-v2-omitted-file",
        status: "running",
        counts: { total: 2, running: 1, cancelled: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-url-accepted"],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-url-accepted",
      jobs: [{ id: 801 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-url-accepted",
          status: "accepted",
          accepted: true,
          job_id: 801,
          batch_id: "batch-url-accepted",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        },
      ],
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "occ-url-accepted",
          url: "https://cached.invalid/ignored",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-url-accepted",
            url: "https://client.invalid/ignored",
          },
          {
            inputKind: "file_stub",
            occurrenceId: "occ-file-omitted",
            name: "missing.mp4",
            contentType: "video/mp4",
            sizeBytes: 1,
          },
        ],
      },
    } as any)

    expect(result).toMatchObject({
      ok: false,
      accepted: true,
      submissionBlocked: true,
      unsentOccurrenceIds: ["occ-file-omitted"],
    })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-v2-omitted-file/cancel",
        body: {
          occurrence_ids: ["occ-file-omitted"],
          reason: "submission_stopped",
        },
      })
    )
  })

  it("cancels a newly created run when the user cancels before create resolves", async () => {
    let resolveCreate!: (value: unknown) => void
    const createPromise = new Promise((resolve) => {
      resolveCreate = resolve
    })
    mocks.bgRequest
      .mockReturnValueOnce(createPromise)
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-cancel-during-create",
        status: "cancelled",
        counts: { total: 1, cancelled: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-must-not-submit",
      jobs: [{ id: 901 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-cancel-during-create",
          status: "accepted",
          accepted: true,
          job_id: 901,
          batch_id: "batch-must-not-submit",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        },
      ],
    })

    const submitted = submitQuickIngestBatch({
      entries: [
        {
          id: "occ-cancel-during-create",
          url: "https://cached.invalid/ignored",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-cancel-during-create",
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-cancel-during-create",
            url: "https://client.invalid/ignored",
          },
        ],
      },
    } as any)
    await vi.waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(1))

    await cancelQuickIngestSession({
      sessionId: "qi-direct-cancel-during-create",
      reason: "user_cancelled",
    })
    resolveCreate({
      contract_version: 2,
      run_id: "run-cancel-during-create",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-cancel-during-create",
      items_url: "/api/v1/media/ingest/runs/run-cancel-during-create/items",
      events_url: "/api/v1/media/ingest/runs/run-cancel-during-create/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-cancel-during-create",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: "https://server.example/cancel-during-create",
          source_kind: "video",
          display_metadata: {},
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 1,
          planned_collection_item_id: null,
        },
      ],
    })

    await expect(submitted).resolves.toMatchObject({
      ok: false,
      accepted: false,
      submissionBlocked: true,
      runId: "run-cancel-during-create",
    })
    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(mocks.bgRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-cancel-during-create/cancel",
        body: { reason: "user_cancelled" },
      })
    )
  })

  it("stops later chunks and cancels the run when cancellation arrives between chunks", async () => {
    const occurrences = Array.from({ length: 51 }, (_, index) => ({
      occurrence_id: `occ-cancel-chunk-${index + 1}`,
      ordinal: index + 1,
      input_kind: "direct_url",
      source_url: `https://server.example/cancel-chunk-${index + 1}`,
      source_kind: "video",
      display_metadata: {},
      state: "staged",
      outcome: null,
      job_id: null,
      batch_id: null,
      attempt: 1,
      planned_collection_item_id: null,
    }))
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-cancel-between-chunks",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-cancel-between-chunks",
        items_url: "/api/v1/media/ingest/runs/run-cancel-between-chunks/items",
        events_url: "/api/v1/media/ingest/runs/run-cancel-between-chunks/events/stream",
        processing_occurrences: occurrences,
      })
      .mockResolvedValue({
        contract_version: 2,
        run_id: "run-cancel-between-chunks",
        status: "cancelled",
        counts: { total: 51, cancelled: 51 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-cancel-first"],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
    mocks.bgUpload.mockImplementation(async () => {
      await cancelQuickIngestSession({
        sessionId: "qi-direct-cancel-between-chunks",
        reason: "user_cancelled",
        tracking: {
          mode: "webui-direct",
          runId: "run-cancel-between-chunks",
        },
      })
      return {
        batch_id: "batch-cancel-first",
        jobs: Array.from({ length: 50 }, (_, index) => ({ id: index + 1 })),
        errors: [],
        submissions: Array.from({ length: 50 }, (_, index) => ({
          occurrence_id: `occ-cancel-chunk-${index + 1}`,
          status: "accepted",
          accepted: true,
          job_id: index + 1,
          batch_id: "batch-cancel-first",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        })),
      }
    })

    const result = await submitQuickIngestBatch({
      entries: occurrences.map((occurrence) => ({
        id: occurrence.occurrence_id,
        url: "https://cached.invalid/ignored",
        type: "video",
      })),
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-cancel-between-chunks",
      pendingRunRequest: {
        inputs: occurrences.map((occurrence) => ({
          inputKind: "direct_url",
          occurrenceId: occurrence.occurrence_id,
          url: "https://client.invalid/ignored",
        })),
      },
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(result).toMatchObject({
      ok: false,
      accepted: true,
      submissionBlocked: true,
      runId: "run-cancel-between-chunks",
      unsentOccurrenceIds: ["occ-cancel-chunk-51"],
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-cancel-between-chunks/cancel",
        body: { reason: "user_cancelled" },
      })
    )
  })

  it("uses direct upload path when extension runtime id is unavailable", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-1",
      jobs: [{ id: 101 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m1" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/article"]
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/101",
        method: "GET"
      })
    )
    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-1",
      status: "ok"
    })
  })

  it("marks duplicate remote file uploads as skipped with guidance", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-duplicate-file",
      jobs: [{ id: 303 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          media_id: "m-duplicate-file",
          db_message: "Media 'existing.pdf' already exists. Overwrite not enabled."
        }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "file-duplicate-1",
          name: "existing.pdf",
          type: "application/pdf",
          data: [1, 2, 3]
        }
      ],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "file-duplicate-1",
      status: "ok",
      outcome: "skipped",
      fileName: "existing.pdf",
      message: DUPLICATE_SKIP_MESSAGE
    })
  })

  it("surfaces backend ingest job submit errors when no jobs are created", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-upload-error",
      jobs: [],
      errors: ["Validation failed: Claimed filename 'upload' has no extension."]
    })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "file-invalid-pdf",
          name: "upload",
          type: "application/pdf",
          data: [37, 80, 68, 70]
        }
      ],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "file-invalid-pdf",
      status: "error",
      fileName: "upload",
      error: "Validation failed: Claimed filename 'upload' has no extension."
    })
  })

  it("surfaces completed ingest jobs with backend error payloads as failed results", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-completed-error",
      jobs: [{ id: 909 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          status: "Error",
          error: "File preparation/download failed: Port not allowed: 3000"
        }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-completed-error",
          url: "http://127.0.0.1:3000/e2e/quick-ingest-source.html",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-completed-error",
      status: "error",
      error: "File preparation/download failed: Port not allowed: 3000"
    })
  })

  it("marks duplicate direct HTML scrape responses as skipped", async () => {
    mocks.bgRequest.mockResolvedValue({
      status: "duplicate",
      media_ids: [],
      total_articles: 1,
      stored_articles: 0,
      skipped_articles: 1,
      duplicate_articles: 1,
      errors: null
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-duplicate-html",
          url: "http://localhost:8080/e2e/quick-ingest-source.html?repeat=1",
          type: "html"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-duplicate-html"
    })

    expect(result.ok).toBe(true)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST"
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-duplicate-html",
      status: "ok",
      outcome: "skipped",
      message: DUPLICATE_SKIP_MESSAGE
    })
  })

  it("surfaces direct HTML scrape responses with zero stored articles as failed", async () => {
    mocks.bgRequest.mockResolvedValue({
      status: "persist-ok",
      media_ids: [],
      total_articles: 1,
      stored_articles: 0,
      errors: ["Failed to extract: http://localhost:8080/e2e/source.html"]
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-failed-html",
          url: "http://localhost:8080/e2e/source.html",
          type: "html"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-failed-html"
    })

    expect(result.ok).toBe(true)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-failed-html",
      status: "error",
      error: "Failed to extract: http://localhost:8080/e2e/source.html"
    })
  })

  it("defaults perform_chunking to true when common options are omitted", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-default-chunking",
      jobs: [{ id: 202 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-default-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-default-chunking",
          url: "https://example.com/default-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          perform_chunking: true
        })
      })
    )
  })

  it("sends auto chunking fields and suppresses stale manual fields", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-auto-chunking",
      jobs: [{ id: 204 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-auto-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-auto-chunking",
          url: "https://example.com/auto-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "qa_search",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        perform_analysis: false,
        overwrite_existing: true,
        chunk_method: "tokens",
        chunk_size: 1200,
        chunk_overlap: 200,
        use_adaptive_chunking: true,
        hierarchical_chunking: true,
        hierarchical_template: { boundaries: [{ kind: "heading" }] },
        transcription_model: "parakeet-standard"
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          perform_chunking: true,
          perform_analysis: true,
          overwrite_existing: false,
          chunking_mode: "auto",
          auto_chunking_goal: "qa_search",
          auto_chunking_use_llm: true,
          transcription_model: "parakeet-standard"
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("chunk_method")
    expect(fields).not.toHaveProperty("chunk_size")
    expect(fields).not.toHaveProperty("chunk_overlap")
    expect(fields).not.toHaveProperty("use_adaptive_chunking")
    expect(fields).not.toHaveProperty("hierarchical_chunking")
    expect(fields).not.toHaveProperty("hierarchical_template")
    expect(fields).not.toHaveProperty("chunking_template_name")
    expect(fields).not.toHaveProperty("auto_apply_template")
  })

  it("sends manual chunking fields and templates in manual mode", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-manual-chunking",
      jobs: [{ id: 205 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-manual-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-manual-chunking",
          url: "https://example.com/manual-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "manual",
        auto_chunking_goal: "qa_search",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        chunk_method: "tokens",
        chunk_size: 900,
        chunk_overlap: 100
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          perform_chunking: true,
          chunking_mode: "manual",
          chunk_method: "tokens",
          chunk_size: 900,
          chunk_overlap: 100,
          chunking_template_name: "manual-template",
          auto_apply_template: true
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("auto_chunking_goal")
    expect(fields).not.toHaveProperty("auto_chunking_use_llm")
  })

  it("omits auto and manual chunking controls when chunking is disabled", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-disabled-chunking",
      jobs: [{ id: 206 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-disabled-chunking" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-disabled-chunking",
          url: "https://example.com/disabled-chunking",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "navigation_summary",
        auto_chunking_use_llm: true
      },
      advancedValues: {
        chunk_size: 900,
        chunk_overlap: 100
      },
      chunkingTemplateName: "manual-template",
      autoApplyTemplate: true
    } as any)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          perform_chunking: false
        })
      })
    )
    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).not.toHaveProperty("chunking_mode")
    expect(fields).not.toHaveProperty("auto_chunking_goal")
    expect(fields).not.toHaveProperty("auto_chunking_use_llm")
    expect(fields).not.toHaveProperty("chunk_size")
    expect(fields).not.toHaveProperty("chunk_overlap")
    expect(fields).not.toHaveProperty("chunking_template_name")
    expect(fields).not.toHaveProperty("auto_apply_template")
  })

  it("captures direct batch tracking metadata before polling completes", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-1",
      jobs: [{ id: 1234 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-track" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-track-1",
          url: "https://example.com/tracked",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-1",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "webui-direct",
        sessionId: "qi-direct-1",
        batchId: "batch-1",
        batchIds: ["batch-1"],
        jobIds: [1234],
        jobIdToItemId: {
          "1234": "entry-track-1"
        },
        startedAt: expect.any(Number)
      })
    )
  })

  it("emits per-item direct tracking metadata for both url and file submissions", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-url-1",
        jobs: [{ id: 501 }]
      })
      .mockResolvedValueOnce({
        batch_id: "batch-file-1",
        jobs: [{ id: 601 }]
      })
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "m-url-1" }
        }
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "m-file-1" }
        }
      })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-501",
          url: "https://example.com/per-item-url",
          type: "document"
        }
      ],
      files: [
        {
          id: "file-601",
          name: "session-restore.mkv",
          type: "video/x-matroska",
          data: [1, 2, 3]
        }
      ],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-per-item",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        sessionId: "qi-direct-per-item",
        batchId: "batch-url-1",
        batchIds: ["batch-url-1"],
        jobIds: [501],
        jobIdToItemId: {
          "501": "entry-501"
        }
      })
    )
    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        sessionId: "qi-direct-per-item",
        batchId: "batch-file-1",
        batchIds: ["batch-file-1"],
        jobIds: [601],
        jobIdToItemId: {
          "601": "file-601"
        }
      })
    )
  })

  it("forces direct transport for direct-session submits and polls when runtime messaging exists", async () => {
    mocks.runtimeId = "ext-runtime-1"

    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-transport",
      jobs: [{ id: 818 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-direct-transport" }
      }
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-direct-transport",
          url: "https://example.com/direct-transport",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-transport",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:quick-ingest-batch"
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        preferDirect: true
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/818",
        method: "GET",
        returnResponse: true,
        preferDirect: true
      })
    )
  })

  it("tracks only submitted direct items when later queue items fail before job creation", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-first-only",
        jobs: [{ id: 701 }]
      })
      .mockRejectedValueOnce(new Error("submit failed for second item"))
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-first-only" }
      }
    })

    const response = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-first-submitted",
          url: "https://example.com/first-submitted",
          type: "document"
        },
        {
          id: "entry-never-submitted",
          url: "https://example.com/never-submitted",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-partial-submit",
      onTrackingMetadata
    } as any)

    expect(onTrackingMetadata).toHaveBeenCalledTimes(1)
    expect(onTrackingMetadata).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: "qi-direct-partial-submit",
        submittedItemIds: ["entry-first-submitted"],
        jobIdToItemId: {
          "701": "entry-first-submitted"
        }
      })
    )
    expect(response.results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "entry-first-submitted",
          status: "ok"
        }),
        expect.objectContaining({
          id: "entry-never-submitted",
          status: "error"
        })
      ])
    )
  })

  it("cancels direct-session tracked batches through backend cancel endpoint", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-cancel",
      jobs: [{ id: 777 }]
    })

    let statusPollCount = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || "")
      if (path.includes("/api/v1/media/ingest/jobs/cancel?batch_id=batch-direct-cancel")) {
        return { ok: true, data: { success: true } }
      }
      if (path === "/api/v1/media/ingest/jobs/777") {
        statusPollCount += 1
        return { ok: true, data: { status: statusPollCount > 1 ? "cancelled" : "processing" } }
      }
      return { ok: false, error: "unexpected path" }
    })

    const runPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-cancel-1",
          url: "https://example.com/cancel-me",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "direct-session-1",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/ingest/jobs/777",
          method: "GET",
          preferDirect: true
        })
      )
    })

    const cancelResponse = await cancelQuickIngestSession({
      sessionId: "direct-session-1",
      reason: "user_cancelled"
    })
    const runResult = await runPromise

    expect(cancelResponse).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining(
          "/api/v1/media/ingest/jobs/cancel?batch_id=batch-direct-cancel"
        ),
        method: "POST",
        preferDirect: true
      })
    )
    expect(runResult.results?.[0]).toMatchObject({
      id: "entry-cancel-1",
      status: "error"
    })
  })

  it("stops submitting later direct items once the session is cancelled", async () => {
    vi.useFakeTimers()
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-stop-1",
      jobs: [{ id: 901 }]
    })

    let statusPollCount = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || "")
      if (path.includes("/api/v1/media/ingest/jobs/cancel?batch_id=batch-stop-1")) {
        return { ok: true, data: { success: true } }
      }
      if (path === "/api/v1/media/ingest/jobs/901") {
        statusPollCount += 1
        return {
          ok: true,
          data: { status: statusPollCount > 1 ? "cancelled" : "processing" }
        }
      }
      return { ok: false, error: "unexpected path" }
    })

    const runPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-stop-1",
          url: "https://example.com/stop-first",
          type: "document"
        },
        {
          id: "entry-stop-2",
          url: "https://example.com/stop-second",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "direct-session-stop",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/media/ingest/jobs/901",
          method: "GET"
        })
      )
    })

    await cancelQuickIngestSession({
      sessionId: "direct-session-stop",
      reason: "user_cancelled"
    })
    await vi.advanceTimersByTimeAsync(2_000)
    const runResult = await runPromise

    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(runResult.results?.map((item) => item.id)).not.toContain("entry-stop-2")
  })

  it("uses extension message transport when extension runtime is available", async () => {
    mocks.runtimeId = "ext-1"
    mocks.sendMessage
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
      ok: true,
      results: [{ id: "entry-1", status: "ok", type: "document" }]
      })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest-batch",
        payload: expect.objectContaining({
          entries: expect.any(Array)
        })
      })
    )
    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(result.ok).toBe(true)
  })

  it("falls back to direct mode when runtime ping preflight times out", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-fallback",
      jobs: [{ id: 808 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-fallback" }
      }
    })

    const resultPromise = submitQuickIngestBatch({
      entries: [
        {
          id: "entry-fallback",
          url: "https://example.com/runtime-fallback",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.advanceTimersByTimeAsync(401)
    const result = await resultPromise

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        preferDirect: true
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-fallback",
      status: "ok"
    })
  })

  it("routes html process-only entries through process-web-scraping", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html",
          url: "https://example.com/page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {
        custom_headers: '{"x-test":"1"}'
      }
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/page",
          scrape_method: "Individual URLs"
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-html",
      status: "ok",
      type: "html"
    })
  })

  it("routes persisted ordinary web URLs through process-web-scraping", async () => {
    mocks.bgRequest.mockResolvedValue({
      status: "persist-ok",
      media_ids: [123],
      total_articles: 1
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-web-persist",
          url: "https://example.com/article",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/article",
          scrape_method: "Individual URLs",
          mode: "persist"
        })
      })
    )
    expect(
      mocks.bgUpload.mock.calls.some(
        ([request]) => request?.path === "/api/v1/media/ingest/jobs"
      )
    ).toBe(false)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-web-persist",
      status: "ok",
      type: "html",
      mediaId: 123,
      persisted: true
    })
  })

  it.each([true, false])(
    "maps perform_analysis=%s to summarize_checkbox without extraction-order fields",
    async (performAnalysis) => {
      mocks.bgRequest.mockResolvedValue({
        status: "persist-ok",
        media_ids: [123],
        total_articles: 1
      })

      await submitQuickIngestBatch({
        entries: [
          {
            id: `entry-analysis-${performAnalysis}`,
            url: "https://example.com/article",
            type: "html"
          }
        ],
        files: [],
        storeRemote: true,
        processOnly: false,
        common: {
          perform_analysis: performAnalysis,
          perform_chunking: true,
          overwrite_existing: false
        },
        advancedValues: {}
      })

      const scrapeCall = mocks.bgRequest.mock.calls.find(
        ([request]) => request?.path === "/api/v1/media/process-web-scraping"
      )
      const body = scrapeCall?.[0]?.body

      expect(body).toMatchObject({ summarize_checkbox: performAnalysis })
      expect(body).not.toHaveProperty("strategy_order")
      expect(body).not.toHaveProperty("extraction_strategy_order")
    }
  )

  it("keeps direct Markdown URLs on the document ingest job route", async () => {
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-markdown-url",
      jobs: [{ id: 778 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-markdown-url" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-markdown-url",
          url: "https://example.com/source.md",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: false,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/source.md"]
        })
      })
    )
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) => request?.path === "/api/v1/media/process-web-scraping"
      )
    ).toBe(false)
    expect(result.results?.[0]).toMatchObject({
      id: "entry-markdown-url",
      status: "ok",
      type: "document"
    })
  })

  it("passes auto chunking fields to process-web-scraping JSON requests", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html-auto",
          url: "https://example.com/page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "auto",
        auto_chunking_goal: "navigation_summary",
        auto_chunking_use_llm: false
      },
      advancedValues: {
        custom_headers: '{"x-test":"1"}',
        chunk_size: 1200,
        chunk_overlap: 200
      }
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-web-scraping",
        method: "POST",
        body: expect.objectContaining({
          url_input: "https://example.com/page",
          scrape_method: "Individual URLs",
          chunking_mode: "auto",
          auto_chunking_goal: "navigation_summary",
          custom_headers: { "x-test": "1" }
        })
      })
    )
    const body = mocks.bgRequest.mock.calls[0][0].body
    expect(body).not.toHaveProperty("auto_chunking_use_llm")
    expect(body).not.toHaveProperty("chunk_size")
    expect(body).not.toHaveProperty("chunk_overlap")
  })

  it("passes manual chunking templates to process-web-scraping JSON requests", async () => {
    mocks.bgRequest.mockResolvedValue({ content: "processed" })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-html-manual",
          url: "https://example.com/manual-page",
          type: "html"
        }
      ],
      files: [],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false,
        chunking_mode: "manual",
        auto_chunking_goal: "balanced",
        auto_chunking_use_llm: false
      },
      advancedValues: {
        chunk_method: "sentences",
        chunk_size: 900
      },
      chunkingTemplateName: "article-template",
      autoApplyTemplate: true
    } as any)

    const body = mocks.bgRequest.mock.calls[0][0].body
    expect(body).toMatchObject({
      url_input: "https://example.com/manual-page",
      chunking_mode: "manual",
      chunk_method: "sentences",
      chunk_size: 900,
      chunking_template_name: "article-template",
      auto_apply_template: true
    })
    expect(body).not.toHaveProperty("auto_chunking_goal")
  })

  it("routes local files through direct process endpoints in web runtime", async () => {
    mocks.bgUpload.mockResolvedValue({ result: "ok" })

    const result = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "file-1",
          name: "notes.txt",
          type: "text/plain",
          data: [1, 2, 3]
        }
      ],
      storeRemote: false,
      processOnly: true,
      common: {
        perform_analysis: true,
        perform_chunking: true,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/process-documents",
        method: "POST",
        file: expect.objectContaining({
          name: "notes.txt"
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "file-1",
      status: "ok"
    })
  })

  it("falls back to persistent /media/add when ingest-job submission is rejected by the concurrent-job limit", async () => {
    const queueLimitError = new Error(
      "User 1 has reached the maximum concurrent job limit (5)"
    ) as Error & { status?: number }
    queueLimitError.status = 429

    mocks.bgUpload
      .mockRejectedValueOnce(queueLimitError)
      .mockResolvedValueOnce({
        results: [
          {
            status: "Success",
            db_id: 321,
            metadata: { title: "Queued article fallback" }
          }
        ]
      })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-queue-limit",
          url: "https://example.com/article.md",
          type: "auto"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST"
      })
    )
    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/add",
        method: "POST",
        fields: expect.objectContaining({
          urls: ["https://example.com/article.md"]
        })
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-queue-limit",
      status: "ok",
      data: {
        results: [
          expect.objectContaining({
            media_id: 321
          })
        ]
      }
    })
  })

  it("creates planned conference collection items before direct job submission", async () => {
    const onTrackingMetadata = vi.fn()

    mocks.bgUpload
      .mockResolvedValueOnce({
        batch_id: "batch-talk-1",
        jobs: [{ id: 501 }]
      })
      .mockResolvedValueOnce({
        batch_id: "batch-talk-2",
        jobs: [{ id: 502 }]
      })
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 7,
          name: "Strange Loop 2012",
          kind: "conference",
          source_url: "https://youtube.com/playlist?list=PL-conf",
          metadata: request.body?.metadata || {},
          default_tags: request.body?.default_tags || [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/7/items") {
        const ordinal = Number(request.body?.ordinal || 0)
        return {
          id: ordinal === 1 ? 11 : 12,
          collection_id: 7,
          ordinal,
          source_url: request.body?.source_url,
          normalized_source_id: request.body?.normalized_source_id,
          source_kind: request.body?.source_kind,
          title: request.body?.title,
          speaker: request.body?.speaker,
          duplicate_status: request.body?.duplicate_status || "new",
          status: "planned",
          retry_count: 0,
          idempotency_key: `conference-7-${ordinal}`,
          warnings: [],
          metadata: request.body?.metadata || {},
          tags: request.body?.tags || [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/ingest/jobs/501" || path === "/api/v1/media/ingest/jobs/502") {
        return {
          ok: true,
          data: {
            status: "completed",
            result: { media_id: path.endsWith("501") ? 901 : 902 }
          }
        }
      }
      if (path === "/api/v1/media/collections/7/items/11" || path === "/api/v1/media/collections/7/items/12") {
        return {
          id: path.endsWith("/11") ? 11 : 12,
          collection_id: 7,
          ordinal: path.endsWith("/11") ? 1 : 2,
          source_url: request.body?.source_url || "https://youtube.com/watch?v=talk",
          duplicate_status: "new",
          status: request.body?.status || "processing",
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "talk-1",
          url: "https://youtube.com/watch?v=talk-1",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Strange Loop 2012",
            ordinal: 1,
            normalizedSourceId: "youtube:video:talk-1",
            duplicateStatus: "new"
          },
          conferenceOverride: {
            selected: true,
            title: "Simplicity Matters",
            speaker: "Rich Hickey",
            tags: ["keynote"]
          }
        },
        {
          id: "talk-2",
          url: "https://youtube.com/watch?v=talk-2",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Strange Loop 2012",
            ordinal: 2,
            normalizedSourceId: "youtube:video:talk-2",
            duplicateStatus: "new"
          },
          conferenceOverride: {
            selected: true,
            speaker: "Alex Miller"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Strange Loop 2012",
        conferenceName: "Strange Loop",
        eventYear: "2012",
        sharedTags: ["conference", "clojure"],
        sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf"
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-conference-run",
      onTrackingMetadata
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections",
        method: "POST",
        body: expect.objectContaining({
          name: "Strange Loop 2012",
          kind: "conference",
          source_url: "https://youtube.com/playlist?list=PL-conf",
          metadata: expect.objectContaining({
            conference_name: "Strange Loop",
            event_year: "2012",
            source_playlist_url: "https://youtube.com/playlist?list=PL-conf"
          }),
          default_tags: ["conference", "clojure"]
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/7/items",
        method: "POST",
        body: expect.objectContaining({
          ordinal: 1,
          source_url: "https://youtube.com/watch?v=talk-1",
          normalized_source_id: "youtube:video:talk-1",
          title: "Simplicity Matters",
          speaker: "Rich Hickey",
          tags: ["conference", "clojure", "keynote"]
        })
      })
    )
    expect(mocks.bgUpload).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        fields: expect.objectContaining({
          media_collection_id: 7,
          media_collection_item_id: 11,
          idempotency_key: "conference-7-1"
        })
      })
    )
    expect(onTrackingMetadata).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        mode: "webui-direct",
        sessionId: "qi-direct-conference-run",
        batchId: "batch-talk-1",
        collectionId: "7",
        plannedItemIds: ["11"],
        jobIdToCollectionItemId: {
          "501": "11"
        },
        durableMode: "durable_collection"
      })
    )
  })

  it.each([
    {
      label: "duplicate",
      terminalData: {
        status: "duplicate",
        media_ids: [],
        stored_articles: 0,
        duplicate_articles: 1,
        errors: null
      },
      expectedStatus: "skipped_existing"
    },
    {
      label: "result error",
      terminalData: {
        status: "persist-ok",
        media_ids: [],
        stored_articles: 0,
        errors: ["Storage failed for article"]
      },
      expectedStatus: "failed"
    },
    {
      label: "duplicate with result error",
      terminalData: {
        status: "duplicate",
        media_ids: [],
        stored_articles: 0,
        duplicate_articles: 1,
        errors: ["Storage failed for article"]
      },
      expectedStatus: "skipped_existing"
    },
    {
      label: "ordinary success",
      terminalData: {
        status: "persist-ok",
        media_ids: [901],
        stored_articles: 1,
        errors: null
      },
      expectedStatus: "completed"
    }
  ])("patches a conference item to $expectedStatus for $label", async ({
    terminalData,
    expectedStatus
  }) => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 13,
          name: "Conference Batch",
          kind: "conference",
          metadata: {},
          default_tags: [],
          items: []
        }
      }
      if (path === "/api/v1/media/collections/13/items") {
        return {
          id: 131,
          collection_id: 13,
          ordinal: 1,
          source_url: request.body?.source_url,
          duplicate_status: "new",
          status: "planned",
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: []
        }
      }
      if (path === "/api/v1/media/process-web-scraping") {
        return terminalData
      }
      if (path === "/api/v1/media/collections/13/items/131") {
        return {
          id: 131,
          collection_id: 13,
          source_url: "https://example.com/talk",
          status: request.body?.status
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    await submitQuickIngestBatch({
      entries: [
        {
          id: "conference-html",
          url: "https://example.com/talk",
          type: "html",
          conferenceOverride: { selected: true, title: "Conference Talk" }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Conference Batch",
        sharedTags: []
      },
      common: {
        perform_analysis: false,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    } as any)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/13/items/131",
        method: "PATCH",
        body: expect.objectContaining({ status: expectedStatus })
      })
    )
    const terminalPatchStatuses = mocks.bgRequest.mock.calls
      .filter(
        ([request]) =>
          request?.path === "/api/v1/media/collections/13/items/131" &&
          request?.method === "PATCH"
      )
      .map(([request]) => request.body?.status)
    expect(terminalPatchStatuses).toEqual([expectedStatus])
  })

  it("skips direct ingest submission for existing conference items when policy includes existing", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 9,
          name: "Conference Batch",
          kind: "conference",
          metadata: {},
          default_tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/9/items") {
        return {
          id: 91,
          collection_id: 9,
          ordinal: 3,
          source_url: request.body?.source_url,
          duplicate_status: request.body?.duplicate_status,
          status: request.body?.status,
          retry_count: 0,
          idempotency_key: "conference-9-3",
          warnings: [],
          metadata: request.body?.metadata || {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/collections/9/items/91") {
        return {
          id: 91,
          collection_id: 9,
          ordinal: 3,
          source_url: "https://youtube.com/watch?v=existing",
          duplicate_status: "duplicate_existing",
          status: request.body?.status,
          retry_count: request.body?.retry_count || 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "existing-talk",
          url: "https://youtube.com/watch?v=existing",
          type: "video",
          playlist: {
            playlistId: "PL-conf",
            playlistTitle: "Conference Batch",
            ordinal: 3,
            normalizedSourceId: "youtube:video:existing",
            duplicateStatus: "duplicate_existing"
          },
          conferenceOverride: {
            selected: true,
            duplicatePolicy: "include_existing",
            title: "Existing Talk"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Conference Batch",
        sharedTags: []
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {},
      __quickIngestSessionId: "qi-direct-duplicate-policy"
    } as any)

    expect(mocks.bgUpload).not.toHaveBeenCalled()
    expect(result.results?.[0]).toMatchObject({
      id: "existing-talk",
      status: "ok",
      outcome: "skipped",
      collectionItemId: 91,
      idempotencyKey: "conference-9-3"
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/9/items",
        method: "POST",
        body: expect.objectContaining({
          status: "skipped_existing",
          metadata: expect.objectContaining({
            duplicate_policy: "include_existing"
          })
        })
      })
    )
  })

  it("marks planned conference items as submit_failed when direct job submission fails", async () => {
    mocks.bgUpload.mockRejectedValueOnce(new Error("job submit failed"))
    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || "")
      if (path === "/api/v1/media/collections") {
        return {
          id: 8,
          name: "Conference Batch",
          kind: "conference",
          metadata: {},
          default_tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z",
          items: []
        }
      }
      if (path === "/api/v1/media/collections/8/items") {
        return {
          id: 81,
          collection_id: 8,
          ordinal: 1,
          source_url: request.body?.source_url,
          duplicate_status: "new",
          status: "planned",
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      if (path === "/api/v1/media/collections/8/items/81") {
        return {
          id: 81,
          collection_id: 8,
          ordinal: 1,
          source_url: "https://youtube.com/watch?v=failed",
          duplicate_status: "new",
          status: request.body?.status,
          error_summary: request.body?.error_summary,
          retry_count: 0,
          warnings: [],
          metadata: {},
          tags: [],
          created_at: "2026-05-01T00:00:00Z",
          updated_at: "2026-05-01T00:00:00Z"
        }
      }
      throw new Error(`Unexpected bgRequest path: ${path}`)
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "failed-talk",
          url: "https://youtube.com/watch?v=failed",
          type: "video",
          conferenceOverride: {
            selected: true,
            title: "Failed Talk"
          }
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      conferenceBatchMetadata: {
        collectionName: "Conference Batch",
        sharedTags: []
      },
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    } as any)

    expect(result.results?.[0]).toMatchObject({
      id: "failed-talk",
      status: "error",
      error: "job submit failed"
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/collections/8/items/81",
        method: "PATCH",
        body: expect.objectContaining({
          status: "submit_failed",
          error_summary: "job submit failed"
        })
      })
    )
  })

  it("returns a direct session ack for mv3 extension pages", async () => {
    mocks.runtimeId = "ext-1"

    const ack = await startQuickIngestSession({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(ack.ok).toBe(true)
    expect(ack.sessionId).toMatch(/^qi-direct-/)
  })

  it("returns a direct session ack when runtime ping preflight times out", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))

    const ackPromise = startQuickIngestSession({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    await vi.advanceTimersByTimeAsync(401)
    const ack = await ackPromise

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(ack.ok).toBe(true)
    expect(ack.sessionId).toMatch(/^qi-direct-/)
  })

  it("bypasses background batch orchestration for explicit direct sessions", async () => {
    mocks.runtimeId = "ext-1"
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-explicit",
      jobs: [{ id: 818 }]
    })
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: { media_id: "m-direct-explicit" }
      }
    })

    const result = await submitQuickIngestBatch({
      entries: [
        {
          id: "entry-direct-explicit",
          url: "https://example.com/direct-explicit",
          type: "document"
        }
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-explicit",
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST"
      })
    )
    expect(result.results?.[0]).toMatchObject({
      id: "entry-direct-explicit",
      status: "ok"
    })
  })

  it("cancels direct sessions without routing through background session runtime", async () => {
    mocks.runtimeId = "ext-1"

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-active",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        batchIds: ["batch-direct-cancel"],
        batchId: "batch-direct-cancel"
      }
    } as any)

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-direct-cancel"),
        method: "POST",
        preferDirect: true
      })
    )
    expect(response).toEqual({ ok: true })
  })

  it("cancels a tracked version-2 run before considering legacy batches", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-cancel-1",
      status: "cancelled",
      counts: { total: 1, cancelled: 1 },
      version: 2,
      collection_id: null,
      batch_ids: ["batch-should-not-run"],
      created_at: "2026-07-13T00:00:00Z",
      updated_at: "2026-07-13T00:00:01Z",
      expires_at: "2026-07-20T00:00:00Z",
    })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-run-cancel",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        runId: "run-cancel-1",
        batchIds: ["batch-should-not-run"],
      },
    } as any)

    expect(response).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-cancel-1/cancel",
        body: { reason: "user_cancelled" },
      }),
    )
  })

  it("reports a live run cancellation failure instead of claiming success", async () => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("service unavailable"), { status: 503 }),
    )

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-run-failure",
      tracking: {
        mode: "webui-direct",
        runId: "run-cancel-failure",
        batchIds: ["batch-must-not-mask-failure"],
      },
    } as any)

    expect(response).toMatchObject({ ok: false })
    expect(response.error).toMatch(/could not (?:connect|be reached)|unavailable/i)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("falls back to tracked batches only when run cancellation is unsupported", async () => {
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("not found"), { status: 404 }),
      )
      .mockResolvedValueOnce({ ok: true })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-run-legacy",
      tracking: {
        mode: "webui-direct",
        runId: "run-cancel-legacy",
        batchIds: ["batch-legacy-1"],
      },
    } as any)

    expect(response).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-legacy-1"),
      }),
    )
  })

  it("sends explicit cancel message with session id", async () => {
    mocks.runtimeId = "ext-1"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({
      ok: true
    })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-session-123",
      reason: "user_cancelled"
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        type: "tldw:ping"
      })
    )
    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest/cancel",
        payload: {
          sessionId: "qi-session-123",
          reason: "user_cancelled"
        }
      })
    )
    expect(response).toEqual({ ok: true })
  })

  it("cancels persisted direct batches after refresh using tracking metadata", async () => {
    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-restored",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        batchIds: ["batch-restore-1", "batch-restore-2"],
        batchId: "batch-restore-2"
      }
    } as any)

    expect(response).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-restore-1"),
        method: "POST",
        preferDirect: true
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("batch_id=batch-restore-2"),
        method: "POST",
        preferDirect: true
      })
    )
  })
})
