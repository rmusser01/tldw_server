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
import { createQuickIngestSessionRuntime } from "@/entries/shared/quick-ingest-session-runtime"

const terminalRunResponse = (runId: string) => ({
  contract_version: 2,
  run_id: runId,
  status: "completed",
  version: 1,
  status_url: `/api/v1/media/ingest/runs/${runId}`,
  items_url: `/api/v1/media/ingest/runs/${runId}/items`,
  events_url: `/api/v1/media/ingest/runs/${runId}/events/stream`,
  processing_occurrences: [],
})

const version2RunRequest = (
  sessionId: string,
  occurrenceId: string,
): Parameters<typeof submitQuickIngestBatch>[0] => ({
  entries: [],
  files: [],
  storeRemote: true,
  processOnly: false,
  __quickIngestSessionId: sessionId,
  pendingRunRequest: {
    inputs: [
      {
        inputKind: "direct_url",
        occurrenceId,
        url: `https://example.com/${occurrenceId}`,
      },
    ],
  },
})

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
      __quickIngestSessionId: "qi-direct-v2-authority",
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

  it("reuses the durable session identity after an ambiguous run-create failure", async () => {
    const sessionId = "qi-direct-ambiguous-create"
    const request = version2RunRequest(sessionId, "occ-ambiguous-create")
    mocks.bgRequest
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(terminalRunResponse("run-ambiguous-create"))

    await expect(submitQuickIngestBatch(request)).rejects.toMatchObject({
      message: expect.any(String),
    })
    await expect(submitQuickIngestBatch(request)).resolves.toMatchObject({
      ok: true,
      runId: "run-ambiguous-create",
    })

    const createBodies = mocks.bgRequest.mock.calls
      .map(([call]) => call)
      .filter((call) => call.path === "/api/v1/media/ingest/runs")
      .map((call) => call.body)
    expect(createBodies).toHaveLength(2)
    expect(createBodies).toEqual([
      expect.objectContaining({ client_request_id: sessionId }),
      expect.objectContaining({ client_request_id: sessionId }),
    ])
  })

  it.each(["", "   "])(
    "rejects a missing durable session identity before run creation (%j)",
    async (__quickIngestSessionId) => {
      const onTrackingMetadata = vi.fn()
      const shouldStop = vi.fn()
      const request = {
        ...version2RunRequest(__quickIngestSessionId, "occ-missing-session"),
        __quickIngestShouldStop: shouldStop,
        onTrackingMetadata,
      }

      await expect(submitQuickIngestBatch(request)).resolves.toEqual({
        ok: false,
        error: "Missing quick ingest session id.",
      })
      expect(mocks.bgRequest).not.toHaveBeenCalled()
      expect(mocks.bgUpload).not.toHaveBeenCalled()
      expect(shouldStop).not.toHaveBeenCalled()
      expect(onTrackingMetadata).not.toHaveBeenCalled()
    },
  )

  it("keeps distinct durable sessions distinct at run creation", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(terminalRunResponse("run-session-a"))
      .mockResolvedValueOnce(terminalRunResponse("run-session-b"))

    await submitQuickIngestBatch(
      version2RunRequest("qi-direct-session-a", "occ-session-a"),
    )
    await submitQuickIngestBatch(
      version2RunRequest("qi-direct-session-b", "occ-session-b"),
    )

    const clientRequestIds = mocks.bgRequest.mock.calls
      .map(([call]) => call)
      .filter((call) => call.path === "/api/v1/media/ingest/runs")
      .map((call) => call.body.client_request_id)
    expect(clientRequestIds).toEqual([
      "qi-direct-session-a",
      "qi-direct-session-b",
    ])
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
      __quickIngestSessionId: "qi-direct-v2-terminal",
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
      __quickIngestSessionId: "qi-direct-v2-mixed",
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
      __quickIngestSessionId: "qi-direct-v2-rate-limited",
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
      __quickIngestSessionId: "qi-direct-v2-partial",
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
      __quickIngestSessionId: "qi-direct-v2-cleanup-failure",
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
      __quickIngestSessionId: "qi-direct-v2-first-cleanup-failure",
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
      __quickIngestSessionId: "qi-direct-v2-omitted-file",
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

  it("records a direct occurrence cancellation during create and cancels the created server row before upload", async () => {
    let resolveCreate!: (value: unknown) => void
    const createPromise = new Promise((resolve) => {
      resolveCreate = resolve
    })
    mocks.bgRequest
      .mockReturnValueOnce(createPromise)
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-row-cancel-during-create",
        status: "running",
        counts: { total: 2, staged: 1, cancellation_requested: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-row-cancel-during-create",
      jobs: [{ id: 902 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-row-continue-during-create",
          status: "accepted",
          accepted: true,
          job_id: 902,
          batch_id: "batch-row-cancel-during-create",
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
          id: "occ-row-cancel-during-create",
          url: "https://cached.invalid/cancel",
          type: "video",
        },
        {
          id: "occ-row-continue-during-create",
          url: "https://cached.invalid/continue",
          type: "video",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
      __quickIngestSessionId: "qi-direct-row-cancel-during-create",
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-row-cancel-during-create",
            url: "https://client.invalid/cancel",
          },
          {
            inputKind: "direct_url",
            occurrenceId: "occ-row-continue-during-create",
            url: "https://client.invalid/continue",
          },
        ],
      },
    } as any)
    await vi.waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(1))

    await expect(
      cancelQuickIngestSession({
        sessionId: "qi-direct-row-cancel-during-create",
        reason: "user_cancelled",
        occurrenceIds: ["occ-row-cancel-during-create"],
      })
    ).resolves.toEqual({ ok: true })

    resolveCreate({
      contract_version: 2,
      run_id: "run-row-cancel-during-create",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-row-cancel-during-create",
      items_url: "/api/v1/media/ingest/runs/run-row-cancel-during-create/items",
      events_url: "/api/v1/media/ingest/runs/run-row-cancel-during-create/events/stream",
      processing_occurrences: [
        {
          occurrence_id: "occ-row-cancel-during-create",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: "https://server.example/cancel",
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
          occurrence_id: "occ-row-continue-during-create",
          ordinal: 2,
          input_kind: "direct_url",
          source_url: "https://server.example/continue",
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
      accepted: true,
      runId: "run-row-cancel-during-create",
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-row-cancel-during-create/cancel",
        body: {
          reason: "user_cancelled",
          occurrence_ids: ["occ-row-cancel-during-create"],
        },
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          occurrence_ids: ["occ-row-continue-during-create"],
        }),
      })
    )
  })

  it("does not create a direct run after its reserved session was already cancelled", async () => {
    const ack = await startQuickIngestSession({
      entries: [],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-cancel-before-create",
            url: "https://example.com/cancel-before-create",
          },
        ],
      },
      storeRemote: true,
      processOnly: false,
    } as any)
    expect(ack).toMatchObject({
      ok: true,
      sessionId: expect.stringMatching(/^qi-direct-/),
    })

    await expect(
      cancelQuickIngestSession({
        sessionId: ack.sessionId,
        reason: "user_cancelled",
      })
    ).resolves.toEqual({ ok: true })

    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-must-not-be-created",
      status: "staged",
      version: 1,
      status_url: "/api/v1/media/ingest/runs/run-must-not-be-created",
      items_url: "/api/v1/media/ingest/runs/run-must-not-be-created/items",
      events_url: "/api/v1/media/ingest/runs/run-must-not-be-created/events/stream",
      processing_occurrences: [],
    })

    await expect(
      submitQuickIngestBatch({
        entries: [],
        files: [],
        pendingRunRequest: {
          inputs: [
            {
              inputKind: "direct_url",
              occurrenceId: "occ-cancel-before-create",
              url: "https://example.com/cancel-before-create",
            },
          ],
        },
        storeRemote: true,
        processOnly: false,
        __quickIngestSessionId: ack.sessionId,
      } as any)
    ).resolves.toMatchObject({
      ok: false,
      accepted: false,
      submissionBlocked: true,
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.bgUpload).not.toHaveBeenCalled()
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

  it("rechecks occurrence cancellation before each chunk and omits a cancelled later row", async () => {
    const occurrences = Array.from({ length: 51 }, (_, index) => ({
      occurrence_id: `occ-row-cancel-chunk-${index + 1}`,
      ordinal: index + 1,
      input_kind: "direct_url",
      source_url: `https://server.example/row-cancel-chunk-${index + 1}`,
      source_kind: "video",
      display_metadata: {},
      state: "staged",
      outcome: null,
      job_id: null,
      batch_id: null,
      attempt: 1,
      planned_collection_item_id: null,
    }))
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (String(request.path).endsWith("/cancel")) {
        return {
          contract_version: 2,
          run_id: "run-row-cancel-between-chunks",
          status: "running",
          counts: { total: 51, queued: 50, cancellation_requested: 1 },
          version: 2,
          collection_id: null,
          batch_ids: ["batch-row-cancel-first"],
          created_at: "2026-07-13T00:00:00Z",
          updated_at: "2026-07-13T00:00:01Z",
          expires_at: "2026-07-20T00:00:00Z",
        }
      }
      return {
        contract_version: 2,
        run_id: "run-row-cancel-between-chunks",
        status: "staged",
        version: 1,
        status_url: "/api/v1/media/ingest/runs/run-row-cancel-between-chunks",
        items_url: "/api/v1/media/ingest/runs/run-row-cancel-between-chunks/items",
        events_url: "/api/v1/media/ingest/runs/run-row-cancel-between-chunks/events/stream",
        processing_occurrences: occurrences,
      }
    })
    let uploadCount = 0
    mocks.bgUpload.mockImplementation(async () => {
      uploadCount += 1
      if (uploadCount === 1) {
        const rowCancel = await cancelQuickIngestSession({
          sessionId: "qi-direct-row-cancel-between-chunks",
          reason: "user_cancelled",
          occurrenceIds: ["occ-row-cancel-chunk-51"],
          tracking: {
            mode: "webui-direct",
            runId: "run-row-cancel-between-chunks",
          },
        } as any)
        expect(rowCancel).toEqual({ ok: true })
      }
      const start = uploadCount === 1 ? 0 : 50
      const count = uploadCount === 1 ? 50 : 1
      return {
        batch_id:
          uploadCount === 1
            ? "batch-row-cancel-first"
            : "batch-row-cancel-second",
        jobs: Array.from({ length: count }, (_, index) => ({
          id: start + index + 1,
        })),
        errors: [],
        submissions: Array.from({ length: count }, (_, index) => ({
          occurrence_id: `occ-row-cancel-chunk-${start + index + 1}`,
          status: "accepted",
          accepted: true,
          job_id: start + index + 1,
          batch_id:
            uploadCount === 1
              ? "batch-row-cancel-first"
              : "batch-row-cancel-second",
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
      __quickIngestSessionId: "qi-direct-row-cancel-between-chunks",
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
      ok: true,
      accepted: true,
      runId: "run-row-cancel-between-chunks",
    })
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "occ-row-cancel-chunk-51"
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-row-cancel-between-chunks/cancel",
        body: {
          reason: "user_cancelled",
          occurrence_ids: ["occ-row-cancel-chunk-51"],
        },
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

  it("keeps fallback identities unique without Web Crypto at a fixed time", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_800_000_000_000)
    vi.stubGlobal("crypto", undefined)
    const input: Parameters<typeof startQuickIngestSession>[0] = {
      entries: [],
      files: [],
      storeRemote: true,
      processOnly: false,
    }

    try {
      const firstDirect = await startQuickIngestSession(input)
      const secondDirect = await startQuickIngestSession(input)

      expect(firstDirect.sessionId).toMatch(/^qi-direct-/)
      expect(secondDirect.sessionId).not.toBe(firstDirect.sessionId)

      __resetQuickIngestRuntimeHealthForTests()
      mocks.runtimeId = "ext-fallback-identities"
      mocks.sendMessage.mockImplementation(async (message: {
        type?: string
        sessionId?: string
      }) => {
        if (message.type === "tldw:ping") return { ok: true, pong: true }
        if (message.type === "tldw:quick-ingest/start") {
          return { ok: true, sessionId: message.sessionId }
        }
        return { ok: false }
      })
      const firstExtension = await startQuickIngestSession(input)
      const secondExtension = await startQuickIngestSession(input)
      const starts = mocks.sendMessage.mock.calls
        .map(([message]) => message)
        .filter((message) => message.type === "tldw:quick-ingest/start")

      expect(secondExtension.sessionId).not.toBe(firstExtension.sessionId)
      expect(starts[1].attemptToken).not.toBe(starts[0].attemptToken)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("routes mv3 extension sessions through the durable background runtime", async () => {
    mocks.runtimeId = "ext-1"
    mocks.sendMessage.mockImplementation(async (message: any) => {
      if (message.type === "tldw:ping") return { ok: true, pong: true }
      if (message.type === "tldw:quick-ingest/start") {
        return { ok: true, sessionId: message.sessionId }
      }
      return { ok: false }
    })

    const ack = await startQuickIngestSession({
      entries: [
        {
          id: "entry-1",
          url: "https://example.com/article",
          type: "document"
        }
      ],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "entry-1",
            url: "https://example.com/article"
          }
        ]
      },
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(1, { type: "tldw:ping" })
    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest/start",
        sessionId: expect.stringMatching(/^qi-/),
        attemptToken: expect.stringMatching(/^qia-/),
        payload: expect.objectContaining({
          entries: [expect.objectContaining({ id: "entry-1" })],
          pendingRunRequest: expect.objectContaining({
            inputs: [expect.objectContaining({ occurrenceId: "entry-1" })]
          })
        })
      })
    )
    expect(ack).toEqual({
      ok: true,
      sessionId: mocks.sendMessage.mock.calls[1]?.[0]?.sessionId,
    })
  })

  it("recovers an accepted start-message timeout by querying the same identity without direct fallback", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    let releaseRun!: () => void
    const runGate = new Promise<void>((resolve) => {
      releaseRun = resolve
    })
    const run = vi.fn(async (_payload: any, context: any) => {
      await context.setRunTracking({
        mode: "extension-runtime",
        runId: "run-timeout-accepted",
        submissionState: "run_created",
        submissionOccurrenceIds: ["occ-timeout-accepted"],
      })
      await runGate
      return { results: [] }
    })
    const worker = createQuickIngestSessionRuntime({
      run,
      emit: vi.fn(),
      saveRunSession: vi.fn(),
    } as any)
    mocks.sendMessage.mockImplementation((message: any) => {
      if (message.type === "tldw:ping") {
        return Promise.resolve({ ok: true, pong: true })
      }
      if (message.type === "tldw:quick-ingest/start") {
        void worker.start(message.payload, {
          sessionId: message.sessionId,
          attemptToken: message.attemptToken,
        } as any)
        return new Promise(() => undefined)
      }
      if (message.type === "tldw:quick-ingest/replay") {
        return worker.replay(message.payload.sessionId)
      }
      return Promise.resolve({ ok: false })
    })

    const ackPromise = startQuickIngestSession({
      entries: [
        {
          id: "occ-timeout-accepted",
          url: "https://example.com/timeout-accepted",
          type: "video",
        },
      ],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-timeout-accepted",
            url: "https://example.com/timeout-accepted",
          },
        ],
      },
      storeRemote: true,
      processOnly: false,
    } as any)

    await vi.advanceTimersByTimeAsync(10_001)
    const ack = await ackPromise

    expect(ack).toMatchObject({
      ok: true,
      sessionId: expect.stringMatching(/^qi-(?!direct-)/),
    })
    const startMessage = mocks.sendMessage.mock.calls.find(
      ([message]) => message.type === "tldw:quick-ingest/start"
    )?.[0]
    const replayMessage = mocks.sendMessage.mock.calls.find(
      ([message]) => message.type === "tldw:quick-ingest/replay"
    )?.[0]
    expect(replayMessage?.payload?.sessionId).toBe(startMessage?.sessionId)
    expect(ack.sessionId).toBe(startMessage?.sessionId)
    expect(run).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.bgUpload).not.toHaveBeenCalled()

    releaseRun()
  })

  it("returns the stable indeterminate extension identity when accepted start and replay responses both time out", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    const run = vi.fn().mockResolvedValue({ results: [] })
    const worker = createQuickIngestSessionRuntime({
      run,
      emit: vi.fn(),
      saveRunSession: vi.fn().mockResolvedValue(true),
    } as any)
    let acceptedSessionId = ""
    let acceptedAttemptToken = ""
    mocks.sendMessage.mockImplementation((message: any) => {
      if (message.type === "tldw:ping") {
        return Promise.resolve({ ok: true, pong: true })
      }
      if (message.type === "tldw:quick-ingest/start") {
        acceptedSessionId = message.sessionId
        acceptedAttemptToken = message.attemptToken
        void worker.start(message.payload, {
          sessionId: message.sessionId,
          attemptToken: message.attemptToken,
        } as any)
        return new Promise(() => undefined)
      }
      if (message.type === "tldw:quick-ingest/replay") {
        return new Promise(() => undefined)
      }
      return Promise.resolve({ ok: false })
    })

    const ackPromise = startQuickIngestSession({
      entries: [
        {
          id: "occ-double-timeout",
          url: "https://example.com/double-timeout",
          type: "video",
        },
      ],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-double-timeout",
            url: "https://example.com/double-timeout",
          },
        ],
      },
      storeRemote: true,
      processOnly: false,
    } as any)

    await vi.advanceTimersByTimeAsync(20_002)

    await expect(ackPromise).resolves.toMatchObject({
      ok: false,
      indeterminate: true,
      sessionId: acceptedSessionId,
      error: expect.stringMatching(/interrupted|delivery|response|replay|timed out/i),
    })
    expect(acceptedSessionId).toMatch(/^qi-(?!direct-)/)
    expect(acceptedAttemptToken).toMatch(/^qia-/)
    expect(run).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("surfaces a never-delivered start timeout without submitting or hanging", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.sendMessage.mockImplementation((message: any) => {
      if (message.type === "tldw:ping") {
        return Promise.resolve({ ok: true, pong: true })
      }
      if (message.type === "tldw:quick-ingest/start") {
        return new Promise(() => undefined)
      }
      if (message.type === "tldw:quick-ingest/replay") {
        return Promise.resolve({ ok: false, error: "Session not found." })
      }
      return Promise.resolve({ ok: false })
    })

    const ackPromise = startQuickIngestSession({
      entries: [
        {
          id: "occ-timeout-never-delivered",
          url: "https://example.com/timeout-never-delivered",
          type: "video",
        },
      ],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-timeout-never-delivered",
            url: "https://example.com/timeout-never-delivered",
          },
        ],
      },
      storeRemote: true,
      processOnly: false,
    } as any)

    await vi.advanceTimersByTimeAsync(10_001)
    await expect(ackPromise).resolves.toMatchObject({
      ok: false,
      sessionId: expect.stringMatching(/^qi-(?!direct-)/),
      error: expect.stringMatching(/not found|interrupted|delivery|start/i),
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("fails closed when extension cancellation times out instead of reporting direct success", async () => {
    vi.useFakeTimers()
    mocks.runtimeId = "ext-1"
    mocks.sendMessage.mockImplementation((message: any) => {
      if (message.type === "tldw:ping") {
        return Promise.resolve({ ok: true, pong: true })
      }
      if (message.type === "tldw:quick-ingest/cancel") {
        return new Promise(() => undefined)
      }
      return Promise.resolve({ ok: false })
    })

    const cancelPromise = cancelQuickIngestSession({
      sessionId: "qi-extension-cancel-timeout",
      reason: "user_cancelled",
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-extension-cancel-timeout",
      },
    } as any)

    await vi.advanceTimersByTimeAsync(10_001)

    await expect(cancelPromise).resolves.toMatchObject({
      ok: false,
      error: expect.stringMatching(/extension|cancel|timeout|respond/i),
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("restores a pending mv3 run after the background worker is recreated", async () => {
    mocks.runtimeId = "ext-1"
    let storedRecords: unknown[] = []
    const saveRunSession = vi.fn(async (record: any) => {
      storedRecords = record ? [record] : []
    })
    const firstWorker = createQuickIngestSessionRuntime({
      run: vi.fn(async (payload: any, context: any) => {
        expect(payload.pendingRunRequest).toMatchObject({
          inputs: [expect.objectContaining({ occurrenceId: "occ-mv3-restore" })]
        })
        await context.setRunTracking({
          mode: "extension-runtime",
          runId: "run-mv3-restore",
          submissionState: "submitting",
          submissionOccurrenceIds: ["occ-mv3-restore"]
        })
        return { results: [] }
      }),
      emit: vi.fn(),
      saveRunSession,
      createSessionId: () => "qi-worker-mv3-restore"
    } as any)
    mocks.sendMessage.mockImplementation(async (message: any) => {
      if (message.type === "tldw:ping") return { ok: true, pong: true }
      if (message.type === "tldw:quick-ingest/start") {
        return firstWorker.start(message.payload, {
          sessionId: message.sessionId,
        })
      }
      return { ok: false }
    })

    const ack = await startQuickIngestSession({
      entries: [
        {
          id: "occ-mv3-restore",
          url: "https://example.com/mv3-restore",
          type: "video"
        }
      ],
      files: [],
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "direct_url",
            occurrenceId: "occ-mv3-restore",
            url: "https://example.com/mv3-restore"
          }
        ]
      },
      storeRemote: true,
      processOnly: false,
      common: {
        perform_analysis: true,
        perform_chunking: false,
        overwrite_existing: false
      },
      advancedValues: {}
    })

    expect(ack).toEqual({
      ok: true,
      sessionId: expect.stringMatching(/^qi-(?!direct-)/),
    })
    await vi.waitFor(() => expect(saveRunSession).toHaveBeenCalledTimes(2))

    const reattachRun = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null
    })
    const recreatedWorker = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn(),
      loadRunSessions: vi.fn(async () => storedRecords),
      saveRunSession,
      reattachRun
    } as any)

    await recreatedWorker.restore()

    expect(reattachRun).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-mv3-restore",
        sessionId: ack.sessionId,
        submissionState: "submitting",
        submissionOccurrenceIds: ["occ-mv3-restore"]
      }),
      { transportPreference: "poll" }
    )
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

  it("cancels only the requested occurrence in a tracked version-2 run", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-cancel-row",
      status: "running",
      counts: { total: 2, running: 1, cancellation_requested: 1 },
      version: 2,
      collection_id: null,
      batch_ids: [],
      created_at: "2026-07-13T00:00:00Z",
      updated_at: "2026-07-13T00:00:01Z",
      expires_at: "2026-07-20T00:00:00Z",
    })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-direct-row-cancel",
      reason: "user_cancelled",
      occurrenceIds: ["occ-cancel-row"],
      tracking: {
        mode: "webui-direct",
        runId: "run-cancel-row",
      },
    } as any)

    expect(response).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-cancel-row/cancel",
        body: {
          reason: "user_cancelled",
          occurrence_ids: ["occ-cancel-row"],
        },
      }),
    )
  })

  it.each([404, 405, 501])(
    "fails an unsupported occurrence-scoped run cancellation without falling back to whole batches (%s)",
    async (status) => {
      mocks.bgRequest.mockRejectedValue(
        Object.assign(new Error("run occurrence cancellation unsupported"), {
          status,
        })
      )

      const response = await cancelQuickIngestSession({
        sessionId: `qi-direct-row-unsupported-${status}`,
        reason: "user_cancelled",
        occurrenceIds: ["occ-row-unsupported"],
        tracking: {
          mode: "webui-direct",
          runId: `run-row-unsupported-${status}`,
          batchIds: ["batch-must-not-be-cancelled"],
        },
      } as any)

      expect(response).toMatchObject({ ok: false })
      expect(response.error).toMatch(/occurrence|unsupported|run|cancel/i)
      expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: `/api/v1/media/ingest/runs/run-row-unsupported-${status}/cancel`,
          body: {
            reason: "user_cancelled",
            occurrence_ids: ["occ-row-unsupported"],
          },
        })
      )
    }
  )

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

  it("routes extension-run row cancellation with run, generation, and occurrence authority", async () => {
    mocks.runtimeId = "ext-row-cancel"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({
      ok: true,
    })

    const response = await cancelQuickIngestSession({
      sessionId: "qi-extension-row-cancel",
      reason: "user_cancelled",
      occurrenceIds: ["occ-extension-row-cancel"],
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-extension-row-cancel",
        runId: "run-extension-row-cancel",
        generation: "generation-extension-row-cancel",
      },
    } as any)

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest/cancel",
        payload: {
          sessionId: "qi-extension-row-cancel",
          runId: "run-extension-row-cancel",
          expectedGeneration: "generation-extension-row-cancel",
          reason: "user_cancelled",
          occurrenceIds: ["occ-extension-row-cancel"],
        },
      })
    )
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(response).toEqual({ ok: true })
  })

  it("routes extension retries through the shared session authority", async () => {
    mocks.runtimeId = "ext-row-retry"
    mocks.manifestVersion = 2
    mocks.sendMessage.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({
      ok: true,
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    expect(retryQuickIngestSession).toBeTypeOf("function")
    if (typeof retryQuickIngestSession !== "function") return

    const response = await retryQuickIngestSession({
      sessionId: "qi-extension-row-retry",
      occurrenceIds: ["occ-extension-row-retry"],
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-extension-row-retry",
        runId: "run-extension-row-retry",
      },
    })

    expect(mocks.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        type: "tldw:quick-ingest/retry",
        payload: {
          sessionId: "qi-extension-row-retry",
          runId: "run-extension-row-retry",
          occurrenceIds: ["occ-extension-row-retry"],
        },
      })
    )
    expect(response).toEqual({ ok: true })
  })

  it("submits direct retry jobs from the authoritative retry response and leaves file stubs awaiting reselection", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-direct-retry-submit",
      version: 2,
      processing_occurrences: [
        {
          occurrence_id: "occ-direct-retry-url",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: "https://server.example/authoritative-retry.mp4",
          source_kind: "video",
          display_metadata: {
            title: "Authoritative retry",
            source_url: "https://cached.invalid/display-only-retry.mp4",
          },
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 2,
          planned_collection_item_id: null,
        },
        {
          occurrence_id: "occ-direct-retry-file",
          ordinal: 2,
          input_kind: "file_stub",
          source_url: null,
          source_kind: "file",
          display_metadata: { title: "Reselect me" },
          state: "awaiting_upload",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 4,
          planned_collection_item_id: null,
        },
      ],
    })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-retry-submit",
      jobs: [{ id: 902 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-direct-retry-url",
          status: "accepted",
          accepted: true,
          job_id: 902,
          batch_id: "batch-direct-retry-submit",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 2,
        },
      ],
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const response = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-submit",
      occurrenceIds: ["occ-direct-retry-url", "occ-direct-retry-file"],
      cachedSourceUrls: {
        "occ-direct-retry-url": "https://cached.invalid/queue-retry.mp4",
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-submit",
        runId: "run-direct-retry-submit",
        generation: "generation-direct-retry-submit-g1",
      },
    })

    expect(response).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        fields: expect.objectContaining({
          run_id: "run-direct-retry-submit",
          occurrence_ids: ["occ-direct-retry-url"],
          attempts: [2],
          urls: ["https://server.example/authoritative-retry.mp4"],
        }),
      }),
    )
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "cached.invalid",
    )
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "occ-direct-retry-file",
    )
  })

  it("refuses a sixty-fifth live direct retry reservation before backend mutation", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
      }
      throw Object.assign(new Error("Run manifest unavailable"), { status: 503 })
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    for (let index = 0; index < 64; index += 1) {
      const response = await retryQuickIngestSession({
        sessionId: `qi-direct-retry-cap-${index}`,
        occurrenceIds: [`occ-direct-retry-cap-${index}`],
        tracking: {
          mode: "webui-direct",
          sessionId: `qi-direct-retry-cap-${index}`,
          runId: `run-direct-retry-cap-${index}`,
          generation: `generation-direct-retry-cap-${index}-g1`,
        },
      })
      expect(response).toMatchObject({
        ok: false,
        indeterminate: true,
        generation: expect.any(String),
      })
    }

    const refused = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-cap-refused",
      occurrenceIds: ["occ-direct-retry-cap-refused"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-cap-refused",
        runId: "run-direct-retry-cap-refused",
        generation: "generation-direct-retry-cap-refused-g1",
      },
    })

    expect(refused).toMatchObject({
      ok: false,
      error: expect.stringMatching(/capacity|reconcile|full/i),
    })
    expect(refused).not.toHaveProperty("generation")
    expect(retryPosts).toBe(64)
  })

  it("prunes expired direct retry reservations and admits a new retry", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(Date.now() + 25 * 60 * 60 * 1_000)
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        return {
          contract_version: 2,
          run_id: "run-direct-retry-cap-after-expiry",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-cap-after-expiry",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-cap-after-expiry",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-cap-after-expiry",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "completed",
              progress_percent: 100,
              progress_message: null,
              job_id: 903,
              batch_id: "batch-direct-retry-cap-after-expiry",
              media_id: 903,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-cap-after-expiry",
        status: "completed",
        counts: { total: 1, completed: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-direct-retry-cap-after-expiry"],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const response = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-cap-after-expiry",
      occurrenceIds: ["occ-direct-retry-cap-after-expiry"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-cap-after-expiry",
        runId: "run-direct-retry-cap-after-expiry",
        generation: "generation-direct-retry-cap-after-expiry-g1",
      },
    })

    expect(response).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(retryPosts).toBe(1)
  })

  it.each([
    Object.assign(new Error("Retry response was lost"), { status: 0 }),
    Object.assign(new Error("Internal Server Error"), { status: 500 }),
    Object.assign(new Error("Bad Gateway"), { status: 502 }),
    Object.assign(new Error("Service Unavailable"), { status: 503 }),
    Object.assign(new Error("Gateway Timeout"), { status: 504 }),
  ])("retains direct retry authority for an ambiguous %s", async (error) => {
    mocks.bgRequest.mockRejectedValue(error)
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession
    const suffix = String((error as any).status)

    const response = await retryQuickIngestSession({
      sessionId: `qi-direct-retry-ambiguous-${suffix}`,
      occurrenceIds: [`occ-direct-retry-ambiguous-${suffix}`],
      tracking: {
        mode: "webui-direct",
        sessionId: `qi-direct-retry-ambiguous-${suffix}`,
        runId: `run-direct-retry-ambiguous-${suffix}`,
        generation: `generation-direct-retry-ambiguous-${suffix}-g1`,
      },
    })

    expect(response).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })
  })

  it("reconciles an ambiguous direct retry from the authoritative manifest and idempotently submits its advanced attempt", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-reconcile",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-reconcile",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://server.example/reconciled-attempt.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {
                source_url: "https://cached.invalid/reconciled-display.mp4",
              },
              action: "ingest",
              state: "staged",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-reconcile",
        status: "running",
        counts: { total: 1, staged: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    mocks.bgUpload
      .mockRejectedValueOnce(
        Object.assign(new Error("Submission response was lost"), { status: 0 }),
      )
      .mockResolvedValueOnce({
        batch_id: "batch-direct-retry-reconcile",
        jobs: [{ id: 903 }],
        errors: [],
        submissions: [
          {
            occurrence_id: "occ-direct-retry-reconcile",
            status: "accepted",
            accepted: true,
            job_id: 903,
            batch_id: "batch-direct-retry-reconcile",
            error_code: null,
            message: null,
            retryable: false,
            attempt: 2,
          },
        ],
      })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const response = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-reconcile",
      occurrenceIds: ["occ-direct-retry-reconcile"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-reconcile",
        runId: "run-direct-retry-reconcile",
        generation: "generation-direct-retry-reconcile-g1",
      },
    })

    expect(response).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(retryPosts).toBe(1)
    expect(mocks.bgUpload).toHaveBeenCalledTimes(2)
    expect(mocks.bgUpload.mock.calls[1]?.[0]).toEqual(
      mocks.bgUpload.mock.calls[0]?.[0],
    )
    expect(mocks.bgUpload.mock.calls[0]?.[0]).toMatchObject({
      fields: {
        run_id: "run-direct-retry-reconcile",
        occurrence_ids: ["occ-direct-retry-reconcile"],
        attempts: [2],
        urls: ["https://server.example/reconciled-attempt.mp4"],
      },
    })
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "cached.invalid",
    )
  })

  it("reconciles a retained ambiguous direct reservation before issuing another backend retry", async () => {
    let manifestAvailable = false
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
      }
      if (!manifestAvailable) {
        throw Object.assign(new Error("Run manifest unavailable"), {
          status: 503,
        })
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-retained",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-retained",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://server.example/retained-attempt.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "staged",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-retained",
        status: "running",
        counts: { total: 1, staged: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-retry-retained",
      jobs: [{ id: 904 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-direct-retry-retained",
          status: "accepted",
          accepted: true,
          job_id: 904,
          batch_id: "batch-direct-retry-retained",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 2,
        },
      ],
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const first = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-retained",
      occurrenceIds: ["occ-direct-retry-retained"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-retained",
        runId: "run-direct-retry-retained",
        generation: "generation-direct-retry-retained-g1",
      },
    })
    manifestAvailable = true
    const second = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-retained",
      occurrenceIds: ["occ-direct-retry-retained"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-retained",
        runId: "run-direct-retry-retained",
        generation: first.generation,
      },
    })

    expect.soft(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(second).toMatchObject({
      ok: true,
      generation: first.generation,
    })
    expect(retryPosts).toBe(1)
    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
  })

  it("retains mixed-manifest retry authority until every selected URL occurrence is active", async () => {
    let secondOccurrenceAdvanced = false
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-mixed-manifest",
          version: secondOccurrenceAdvanced ? 3 : 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-mixed-first",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://server.example/mixed-first.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: secondOccurrenceAdvanced ? "queued" : "staged",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: secondOccurrenceAdvanced ? 920 : null,
              batch_id: secondOccurrenceAdvanced ? "batch-mixed-first" : null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
            {
              occurrence_id: "occ-direct-retry-mixed-second",
              ordinal: 2,
              input_kind: "direct_url",
              source_url: "https://server.example/mixed-second.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: secondOccurrenceAdvanced ? "staged" : "terminal",
              outcome: secondOccurrenceAdvanced ? null : "processing_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: secondOccurrenceAdvanced ? 2 : 1,
              retryable: !secondOccurrenceAdvanced,
            },
            {
              occurrence_id: "occ-direct-retry-mixed-file",
              ordinal: 3,
              input_kind: "file_stub",
              source_url: null,
              normalized_source_id: null,
              source_kind: "file",
              display_metadata: {},
              action: "ingest",
              state: "awaiting_upload",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-mixed-manifest",
        status: "running",
        counts: { total: 3 },
        version: secondOccurrenceAdvanced ? 3 : 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    mocks.bgUpload
      .mockRejectedValueOnce(
        Object.assign(new Error("Submission response was lost"), { status: 0 }),
      )
      .mockImplementation(async (request: any) => ({
        batch_id: `batch-${request.fields.occurrence_ids[0]}`,
        jobs: [{ id: request.fields.occurrence_ids[0].endsWith("first") ? 920 : 921 }],
        errors: [],
        submissions: request.fields.occurrence_ids.map(
          (occurrenceId: string, index: number) => ({
            occurrence_id: occurrenceId,
            status: "accepted",
            accepted: true,
            job_id: 920 + index,
            batch_id: `batch-${occurrenceId}`,
            error_code: null,
            message: null,
            retryable: false,
            attempt: request.fields.attempts[index],
          }),
        ),
      }))
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession
    const occurrenceIds = [
      "occ-direct-retry-mixed-first",
      "occ-direct-retry-mixed-second",
      "occ-direct-retry-mixed-file",
    ]

    const first = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-mixed-manifest",
      occurrenceIds,
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-mixed-manifest",
        runId: "run-direct-retry-mixed-manifest",
        generation: "generation-direct-retry-mixed-manifest-g1",
      },
    })
    expect.soft(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect.soft(mocks.bgUpload).toHaveBeenCalledTimes(2)
    expect.soft(mocks.bgUpload.mock.calls[1]?.[0]).toEqual(
      mocks.bgUpload.mock.calls[0]?.[0],
    )

    secondOccurrenceAdvanced = true
    const second = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-mixed-manifest",
      occurrenceIds,
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-mixed-manifest",
        runId: "run-direct-retry-mixed-manifest",
        generation: first.generation,
      },
    })

    expect(second).toMatchObject({ ok: true, generation: first.generation })
    expect(retryPosts).toBe(1)
    expect(mocks.bgUpload).toHaveBeenCalledTimes(3)
    expect(mocks.bgUpload.mock.calls[0]?.[0]).toMatchObject({
      fields: {
        occurrence_ids: ["occ-direct-retry-mixed-first"],
        attempts: [2],
      },
    })
    expect(mocks.bgUpload.mock.calls[2]?.[0]).toMatchObject({
      fields: {
        occurrence_ids: ["occ-direct-retry-mixed-second"],
        attempts: [2],
      },
    })
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "occ-direct-retry-mixed-file",
    )
  })

  it("fails closed when a direct retry response lacks an authoritative source or attempt", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-direct-retry-invalid-authority",
      version: 2,
      processing_occurrences: [
        {
          occurrence_id: "occ-direct-retry-invalid-authority",
          ordinal: 1,
          input_kind: "direct_url",
          source_url: null,
          source_kind: "video",
          display_metadata: {
            source_url: "https://cached.invalid/must-not-be-promoted.mp4",
          },
          state: "staged",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 0,
          planned_collection_item_id: null,
        },
      ],
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const response = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-invalid-authority",
      occurrenceIds: ["occ-direct-retry-invalid-authority"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-invalid-authority",
        runId: "run-direct-retry-invalid-authority",
        generation: "generation-direct-retry-invalid-authority-g1",
      },
    })

    expect(response).toMatchObject({
      ok: false,
      generation: expect.any(String),
      error: expect.stringMatching(/authoritative|source|attempt/i),
    })
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("rolls direct retry authority back only after a deterministic rejection", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) {
          throw Object.assign(new Error("Retry conflict"), { status: 409 })
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-deterministic",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-deterministic",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-deterministic",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-deterministic",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "completed",
              progress_percent: 100,
              progress_message: null,
              job_id: 904,
              batch_id: "batch-direct-retry-deterministic",
              media_id: 904,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-deterministic",
        status: "completed",
        counts: { total: 1, completed: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-direct-retry-deterministic"],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession
    const tracking = {
      mode: "webui-direct",
      sessionId: "qi-direct-retry-deterministic",
      runId: "run-direct-retry-deterministic",
      generation: "generation-direct-retry-deterministic-g1",
    }

    const rejected = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds: ["occ-direct-retry-deterministic"],
      tracking,
    })
    const retried = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds: ["occ-direct-retry-deterministic"],
      tracking,
    })

    expect(rejected).toMatchObject({
      ok: false,
      error: expect.stringMatching(/retry|unavailable|conflict/i),
    })
    expect(rejected).not.toHaveProperty("generation")
    expect(rejected).not.toHaveProperty("indeterminate", true)
    expect(retried).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
  })

  it("advances direct retry authority so a later cancellation cannot match the old generation", async () => {
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-generation",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-generation",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "completed",
              progress_percent: 100,
              progress_message: null,
              job_id: 900,
              batch_id: "batch-direct-retry-generation",
              media_id: 900,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-generation",
        status: "completed",
        counts: { total: 1, completed: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-direct-retry-generation"],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    expect(retryQuickIngestSession).toBeTypeOf("function")
    if (typeof retryQuickIngestSession !== "function") return

    const first = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-generation",
      occurrenceIds: ["occ-direct-retry-generation"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-generation",
        runId: "run-direct-retry-generation",
        generation: "generation-direct-retry-old",
      },
    })
    const second = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-generation",
      occurrenceIds: ["occ-direct-retry-generation"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-generation",
        runId: "run-direct-retry-generation",
        generation: first.generation,
      },
    })

    expect(first).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/old/),
    })
    expect(second).toMatchObject({
      ok: true,
      generation: expect.any(String),
    })
    expect(second.generation).not.toBe(first.generation)
  })

  it("keeps direct retry recovery indeterminate when an empty retry response has not advanced the selected URL", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        return {
          contract_version: 2,
          run_id: "run-direct-retry-empty-not-advanced",
          version: 1,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-empty-not-advanced",
          version: 1,
          items: [
            {
              occurrence_id: "occ-direct-retry-empty-not-advanced",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-empty-not-advanced",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "processing_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-empty-not-advanced",
        status: "failed",
        counts: { total: 1, failed: 1 },
        version: 1,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const originalTracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-empty-not-advanced",
      runId: "run-direct-retry-empty-not-advanced",
      generation: "generation-direct-retry-empty-not-advanced-g1",
    }

    const first = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-empty-not-advanced"],
      tracking: originalTracking,
    })

    expect(first).toMatchObject({
      ok: false,
      indeterminate: true,
      notAdvanced: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    const second = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-empty-not-advanced"],
      tracking: { ...originalTracking, generation: first.generation },
    })
    expect(second).toMatchObject({
      ok: false,
      indeterminate: true,
      notAdvanced: true,
      generation: first.generation,
    })
    expect(retryPosts).toBe(2)
  })

  it("rejects a stale direct cancellation after retry advances the session generation", async () => {
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        return {
          contract_version: 2,
          run_id: "run-direct-cancel-generation",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-cancel-generation",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-cancel-generation",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-cancel-generation",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "running",
              outcome: null,
              progress_percent: 30,
              progress_message: "Processing",
              job_id: 905,
              batch_id: "batch-direct-cancel-generation",
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-cancel-generation",
        status: "running",
        counts: { total: 1, running: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-direct-cancel-generation"],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      }
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const retry = await retryQuickIngestSession({
      sessionId: "qi-direct-cancel-generation",
      occurrenceIds: ["occ-direct-cancel-generation"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-cancel-generation",
        runId: "run-direct-cancel-generation",
        generation: "generation-direct-cancel-g1",
      },
    })
    expect(retry).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    mocks.bgRequest.mockClear()
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-direct-cancel-generation",
      status: "cancelled",
      counts: { total: 1, cancelled: 1 },
      version: 3,
      collection_id: null,
      batch_ids: [],
      created_at: "2026-07-13T00:00:00Z",
      updated_at: "2026-07-13T00:00:01Z",
      expires_at: "2026-07-20T00:00:00Z",
    })

    const stale = await cancelQuickIngestSession({
      sessionId: "qi-direct-cancel-generation",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-cancel-generation",
        runId: "run-direct-cancel-generation",
        generation: "generation-direct-cancel-g1",
      },
    })
    expect(stale).toMatchObject({
      ok: false,
      error: expect.stringMatching(/generation|stale|superseded/i),
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()

    const current = await cancelQuickIngestSession({
      sessionId: "qi-direct-cancel-generation",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-cancel-generation",
        runId: "run-direct-cancel-generation",
        generation: retry.generation,
      },
    })
    expect(current).toEqual({ ok: true })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("keeps direct retry generation authority across an existing-run replacement upload", async () => {
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-direct-file-generation",
      version: 2,
      processing_occurrences: [
        {
          occurrence_id: "occ-direct-file-generation",
          ordinal: 1,
          input_kind: "file_stub",
          source_url: null,
          source_kind: "file",
          display_metadata: { title: "Replacement required" },
          state: "awaiting_upload",
          outcome: null,
          job_id: null,
          batch_id: null,
          attempt: 2,
          planned_collection_item_id: null,
        },
      ],
    })
    const quickIngestBatchModule = await import("@/services/tldw/quick-ingest-batch")
    const retryQuickIngestSession = (quickIngestBatchModule as any)
      .retryQuickIngestSession

    const retry = await retryQuickIngestSession({
      sessionId: "qi-direct-file-generation",
      occurrenceIds: ["occ-direct-file-generation"],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-file-generation",
        runId: "run-direct-file-generation",
        generation: "generation-direct-file-generation-g1",
      },
    })
    expect(retry).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-file-generation",
      jobs: [{ id: 905 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-direct-file-generation",
          status: "accepted",
          accepted: true,
          job_id: 905,
          batch_id: "batch-direct-file-generation",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 2,
        },
      ],
    })
    const upload = await submitQuickIngestBatch({
      entries: [],
      files: [
        {
          id: "occ-direct-file-generation",
          name: "replacement.mp4",
          type: "video/mp4",
          data: [1, 2, 3],
        },
      ],
      storeRemote: true,
      processOnly: false,
      pendingRunRequest: {
        inputs: [
          {
            inputKind: "file_stub",
            occurrenceId: "occ-direct-file-generation",
            attempt: 2,
            name: "replacement.mp4",
            contentType: "video/mp4",
            sizeBytes: 3,
          },
        ],
      },
      __quickIngestSessionId: "qi-direct-file-generation",
      __quickIngestRunId: "run-direct-file-generation",
    } as any)
    expect(upload.accepted).toBe(true)

    mocks.bgRequest.mockClear()
    mocks.bgRequest.mockResolvedValue({
      contract_version: 2,
      run_id: "run-direct-file-generation",
      status: "cancelled",
      counts: { total: 1, cancelled: 1 },
      version: 3,
      collection_id: null,
      batch_ids: ["batch-direct-file-generation"],
      created_at: "2026-07-14T00:00:00Z",
      updated_at: "2026-07-14T00:00:01Z",
      expires_at: "2026-07-21T00:00:00Z",
    })
    const stale = await cancelQuickIngestSession({
      sessionId: "qi-direct-file-generation",
      reason: "user_cancelled",
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-file-generation",
        runId: "run-direct-file-generation",
        generation: "generation-direct-file-generation-g1",
      },
    })

    expect(stale).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|stale|superseded/i),
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("coalesces a direct g2 retry while the g1-to-g2 owner is still awaiting the backend", async () => {
    let releaseOwner!: () => void
    const ownerGate = new Promise<void>((resolve) => {
      releaseOwner = resolve
    })
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) await ownerGate
        return {
          contract_version: 2,
          run_id: "run-direct-retry-owner",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-owner",
          version: 1,
          items: [
            {
              occurrence_id: "occ-direct-retry-owner",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://server.example/retry-owner.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "processing_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-owner",
        status: "failed",
        counts: { total: 1, terminal: 1 },
        version: 1,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const input = {
      sessionId: "qi-direct-retry-owner",
      occurrenceIds: ["occ-direct-retry-owner"],
      tracking: {
        mode: "webui-direct" as const,
        sessionId: "qi-direct-retry-owner",
        runId: "run-direct-retry-owner",
        generation: "generation-direct-retry-owner-g1",
      },
    }

    const owner = retryQuickIngestSession(input)
    await vi.waitFor(() => expect(retryPosts).toBe(1))
    const stale = await retryQuickIngestSession(input)
    expect(stale).toMatchObject({
      ok: false,
      generation: expect.not.stringMatching(/g1$/),
      error: expect.stringMatching(/generation|superseded/i),
    })

    const concurrent = retryQuickIngestSession({
      ...input,
      tracking: { ...input.tracking, generation: stale.generation },
    })
    await new Promise((resolve) => setTimeout(resolve, 25))
    const postsBeforeOwnerSettled = retryPosts
    releaseOwner()
    await Promise.allSettled([owner, concurrent])

    expect(postsBeforeOwnerSettled).toBe(1)
    expect(retryPosts).toBe(1)
  })

  it("does not let an awaiting file mask a status-unavailable URL during direct retry reconciliation", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-file-status",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-file-status-file",
              ordinal: 1,
              input_kind: "file_stub",
              source_url: null,
              normalized_source_id: null,
              source_kind: "file",
              display_metadata: {},
              action: "ingest",
              state: "awaiting_upload",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
            {
              occurrence_id: "occ-direct-retry-file-status-url",
              ordinal: 2,
              input_kind: "direct_url",
              source_url: "https://server.example/status-unavailable.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "status_unavailable",
              outcome: null,
              progress_percent: null,
              progress_message: "Status unavailable",
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-file-status",
        status: "running",
        counts: { total: 2 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )

    const response = await retryQuickIngestSession({
      sessionId: "qi-direct-retry-file-status",
      occurrenceIds: [
        "occ-direct-retry-file-status-file",
        "occ-direct-retry-file-status-url",
      ],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-retry-file-status",
        runId: "run-direct-retry-file-status",
        generation: "generation-direct-retry-file-status-g1",
      },
    })

    expect(response).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
      error: expect.stringMatching(/status|manifest|unavailable/i),
    })
    expect(retryPosts).toBe(1)
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("retries only the still-terminal URL after a partial mixed manifest without reselecting its file", async () => {
    let firstUrlSubmitted = false
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) {
          throw Object.assign(new Error("Retry response was lost"), { status: 504 })
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-partial-mixed",
          version: 3,
          processing_occurrences: [
            {
              occurrence_id: "occ-direct-retry-partial-b",
              ordinal: 2,
              input_kind: "direct_url",
              source_url: "https://server.example/partial-b.mp4",
              source_kind: "video",
              display_metadata: {},
              state: "staged",
              outcome: null,
              job_id: null,
              batch_id: null,
              attempt: 2,
              planned_collection_item_id: null,
            },
          ],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-partial-mixed",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-partial-a",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://server.example/partial-a.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: firstUrlSubmitted ? "queued" : "staged",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: firstUrlSubmitted ? 930 : null,
              batch_id: firstUrlSubmitted ? "batch-partial-a" : null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
            {
              occurrence_id: "occ-direct-retry-partial-b",
              ordinal: 2,
              input_kind: "direct_url",
              source_url: "https://server.example/partial-b.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "processing_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
            {
              occurrence_id: "occ-direct-retry-partial-file",
              ordinal: 3,
              input_kind: "file_stub",
              source_url: null,
              normalized_source_id: null,
              source_kind: "file",
              display_metadata: {},
              action: "ingest",
              state: "awaiting_upload",
              outcome: null,
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-partial-mixed",
        status: "running",
        counts: { total: 3 },
        version: 2,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    mocks.bgUpload.mockImplementation(async (request: any) => {
      if (request.fields.occurrence_ids.includes("occ-direct-retry-partial-a")) {
        firstUrlSubmitted = true
      }
      return {
        batch_id: `batch-${request.fields.occurrence_ids[0]}`,
        jobs: [{ id: request.fields.occurrence_ids[0].endsWith("a") ? 930 : 931 }],
        errors: [],
        submissions: request.fields.occurrence_ids.map(
          (occurrenceId: string, index: number) => ({
            occurrence_id: occurrenceId,
            status: "accepted",
            accepted: true,
            job_id: 930 + index,
            batch_id: `batch-${occurrenceId}`,
            error_code: null,
            message: null,
            retryable: false,
            attempt: request.fields.attempts[index],
          }),
        ),
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const occurrenceIds = [
      "occ-direct-retry-partial-a",
      "occ-direct-retry-partial-b",
      "occ-direct-retry-partial-file",
    ]
    const tracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-partial-mixed",
      runId: "run-direct-retry-partial-mixed",
      generation: "generation-direct-retry-partial-mixed-g1",
    }

    const first = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds,
      tracking,
    })
    expect(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const second = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds,
      tracking: { ...tracking, generation: first.generation },
    })

    expect(second).toMatchObject({ ok: true, generation: first.generation })
    expect(retryPosts).toBe(2)
    expect(mocks.bgUpload).toHaveBeenCalledTimes(2)
    expect(mocks.bgUpload.mock.calls[0]?.[0]).toMatchObject({
      fields: {
        occurrence_ids: ["occ-direct-retry-partial-a"],
        attempts: [2],
      },
    })
    expect(mocks.bgUpload.mock.calls[1]?.[0]).toMatchObject({
      fields: {
        occurrence_ids: ["occ-direct-retry-partial-b"],
        attempts: [2],
      },
    })
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "occ-direct-retry-partial-file",
    )
  })

  it("keeps a lost file-only retry reserved until one explicit retry returns awaiting upload", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) {
          throw Object.assign(new Error("Retry response was lost"), { status: 504 })
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-file-only",
          version: 2,
          processing_occurrences: [
            {
              occurrence_id: "occ-direct-retry-file-only",
              ordinal: 0,
              input_kind: "file_stub",
              source_url: null,
              source_kind: "file",
              display_metadata: { file_name: "retry.mp4" },
              state: "awaiting_upload",
              outcome: null,
              job_id: null,
              batch_id: null,
              attempt: 2,
              planned_collection_item_id: null,
            },
          ],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-file-only",
          version: 1,
          items: [
            {
              occurrence_id: "occ-direct-retry-file-only",
              ordinal: 0,
              input_kind: "file_stub",
              source_url: null,
              normalized_source_id: null,
              source_kind: "file",
              display_metadata: { file_name: "retry.mp4" },
              action: "ingest",
              state: "terminal",
              outcome: "submit_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-file-only",
        status: "failed",
        counts: { total: 1, terminal: 1, submit_failed: 1 },
        version: 1,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const tracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-file-only",
      runId: "run-direct-retry-file-only",
      generation: "generation-direct-retry-file-only-g1",
    }

    const lost = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds: ["occ-direct-retry-file-only"],
      tracking,
    })
    expect(lost).toMatchObject({
      ok: false,
      indeterminate: true,
      notAdvanced: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const recovered = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds: ["occ-direct-retry-file-only"],
      tracking: { ...tracking, generation: lost.generation },
    })

    expect(recovered).toMatchObject({ ok: true, generation: lost.generation })
    expect(retryPosts).toBe(2)
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("keeps a partial mixed retry reserved when only its file stub has not advanced", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) {
          throw Object.assign(new Error("Retry response was lost"), { status: 504 })
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-partial-file",
          version: 3,
          processing_occurrences: [
            {
              occurrence_id: "occ-direct-retry-partial-file",
              ordinal: 1,
              input_kind: "file_stub",
              source_url: null,
              source_kind: "file",
              display_metadata: { file_name: "partial.mp4" },
              state: "awaiting_upload",
              outcome: null,
              job_id: null,
              batch_id: null,
              attempt: 2,
              planned_collection_item_id: null,
            },
          ],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-partial-file",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-partial-url",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://server.example/partial-file-url.mp4",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "queued",
              outcome: null,
              progress_percent: 0,
              progress_message: "Queued",
              job_id: 960,
              batch_id: "batch-partial-file-url",
              media_id: null,
              planned_collection_item_id: null,
              attempt: 2,
              retryable: false,
            },
            {
              occurrence_id: "occ-direct-retry-partial-file",
              ordinal: 1,
              input_kind: "file_stub",
              source_url: null,
              normalized_source_id: null,
              source_kind: "file",
              display_metadata: { file_name: "partial.mp4" },
              action: "ingest",
              state: "terminal",
              outcome: "submit_failed",
              progress_percent: null,
              progress_message: null,
              job_id: null,
              batch_id: null,
              media_id: null,
              planned_collection_item_id: null,
              attempt: 1,
              retryable: true,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-partial-file",
        status: "running",
        counts: { total: 2, queued: 1, terminal: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-partial-file-url"],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const occurrenceIds = [
      "occ-direct-retry-partial-url",
      "occ-direct-retry-partial-file",
    ]
    const tracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-partial-file",
      runId: "run-direct-retry-partial-file",
      generation: "generation-direct-retry-partial-file-g1",
    }

    const lost = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds,
      tracking,
    })
    expect(lost).toMatchObject({
      ok: false,
      indeterminate: true,
      notAdvanced: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const recovered = await retryQuickIngestSession({
      sessionId: tracking.sessionId,
      occurrenceIds,
      tracking: { ...tracking, generation: lost.generation },
    })

    expect(recovered).toMatchObject({ ok: true, generation: lost.generation })
    expect(retryPosts).toBe(2)
    expect(mocks.bgUpload).not.toHaveBeenCalled()
  })

  it("prunes an expired direct retry generation before accepting the original fenced generation", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-14T00:00:00Z"))
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        if (retryPosts === 1) {
          throw Object.assign(new Error("Gateway Timeout"), { status: 504 })
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation-expiry",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        const advanced = retryPosts > 1
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation-expiry",
          version: advanced ? 2 : 1,
          items: [
            {
              occurrence_id: "occ-direct-retry-generation-expiry",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-generation-expiry",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: advanced ? "completed" : "processing_failed",
              progress_percent: advanced ? 100 : null,
              progress_message: null,
              job_id: advanced ? 901 : null,
              batch_id: advanced ? "batch-direct-retry-generation-expiry" : null,
              media_id: advanced ? 901 : null,
              planned_collection_item_id: null,
              attempt: advanced ? 2 : 1,
              retryable: !advanced,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-generation-expiry",
        status: retryPosts > 1 ? "completed" : "failed",
        counts: { total: 1 },
        version: retryPosts > 1 ? 2 : 1,
        collection_id: null,
        batch_ids:
          retryPosts > 1 ? ["batch-direct-retry-generation-expiry"] : [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const originalTracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-generation-expiry",
      runId: "run-direct-retry-generation-expiry",
      generation: "generation-direct-retry-generation-expiry-g1",
    }

    const first = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-generation-expiry"],
      tracking: originalTracking,
    })
    expect(first).toMatchObject({
      ok: false,
      indeterminate: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    vi.setSystemTime(Date.now() + 25 * 60 * 60 * 1_000)
    const afterExpiry = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-generation-expiry"],
      tracking: originalTracking,
    })

    expect(afterExpiry).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(retryPosts).toBe(2)
  })

  it("retires direct generation authority only after current-generation whole-run cancellation", async () => {
    let retryPosts = 0
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        retryPosts += 1
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation-retire",
          version: retryPosts + 1,
          processing_occurrences: [],
        }
      }
      if (request.path.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-generation-retire",
          version: retryPosts + 1,
          items: [
            {
              occurrence_id: "occ-direct-retry-generation-retire",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-generation-retire",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "running",
              outcome: null,
              progress_percent: 20,
              progress_message: "Processing",
              job_id: 906,
              batch_id: "batch-direct-retry-generation-retire",
              media_id: null,
              planned_collection_item_id: null,
              attempt: retryPosts + 1,
              retryable: false,
            },
          ],
          next_cursor: null,
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-generation-retire",
        status: "running",
        counts: { total: 1, running: 1 },
        version: retryPosts + 1,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const originalTracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-generation-retire",
      runId: "run-direct-retry-generation-retire",
      generation: "generation-direct-retry-generation-retire-g1",
    }
    const retry = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-generation-retire"],
      tracking: originalTracking,
    })

    const stale = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: originalTracking,
    })
    expect(stale).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|superseded/i),
    })

    const current = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: { ...originalTracking, generation: retry.generation },
    })
    expect(current).toEqual({ ok: true })

    const reused = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-generation-retire"],
      tracking: originalTracking,
    })
    expect(reused).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })
    expect(retryPosts).toBe(2)
  })

  it("keeps whole-run retry generation authority when only the selected occurrence is resolved", async () => {
    let cancelPosts = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path?.endsWith("/retry")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-selected-resolved",
          version: 2,
          processing_occurrences: [],
        }
      }
      if (request.path?.includes("/items")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-selected-resolved",
          version: 2,
          items: [
            {
              occurrence_id: "occ-direct-retry-selected-resolved",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-selected-resolved",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "terminal",
              outcome: "completed",
              progress_percent: 100,
              progress_message: null,
              job_id: 902,
              batch_id: "batch-direct-retry-selected-resolved",
              media_id: 902,
              planned_collection_item_id: null,
              retryable: false,
              attempt: 2,
            },
            {
              occurrence_id: "occ-direct-retry-active-sibling",
              ordinal: 1,
              input_kind: "direct_url",
              source_url: "https://example.com/direct-retry-active-sibling",
              normalized_source_id: null,
              source_kind: "video",
              display_metadata: {},
              action: "ingest",
              state: "running",
              outcome: null,
              progress_percent: 40,
              progress_message: "Processing",
              job_id: 903,
              batch_id: "batch-direct-retry-active-sibling",
              media_id: null,
              planned_collection_item_id: null,
              retryable: false,
              attempt: 1,
            },
          ],
          next_cursor: null,
        }
      }
      if (request.path?.endsWith("/cancel")) {
        cancelPosts += 1
        return {
          contract_version: 2,
          run_id: "run-direct-retry-selected-resolved",
          status: "cancelled",
          counts: { total: 2, cancelled: 2 },
          version: 3,
          collection_id: null,
          batch_ids: [],
          created_at: "2026-07-14T00:00:00Z",
          updated_at: "2026-07-14T00:00:02Z",
          expires_at: "2026-07-21T00:00:00Z",
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-selected-resolved",
        status: "running",
        counts: { total: 2, completed: 1, running: 1 },
        version: 2,
        collection_id: null,
        batch_ids: [
          "batch-direct-retry-selected-resolved",
          "batch-direct-retry-active-sibling",
        ],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:01Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    const { retryQuickIngestSession } = await import(
      "@/services/tldw/quick-ingest-batch"
    )
    const originalTracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-selected-resolved",
      runId: "run-direct-retry-selected-resolved",
      generation: "generation-direct-retry-selected-resolved-g1",
    }
    const retry = await retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: ["occ-direct-retry-selected-resolved"],
      tracking: originalTracking,
    })
    expect(retry).toMatchObject({
      ok: true,
      generation: expect.not.stringMatching(/g1$/),
    })

    const stale = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: originalTracking,
    })
    expect(stale).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|superseded/i),
    })
    expect(cancelPosts).toBe(0)

    const current = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: { ...originalTracking, generation: retry.generation },
    })
    expect(current).toEqual({ ok: true })
    expect(cancelPosts).toBe(1)
  })

  it.each(["completed", "cancelled", "partial_failure"] as const)(
    "retires direct generation authority when retry reconciliation has authoritative %s run status",
    async (runStatus) => {
      const terminalOutcome =
        runStatus === "completed"
          ? "completed"
          : runStatus === "cancelled"
            ? "cancelled"
            : "processing_failed"
      mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
        if (request.path?.endsWith("/retry")) {
          return {
            contract_version: 2,
            run_id: "run-direct-retry-terminal-retire",
            version: 2,
            processing_occurrences: [],
          }
        }
        if (request.path?.includes("/items")) {
          return {
            contract_version: 2,
            run_id: "run-direct-retry-terminal-retire",
            version: 2,
            items: [
              {
                occurrence_id: "occ-direct-retry-terminal-retire",
                ordinal: 0,
                input_kind: "direct_url",
                source_url: "https://example.com/direct-retry-terminal-retire",
                normalized_source_id: null,
                source_kind: "video",
                display_metadata: {},
                action: "ingest",
                state: "terminal",
                outcome: terminalOutcome,
                progress_percent: 100,
                progress_message: null,
                job_id: 902,
                batch_id: "batch-direct-retry-terminal-retire",
                media_id: 902,
                planned_collection_item_id: null,
                retryable: false,
                attempt: 2,
              },
            ],
            next_cursor: null,
          }
        }
        return {
          contract_version: 2,
          run_id: "run-direct-retry-terminal-retire",
          status: runStatus,
          counts:
            runStatus === "partial_failure"
              ? { total: 1, processing_failed: 1 }
              : { total: 1, [runStatus]: 1 },
          version: 2,
          collection_id: null,
          batch_ids: [],
          created_at: "2026-07-14T00:00:00Z",
          updated_at: "2026-07-14T00:00:01Z",
          expires_at: "2026-07-21T00:00:00Z",
        }
      })
      const { retryQuickIngestSession } = await import(
        "@/services/tldw/quick-ingest-batch"
      )
      const originalTracking = {
        mode: "webui-direct" as const,
        sessionId: "qi-direct-retry-terminal-retire",
        runId: "run-direct-retry-terminal-retire",
        generation: "generation-direct-retry-terminal-retire-g1",
      }
      const retry = await retryQuickIngestSession({
        sessionId: originalTracking.sessionId,
        occurrenceIds: ["occ-direct-retry-terminal-retire"],
        tracking: originalTracking,
      })
      expect(retry).toMatchObject({
        ok: true,
        generation: expect.not.stringMatching(/g1$/),
      })

      mocks.bgRequest.mockResolvedValue({
        contract_version: 2,
        run_id: "run-direct-retry-terminal-retire",
        status: "cancelled",
        counts: { total: 1, cancelled: 1 },
        version: 3,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:02Z",
        expires_at: "2026-07-21T00:00:00Z",
      })
      const retired = await cancelQuickIngestSession({
        sessionId: originalTracking.sessionId,
        tracking: originalTracking,
      })

      expect(retired).toEqual({ ok: true })
      expect(mocks.bgRequest).toHaveBeenCalledTimes(4)
    },
  )

  it("retires a successful nonempty direct retry generation only after terminal reattach", async () => {
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path.endsWith("/retry")) {
        return {
          contract_version: 2,
          run_id: "run-direct-retry-terminal-hook",
          version: 2,
          processing_occurrences: [
            {
              occurrence_id: "occ-direct-retry-terminal-hook-url",
              ordinal: 0,
              input_kind: "direct_url",
              source_url: "https://server.example/terminal-hook.mp4",
              source_kind: "video",
              display_metadata: {},
              state: "staged",
              outcome: null,
              job_id: null,
              batch_id: null,
              attempt: 2,
              planned_collection_item_id: null,
            },
            {
              occurrence_id: "occ-direct-retry-terminal-hook-file",
              ordinal: 1,
              input_kind: "file_stub",
              source_url: null,
              source_kind: "file",
              display_metadata: { file_name: "replacement.mp4" },
              state: "awaiting_upload",
              outcome: null,
              job_id: null,
              batch_id: null,
              attempt: 2,
              planned_collection_item_id: null,
            },
          ],
        }
      }
      return {
        contract_version: 2,
        run_id: "run-direct-retry-terminal-hook",
        status: "cancelled",
        counts: { total: 2, cancelled: 2 },
        version: 3,
        collection_id: null,
        batch_ids: ["batch-direct-retry-terminal-hook"],
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:02Z",
        expires_at: "2026-07-21T00:00:00Z",
      }
    })
    mocks.bgUpload.mockResolvedValue({
      batch_id: "batch-direct-retry-terminal-hook",
      jobs: [{ id: 961 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-direct-retry-terminal-hook-url",
          status: "accepted",
          accepted: true,
          job_id: 961,
          batch_id: "batch-direct-retry-terminal-hook",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 2,
        },
      ],
    })
    const quickIngest = await import("@/services/tldw/quick-ingest-batch")
    const originalTracking = {
      mode: "webui-direct" as const,
      sessionId: "qi-direct-retry-terminal-hook",
      runId: "run-direct-retry-terminal-hook",
      generation: "generation-direct-retry-terminal-hook-g1",
    }

    const retry = await quickIngest.retryQuickIngestSession({
      sessionId: originalTracking.sessionId,
      occurrenceIds: [
        "occ-direct-retry-terminal-hook-url",
        "occ-direct-retry-terminal-hook-file",
      ],
      tracking: originalTracking,
    })
    const fenced = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: originalTracking,
    })
    expect(fenced).toMatchObject({
      ok: false,
      generation: retry.generation,
      error: expect.stringMatching(/generation|superseded/i),
    })

    const retireDirectQuickIngestSessionAuthority = (quickIngest as any)
      .retireDirectQuickIngestSessionAuthority
    expect(retireDirectQuickIngestSessionAuthority).toBeTypeOf("function")
    if (typeof retireDirectQuickIngestSessionAuthority !== "function") return
    retireDirectQuickIngestSessionAuthority(
      originalTracking.sessionId,
      retry.generation,
    )

    const retired = await cancelQuickIngestSession({
      sessionId: originalTracking.sessionId,
      tracking: originalTracking,
    })
    expect(retired).toEqual({ ok: true })
    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(mocks.bgUpload.mock.calls)).not.toContain(
      "occ-direct-retry-terminal-hook-file",
    )
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
