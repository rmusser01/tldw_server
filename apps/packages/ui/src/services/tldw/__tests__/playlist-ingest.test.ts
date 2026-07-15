import { describe, expect, expectTypeOf, it, vi } from "vitest";

import {
  PLAYLIST_INGEST_SUBMIT_CHUNK_SIZE,
  cancelRun,
  createRun,
  getRun,
  loadCompletePlaylistPreflightItems,
  listRunItems,
  normalizePlaylistPreflightSummary,
  pollRunSnapshot,
  retryRunItems,
  streamRunEvents,
  submitPendingChunks,
  type ApiPlaylistPreflightSummaryResponse,
  type PlaylistIngestRunApi,
  type PlaylistIngestRunSubmissionRequest,
  type PlaylistIngestRunCreateResult,
  type PlaylistIngestRunItem,
  type PlaylistIngestRunSnapshot,
  type PlaylistIngestPageParams,
  type PlaylistIngestSubmissionRequest,
  type PlaylistIngestRequestOptions,
  type PlaylistPreflightItem,
  type PlaylistPreflightItemsPage,
  type PlaylistPreflightSummary,
} from "../playlist-ingest";

const item = (
  ordinal: number,
  overrides: Partial<PlaylistPreflightItem> = {},
): PlaylistPreflightItem => ({
  occurrenceId: `occ-${ordinal}`,
  ordinal,
  occurrenceIndexForSource: 1,
  sourceUrl: `https://www.youtube.com/watch?v=video-${ordinal}`,
  normalizedSourceId: `youtube:video:${ordinal}`,
  sourceKind: "youtube_video",
  availability: "available",
  duplicateStatus: "new",
  duplicateOfOccurrenceId: null,
  selectedByDefault: true,
  displayMetadata: {
    title: `Video ${ordinal}`,
    channelOrUploader: "Conference channel",
  },
  ...overrides,
});

const summary = (
  loadedCount: number,
  totalCount: number | null = loadedCount,
): PlaylistPreflightSummary => ({
  contractVersion: 2,
  preflightId: "preflight-1",
  status: "ready",
  sourceUrl: "https://www.youtube.com/playlist?list=PL500",
  sourceKind: "youtube_playlist",
  playlistId: "PL500",
  summary: {
    playlistTitle: "Conference 500",
    totalCount,
    loadedCount,
    ingestibleCount: loadedCount,
    unavailableCount: 0,
    duplicateCount: 0,
    selectedCount: loadedCount,
    warnings: [],
  },
  error: null,
  createdAt: "2026-07-13T00:00:00Z",
  updatedAt: "2026-07-13T00:00:01Z",
  expiresAt: "2026-07-14T00:00:00Z",
});

const page = (
  items: PlaylistPreflightItem[],
  nextCursor: string | null,
  preflightId = "preflight-1",
): PlaylistPreflightItemsPage => ({
  contractVersion: 2,
  preflightId,
  items,
  nextCursor,
});

const apiReadySummary = (
  summaryCounts: Record<string, unknown>,
): ApiPlaylistPreflightSummaryResponse =>
  ({
    contract_version: 2,
    preflight_id: "preflight-1",
    status: "ready",
    source_url: "https://www.youtube.com/playlist?list=PL500",
    source_kind: "youtube_playlist",
    playlist_id: "PL500",
    summary: {
      playlist_title: "Conference 500",
      total_count: 0,
      ingestible_count: 0,
      unavailable_count: 0,
      duplicate_count: 0,
      selected_count: 0,
      warnings: [],
      ...summaryCounts,
    },
    error: null,
    created_at: "2026-07-13T00:00:00Z",
    updated_at: "2026-07-13T00:00:01Z",
    expires_at: "2026-07-14T00:00:00Z",
  }) as unknown as ApiPlaylistPreflightSummaryResponse;

const runResult = (
  occurrences: PlaylistIngestRunCreateResult["processingOccurrences"],
): PlaylistIngestRunCreateResult => ({
  contractVersion: 2,
  runId: "run-1",
  status: "staged",
  version: 1,
  statusUrl: "/api/v1/media/ingest/runs/run-1",
  itemsUrl: "/api/v1/media/ingest/runs/run-1/items",
  eventsUrl: "/api/v1/media/ingest/runs/run-1/events/stream",
  processingOccurrences: occurrences,
});

const processingOccurrence = (
  occurrenceId: string,
  overrides: Partial<
    PlaylistIngestRunCreateResult["processingOccurrences"][number]
  > = {},
): PlaylistIngestRunCreateResult["processingOccurrences"][number] => ({
  occurrenceId,
  ordinal: Number(occurrenceId.replace(/\D/g, "")) || 1,
  inputKind: "materialized_playlist_item",
  sourceUrl: `https://www.youtube.com/watch?v=server-${occurrenceId}`,
  sourceKind: "youtube_video",
  displayMetadata: { title: occurrenceId },
  state: "staged",
  outcome: null,
  jobId: null,
  batchId: null,
  attempt: 1,
  plannedCollectionItemId: null,
  ...overrides,
});

const runItem = (
  occurrenceId: string,
  overrides: Partial<PlaylistIngestRunItem> = {},
): PlaylistIngestRunItem => ({
  occurrenceId,
  ordinal: Number(occurrenceId.replace(/\D/g, "")) || 1,
  inputKind: "direct_url",
  sourceUrl: `https://server.example/${occurrenceId}`,
  normalizedSourceId: `url:${occurrenceId}`,
  sourceKind: "video",
  displayMetadata: { title: occurrenceId },
  action: "ingest",
  state: "running",
  outcome: null,
  progressPercent: 10,
  progressMessage: "Running",
  jobId: null,
  batchId: null,
  mediaId: null,
  plannedCollectionItemId: null,
  attempt: 1,
  retryable: false,
  ...overrides,
});

const runSummary = (version: number) => ({
  contractVersion: 2 as const,
  runId: "run-1",
  status: "running",
  counts: { total: 2, running: 2 },
  version,
  collectionId: null,
  batchIds: [],
  createdAt: "2026-07-13T00:00:00Z",
  updatedAt: `2026-07-13T00:00:0${version}Z`,
  expiresAt: "2026-07-20T00:00:00Z",
});

describe("playlist ingest run client", () => {
  it("delegates create/get and preserves structured review-required errors", async () => {
    const reviewRequired = Object.assign(new Error("review required"), {
      code: "review_required",
      recovery: {
        kind: "reviewRequired",
        items: [{ occurrenceId: "occ-1" }],
      },
    });
    const request: PlaylistIngestRunSubmissionRequest = {
      clientRequestId: "quick-ingest-session-1",
      inputs: [
        {
          inputKind: "materialized_playlist_item",
          occurrenceId: "occ-1",
          materializationId: "materialization-1",
        },
      ],
    };
    const summary = {
      contractVersion: 2 as const,
      runId: "run-1",
      status: "running",
      counts: { total: 1, running: 1 },
      version: 2,
      collectionId: null,
      batchIds: [],
      createdAt: "2026-07-13T00:00:00Z",
      updatedAt: "2026-07-13T00:00:01Z",
      expiresAt: "2026-07-20T00:00:00Z",
    };
    const api = {
      createPlaylistIngestRun: vi.fn().mockRejectedValue(reviewRequired),
      getPlaylistIngestRun: vi.fn().mockResolvedValue(summary),
    } as unknown as PlaylistIngestRunApi;

    await expect(createRun(api, request)).rejects.toBe(reviewRequired);
    await expect(getRun(api, "run-1")).resolves.toBe(summary);
    expect(api.createPlaylistIngestRun).toHaveBeenCalledWith(request);
    expect(api.getPlaylistIngestRun).toHaveBeenCalledWith("run-1");
  });

  it("requires the client request identity at the public create boundary", () => {
    expectTypeOf<Parameters<typeof createRun>[1]>().toEqualTypeOf<
      PlaylistIngestRunSubmissionRequest
    >();
  });

  it("restarts opaque-cursor paging when run item versions change", async () => {
    const first = {
      ...processingOccurrence("occ-1"),
      normalizedSourceId: "youtube:occ-1",
      action: "ingest",
      progressPercent: null,
      progressMessage: null,
      mediaId: null,
      retryable: false,
    };
    const second = {
      ...first,
      occurrenceId: "occ-2",
      ordinal: 2,
      normalizedSourceId: "youtube:occ-2",
    };
    const api = {
      listPlaylistIngestRunItems: vi
        .fn()
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 1,
          items: [first],
          nextCursor: "opaque:first",
        })
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 2,
          items: [second],
          nextCursor: null,
        })
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 2,
          items: [{ ...first, progressPercent: 25 }],
          nextCursor: "opaque:second",
        })
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 2,
          items: [second],
          nextCursor: null,
        }),
    } as unknown as PlaylistIngestRunApi;

    const snapshot = await listRunItems(api, "run-1", { pageSize: 1 });

    expect(snapshot).toMatchObject({
      runId: "run-1",
      version: 2,
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          progressPercent: 25,
        }),
        expect.objectContaining({ occurrenceId: "occ-2" }),
      ],
    });
    expect(api.listPlaylistIngestRunItems).toHaveBeenCalledTimes(4);
    expect(
      vi.mocked(api.listPlaylistIngestRunItems).mock.calls.map(
        (call) => call[1]?.cursor ?? null,
      ),
    ).toEqual([null, "opaque:first", null, "opaque:second"]);
  });

  it("fails closed when run item paging exceeds 500 unique occurrences", async () => {
    const api = {
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 1,
        items: Array.from({ length: 501 }, (_, index) =>
          runItem(`occ-${index + 1}`),
        ),
        nextCursor: null,
      }),
    } as unknown as PlaylistIngestRunApi;

    await expect(listRunItems(api, "run-1", { pageSize: 500 })).rejects.toMatchObject({
      code: "run_status_unavailable",
    });
  });

  it("fails closed on endless unique run cursors and oversized opaque cursors", async () => {
    let pageNumber = 0;
    const endlessApi = {
      listPlaylistIngestRunItems: vi.fn(async () => {
        pageNumber += 1;
        if (pageNumber > 500) {
          throw Object.assign(new Error("client exceeded paging bound"), {
            code: "test_page_bound_exceeded",
          });
        }
        return {
          contractVersion: 2 as const,
          runId: "run-1",
          version: 1,
          items: [runItem(`occ-${pageNumber}`)],
          nextCursor: `cursor-${pageNumber}`,
        };
      }),
    } as unknown as PlaylistIngestRunApi;

    await expect(
      listRunItems(endlessApi, "run-1", { pageSize: 1 }),
    ).rejects.toMatchObject({ code: "run_status_unavailable" });
    expect(endlessApi.listPlaylistIngestRunItems).toHaveBeenCalledTimes(500);

    const oversizedApi = {
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 1,
        items: [runItem("occ-1")],
        nextCursor: "x".repeat(4097),
      }),
    } as unknown as PlaylistIngestRunApi;
    await expect(listRunItems(oversizedApi, "run-1")).rejects.toMatchObject({
      code: "run_status_unavailable",
    });
  });

  it("delegates occurrence cancellation and deliberate retry unchanged", async () => {
    const cancelled = {
      contractVersion: 2 as const,
      runId: "run-1",
      status: "running",
      counts: { total: 2, cancellation_requested: 1 },
      version: 3,
      collectionId: null,
      batchIds: ["batch-1"],
      createdAt: "2026-07-13T00:00:00Z",
      updatedAt: "2026-07-13T00:00:02Z",
      expiresAt: "2026-07-20T00:00:00Z",
    };
    const retried = {
      contractVersion: 2 as const,
      runId: "run-1",
      version: 4,
      processingOccurrences: [
        processingOccurrence("occ-2", { attempt: 2 }),
      ],
    };
    const api = {
      cancelPlaylistIngestRun: vi.fn().mockResolvedValue(cancelled),
      retryPlaylistIngestRunItems: vi.fn().mockResolvedValue(retried),
    } as unknown as PlaylistIngestRunApi;

    await expect(
      cancelRun(api, "run-1", {
        occurrenceIds: ["occ-1"],
        reason: "user_cancelled",
      }),
    ).resolves.toBe(cancelled);
    await expect(retryRunItems(api, "run-1", ["occ-2"])).resolves.toBe(
      retried,
    );
    expect(api.cancelPlaylistIngestRun).toHaveBeenCalledWith("run-1", {
      occurrenceIds: ["occ-1"],
      reason: "user_cancelled",
    });
    expect(api.retryPlaylistIngestRunItems).toHaveBeenCalledWith("run-1", [
      "occ-2",
    ]);
  });

  it("polls a full snapshot and merges rows only by occurrence id", async () => {
    const previous: PlaylistIngestRunSnapshot = {
      summary: runSummary(1),
      items: [
        runItem("occ-1", {
          sourceUrl: "https://cached.invalid/occ-1",
          jobId: 77,
        }),
        runItem("occ-2", { jobId: 88 }),
      ],
      lastEventId: null,
    };
    const api = {
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(2)),
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 2,
        items: [
          runItem("occ-1", {
            sourceUrl: "https://server.example/authoritative-occ-1",
            jobId: 88,
            progressPercent: 50,
          }),
          runItem("occ-2", { jobId: 77, progressPercent: 60 }),
        ],
        nextCursor: null,
      }),
    } as unknown as PlaylistIngestRunApi;

    const next = await pollRunSnapshot(api, "run-1", previous);

    expect(next.items).toEqual([
      expect.objectContaining({
        occurrenceId: "occ-1",
        sourceUrl: "https://server.example/authoritative-occ-1",
        jobId: 88,
        progressPercent: 50,
      }),
      expect.objectContaining({
        occurrenceId: "occ-2",
        jobId: 77,
        progressPercent: 60,
      }),
    ]);
  });

  it("applies SSE state/outcome by occurrence and fully reloads on resync", async () => {
    const initial: PlaylistIngestRunSnapshot = {
      summary: runSummary(2),
      items: [runItem("occ-1", { jobId: 77 }), runItem("occ-2", { jobId: 77 })],
      lastEventId: 40,
    };
    const api = {
      streamPlaylistIngestRunEvents: vi.fn(async function* () {
        yield {
          kind: "occurrence" as const,
          event: {
            eventId: 41,
            runId: "run-1",
            occurrenceId: "occ-2",
            jobId: 77,
            batchId: "batch-2",
            eventType: "terminal",
            state: "terminal" as const,
            outcome: "completed" as const,
            progressPercent: 100,
            progressMessage: "Done",
            occurredAt: "2026-07-13T00:00:03Z",
          },
        };
        yield {
          kind: "resyncRequired" as const,
          runId: "run-1",
          minEventId: 50,
          latestEventId: 55,
        };
      }),
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(5)),
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 5,
        items: [
          runItem("occ-1", { progressPercent: 75 }),
          runItem("occ-2", {
            state: "terminal",
            outcome: "completed",
            progressPercent: 100,
          }),
        ],
        nextCursor: null,
      }),
    } as unknown as PlaylistIngestRunApi;

    const snapshots: PlaylistIngestRunSnapshot[] = [];
    for await (const snapshot of streamRunEvents(api, initial)) {
      snapshots.push(snapshot);
    }

    expect(snapshots[0]).toMatchObject({
      lastEventId: 41,
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          state: "running",
          outcome: null,
        }),
        expect.objectContaining({
          occurrenceId: "occ-2",
          state: "terminal",
          outcome: "completed",
        }),
      ],
    });
    expect(snapshots[1]).toMatchObject({
      summary: { version: 5 },
      lastEventId: 55,
    });
    expect(
      snapshots[1]?.items.find((item) => item.occurrenceId === "occ-1"),
    ).toMatchObject({ occurrenceId: "occ-1", progressPercent: 75 });
    expect(api.getPlaylistIngestRun).toHaveBeenCalledWith("run-1");
    expect(api.listPlaylistIngestRunItems).toHaveBeenCalledWith("run-1", {
      limit: 100,
    });
  });

  it("keeps direct transport on every summary and paged item request during resync", async () => {
    const initial: PlaylistIngestRunSnapshot = {
      summary: runSummary(1),
      items: [runItem("occ-1")],
      lastEventId: 10,
    };
    const api = {
      streamPlaylistIngestRunEvents: vi.fn(async function* () {
        yield {
          kind: "resyncRequired" as const,
          runId: "run-1",
          minEventId: 11,
          latestEventId: 12,
        };
      }),
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(2)),
      listPlaylistIngestRunItems: vi
        .fn()
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 2,
          items: [runItem("occ-1")],
          nextCursor: "page-2",
        })
        .mockResolvedValueOnce({
          contractVersion: 2,
          runId: "run-1",
          version: 2,
          items: [runItem("occ-2")],
          nextCursor: null,
        }),
    } as unknown as PlaylistIngestRunApi;

    const snapshots: PlaylistIngestRunSnapshot[] = [];
    for await (const snapshot of streamRunEvents(api, initial, {
      preferDirect: true,
    })) {
      snapshots.push(snapshot);
    }

    expect(snapshots[0]?.items).toHaveLength(2);
    expect(api.getPlaylistIngestRun).toHaveBeenCalledWith("run-1", {
      preferDirect: true,
    });
    expect(api.listPlaylistIngestRunItems).toHaveBeenNthCalledWith(
      1,
      "run-1",
      { limit: 100 },
      { preferDirect: true },
    );
    expect(api.listPlaylistIngestRunItems).toHaveBeenNthCalledWith(
      2,
      "run-1",
      { limit: 100, cursor: "page-2" },
      { preferDirect: true },
    );
  });

  it("reloads authoritatively instead of applying same-state retained replay", async () => {
    const authoritative = runItem("occ-1", {
      state: "running",
      progressPercent: 65,
      progressMessage: "Transcribing",
      jobId: 88,
      batchId: "batch-current",
    });
    const initial: PlaylistIngestRunSnapshot = {
      summary: runSummary(4),
      items: [authoritative],
      lastEventId: null,
    };
    const api = {
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(4)),
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 4,
        items: [authoritative],
        nextCursor: null,
      }),
      streamPlaylistIngestRunEvents: vi.fn(async function* () {
        yield {
          kind: "occurrence" as const,
          event: {
            eventId: 2,
            runId: "run-1",
            occurrenceId: "occ-1",
            jobId: 77,
            batchId: "batch-old",
            eventType: "running",
            state: "running" as const,
            outcome: null,
            progressPercent: 10,
            progressMessage: "Downloading",
            occurredAt: "2026-07-13T00:00:02Z",
          },
        };
      }),
    } as unknown as PlaylistIngestRunApi;

    const snapshots: PlaylistIngestRunSnapshot[] = [];
    for await (const snapshot of streamRunEvents(api, initial)) {
      snapshots.push(snapshot);
    }

    expect(snapshots[0]).toMatchObject({
      lastEventId: 2,
      items: [
        expect.objectContaining({
          state: "running",
          progressPercent: 65,
          progressMessage: "Transcribing",
          jobId: 88,
          batchId: "batch-current",
        }),
      ],
    });
    expect(api.getPlaylistIngestRun).toHaveBeenCalledWith("run-1");
  });

  it("reloads authoritative items when an SSE summary advances the run version", async () => {
    const initial: PlaylistIngestRunSnapshot = {
      summary: runSummary(2),
      items: [runItem("occ-1", { progressPercent: 10 })],
      lastEventId: 40,
    };
    const api = {
      streamPlaylistIngestRunEvents: vi.fn(async function* () {
        yield { kind: "snapshot" as const, summary: runSummary(3) };
      }),
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(3)),
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 3,
        items: [runItem("occ-1", { progressPercent: 65 })],
        nextCursor: null,
      }),
    } as unknown as PlaylistIngestRunApi;

    const snapshots: PlaylistIngestRunSnapshot[] = [];
    for await (const snapshot of streamRunEvents(api, initial)) {
      snapshots.push(snapshot);
    }

    expect(snapshots).toEqual([
      expect.objectContaining({
        summary: expect.objectContaining({ version: 3 }),
        items: [expect.objectContaining({ progressPercent: 65 })],
        lastEventId: 40,
      }),
    ]);
    expect(api.getPlaylistIngestRun).toHaveBeenCalledWith("run-1");
    expect(api.listPlaylistIngestRunItems).toHaveBeenCalledWith("run-1", {
      limit: 100,
    });
  });

  it("does not regress an authoritative running item from retained queued replay", async () => {
    const initial: PlaylistIngestRunSnapshot = {
      summary: runSummary(4),
      items: [runItem("occ-1", { state: "running", progressPercent: 65 })],
      lastEventId: null,
    };
    const api = {
      getPlaylistIngestRun: vi.fn().mockResolvedValue(runSummary(4)),
      listPlaylistIngestRunItems: vi.fn().mockResolvedValue({
        contractVersion: 2,
        runId: "run-1",
        version: 4,
        items: [runItem("occ-1", { state: "running", progressPercent: 65 })],
        nextCursor: null,
      }),
      streamPlaylistIngestRunEvents: vi.fn(async function* () {
        yield {
          kind: "occurrence" as const,
          event: {
            eventId: 1,
            runId: "run-1",
            occurrenceId: "occ-1",
            jobId: 77,
            batchId: "batch-1",
            eventType: "queued",
            state: "queued" as const,
            outcome: null,
            progressPercent: 0,
            progressMessage: "Queued",
            occurredAt: "2026-07-13T00:00:01Z",
          },
        };
      }),
    } as unknown as PlaylistIngestRunApi;

    const snapshots: PlaylistIngestRunSnapshot[] = [];
    for await (const snapshot of streamRunEvents(api, initial)) {
      snapshots.push(snapshot);
    }

    expect(snapshots[0]).toMatchObject({
      lastEventId: 1,
      items: [
        expect.objectContaining({
          occurrenceId: "occ-1",
          state: "running",
          progressPercent: 65,
          progressMessage: "Running",
        }),
      ],
    });
  });

  it("submits only authoritative processing occurrences in bounded URL chunks", async () => {
    expect(PLAYLIST_INGEST_SUBMIT_CHUNK_SIZE).toBeGreaterThan(0);
    expect(PLAYLIST_INGEST_SUBMIT_CHUNK_SIZE).toBeLessThan(500);
    const authoritative = [
      processingOccurrence("occ-1", {
        sourceUrl: "https://www.youtube.com/watch?v=authoritative-1",
        plannedCollectionItemId: 101,
      }),
      processingOccurrence("occ-2", {
        sourceUrl: "https://www.youtube.com/watch?v=authoritative-2",
        plannedCollectionItemId: 102,
      }),
      processingOccurrence("occ-3", {
        sourceUrl: "https://www.youtube.com/watch?v=authoritative-3",
        plannedCollectionItemId: 103,
      }),
    ];
    const submitChunk = vi.fn(
      async (request: PlaylistIngestSubmissionRequest) => ({
        batch_id: `batch-${request.fields.occurrence_ids?.[0]}`,
        jobs: [],
        errors: [],
        submissions: (request.fields.occurrence_ids as string[]).map(
          (occurrenceId, index) => ({
            occurrence_id: occurrenceId,
            status: "accepted",
            accepted: true,
            job_id: index + 1,
            batch_id: `batch-${occurrenceId}`,
            error_code: null,
            message: null,
            retryable: false,
            attempt: 1,
          }),
        ),
      }),
    );

    const result = await submitPendingChunks({
      run: runResult(authoritative),
      chunkSize: 2,
      cachedSourceUrls: {
        "occ-1": "https://cached.invalid/never-submit-1",
        "occ-2": "https://cached.invalid/never-submit-2",
        "occ-3": "https://cached.invalid/never-submit-3",
      },
      baseFields: { media_type: "video", perform_analysis: true },
      submitChunk,
    });

    expect(submitChunk).toHaveBeenCalledTimes(2);
    expect(submitChunk.mock.calls[0]?.[0]).toMatchObject({
      fields: {
        run_id: "run-1",
        occurrence_ids: ["occ-1", "occ-2"],
        attempts: [1, 1],
        planned_item_ids: [101, 102],
        urls: [
          "https://www.youtube.com/watch?v=authoritative-1",
          "https://www.youtube.com/watch?v=authoritative-2",
        ],
      },
    });
    expect(submitChunk.mock.calls[1]?.[0]).toMatchObject({
      fields: {
        run_id: "run-1",
        occurrence_ids: ["occ-3"],
        attempts: [1],
        planned_item_ids: [103],
        urls: ["https://www.youtube.com/watch?v=authoritative-3"],
      },
    });
    expect(JSON.stringify(submitChunk.mock.calls)).not.toContain(
      "cached.invalid",
    );
    expect(result.submissions).toHaveLength(3);
  });

  it("keeps URL and file occurrence arrays aligned and preserves structured partial acceptance", async () => {
    const submitChunk = vi.fn(
      async (request: PlaylistIngestSubmissionRequest) => ({
        batch_id: "batch-mixed",
        jobs: [],
        errors: [],
        submissions: [
          {
            occurrence_id: "occ-url",
            status: "rejected",
            accepted: false,
            job_id: null,
            batch_id: "batch-mixed",
            error_code: "occurrence_not_processable",
            message: "No longer processable.",
            retryable: false,
            attempt: 1,
          },
          {
            occurrence_id: "occ-file",
            status: "accepted",
            accepted: true,
            job_id: 22,
            batch_id: "batch-mixed",
            error_code: null,
            message: null,
            retryable: false,
            attempt: 2,
          },
        ],
      }),
    );
    const run = runResult([
      processingOccurrence("occ-url"),
      processingOccurrence("occ-file", {
        inputKind: "file_stub",
        sourceUrl: null,
        sourceKind: "file",
        state: "awaiting_upload",
        attempt: 2,
        plannedCollectionItemId: 202,
      }),
    ]);

    const result = await submitPendingChunks({
      run,
      baseFields: { media_type: "video" },
      filesByOccurrenceId: {
        "occ-file": {
          name: "talk.mp4",
          type: "video/mp4",
          data: [1, 2, 3],
        },
      },
      submitChunk,
    });

    expect(submitChunk).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          occurrence_ids: ["occ-url"],
          attempts: [1],
          urls: ["https://www.youtube.com/watch?v=server-occ-url"],
          file_occurrence_ids: ["occ-file"],
          file_attempts: [2],
          file_planned_item_ids: [202],
        }),
        files: [
          expect.objectContaining({
            fieldName: "files",
            name: "talk.mp4",
          }),
        ],
      }),
    );
    expect(result.submissions).toEqual([
      expect.objectContaining({
        occurrenceId: "occ-url",
        accepted: false,
        errorCode: "occurrence_not_processable",
      }),
      expect.objectContaining({
        occurrenceId: "occ-file",
        accepted: true,
        jobId: 22,
        attempt: 2,
      }),
    ]);
  });

  it("stops without submitting when an awaiting-upload occurrence has no local file", async () => {
    const submitChunk = vi.fn();
    const result = await submitPendingChunks({
      run: runResult([
        processingOccurrence("occ-file", {
          inputKind: "file_stub",
          sourceUrl: null,
          sourceKind: "file",
          state: "awaiting_upload",
        }),
      ]),
      baseFields: { media_type: "video" },
      submitChunk,
    });

    expect(submitChunk).not.toHaveBeenCalled();
    expect(result).toMatchObject({
      stopped: true,
      unsentOccurrenceIds: ["occ-file"],
    });
  });

  it("preserves accepted work while stopping on an omitted processing occurrence", async () => {
    const submitChunk = vi.fn().mockResolvedValue({
      batch_id: "batch-accepted",
      jobs: [{ id: 91 }],
      errors: [],
      submissions: [
        {
          occurrence_id: "occ-url",
          status: "accepted",
          accepted: true,
          job_id: 91,
          batch_id: "batch-accepted",
          error_code: null,
          message: null,
          retryable: false,
          attempt: 1,
        },
      ],
    });
    const result = await submitPendingChunks({
      run: runResult([
        processingOccurrence("occ-url"),
        processingOccurrence("occ-file", {
          inputKind: "file_stub",
          sourceUrl: null,
          sourceKind: "file",
          state: "awaiting_upload",
        }),
      ]),
      baseFields: { media_type: "video" },
      submitChunk,
    });

    expect(result).toMatchObject({
      stopped: true,
      submissions: [expect.objectContaining({ occurrenceId: "occ-url", accepted: true })],
      unsentOccurrenceIds: ["occ-file"],
    });
  });

  it("stops later chunks on a global failure and exposes Retry-After", async () => {
    const submitChunk = vi
      .fn<(request: PlaylistIngestSubmissionRequest) => Promise<any>>()
      .mockRejectedValue(
        Object.assign(new Error("rate limited"), {
          status: 429,
          retryAfterMs: 19_000,
        }),
      );

    const result = await submitPendingChunks({
      run: runResult([
        processingOccurrence("occ-1"),
        processingOccurrence("occ-2"),
      ]),
      chunkSize: 1,
      baseFields: { media_type: "video" },
      submitChunk,
    });

    expect(submitChunk).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      stopped: true,
      retryAfterMs: 19_000,
      unsentOccurrenceIds: ["occ-1", "occ-2"],
    });
  });

  it("retries one ambiguous transport failure with the identical attempt request", async () => {
    const submitChunk = vi
      .fn<(request: PlaylistIngestSubmissionRequest) => Promise<any>>()
      .mockRejectedValueOnce(
        Object.assign(new Error("connection reset"), { status: 0 }),
      )
      .mockResolvedValueOnce({
        batch_id: "batch-original",
        jobs: [],
        errors: [],
        submissions: [
          {
            occurrence_id: "occ-1",
            status: "accepted",
            accepted: true,
            job_id: 44,
            batch_id: "batch-original",
            error_code: null,
            message: null,
            retryable: false,
            attempt: 1,
          },
        ],
      });

    await submitPendingChunks({
      run: runResult([processingOccurrence("occ-1")]),
      baseFields: { media_type: "video" },
      submitChunk,
    });

    expect(submitChunk).toHaveBeenCalledTimes(2);
    expect(submitChunk.mock.calls[1]?.[0]).toBe(
      submitChunk.mock.calls[0]?.[0],
    );
    expect(submitChunk.mock.calls[1]?.[0].fields.attempts).toEqual([1]);
  });

  it("never retries an explicit HTTP submission failure", async () => {
    const submitChunk = vi
      .fn<(request: PlaylistIngestSubmissionRequest) => Promise<any>>()
      .mockRejectedValue(Object.assign(new Error("unavailable"), { status: 503 }));

    const result = await submitPendingChunks({
      run: runResult([processingOccurrence("occ-1")]),
      baseFields: { media_type: "video" },
      submitChunk,
    });

    expect(submitChunk).toHaveBeenCalledTimes(1);
    expect(result.stopped).toBe(true);
  });
});

type PageLoader = (
  preflightId: string,
  params: PlaylistIngestPageParams,
  options: PlaylistIngestRequestOptions,
) => Promise<PlaylistPreflightItemsPage>;

describe("loadCompletePlaylistPreflightItems", () => {
  it("loads a 500-item immutable snapshot in exact opaque-cursor order", async () => {
    const items = Array.from({ length: 500 }, (_, index) => item(index + 1));
    const cursors = [
      "opaque:page/+2==",
      "do-not-decode::3?",
      "cursor%2Ffour+raw",
      "final cursor = five",
    ];
    const pages = [
      page(items.slice(0, 100), cursors[0]),
      page(items.slice(100, 200), cursors[1]),
      page(items.slice(200, 300), cursors[2]),
      page(items.slice(300, 400), cursors[3]),
      page(items.slice(400), null),
    ];
    const requestedCursors: Array<string | null> = [];
    const observedSignals: AbortSignal[] = [];
    const loadPage = vi.fn<PageLoader>(async (preflightId, params, options) => {
      expect(preflightId).toBe("preflight-1");
      requestedCursors.push(params.cursor ?? null);
      observedSignals.push(options.signal as AbortSignal);
      const index = requestedCursors.length - 1;
      return pages[index];
    });
    const controller = new AbortController();

    const result = await loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(500),
      signal: controller.signal,
      loadPage,
      pageSize: 100,
    });

    expect(requestedCursors).toEqual([null, ...cursors]);
    expect(new Set(requestedCursors).size).toBe(requestedCursors.length);
    expect(observedSignals).toEqual(Array(5).fill(controller.signal));
    expect(result).toHaveLength(500);
    expect(result.map(({ occurrenceId }) => occurrenceId)).toEqual(
      items.map(({ occurrenceId }) => occurrenceId),
    );
    expect(result.map(({ ordinal }) => ordinal)).toEqual(
      Array.from({ length: 500 }, (_, index) => index + 1),
    );
  });

  it("cancels sequential paging with the shared AbortSignal and returns no partial snapshot", async () => {
    const controller = new AbortController();
    const loadPage = vi
      .fn<PageLoader>()
      .mockResolvedValueOnce(page([item(1)], "next-page"))
      .mockImplementationOnce(
        async (_preflightId, _params, options) =>
          new Promise<PlaylistPreflightItemsPage>((_resolve, reject) => {
            options.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true },
            );
          }),
      );

    const result = loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(2),
      signal: controller.signal,
      loadPage,
      pageSize: 1,
    });
    await vi.waitFor(() => expect(loadPage).toHaveBeenCalledTimes(2));

    controller.abort();

    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    expect(loadPage.mock.calls[0]?.[2].signal).toBe(controller.signal);
    expect(loadPage.mock.calls[1]?.[2].signal).toBe(controller.signal);
  });

  it("rejects a repeated cursor without refetching a completed page", async () => {
    const loadPage = vi
      .fn<PageLoader>()
      .mockResolvedValueOnce(page([item(1)], "repeat-me"))
      .mockResolvedValueOnce(page([item(2)], "repeat-me"));

    const result = loadCompletePlaylistPreflightItems({
      preflightId: "preflight-1",
      summary: summary(2),
      signal: new AbortController().signal,
      loadPage,
      pageSize: 1,
    });

    await expect(result).rejects.toMatchObject({
      code: "preflight_incomplete",
      message: "Playlist inspection is incomplete. Try again.",
      retryable: true,
    });
    expect(loadPage).toHaveBeenCalledTimes(2);
    expect(loadPage.mock.calls.map((call) => call[1].cursor ?? null)).toEqual([
      null,
      "repeat-me",
    ]);
  });

  it.each([
    {
      name: "an empty continuing page",
      readySummary: summary(1),
      firstPage: page([], "must-not-be-followed"),
    },
    {
      name: "a cursor after reaching loaded count",
      readySummary: summary(1),
      firstPage: page([item(1)], "must-not-be-followed"),
    },
    {
      name: "items exceeding loaded count",
      readySummary: summary(1),
      firstPage: page([item(1), item(2)], "must-not-be-followed"),
    },
  ])(
    "rejects $name without following its cursor",
    async ({ readySummary, firstPage }) => {
      const loadPage = vi
        .fn<PageLoader>()
        .mockResolvedValueOnce(firstPage)
        .mockResolvedValueOnce(page([], null));

      const result = loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: readySummary,
        signal: new AbortController().signal,
        loadPage,
        pageSize: 100,
      });

      await expect(result).rejects.toMatchObject({
        code: "preflight_incomplete",
      });
      expect(loadPage).toHaveBeenCalledTimes(1);
    },
  );

  it("rejects a non-v2 ready summary before loading items", async () => {
    const loadPage = vi.fn<PageLoader>().mockResolvedValue(page([], null));
    const invalidSummary = {
      ...summary(0),
      contractVersion: 1,
    } as unknown as PlaylistPreflightSummary;

    await expect(
      loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: invalidSummary,
        signal: new AbortController().signal,
        loadPage,
      }),
    ).rejects.toMatchObject({ code: "preflight_incomplete" });
    expect(loadPage).not.toHaveBeenCalled();
  });

  it("rejects a non-v2 items page without following its cursor", async () => {
    const loadPage = vi.fn<PageLoader>().mockResolvedValue({
      ...page([item(1)], "must-not-be-followed"),
      contractVersion: 1,
    } as unknown as PlaylistPreflightItemsPage);

    await expect(
      loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: summary(1),
        signal: new AbortController().signal,
        loadPage,
      }),
    ).rejects.toMatchObject({ code: "preflight_incomplete" });
    expect(loadPage).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["missing", {}],
    ["invalid", { loaded_count: "0" }],
  ])(
    "rejects a ready summary with %s loaded_count before loading items",
    async (_name, summaryCounts) => {
      const loadPage = vi.fn<PageLoader>().mockResolvedValue(page([], null));
      const normalized = normalizePlaylistPreflightSummary(
        apiReadySummary(summaryCounts),
      );

      await expect(
        loadCompletePlaylistPreflightItems({
          preflightId: "preflight-1",
          summary: normalized,
          signal: new AbortController().signal,
          loadPage,
        }),
      ).rejects.toMatchObject({ code: "preflight_incomplete" });
      expect(loadPage).not.toHaveBeenCalled();
    },
  );

  it.each([
    {
      name: "mismatched preflight ids",
      readySummary: summary(1),
      pages: [page([item(1)], null, "other-preflight")],
    },
    {
      name: "duplicate occurrence ids",
      readySummary: summary(2),
      pages: [
        page([item(1)], "next"),
        page([item(2, { occurrenceId: "occ-1" })], null),
      ],
    },
    {
      name: "loaded-count mismatches",
      readySummary: summary(3, null),
      pages: [page([item(1), item(2)], null)],
    },
    {
      name: "known total-count mismatches",
      readySummary: summary(2, 3),
      pages: [page([item(1), item(2)], null)],
    },
  ])(
    "rejects $name as a safe incomplete snapshot",
    async ({ readySummary, pages }) => {
      const loadPage = vi.fn<PageLoader>(
        async () => pages.shift() as PlaylistPreflightItemsPage,
      );

      const result = loadCompletePlaylistPreflightItems({
        preflightId: "preflight-1",
        summary: readySummary,
        signal: new AbortController().signal,
        loadPage,
        pageSize: 100,
      });

      await expect(result).rejects.toMatchObject({
        code: "preflight_incomplete",
        message: "Playlist inspection is incomplete. Try again.",
        retryable: true,
      });
    },
  );
});
