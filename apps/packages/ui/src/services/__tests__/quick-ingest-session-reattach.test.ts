import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn(),
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args),
}))

import { reattachQuickIngestSession } from "@/services/tldw/quick-ingest-session-reattach"

const runSummaryResponse = (status: string, version: number) => ({
  contract_version: 2,
  run_id: "run-reload-submission",
  status,
  counts: { total: 2 },
  version,
  collection_id: null,
  batch_ids: [],
  created_at: "2026-07-13T00:00:00Z",
  updated_at: "2026-07-13T00:00:01Z",
  expires_at: "2026-07-20T00:00:00Z",
})

const runItemResponse = (
  occurrenceId: string,
  state: string,
  overrides: Record<string, unknown> = {}
) => ({
  occurrence_id: occurrenceId,
  ordinal: occurrenceId.endsWith("2") ? 2 : 1,
  input_kind: "direct_url",
  source_url: `https://server.example/${occurrenceId}`,
  normalized_source_id: `url:${occurrenceId}`,
  source_kind: "video",
  display_metadata: { title: occurrenceId },
  action: "ingest",
  state,
  outcome: null,
  progress_percent: state === "running" ? 40 : 0,
  progress_message: null,
  job_id: state === "running" ? 77 : null,
  batch_id: state === "running" ? "batch-accepted" : null,
  media_id: null,
  planned_collection_item_id: null,
  attempt: 1,
  retryable: false,
  ...overrides,
})

describe("reattachQuickIngestSession", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgStream.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("reattaches active direct jobs into a processing snapshot", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "processing",
      },
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("processing")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 77,
        status: "processing",
      }),
    ])
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/77",
        method: "GET",
        returnResponse: true,
        preferDirect: true,
      })
    )
  })

  it("retries a thrown status read and preserves direct transport", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(new Error("network timeout"))
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    for (const [request] of mocks.bgRequest.mock.calls) {
      expect(request).toEqual(expect.objectContaining({ preferDirect: true }))
    }
  })

  it("retries a thrown transient numeric status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("service unavailable"), { status: 503 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("retries a thrown status-zero transport failure", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("network unavailable"), { status: 0 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("does not retry a thrown permanent numeric status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockRejectedValueOnce(
        Object.assign(new Error("unauthorized"), { status: 401 })
      )
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("retries an HTTP 503 status read and returns the completed job", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        error: "service unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-77" },
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "batch-1",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("completed")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    for (const [request] of mocks.bgRequest.mock.calls) {
      expect(request).toEqual(expect.objectContaining({ preferDirect: true }))
    }
  })

  it("retries a resolved status-zero transport failure", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: 0,
        error: "network unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "processing",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("does not retry a string HTTP status", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: false,
        status: "503",
        error: "service unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("does not retry an ok response carrying a transient status code", async () => {
    vi.useFakeTimers()
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 503,
        data: {
          status: "processing",
        },
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
        },
      })

    const pendingSnapshot = reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    await vi.runAllTimersAsync()
    const snapshot = await pendingSnapshot

    expect(snapshot.lifecycle).toBe("processing")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("prefers a polling run snapshot over per-job fan-out when runId is present", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reattach-1",
        status: "running",
        counts: { total: 1, running: 1 },
        version: 2,
        collection_id: null,
        batch_ids: ["batch-1"],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reattach-1",
        version: 2,
        items: [
          {
            occurrence_id: "occ-reattach-1",
            ordinal: 1,
            input_kind: "direct_url",
            source_url: "https://server.example/authoritative",
            normalized_source_id: "url:authoritative",
            source_kind: "video",
            display_metadata: { title: "Authoritative" },
            action: "ingest",
            state: "running",
            outcome: null,
            progress_percent: 40,
            progress_message: "Downloading",
            job_id: 77,
            batch_id: "batch-1",
            media_id: null,
            planned_collection_item_id: null,
            attempt: 4,
            retryable: false,
          },
        ],
        next_cursor: null,
      })

    const snapshot = await reattachQuickIngestSession(
      {
        mode: "extension-runtime",
        runId: "run-reattach-1",
        jobIds: [999],
        startedAt: Date.now(),
      } as any,
      { transportPreference: "poll" },
    )

    expect(snapshot.lifecycle).toBe("processing")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 77,
        sourceItemId: "occ-reattach-1",
        status: "running",
        lifecycleState: "running",
        terminalOutcome: null,
        progressPercent: 40,
        progressMessage: "Downloading",
        retryable: false,
        attempt: 4,
      }),
    ])
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/media/ingest/jobs/999" }),
    )
  })

  it("keeps the authoritative poll when the SSE event boundary is unknown", async () => {
    const summary = {
      contract_version: 2,
      run_id: "run-sse-progress",
      status: "running",
      counts: { total: 1, running: 1 },
      version: 2,
      collection_id: null,
      batch_ids: ["batch-sse"],
      created_at: "2026-07-13T00:00:00Z",
      updated_at: "2026-07-13T00:00:01Z",
      expires_at: "2026-07-20T00:00:00Z",
    }
    mocks.bgRequest
      .mockResolvedValueOnce(summary)
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-sse-progress",
        version: 2,
        items: [
          {
            occurrence_id: "occ-sse-progress",
            ordinal: 1,
            input_kind: "direct_url",
            source_url: "https://server.example/authoritative",
            normalized_source_id: "url:authoritative",
            source_kind: "video",
            display_metadata: { title: "Authoritative" },
            action: "ingest",
            state: "running",
            outcome: null,
            progress_percent: 40,
            progress_message: "Downloading",
            job_id: 77,
            batch_id: "batch-sse",
            media_id: null,
            planned_collection_item_id: null,
            attempt: 1,
            retryable: false,
          },
        ],
        next_cursor: null,
      })
    let emitted = 0
    mocks.bgStream.mockImplementation(async function* () {
      emitted += 1
      yield JSON.stringify(summary)
      emitted += 1
      yield JSON.stringify({
        event_id: 3,
        run_id: "run-sse-progress",
        occurrence_id: "occ-sse-progress",
        job_id: 77,
        batch_id: "batch-sse",
        event_type: "terminal",
        state: "terminal",
        outcome: "completed",
        progress_percent: 100,
        progress_message: "Done",
        occurred_at: "2026-07-13T00:00:03Z",
      })
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      runId: "run-sse-progress",
      jobIds: [77],
    })

    expect(emitted).toBe(0)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(snapshot).toMatchObject({
      lifecycle: "processing",
      jobs: [
        expect.objectContaining({
          sourceItemId: "occ-sse-progress",
          status: "running",
        }),
      ],
    })
  })

  it("bounds unknown-cursor recovery to one authoritative poll for 500 retained events", async () => {
    const summary = {
      contract_version: 2,
      run_id: "run-unknown-cursor-bound",
      status: "running",
      counts: { total: 500, running: 500 },
      version: 2,
      collection_id: null,
      batch_ids: ["batch-bound"],
      created_at: "2026-07-13T00:00:00Z",
      updated_at: "2026-07-13T00:00:01Z",
      expires_at: "2026-07-20T00:00:00Z",
    }
    const items = Array.from({ length: 500 }, (_, index) =>
      runItemResponse(`occ-bound-${index + 1}`, "running", {
        ordinal: index + 1,
        job_id: index + 1,
        batch_id: "batch-bound",
      })
    )
    mocks.bgRequest.mockImplementation(
      ({ path }: { path: string }) =>
        path.includes("/items")
          ? Promise.resolve({
              contract_version: 2,
              run_id: "run-unknown-cursor-bound",
              version: 2,
              items,
              next_cursor: null,
            })
          : Promise.resolve(summary)
    )
    mocks.bgStream.mockImplementation(async function* () {
      for (let index = 0; index < items.length; index += 1) {
        yield JSON.stringify({
          event_id: index + 1,
          run_id: "run-unknown-cursor-bound",
          occurrence_id: `occ-bound-${index + 1}`,
          job_id: index + 1,
          batch_id: "batch-bound",
          event_type: "progress",
          state: "running",
          outcome: null,
          progress_percent: 40,
          progress_message: "Downloading",
          occurred_at: "2026-07-13T00:00:02Z",
        })
      }
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      runId: "run-unknown-cursor-bound",
      jobIds: items.map((_, index) => index + 1),
    })

    expect(snapshot.lifecycle).toBe("processing")
    expect(snapshot.jobs).toHaveLength(500)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(mocks.bgStream).not.toHaveBeenCalled()
  })

  it("represents terminal run occurrences that never created jobs", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-terminal-no-job",
        status: "completed",
        counts: { total: 1, completed: 1 },
        version: 3,
        collection_id: null,
        batch_ids: [],
        created_at: "2026-07-13T00:00:00Z",
        updated_at: "2026-07-13T00:00:01Z",
        expires_at: "2026-07-20T00:00:00Z",
      })
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-terminal-no-job",
        version: 3,
        items: [
          {
            occurrence_id: "occ-included-existing",
            ordinal: 1,
            input_kind: "materialized_playlist_item",
            source_url: null,
            normalized_source_id: "youtube:video:existing",
            source_kind: "youtube_video",
            display_metadata: { title: "Existing item" },
            action: "include_existing",
            state: "terminal",
            outcome: "included_existing",
            progress_percent: 100,
            progress_message: null,
            job_id: null,
            batch_id: null,
            media_id: 42,
            planned_collection_item_id: null,
            attempt: 1,
            retryable: false,
          },
        ],
        next_cursor: null,
      })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      runId: "run-terminal-no-job",
      jobIds: [],
    })

    expect(snapshot.lifecycle).toBe("completed")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: null,
        sourceItemId: "occ-included-existing",
        status: "completed",
        result: expect.objectContaining({
          media_id: 42,
          outcome: "included_existing",
        }),
      }),
    ])
  })

  it("cancels and repolls authoritative staged occurrences after post-create reload", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(runSummaryResponse("running", 1))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 1,
        items: [runItemResponse("occ-staged-1", "staged")],
        next_cursor: null,
      })
      .mockResolvedValueOnce(runSummaryResponse("cancelled", 2))
      .mockResolvedValueOnce(runSummaryResponse("cancelled", 2))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 2,
        items: [
          runItemResponse("occ-staged-1", "terminal", {
            outcome: "cancelled",
            progress_percent: 100,
          }),
        ],
        next_cursor: null,
      })

    const snapshot = await reattachQuickIngestSession(
      {
        mode: "webui-direct",
        submissionState: "run_created",
        runId: "run-reload-submission",
        submittedItemIds: ["occ-staged-1"],
      } as any,
      { transportPreference: "poll" }
    )

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-reload-submission/cancel",
        body: {
          occurrence_ids: ["occ-staged-1"],
          reason: "submission_interrupted",
        },
      })
    )
    expect(snapshot).toMatchObject({
      lifecycle: "cancelled",
      jobs: [
        expect.objectContaining({
          sourceItemId: "occ-staged-1",
          status: "cancelled",
        }),
      ],
    })
  })

  it("retries authoritative unsent cleanup from a persisted cleanup-required phase", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(runSummaryResponse("running", 1))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 1,
        items: [runItemResponse("occ-cleanup-retry", "submit_pending")],
        next_cursor: null,
      })
      .mockResolvedValueOnce(runSummaryResponse("cancelled", 2))
      .mockResolvedValueOnce(runSummaryResponse("cancelled", 2))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 2,
        items: [
          runItemResponse("occ-cleanup-retry", "terminal", {
            outcome: "cancelled",
            progress_percent: 100,
          }),
        ],
        next_cursor: null,
      })

    const snapshot = await reattachQuickIngestSession(
      {
        mode: "extension-runtime",
        submissionState: "cleanup_required",
        runId: "run-reload-submission",
        submissionOccurrenceIds: ["occ-cleanup-retry"],
      } as any,
      { transportPreference: "poll" }
    )

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/media/ingest/runs/run-reload-submission/cancel",
        body: {
          occurrence_ids: ["occ-cleanup-retry"],
          reason: "submission_interrupted",
        },
      })
    )
    expect(snapshot).toMatchObject({
      lifecycle: "cancelled",
      jobs: [
        expect.objectContaining({
          sourceItemId: "occ-cleanup-retry",
          status: "cancelled",
        }),
      ],
    })
  })

  it("cancels only unsent staged occurrences while accepted jobs keep running", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(runSummaryResponse("running", 2))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 2,
        items: [
          runItemResponse("occ-accepted-1", "running"),
          runItemResponse("occ-unsent-2", "submit_pending"),
        ],
        next_cursor: null,
      })
      .mockResolvedValueOnce(runSummaryResponse("running", 3))
      .mockResolvedValueOnce(runSummaryResponse("running", 3))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 3,
        items: [
          runItemResponse("occ-accepted-1", "running"),
          runItemResponse("occ-unsent-2", "terminal", {
            outcome: "cancelled",
            progress_percent: 100,
          }),
        ],
        next_cursor: null,
      })

    const snapshot = await reattachQuickIngestSession(
      {
        mode: "webui-direct",
        submissionState: "submitting",
        runId: "run-reload-submission",
        jobIds: [77],
        submittedItemIds: ["occ-accepted-1", "occ-unsent-2"],
        jobIdToItemId: { "77": "occ-accepted-1" },
      } as any,
      { transportPreference: "poll" }
    )

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        body: {
          occurrence_ids: ["occ-unsent-2"],
          reason: "submission_interrupted",
        },
      })
    )
    expect(snapshot).toMatchObject({
      lifecycle: "processing",
      jobs: [
        expect.objectContaining({
          sourceItemId: "occ-accepted-1",
          status: "running",
        }),
        expect.objectContaining({
          sourceItemId: "occ-unsent-2",
          status: "cancelled",
        }),
      ],
    })
  })

  it("keeps accepted work retryable when interrupted-submission cancellation fails", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(runSummaryResponse("running", 2))
      .mockResolvedValueOnce({
        contract_version: 2,
        run_id: "run-reload-submission",
        version: 2,
        items: [
          runItemResponse("occ-accepted-1", "running"),
          runItemResponse("occ-unsent-2", "awaiting_upload", {
            input_kind: "file_stub",
            source_url: null,
          }),
        ],
        next_cursor: null,
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("cleanup unavailable"), { status: 503 })
      )

    const snapshot = await reattachQuickIngestSession(
      {
        mode: "webui-direct",
        submissionState: "submitting",
        runId: "run-reload-submission",
        jobIds: [77],
        submittedItemIds: ["occ-accepted-1", "occ-unsent-2"],
        jobIdToItemId: { "77": "occ-accepted-1" },
      } as any,
      { transportPreference: "poll" }
    )

    expect(snapshot).toMatchObject({
      lifecycle: "processing",
      jobs: [
        expect.objectContaining({
          sourceItemId: "occ-accepted-1",
          status: "status_unavailable",
          lifecycleState: "status_unavailable",
          retryable: true,
        }),
      ],
    })
    expect(snapshot.errorMessage).toMatch(/temporar|unavailable|try again|retry/i)
  })

  it("marks a persisted processing session as interrupted when reattachment cannot prove live progress", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: false,
      status: 404,
      error: "not found",
    })

    const result = await reattachQuickIngestSession({
      mode: "webui-direct",
      batchId: "missing",
      jobIds: [77],
      startedAt: Date.now()
    })

    expect(result.lifecycle).toBe("interrupted")
    expect(result.errorMessage).toMatch(/could not reconnect/i)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it.each([401, 403])(
    "does not retry permanent HTTP %i status reads",
    async (status) => {
      mocks.bgRequest.mockResolvedValue({
        ok: false,
        status,
        error: "permanent failure",
      })

      const result = await reattachQuickIngestSession({
        mode: "webui-direct",
        jobIds: [77],
        startedAt: Date.now(),
      })

      expect(result.lifecycle).toBe("interrupted")
      expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    }
  )

  it.each([
    [429, /too many|rate|try again/i],
    [503, /unavailable|try again/i],
    [undefined, /network|connect|unavailable/i],
  ])(
    "returns canonical status-unavailable evidence when run reattachment is retryable (%s)",
    async (status, expectedMessage) => {
      mocks.bgRequest.mockRejectedValue(
        Object.assign(
          new Error(
            status == null
              ? "network disconnected"
              : "temporarily unavailable"
          ),
          status == null ? {} : { status }
        )
      )

      const result = await reattachQuickIngestSession({
        mode: "webui-direct",
        runId: `run-retryable-${status}`,
        jobIds: [77],
        submittedItemIds: ["occ-retryable"],
      })

      expect(result).toMatchObject({
        lifecycle: "processing",
        jobs: [
          expect.objectContaining({
            jobId: 77,
            sourceItemId: "occ-retryable",
            status: "status_unavailable",
            lifecycleState: "status_unavailable",
            progressMessage: expect.stringMatching(expectedMessage),
            retryable: true,
          }),
        ],
      })
      expect(result.errorMessage).toMatch(expectedMessage)
      expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    }
  )

  it("does not retry a successful response with no job status", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {},
    })

    const result = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      startedAt: Date.now(),
    })

    expect(result.lifecycle).toBe("interrupted")
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it.each([408, 429])(
    "stops after three transient HTTP %i status-read attempts",
    async (status) => {
      vi.useFakeTimers()
      mocks.bgRequest.mockResolvedValue({
        ok: false,
        status,
        error: "transient failure",
      })

      const pendingSnapshot = reattachQuickIngestSession({
        mode: "webui-direct",
        jobIds: [77],
        startedAt: Date.now(),
      })

      await vi.runAllTimersAsync()
      const result = await pendingSnapshot

      expect(result.lifecycle).toBe("interrupted")
      expect(mocks.bgRequest).toHaveBeenCalledTimes(3)
    }
  )

  it("reports run authorization failure distinctly without legacy job fallback", async () => {
    mocks.bgRequest.mockRejectedValue(
      Object.assign(new Error("unauthorized"), { status: 401 })
    )

    const result = await reattachQuickIngestSession({
      mode: "webui-direct",
      runId: "run-auth-failure",
      jobIds: [77],
      submittedItemIds: ["occ-auth-failure"],
    })

    expect(result.lifecycle).toBe("interrupted")
    expect(result.errorMessage).toMatch(/authoriz|sign in|authentication/i)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("maps reattached jobs back to submitted queue item identities", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-88" },
        },
      })
      .mockResolvedValueOnce({
        ok: true,
        data: {
          status: "completed",
          result: { media_id: "media-99" },
        },
      })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [88, 99],
      submittedItemIds: ["queued-url-1", "queued-file-1"],
      jobIdToItemId: {
        "99": "queued-file-1",
      },
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("completed")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 88,
        sourceItemId: "queued-url-1",
      }),
      expect.objectContaining({
        jobId: 99,
        sourceItemId: "queued-file-1",
      }),
    ])
  })

  it("treats completed jobs with error payloads as partial failures during reattach", async () => {
    mocks.bgRequest.mockResolvedValue({
      ok: true,
      data: {
        status: "completed",
        result: {
          status: "Error",
          error: "File preparation/download failed: Port not allowed: 3000"
        }
      },
    })

    const snapshot = await reattachQuickIngestSession({
      mode: "webui-direct",
      jobIds: [77],
      submittedItemIds: ["queued-url-1"],
      startedAt: Date.now(),
    })

    expect(snapshot.lifecycle).toBe("partial_failure")
    expect(snapshot.jobs).toEqual([
      expect.objectContaining({
        jobId: 77,
        status: "completed",
        error: "File preparation/download failed: Port not allowed: 3000",
        sourceItemId: "queued-url-1",
      }),
    ])
  })
})
