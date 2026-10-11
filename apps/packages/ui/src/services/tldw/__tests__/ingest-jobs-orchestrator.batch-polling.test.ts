import { describe, expect, it, vi } from "vitest"

import {
  createIngestJobsTracker,
  pollTrackedIngestJobs,
  type IngestJobStatusResponse
} from "@/services/tldw/ingest-jobs-orchestrator"

type BatchCall = {
  batchId: string
  jobIds: number[]
}

const runPoll = async (options: {
  batchResponses?: Array<IngestJobStatusResponse | undefined>
  fetchJob?: ReturnType<typeof vi.fn>
  fetchJobs?: ReturnType<typeof vi.fn>
}) => {
  const tracker = createIngestJobsTracker<{ id: string }>()
  tracker.trackJobs(
    "batch-1",
    [1, 2, 3],
    { id: "a" }
  )
  tracker.trackJobs(
    "batch-1",
    [4],
    { id: "b" }
  )

  const calls: BatchCall[] = []
  const fetchJobs =
    options.fetchJobs ??
    vi.fn(async (batchId: string, jobIds: number[]) => {
      calls.push({ batchId, jobIds })
      const response = options.batchResponses?.[calls.length - 1]
      return response ?? { ok: true, data: { jobs: [] } }
    })

  const results = await pollTrackedIngestJobs({
    tracker,
    fetchJobs,
    fetchJob: options.fetchJob,
    timeoutMs: 1000,
    pollIntervalMs: 1,
    isCancelled: () => false,
    onCancel: async () => {},
    mapCompleted: (_item, data) => ({ kind: "completed" as const, data }),
    mapFailure: (_item, details) => ({
      kind: "failed" as const,
      status: details.status,
      error: details.error
    }),
    mapCancelled: () => ({ kind: "cancelled" as const })
  })

  return { tracker, calls, fetchJobs, results }
}

describe("pollTrackedIngestJobs batch polling", () => {
  it("resolves an idempotently replayed job retained in its original batch through its scoped per-job fetch", async () => {
    const fetchJob = vi.fn(async (id: number) => ({ ok: true, data: {
      id, batch_id: "original-batch", status: "completed", result: { media_id: 22 }
    } }))
    const { calls, results, tracker } = await runPoll({
      batchResponses: [{ ok: true, data: { jobs: [
        { id: 1, status: "completed", result: { media_id: 11 } },
        { id: 3, status: "completed", result: { media_id: 33 } },
        { id: 4, status: "completed", result: { media_id: 44 } },
      ] } }], fetchJob,
    })
    expect(fetchJob).toHaveBeenCalledExactlyOnceWith(2)
    expect(calls).toEqual([{ batchId: "batch-1", jobIds: [1, 2, 3, 4] }])
    expect(results).toHaveLength(4)
    expect(results.every(result => result.kind === "completed")).toBe(true)
    expect(results).toContainEqual({ kind: "completed", data: { media_id: 22 } })
    expect(tracker.getItems()).toEqual([])
  })

  it("issues exactly one batch request per cycle for all unresolved jobs", async () => {
    const fetchJob = vi.fn(async () => ({ ok: true, data: { status: "completed" } }))
    const { calls, results } = await runPoll({
      fetchJob,
      batchResponses: [
        {
          ok: true,
          data: {
            jobs: [
              { id: 1, status: "completed", result: { media_id: 11 } },
              { id: 2, status: "running" },
              { id: 3, status: "failed", error_message: "boom" },
              { id: 4, status: "completed", result: { media_id: 44 } }
            ]
          }
        },
        {
          ok: true,
          data: { jobs: [{ id: 2, status: "completed", result: { media_id: 22 } }] }
        }
      ]
    })

    // 4 unresolved jobs across two submits of the same batch, but each cycle
    // must produce a single batched request.
    expect(calls).toHaveLength(2)
    expect(fetchJob).not.toHaveBeenCalled()
    expect(calls[0]).toEqual({ batchId: "batch-1", jobIds: [1, 2, 3, 4] })
    expect(calls[1]).toEqual({ batchId: "batch-1", jobIds: [2] })

    expect(results).toHaveLength(4)
    expect(results.filter((r) => r.kind === "completed")).toHaveLength(3)
    expect(results).toContainEqual(
      expect.objectContaining({ kind: "failed", status: "failed", error: "boom" })
    )
  })

  it("keeps a missing job pending when its scoped per-job fallback returns 404", async () => {
    const fetchJob = vi.fn(async () => ({ ok: false, status: 404 }))
    const { calls, results } = await runPoll({
      // Both lookups omit job 2, so the next cycle still tracks it.
      batchResponses: [
        {
          ok: true,
          data: {
            jobs: [
              { id: 1, status: "completed", result: { media_id: 11 } },
              { id: 3, status: "completed", result: { media_id: 33 } },
              { id: 4, status: "completed", result: { media_id: 44 } }
            ]
          }
        },
        {
          ok: true,
          data: { jobs: [{ id: 2, status: "completed", result: { media_id: 22 } }] }
        }
      ],
      fetchJob
    })

    expect(fetchJob).toHaveBeenCalledExactlyOnceWith(2)
    // Missing job 2 stayed pending: the second cycle re-requests only it.
    expect(calls[1]).toEqual({ batchId: "batch-1", jobIds: [2] })
    expect(results).toHaveLength(4)
    expect(results.every((r) => r.kind === "completed")).toBe(true)
  })

  it("propagates a failed batch response to every tracked job of that batch (auth-error mapping preserved)", async () => {
    const tracker = createIngestJobsTracker<{ id: string }>()
    tracker.trackJobs("batch-x", [7, 8], { id: "a" })

    const authError = { ok: false, status: 401, error: "Not authenticated." }
    const fetchJob = vi.fn()
    const results = await pollTrackedIngestJobs({
      tracker,
      fetchJobs: vi.fn(async () => authError),
      fetchJob,
      timeoutMs: 10,
      pollIntervalMs: 1,
      isCancelled: () => false,
      onCancel: async () => {},
      mapCompleted: () => ({ kind: "completed" as const }),
      mapCancelled: () => ({ kind: "cancelled" as const }),
      mapFailure: (_item, details) => ({
        kind: "failed" as const,
        status: details.status
      }),
      mapRequestError: (item, response) =>
        Number(response?.status || 0) === 401
          ? { kind: "failed" as const, status: "auth", id: item.meta.id }
          : undefined
    })

    expect(results).toHaveLength(2)
    expect(fetchJob).not.toHaveBeenCalled()
    expect(results.every((r) => (r as any).status === "auth")).toBe(true)
  })

  it("still supports the legacy per-job fetchJob path when fetchJobs is absent", async () => {
    const tracker = createIngestJobsTracker<{ id: string }>()
    tracker.trackJobs("batch-legacy", [5, 6], { id: "a" })

    const perJobCalls: number[] = []
    const results = await pollTrackedIngestJobs({
      tracker,
      fetchJob: async (jobId) => {
        perJobCalls.push(jobId)
        return {
          ok: true,
          data: { status: "completed", result: { media_id: jobId } }
        }
      },
      timeoutMs: 1000,
      pollIntervalMs: 1,
      isCancelled: () => false,
      onCancel: async () => {},
      mapCompleted: (_item, data) => ({ kind: "completed" as const, data }),
      mapFailure: () => ({ kind: "failed" as const }),
      mapCancelled: () => ({ kind: "cancelled" as const })
    })

    expect(perJobCalls).toEqual([5, 6])
    expect(results).toHaveLength(2)
    expect(results.every((r) => r.kind === "completed")).toBe(true)
  })
})
