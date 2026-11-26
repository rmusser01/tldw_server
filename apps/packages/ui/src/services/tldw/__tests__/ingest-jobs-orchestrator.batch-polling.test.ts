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

type JobMeta = { id: string }
type PollResult =
  | { kind: "completed"; data?: unknown }
  | { kind: "failed"; status?: string; error?: string; id?: string }
  | { kind: "cancelled" }
type PollOptions = Parameters<typeof pollTrackedIngestJobs<JobMeta, PollResult>>[0]

const runPoll = async (options: {
  batchResponses?: Array<IngestJobStatusResponse | undefined>
} & Partial<Pick<PollOptions, "fetchJob" | "fetchJobs">>) => {
  const tracker = createIngestJobsTracker<JobMeta>()
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

  const results = await pollTrackedIngestJobs<JobMeta, PollResult>({
    tracker,
    fetchJobs,
    fetchJob: options.fetchJob,
    timeoutMs: 1000,
    pollIntervalMs: 1,
    isCancelled: () => false,
    onCancel: async () => {},
    mapCompleted: (_item, data) => ({ kind: "completed", data }),
    mapFailure: (_item, details) => ({
      kind: "failed",
      status: details.status,
      error: details.error
    }),
    mapCancelled: () => ({ kind: "cancelled" })
  })

  return { tracker, calls, fetchJobs, results }
}

describe("pollTrackedIngestJobs batch polling", () => {
  it("issues exactly one batch request per cycle for all unresolved jobs", async () => {
    const { calls, results } = await runPoll({
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
    expect(calls[0]).toEqual({ batchId: "batch-1", jobIds: [1, 2, 3, 4] })
    expect(calls[1]).toEqual({ batchId: "batch-1", jobIds: [2] })

    expect(results).toHaveLength(4)
    expect(results.filter((r) => r.kind === "completed")).toHaveLength(3)
    expect(results).toContainEqual(
      expect.objectContaining({ kind: "failed", status: "failed", error: "boom" })
    )
  })

  it("treats a job missing from the batch response like a per-job 404 (stays pending, never falls back to fetchJob)", async () => {
    const fetchJob = vi.fn(async () => ({ ok: true, data: { status: "completed" } }))
    const { calls, results } = await runPoll({
      // Job 2 is missing from the first batch response; exactly like the
      // per-job 404 path it must stay pending and be re-requested next
      // cycle instead of being failed or falling back to fetchJob.
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

    expect(fetchJob).not.toHaveBeenCalled()
    // Missing job 2 stayed pending: the second cycle re-requests only it.
    expect(calls[1]).toEqual({ batchId: "batch-1", jobIds: [2] })
    expect(results).toHaveLength(4)
    expect(results.every((r) => r.kind === "completed")).toBe(true)
  })

  it("propagates a failed batch response to every tracked job of that batch (auth-error mapping preserved)", async () => {
    const tracker = createIngestJobsTracker<JobMeta>()
    tracker.trackJobs("batch-x", [7, 8], { id: "a" })

    const authError = { ok: false, status: 401, error: "Not authenticated." }
    const results = await pollTrackedIngestJobs<JobMeta, PollResult>({
      tracker,
      fetchJobs: vi.fn(async () => authError),
      timeoutMs: 10,
      pollIntervalMs: 1,
      isCancelled: () => false,
      onCancel: async () => {},
      mapCompleted: () => ({ kind: "completed" }),
      mapCancelled: () => ({ kind: "cancelled" }),
      mapFailure: (_item, details) => ({
        kind: "failed",
        status: details.status
      }),
      mapRequestError: (item, response) =>
        Number(response?.status || 0) === 401
          ? { kind: "failed", status: "auth", id: item.meta.id }
          : undefined
    })

    expect(results).toHaveLength(2)
    expect(results.every((r) => r.kind === "failed" && r.status === "auth")).toBe(true)
  })

  it("still supports the legacy per-job fetchJob path when fetchJobs is absent", async () => {
    const tracker = createIngestJobsTracker<JobMeta>()
    tracker.trackJobs("batch-legacy", [5, 6], { id: "a" })

    const perJobCalls: number[] = []
    const results = await pollTrackedIngestJobs<JobMeta, PollResult>({
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
      mapCompleted: (_item, data) => ({ kind: "completed", data }),
      mapFailure: () => ({ kind: "failed" }),
      mapCancelled: () => ({ kind: "cancelled" })
    })

    expect(perJobCalls).toEqual([5, 6])
    expect(results).toHaveLength(2)
    expect(results.every((r) => r.kind === "completed")).toBe(true)
  })
})
