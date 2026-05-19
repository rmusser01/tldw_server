import {
  completedIngestJobIndicatesFailure,
  extractCompletedIngestJobError,
  extractCompletedIngestJobTerminalData,
} from "@/services/tldw/ingest-job-results"

export const TERMINAL_INGEST_JOB_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])

export type IngestJobStatusResponse = {
  ok: boolean
  status?: number
  data?: any
  error?: string
}

export type IngestJobTrackerItem<TMeta> = {
  jobId: number
  batchId: string
  meta: TMeta
}

export type IngestJobFailureDetails = {
  status: string
  data?: any
  error?: string
  response?: IngestJobStatusResponse
}

export type PollSingleIngestJobResult = {
  terminalStatus: "completed" | "failed" | "cancelled" | "timeout"
  data?: any
  error?: string
  response?: IngestJobStatusResponse
}

const normalizeBatchId = (value: unknown): string => String(value || "").trim()

export const extractIngestJobIds = (data: any): number[] => {
  const jobs = Array.isArray(data?.jobs) ? data.jobs : []
  const ids: number[] = []
  for (const item of jobs) {
    const id = Number(item?.id)
    if (Number.isFinite(id) && id > 0) {
      ids.push(Math.trunc(id))
    }
  }
  return ids
}

export const createIngestJobsTracker = <TMeta>() => {
  const tracked = new Map<number, IngestJobTrackerItem<TMeta>>()
  const cancelledBatches = new Set<string>()

  const trackJobs = (batchId: string, jobIds: number[], meta: TMeta): number[] => {
    const normalizedBatchId = normalizeBatchId(batchId)
    if (!normalizedBatchId || !Array.isArray(jobIds) || jobIds.length === 0) {
      throw new Error("Ingest job submission returned no job IDs.")
    }

    const accepted: number[] = []
    for (const rawJobId of jobIds) {
      const jobId = Number(rawJobId)
      if (!Number.isFinite(jobId) || jobId <= 0) continue
      const normalizedJobId = Math.trunc(jobId)
      tracked.set(normalizedJobId, {
        jobId: normalizedJobId,
        batchId: normalizedBatchId,
        meta
      })
      accepted.push(normalizedJobId)
    }

    if (accepted.length === 0) {
      throw new Error("Ingest job submission returned no job IDs.")
    }
    return accepted
  }

  const trackSubmit = (submitData: any, meta: TMeta): number[] => {
    const batchId = normalizeBatchId(submitData?.batch_id)
    const jobIds = extractIngestJobIds(submitData)
    return trackJobs(batchId, jobIds, meta)
  }

  return {
    trackJobs,
    trackSubmit,
    clear: () => {
      tracked.clear()
      cancelledBatches.clear()
    },
    clearJob: (jobId: number) => {
      tracked.delete(Math.trunc(Number(jobId)))
    },
    hasItems: () => tracked.size > 0,
    getItem: (jobId: number) => tracked.get(Math.trunc(Number(jobId))),
    getItems: () => Array.from(tracked.values()),
    getJobIds: () => Array.from(tracked.keys()),
    getBatchIds: () => Array.from(new Set(Array.from(tracked.values()).map((item) => item.batchId))),
    cancelTrackedBatches: async (
      cancelBatch: (batchId: string) => Promise<void> | void
    ): Promise<void> => {
      const pending = new Set<string>()
      for (const item of tracked.values()) {
        pending.add(item.batchId)
      }
      for (const batchId of pending) {
        if (!batchId || cancelledBatches.has(batchId)) continue
        cancelledBatches.add(batchId)
        await cancelBatch(batchId)
      }
    }
  }
}

type PollTrackedIngestJobsOptions<TMeta, TResult> = {
  tracker: ReturnType<typeof createIngestJobsTracker<TMeta>>
  fetchJob: (jobId: number) => Promise<IngestJobStatusResponse | undefined>
  timeoutMs: number
  pollIntervalMs?: number
  isCancelled: () => boolean
  onCancel: () => Promise<void> | void
  onPendingJobIds?: (jobIds: number[]) => void
  mapCompleted: (
    item: IngestJobTrackerItem<TMeta>,
    data: any,
    response?: IngestJobStatusResponse
  ) => TResult
  mapCancelled: (
    item: IngestJobTrackerItem<TMeta>,
    data?: any,
    response?: IngestJobStatusResponse
  ) => TResult
  mapFailure: (item: IngestJobTrackerItem<TMeta>, details: IngestJobFailureDetails) => TResult
  mapRequestError?: (
    item: IngestJobTrackerItem<TMeta>,
    response: IngestJobStatusResponse | undefined
  ) => TResult | undefined
}

export const pollTrackedIngestJobs = async <TMeta, TResult>(
  options: PollTrackedIngestJobsOptions<TMeta, TResult>
): Promise<TResult[]> => {
  const pollIntervalMs = Math.max(1, Number(options.pollIntervalMs || 1200))
  const deadline = Date.now() + Math.max(10_000, Number(options.timeoutMs || 0))
  const unresolved = new Map<number, IngestJobTrackerItem<TMeta>>(
    options.tracker.getItems().map((item) => [item.jobId, item])
  )
  const results: TResult[] = []

  while (unresolved.size > 0) {
    if (options.isCancelled()) {
      await options.onCancel()
      for (const item of unresolved.values()) {
        results.push(options.mapCancelled(item))
        options.tracker.clearJob(item.jobId)
      }
      unresolved.clear()
      break
    }

    let observedTerminal = false
    for (const [jobId, item] of Array.from(unresolved.entries())) {
      const response = await options.fetchJob(jobId)
      if (!response?.ok) {
        const mapped = options.mapRequestError?.(item, response)
        if (typeof mapped !== "undefined") {
          observedTerminal = true
          unresolved.delete(jobId)
          options.tracker.clearJob(jobId)
          results.push(mapped)
        }
        continue
      }

      const status = String(response.data?.status || "").toLowerCase()
      if (!TERMINAL_INGEST_JOB_STATUSES.has(status)) {
        continue
      }

      observedTerminal = true
      unresolved.delete(jobId)
      options.tracker.clearJob(jobId)

      if (status === "completed") {
        if (completedIngestJobIndicatesFailure(response.data)) {
          results.push(
            options.mapFailure(item, {
              status: "failed",
              data: extractCompletedIngestJobTerminalData(response.data),
              error:
                extractCompletedIngestJobError(response.data) ||
                "Ingest completed with an error result.",
              response
            })
          )
        } else {
          results.push(
            options.mapCompleted(
              item,
              extractCompletedIngestJobTerminalData(response.data),
              response
            )
          )
        }
        continue
      }
      if (status === "cancelled") {
        results.push(options.mapCancelled(item, response.data, response))
        continue
      }

      const errorText =
        String(response.data?.error_message || "").trim() ||
        String(response.data?.cancellation_reason || "").trim() ||
        `Ingest ${status || "failed"}`
      results.push(
        options.mapFailure(item, {
          status,
          data: response.data,
          error: errorText,
          response
        })
      )
    }

    options.onPendingJobIds?.(Array.from(unresolved.keys()))
    if (unresolved.size === 0) {
      break
    }

    if (Date.now() >= deadline) {
      for (const item of unresolved.values()) {
        results.push(
          options.mapFailure(item, {
            status: "timeout",
            error: "Timed out while waiting for media ingest jobs."
          })
        )
        options.tracker.clearJob(item.jobId)
      }
      unresolved.clear()
      break
    }

    if (!observedTerminal) {
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))
    }
  }

  return results
}

type PollSingleIngestJobOptions = {
  jobId: number
  fetchJob: (jobId: number) => Promise<IngestJobStatusResponse | undefined>
  timeoutMs: number
  pollIntervalMs?: number
  isCancelled: () => boolean
  onCancel: () => Promise<void> | void
  onRequestError?: (
    response: IngestJobStatusResponse | undefined
  ) =>
    | {
        terminalStatus: "failed" | "cancelled"
        error?: string
      }
    | undefined
}

export const pollSingleIngestJob = async (
  options: PollSingleIngestJobOptions
): Promise<PollSingleIngestJobResult> => {
  const pollIntervalMs = Math.max(1, Number(options.pollIntervalMs || 1200))
  const deadline = Date.now() + Math.max(10_000, Number(options.timeoutMs || 0))

  while (Date.now() < deadline) {
    if (options.isCancelled()) {
      await options.onCancel()
      return {
        terminalStatus: "cancelled",
        error: "Cancelled by user."
      }
    }

    const response = await options.fetchJob(options.jobId)
    if (!response?.ok) {
      const mapped = options.onRequestError?.(response)
      if (mapped) {
        return {
          terminalStatus: mapped.terminalStatus,
          error: mapped.error,
          response
        }
      }
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))
      continue
    }

    const status = String(response.data?.status || "").toLowerCase()
    if (!TERMINAL_INGEST_JOB_STATUSES.has(status)) {
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))
      continue
    }
    if (status === "completed") {
      if (completedIngestJobIndicatesFailure(response.data)) {
        return {
          terminalStatus: "failed",
          data: extractCompletedIngestJobTerminalData(response.data),
          error:
            extractCompletedIngestJobError(response.data) ||
            "Ingest completed with an error result.",
          response
        }
      }
      return {
        terminalStatus: "completed",
        data: extractCompletedIngestJobTerminalData(response.data),
        response
      }
    }
    if (status === "cancelled") {
      return {
        terminalStatus: "cancelled",
        data: response.data,
        error: "Cancelled by user.",
        response
      }
    }
    const errorText =
      String(response.data?.error_message || "").trim() ||
      String(response.data?.cancellation_reason || "").trim() ||
      `Ingest ${status || "failed"}`
    return {
      terminalStatus: "failed",
      data: response.data,
      error: errorText,
      response
    }
  }

  return {
    terminalStatus: "timeout",
    error: "Timed out while waiting for media ingest jobs."
  }
}
