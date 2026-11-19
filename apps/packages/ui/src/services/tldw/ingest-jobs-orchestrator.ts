import {
  completedIngestJobIndicatesFailure,
  extractCompletedIngestJobError,
  extractCompletedIngestJobTerminalData,
} from "@/services/tldw/ingest-job-results"

export const ACTIVE_INGEST_JOB_STATUSES = new Set([
  "pending",
  "queued",
  "running",
  "processing",
  "uploading",
  "analyzing",
  "storing"
])

export const getQuickIngestJobPollInterval = (
  startedAt: number,
  initialIntervalMs: number,
  statusUnavailable = false
): number =>
  statusUnavailable || Date.now() - startedAt >= 5 * 60_000 ? 10_000 : initialIntervalMs

export const INGEST_JOB_STATUS_INTERRUPTED =
  "Quick Ingest could not confirm the final status of one or more server jobs. Check Media before importing those items again."

export const isTransientIngestStatusRead = (status: unknown): boolean =>
  typeof status === "number" && Number.isFinite(status) &&
  (status === 0 || status === 408 || status === 429 || (status >= 500 && status < 600))

export const TERMINAL_INGEST_JOB_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])

export type IngestJobProgress = {
  status: string
  progressPercent?: number
  progressMessage?: string
}

export const readIngestJobProgress = (data?: {
  status?: unknown
  progress_percent?: unknown
  progress_message?: unknown
}): IngestJobProgress => {
  const percent = data?.progress_percent
  const message = data?.progress_message
  return {
    status: String(data?.status || "").trim().toLowerCase(),
    progressPercent: typeof percent === "number" && Number.isFinite(percent) && percent >= 0 && percent <= 100
      ? percent
      : undefined,
    progressMessage: typeof message === "string" ? message.trim() || undefined : undefined,
  }
}

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
  terminalStatus: "completed" | "failed" | "cancelled" | "timeout" | "interrupted"
  data?: any
  error?: string
  response?: IngestJobStatusResponse
}

const normalizeBatchId = (value: unknown): string => String(value || "").trim()

const formatIngestSubmitDetail = (item: unknown): string => {
  if (typeof item === "string") return item.trim()
  if (!item || typeof item !== "object") return ""

  const detail = item as { loc?: unknown; msg?: unknown }
  const msg = typeof detail.msg === "string" ? detail.msg.trim() : ""
  const loc = Array.isArray(detail.loc)
    ? detail.loc
        .map((part: unknown) => String(part).trim())
        .filter(Boolean)
        .join(".")
    : ""

  return loc && msg ? `${loc}: ${msg}` : msg
}

export const extractIngestJobSubmitError = (data: any): string | null => {
  const errors = Array.isArray(data?.errors)
    ? data.errors
        .map((item: unknown) => String(item || "").trim())
        .filter(Boolean)
    : []
  if (errors.length > 0) return errors.join("; ")

  const detail = data?.detail
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim()
  }
  if (Array.isArray(detail)) {
    const detailMessages = detail.map(formatIngestSubmitDetail).filter(Boolean)
    if (detailMessages.length > 0) return detailMessages.join("; ")
  }

  return null
}

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

export const requireSubmittedIngestJobs = (
  submitData: any
): { batchId: string; jobIds: number[] } => {
  const batchId = normalizeBatchId(submitData?.batch_id)
  const jobIds = extractIngestJobIds(submitData)
  if (!batchId || jobIds.length === 0) {
    throw new Error(
      extractIngestJobSubmitError(submitData) ||
        "Ingest job submission returned no job IDs."
    )
  }
  return { batchId, jobIds }
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
    const { batchId, jobIds } = requireSubmittedIngestJobs(submitData)
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
      // Batch cancels are independent per batch: fan them out together
      // instead of paying one round-trip per tracked batch (WP-38).
      await Promise.all(
        Array.from(pending).map(async (batchId) => {
          if (!batchId || cancelledBatches.has(batchId)) return
          cancelledBatches.add(batchId)
          await cancelBatch(batchId)
        })
      )
    }
  }
}

type LegacyPerJobFetch = {
  /**
   * Per-job status fetch (legacy N+1 path). At least one of `fetchJob` /
   * `fetchJobs` must be provided; `fetchJob` is ignored when `fetchJobs` is
   * present. Omitting both is a caller bug that previously spun silently to
   * the poll timeout (PR #3210 review follow-up).
   */
  fetchJob: (jobId: number) => Promise<IngestJobStatusResponse | undefined>
  fetchJobs?: never
}

type BatchedJobsFetch = {
  fetchJob?: LegacyPerJobFetch["fetchJob"]
  /**
   * Batched status fetch: ONE request per tracked batch per poll cycle. The
   * returned response is expected to carry the reconciled status of every
   * requested job under `data.jobs` (an array of job status objects with an
   * `id` field, matching `GET /api/v1/media/ingest/jobs?batch_id=...`).
   * Callers can opt in to individual reads for omitted jobs with `fetchMissingJob`.
   */
  fetchJobs: (
    batchId: string,
    jobIds: number[]
  ) => Promise<IngestJobStatusResponse | undefined>
}

type PollTrackedIngestJobsBase<TMeta, TResult> = {
  tracker: ReturnType<typeof createIngestJobsTracker<TMeta>>
  /**
   * Batched status fetch: ONE request per tracked batch per poll cycle. The
   * returned response is expected to carry the reconciled status of every
   * requested job under `data.jobs` (an array of job status objects with an
   * `id` field, matching `GET /api/v1/media/ingest/jobs?batch_id=...`).
   * Callers can opt in to individual reads for omitted jobs with `fetchMissingJob`.
   */
  fetchJobs?: (
    batchId: string,
    jobIds: number[]
  ) => Promise<IngestJobStatusResponse | undefined>
  fetchMissingJob?: LegacyPerJobFetch["fetchJob"]
  timeoutMs: number
  pollIntervalMs?: number | (() => number)
  signal?: AbortSignal
  isCancelled: () => boolean
  onCancel: () => Promise<void> | void
  onPendingJobIds?: (jobIds: number[]) => void
  onResult?: (result: TResult) => void
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

/**
 * At least one fetch strategy is required; batching takes priority. Omitting both is a
 * compile error instead of a silent spin to the poll timeout (PR #3210
 * review follow-up).
 */
type PollTrackedIngestJobsOptions<TMeta, TResult> = PollTrackedIngestJobsBase<TMeta, TResult> &
  (LegacyPerJobFetch | BatchedJobsFetch)

const waitForIngestPoll = (
  pollIntervalMs: number | (() => number) | undefined,
  signal?: AbortSignal
): Promise<void> => {
  const interval = typeof pollIntervalMs === "function" ? pollIntervalMs() : pollIntervalMs
  signal?.throwIfAborted()
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, Math.max(1, Number(interval || 1200)))
    signal?.addEventListener("abort", onAbort, { once: true })
  })
}

export const pollTrackedIngestJobs = async <TMeta, TResult>(
  options: PollTrackedIngestJobsOptions<TMeta, TResult>
): Promise<TResult[]> => {
  const deadline = Date.now() + Math.max(10_000, Number(options.timeoutMs || 0))
  const unresolved = new Map<number, IngestJobTrackerItem<TMeta>>(
    options.tracker.getItems().map((item) => [item.jobId, item])
  )
  const results: TResult[] = []
  const recordResult = (result: TResult) => {
    results.push(result)
    options.onResult?.(result)
  }

  const consumeJobResponse = (
    jobId: number,
    item: IngestJobTrackerItem<TMeta>,
    response: IngestJobStatusResponse | undefined
  ): boolean => {
    if (!response?.ok) {
      const mapped = options.mapRequestError?.(item, response)
      if (typeof mapped !== "undefined") {
        unresolved.delete(jobId)
        options.tracker.clearJob(jobId)
        recordResult(mapped)
        return true
      }
      return false
    }

    const { status } = readIngestJobProgress(response.data)
    if (!TERMINAL_INGEST_JOB_STATUSES.has(status)) {
      return false
    }

    unresolved.delete(jobId)
    options.tracker.clearJob(jobId)

    if (status === "completed") {
      if (completedIngestJobIndicatesFailure(response.data)) {
        recordResult(
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
        recordResult(
          options.mapCompleted(
            item,
            extractCompletedIngestJobTerminalData(response.data),
            response
          )
        )
      }
      return true
    }
    if (status === "cancelled") {
      recordResult(options.mapCancelled(item, response.data, response))
      return true
    }

    const errorText =
      String(response.data?.error_message || "").trim() ||
      String(response.data?.cancellation_reason || "").trim() ||
      `Ingest ${status || "failed"}`
    recordResult(
      options.mapFailure(item, {
        status,
        data: response.data,
        error: errorText,
        response
      })
    )
    return true
  }

  // Fetch the current status of every unresolved job with ONE request per
  // tracked batch (instead of one request per job) and reconcile each job
  // from the batch response.
  const consumeBatchResponses = async (): Promise<boolean> => {
    const jobsByBatch = new Map<string, number[]>()
    for (const [jobId, item] of unresolved.entries()) {
      const batchJobIds = jobsByBatch.get(item.batchId) ?? []
      batchJobIds.push(jobId)
      jobsByBatch.set(item.batchId, batchJobIds)
    }

    const responsesByJob = new Map<
      number,
      IngestJobStatusResponse | undefined
    >()
    // Batch status fetches are independent per batch: issue them together so
    // a poll round costs one round-trip, not one per tracked batch (WP-37).
    const batchEntries = Array.from(jobsByBatch.entries())
    const batchResponses = await Promise.all(
      batchEntries.map(([batchId, batchJobIds]) =>
        options.fetchJobs?.(batchId, batchJobIds)
      )
    )
    batchEntries.forEach(([, batchJobIds], index) => {
      const response = batchResponses[index]
      const jobs = response?.ok ? response.data?.jobs : null
      if (Array.isArray(jobs)) {
        for (const job of jobs) {
          const id = Number((job as any)?.id)
          if (!Number.isFinite(id) || !unresolved.has(Math.trunc(id))) continue
          responsesByJob.set(Math.trunc(id), {
            ok: true,
            status: response?.status,
            data: job,
            error: response?.error
          })
        }
      } else if (response || options.fetchJobs) {
        // Batch request itself failed: surface the same error response to
        // every tracked job of the batch, mirroring the per-job error path.
        for (const jobId of batchJobIds) {
          responsesByJob.set(jobId, response)
        }
      }
    })

    let observedTerminal = false
    for (const [jobId, item] of Array.from(unresolved.entries())) {
      const response = responsesByJob.has(jobId)
        ? responsesByJob.get(jobId)
        : options.fetchMissingJob
          ? await options.fetchMissingJob(jobId)
          : { ok: false, status: 404, error: "Ingest job missing from batch status response." }
      if (consumeJobResponse(jobId, item, response)) {
        observedTerminal = true
      }
    }
    return observedTerminal
  }

  while (unresolved.size > 0) {
    options.signal?.throwIfAborted()
    if (options.isCancelled()) {
      await options.onCancel()
      for (const item of unresolved.values()) {
        recordResult(options.mapCancelled(item))
        options.tracker.clearJob(item.jobId)
      }
      unresolved.clear()
      break
    }

    let observedTerminal = false
    if (options.fetchJobs) {
      observedTerminal = await consumeBatchResponses()
    } else {
      for (const [jobId, item] of Array.from(unresolved.entries())) {
        const response = await options.fetchJob?.(jobId)
        if (consumeJobResponse(jobId, item, response)) {
          observedTerminal = true
        }
      }
    }

    options.onPendingJobIds?.(Array.from(unresolved.keys()))
    if (unresolved.size === 0) {
      break
    }

    if (Date.now() >= deadline) {
      for (const item of unresolved.values()) {
        recordResult(
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
      await waitForIngestPoll(options.pollIntervalMs, options.signal)
    }
  }

  return results
}

type PollSingleIngestJobOptions = {
  jobId: number
  fetchJob: (jobId: number) => Promise<IngestJobStatusResponse | undefined>
  timeoutMs: number
  pollIntervalMs?: number | (() => number)
  signal?: AbortSignal
  isCancelled: () => boolean
  onCancel: () => Promise<void> | void
  onProgress?: (progress: IngestJobProgress) => void
  onRequestError?: (
    response: IngestJobStatusResponse | undefined
  ) =>
    | {
        terminalStatus: "failed" | "cancelled" | "interrupted"
        error?: string
      }
    | undefined
}

export const pollSingleIngestJob = async (
  options: PollSingleIngestJobOptions
): Promise<PollSingleIngestJobResult> => {
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
      await waitForIngestPoll(options.pollIntervalMs, options.signal)
      continue
    }

    const progress = readIngestJobProgress(response.data)
    const { status } = progress
    if (!TERMINAL_INGEST_JOB_STATUSES.has(status)) {
      if (status && !options.isCancelled()) options.onProgress?.(progress)
      await waitForIngestPoll(options.pollIntervalMs, options.signal)
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
