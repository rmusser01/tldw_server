import { ACTIVE_INGEST_JOB_STATUSES, INGEST_JOB_STATUS_INTERRUPTED, TERMINAL_INGEST_JOB_STATUSES, isTransientIngestStatusRead, readIngestJobProgress, type IngestJobStatusResponse } from "./ingest-jobs-orchestrator"
import { assertQuickIngestRequestCurrent, scopedQuickIngestRequest, type QuickIngestRequestContext } from "./quick-ingest-request-scope"
import {
  completedIngestJobIndicatesFailure,
  extractCompletedIngestJobError,
  extractCompletedIngestJobTerminalData,
} from "@/services/tldw/ingest-job-results"
import type {
  PersistedQuickIngestTracking,
  ReattachedQuickIngestJob,
  ReattachedQuickIngestSnapshot,
} from "@/components/Common/QuickIngest/types"

const FAILED_JOB_STATUSES = new Set(["failed", "quarantined", "timeout"])
const STATUS_READ_ATTEMPTS = 3
const STATUS_READ_RETRY_DELAY_MS = 100

const normalizeJobIds = (jobIds?: number[]): number[] =>
  Array.isArray(jobIds)
    ? jobIds
        .map((jobId) => Number(jobId))
        .filter((jobId) => Number.isFinite(jobId) && jobId > 0)
        .map((jobId) => Math.trunc(jobId))
    : []

const normalizeStringIds = (values?: unknown[]): string[] =>
  Array.from(
    new Set(
      Array.isArray(values)
        ? values
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        : []
    )
  )

const normalizeJobIdToItemId = (
  jobIdToItemId?: Record<string, string>
): Record<string, string> => {
  const entries = Object.entries(jobIdToItemId || {})
    .map(([jobId, itemId]) => [String(jobId || "").trim(), String(itemId || "").trim()] as const)
    .filter(([jobId, itemId]) => jobId && itemId)
  return entries.length > 0 ? Object.fromEntries(entries) : {}
}

const resolveSubmittedItemIds = (
  tracking: PersistedQuickIngestTracking
): string[] =>
  normalizeStringIds([
    ...(Array.isArray(tracking.submittedItemIds) ? tracking.submittedItemIds : []),
    ...(Array.isArray(tracking.itemIds) ? tracking.itemIds : []),
  ])

const interruptedSnapshot = (
  errorMessage: string = INGEST_JOB_STATUS_INTERRUPTED
): ReattachedQuickIngestSnapshot => ({
  lifecycle: "interrupted",
  jobs: [],
  errorMessage,
})

const isTransientStatusRead = (
  response: { ok?: unknown; status?: unknown } | null
): boolean => response?.ok === false && isTransientIngestStatusRead(response.status)

const isTransientThrownStatusRead = (error: unknown): boolean => {
  const status = (error as { status?: unknown } | null)?.status
  return typeof status === "undefined" || isTransientIngestStatusRead(status)
}

const readJobStatus = async (
  jobId: number,
  context: QuickIngestRequestContext
): Promise<IngestJobStatusResponse | "unavailable" | null> => {
  for (let attempt = 0; attempt < STATUS_READ_ATTEMPTS; attempt += 1) {
    try {
      const response = await scopedQuickIngestRequest<IngestJobStatusResponse | null>(context, {
        path: `/api/v1/media/ingest/jobs/${jobId}`,
        method: "GET",
        timeoutMs: 10_000,
        returnResponse: true,
        preferDirect: true,
      })

      if (!isTransientStatusRead(response)) return response
    } catch (error) {
      assertQuickIngestRequestCurrent(context, error);
      if (!isTransientThrownStatusRead(error)) return null
    }

    if (attempt === STATUS_READ_ATTEMPTS - 1) return "unavailable"
    await new Promise((resolve) => setTimeout(resolve, STATUS_READ_RETRY_DELAY_MS))
  }

  return null
}

const isLogicalFailure = (job: ReattachedQuickIngestJob): boolean =>
  FAILED_JOB_STATUSES.has(job.status) ||
  (job.status === "completed" && completedIngestJobIndicatesFailure(job.result))

const buildJobSnapshot = (
  jobId: number,
  response: { data?: any },
  sourceItemId?: string
): ReattachedQuickIngestJob => {
  const progress = readIngestJobProgress(response.data)
  const { status } = progress
  const terminalData =
    status === "completed"
      ? extractCompletedIngestJobTerminalData(response.data)
      : undefined
  const error =
    status === "cancelled"
      ? String(response.data?.cancellation_reason || "Cancelled by user.").trim()
      : FAILED_JOB_STATUSES.has(status)
        ? String(response.data?.error_message || `Ingest ${status}`).trim()
        : status === "completed" && completedIngestJobIndicatesFailure(response.data)
          ? extractCompletedIngestJobError(response.data) ||
            "Ingest completed with an error result."
        : undefined

  return {
    jobId,
    ...progress,
    result: terminalData,
    error: error || undefined,
    sourceItemId: sourceItemId || undefined,
  }
}

const deriveLifecycle = (
  jobs: ReattachedQuickIngestJob[]
): ReattachedQuickIngestSnapshot["lifecycle"] => {
  if (jobs.length === 0) return "interrupted"
  if (jobs.some((job) => ACTIVE_INGEST_JOB_STATUSES.has(job.status))) {
    return "processing"
  }

  const completedCount = jobs.filter(
    (job) => job.status === "completed" && !isLogicalFailure(job)
  ).length
  const cancelledCount = jobs.filter((job) => job.status === "cancelled").length
  const failedCount = jobs.filter((job) => isLogicalFailure(job)).length

  if (completedCount === jobs.length) return "completed"
  if (cancelledCount === jobs.length) return "cancelled"
  if (completedCount > 0 || cancelledCount > 0 || failedCount > 0) {
    return "partial_failure"
  }
  return "interrupted"
}

export const reattachQuickIngestSession = async (
  tracking: PersistedQuickIngestTracking,
  context: QuickIngestRequestContext = {},
  confirmedCancelledJobIds?: ReadonlySet<number>,
): Promise<ReattachedQuickIngestSnapshot> => {
  const jobIds = normalizeJobIds(tracking.jobIds)
  if (tracking.mode !== "webui-direct" || jobIds.length === 0) {
    return interruptedSnapshot()
  }

  const jobs: ReattachedQuickIngestJob[] = []
  let unavailable = false
  let interrupted = false
  const submittedItemIds = resolveSubmittedItemIds(tracking)
  const jobIdToItemId = normalizeJobIdToItemId(tracking.jobIdToItemId)

  try {
    for (const [index, jobId] of jobIds.entries()) {
      const sourceItemId = jobIdToItemId[String(jobId)] || submittedItemIds[index]
      if (confirmedCancelledJobIds?.has(jobId)) {
        assertQuickIngestRequestCurrent(context)
        jobs.push({ jobId, sourceItemId, status: "cancelled", error: "Cancelled by user." })
        continue
      }
      const response = await readJobStatus(jobId, context)

      if (response === "unavailable") {
        unavailable = true
        continue
      }
      if (!response?.ok) {
        interrupted = true
        continue
      }

      const job = buildJobSnapshot(jobId, response, sourceItemId)
      if (ACTIVE_INGEST_JOB_STATUSES.has(job.status) || TERMINAL_INGEST_JOB_STATUSES.has(job.status) || FAILED_JOB_STATUSES.has(job.status)) {
        jobs.push(job)
      } else {
        interrupted = true
      }
    }
  } catch (error) {
    assertQuickIngestRequestCurrent(context, error)
    return interruptedSnapshot()
  }

  const lifecycle = deriveLifecycle(jobs)
  if (unavailable || lifecycle === "processing") {
    return { lifecycle: "processing", jobs, statusUnavailable: unavailable || interrupted }
  }
  if (interrupted || lifecycle === "interrupted") return { ...interruptedSnapshot(), jobs }
  return { lifecycle, jobs, errorMessage: null }
}

export const createInterruptedQuickIngestSnapshot = interruptedSnapshot
