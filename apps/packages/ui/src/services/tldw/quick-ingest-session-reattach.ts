import { bgRequest } from "@/services/background-proxy"
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

const INTERRUPTED_REATTACH_MESSAGE =
  "Quick ingest could not reconnect to live job status."

const ACTIVE_JOB_STATUSES = new Set([
  "pending",
  "queued",
  "running",
  "processing",
  "uploading",
  "analyzing",
  "storing",
])

const FAILED_JOB_STATUSES = new Set(["failed", "quarantined", "timeout"])

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
  errorMessage: string = INTERRUPTED_REATTACH_MESSAGE
): ReattachedQuickIngestSnapshot => ({
  lifecycle: "interrupted",
  jobs: [],
  errorMessage,
})

const normalizeJobStatus = (value: unknown): string =>
  String(value || "").trim().toLowerCase()

const isLogicalFailure = (job: ReattachedQuickIngestJob): boolean =>
  FAILED_JOB_STATUSES.has(job.status) ||
  (job.status === "completed" && completedIngestJobIndicatesFailure(job.result))

const buildJobSnapshot = (
  jobId: number,
  response: { data?: any },
  sourceItemId?: string
): ReattachedQuickIngestJob => {
  const status = normalizeJobStatus(response.data?.status)
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
    status,
    result: terminalData,
    error: error || undefined,
    sourceItemId: sourceItemId || undefined,
  }
}

const deriveLifecycle = (
  jobs: ReattachedQuickIngestJob[]
): ReattachedQuickIngestSnapshot["lifecycle"] => {
  if (jobs.length === 0) return "interrupted"
  if (jobs.some((job) => ACTIVE_JOB_STATUSES.has(job.status))) {
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
  tracking: PersistedQuickIngestTracking
): Promise<ReattachedQuickIngestSnapshot> => {
  const jobIds = normalizeJobIds(tracking.jobIds)
  if (tracking.mode !== "webui-direct" || jobIds.length === 0) {
    return interruptedSnapshot()
  }

  const jobs: ReattachedQuickIngestJob[] = []
  const submittedItemIds = resolveSubmittedItemIds(tracking)
  const jobIdToItemId = normalizeJobIdToItemId(tracking.jobIdToItemId)

  try {
    for (const [index, jobId] of jobIds.entries()) {
      const response = await bgRequest<any>({
        path: `/api/v1/media/ingest/jobs/${jobId}`,
        method: "GET",
        timeoutMs: 10_000,
        returnResponse: true,
      })

      if (!response?.ok || !normalizeJobStatus(response.data?.status)) {
        return interruptedSnapshot()
      }

      jobs.push(
        buildJobSnapshot(
          jobId,
          response,
          jobIdToItemId[String(jobId)] || submittedItemIds[index]
        )
      )
    }
  } catch {
    return interruptedSnapshot()
  }

  return {
    lifecycle: deriveLifecycle(jobs),
    jobs,
    errorMessage: null,
  }
}

export const createInterruptedQuickIngestSnapshot = interruptedSnapshot
