import { bgRequest } from "@/services/background-proxy"
import { mediaMethods } from "@/services/tldw/domains/media"
import {
  completedIngestJobIndicatesFailure,
  extractCompletedIngestJobError,
  extractCompletedIngestJobTerminalData,
} from "@/services/tldw/ingest-job-results"
import {
  PlaylistIngestPublicError,
  cancelRun,
  pollRunSnapshot,
  streamRunEvents,
  type PlaylistIngestRunItem,
  type PlaylistIngestRunSnapshot,
} from "@/services/tldw/playlist-ingest"
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
  errorMessage: string = INTERRUPTED_REATTACH_MESSAGE
): ReattachedQuickIngestSnapshot => ({
  lifecycle: "interrupted",
  jobs: [],
  errorMessage,
})

const normalizeJobStatus = (value: unknown): string =>
  String(value || "").trim().toLowerCase()

const isTransientHttpStatus = (status: unknown): status is number =>
  typeof status === "number" &&
  Number.isFinite(status) &&
  (status === 0 ||
    status === 408 ||
    status === 429 ||
    (status >= 500 && status < 600))

const isTransientStatusRead = (
  response: { ok?: unknown; status?: unknown } | null
): boolean => response?.ok === false && isTransientHttpStatus(response.status)

const isTransientThrownStatusRead = (error: unknown): boolean => {
  const status = (error as { status?: unknown } | null)?.status
  return typeof status === "undefined" || isTransientHttpStatus(status)
}

const readJobStatus = async (jobId: number): Promise<any> => {
  for (let attempt = 0; attempt < STATUS_READ_ATTEMPTS; attempt += 1) {
    try {
      const response = await bgRequest<any>({
        path: `/api/v1/media/ingest/jobs/${jobId}`,
        method: "GET",
        timeoutMs: 10_000,
        returnResponse: true,
        preferDirect: true,
      })

      if (!isTransientStatusRead(response) || attempt === STATUS_READ_ATTEMPTS - 1) {
        return response
      }
    } catch (error) {
      if (
        !isTransientThrownStatusRead(error) ||
        attempt === STATUS_READ_ATTEMPTS - 1
      ) {
        return null
      }
    }

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

type QuickIngestReattachOptions = {
  transportPreference?: "sse" | "poll"
}

const successfulRunOutcomes = new Set([
  "completed",
  "included_existing",
  "metadata_updated",
  "skipped_existing",
])

const failedRunOutcomes = new Set([
  "submit_failed",
  "processing_failed",
  "metadata_update_failed",
])

const runItemStatus = (item: PlaylistIngestRunItem): string => {
  if (item.state !== "terminal") {
    if (item.state === "awaiting_upload") return "uploading"
    if (
      item.state === "staged" ||
      item.state === "preparing" ||
      item.state === "submit_pending"
    ) {
      return "queued"
    }
    if (
      item.state === "cancellation_requested" ||
      item.state === "status_unavailable"
    ) {
      return "processing"
    }
    return item.state
  }
  if (item.outcome === "cancelled") return "cancelled"
  if (item.outcome && failedRunOutcomes.has(item.outcome)) return "failed"
  if (item.outcome && successfulRunOutcomes.has(item.outcome)) return "completed"
  return "processing"
}

const deriveRunLifecycle = (
  items: PlaylistIngestRunItem[],
): ReattachedQuickIngestSnapshot["lifecycle"] => {
  if (
    items.some(
      (item) => item.state !== "terminal" || item.outcome === null,
    )
  ) {
    return "processing"
  }
  const completedCount = items.filter(
    (item) => item.outcome && successfulRunOutcomes.has(item.outcome),
  ).length
  const cancelledCount = items.filter(
    (item) => item.outcome === "cancelled",
  ).length
  const failedCount = items.filter(
    (item) => item.outcome && failedRunOutcomes.has(item.outcome),
  ).length
  if (items.length > 0 && completedCount === items.length) return "completed"
  if (items.length > 0 && cancelledCount === items.length) return "cancelled"
  if (completedCount > 0 || cancelledCount > 0 || failedCount > 0) {
    return "partial_failure"
  }
  return "interrupted"
}

const snapshotFromRun = (
  snapshot: PlaylistIngestRunSnapshot,
): ReattachedQuickIngestSnapshot => ({
  lifecycle: deriveRunLifecycle(snapshot.items),
  jobs: snapshot.items.map((item) => ({
    jobId: item.jobId,
    status: runItemStatus(item),
    result: {
      media_id: item.mediaId,
      outcome: item.outcome,
      title: item.displayMetadata.title,
    },
    error:
      item.outcome && failedRunOutcomes.has(item.outcome)
        ? item.progressMessage || `Ingest ${item.outcome}`
        : undefined,
    sourceItemId: item.occurrenceId,
  })),
  errorMessage: null,
})

const runSnapshotSignature = (snapshot: PlaylistIngestRunSnapshot): string =>
  JSON.stringify({
    status: snapshot.summary.status,
    version: snapshot.summary.version,
    items: snapshot.items.map((item) => [
      item.occurrenceId,
      item.state,
      item.outcome,
      item.progressPercent,
      item.progressMessage,
      item.jobId,
      item.batchId,
      item.mediaId,
    ]),
  })

const reattachRun = async (
  runId: string,
  options: QuickIngestReattachOptions,
  submissionState?: PersistedQuickIngestTracking["submissionState"],
): Promise<ReattachedQuickIngestSnapshot> => {
  let polled = await pollRunSnapshot(mediaMethods, runId)
  if (submissionState === "run_created" || submissionState === "submitting") {
    const unsentOccurrenceIds = polled.items.flatMap((item) =>
      item.state === "staged" ||
      item.state === "awaiting_upload" ||
      item.state === "submit_pending"
        ? [item.occurrenceId]
        : []
    )
    if (unsentOccurrenceIds.length > 0) {
      await cancelRun(mediaMethods, runId, {
        occurrenceIds: unsentOccurrenceIds,
        reason: "submission_interrupted",
      })
      polled = await pollRunSnapshot(mediaMethods, runId)
    }
  }
  const hasSafeEventBoundary = polled.lastEventId !== null
  if (
    options.transportPreference === "poll" ||
    !hasSafeEventBoundary ||
    deriveRunLifecycle(polled.items) !== "processing"
  ) {
    return snapshotFromRun(polled)
  }

  try {
    const initialSignature = runSnapshotSignature(polled)
    for await (const streamed of streamRunEvents(mediaMethods, polled, {
      streamIdleTimeoutMs: 10_000,
    })) {
      if (runSnapshotSignature(streamed) === initialSignature) continue
      return snapshotFromRun(streamed)
    }
  } catch {
    // A complete polling snapshot remains authoritative when SSE is unavailable.
  }
  return snapshotFromRun(polled)
}

export const reattachQuickIngestSession = async (
  tracking: PersistedQuickIngestTracking,
  options: QuickIngestReattachOptions = {},
): Promise<ReattachedQuickIngestSnapshot> => {
  if (tracking.runId) {
    try {
      return await reattachRun(tracking.runId, {
        transportPreference:
          options.transportPreference ??
          (tracking.mode === "extension-runtime" ? "poll" : "sse"),
      }, tracking.submissionState)
    } catch (error) {
      const status =
        error instanceof PlaylistIngestPublicError
          ? error.status
          : typeof (error as { status?: unknown } | null)?.status === "number"
            ? (error as { status: number }).status
            : null
      const compatibilityFallback =
        status === 404 || status === 405 || status === 501
      if (!compatibilityFallback) {
        if (status === 401 || status === 403) {
          return interruptedSnapshot(
            "Quick ingest could not reconnect because authorization is required. Sign in or update your API key."
          )
        }
        const retryable =
          error instanceof PlaylistIngestPublicError
            ? error.retryable
            : status === 0 ||
              status === 408 ||
              status === 429 ||
              status === 503 ||
              status === 504 ||
              (status === null &&
                /timeout|timed out|network|fetch|connect|unavailable/i.test(
                  error instanceof Error ? `${error.name} ${error.message}` : ""
                ))
        if (retryable) {
          const jobIds = normalizeJobIds(tracking.jobIds)
          const submittedItemIds = resolveSubmittedItemIds(tracking)
          const jobIdToItemId = normalizeJobIdToItemId(tracking.jobIdToItemId)
          const jobs =
            jobIds.length > 0
              ? jobIds.map((jobId, index) => ({
                  jobId,
                  status: "processing",
                  sourceItemId:
                    jobIdToItemId[String(jobId)] || submittedItemIds[index],
                }))
              : submittedItemIds.map((sourceItemId) => ({
                  jobId: null,
                  status: "processing",
                  sourceItemId,
                }))
          return {
            lifecycle: "processing",
            jobs,
            errorMessage:
              "Run status is temporarily unavailable. Quick ingest will retry.",
          }
        }
        return interruptedSnapshot()
      }
      // Older servers may not expose run status; preserve legacy job reattachment.
    }
  }

  const jobIds = normalizeJobIds(tracking.jobIds)
  if (tracking.mode !== "webui-direct" || jobIds.length === 0) {
    return interruptedSnapshot()
  }

  const jobs: ReattachedQuickIngestJob[] = []
  const submittedItemIds = resolveSubmittedItemIds(tracking)
  const jobIdToItemId = normalizeJobIdToItemId(tracking.jobIdToItemId)

  try {
    for (const [index, jobId] of jobIds.entries()) {
      const response = await readJobStatus(jobId)

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
