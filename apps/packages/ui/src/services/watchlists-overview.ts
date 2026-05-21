import {
  fetchScrapedItems,
  fetchWatchlistContentAlerts,
  fetchWatchlistJobs,
  fetchWatchlistOutputs,
  fetchWatchlistRuns,
  fetchWatchlistSources
} from "@/services/watchlists"
import type {
  PaginatedResponse,
  WatchlistJob,
  WatchlistOutput,
  WatchlistRun,
  WatchlistSource
} from "@/types/watchlists"

const OVERVIEW_PAGE_SIZE = 200
const OVERVIEW_MAX_PAGES = 15

const HEALTHY_SOURCE_STATUSES = new Set([
  "ok",
  "healthy",
  "ready",
  "running",
  "pending",
  "queued"
])

const DEGRADED_SOURCE_STATUSES = new Set([
  "backoff",
  "deferred",
  "stale",
  "warning",
  "error",
  "failed",
  "unreachable",
  "forum_disabled"
])

export type SourceHealthBucket = "healthy" | "degraded" | "inactive" | "unknown"
export type OverviewHealthStatus = "healthy" | "attention" | "inactive" | "unknown"

export interface WatchlistsOverviewFailedRun extends WatchlistRun {
  job_name?: string
}

export interface WatchlistsOverviewAttention {
  total: number
  sources: number
  jobs: number
  runs: number
  outputs: number
}

export interface WatchlistsOverviewTabBadges {
  sources: number
  runs: number
  outputs: number
}

export interface WatchlistsOverviewHealthModel {
  statuses: {
    sources: OverviewHealthStatus
    jobs: OverviewHealthStatus
    runs: OverviewHealthStatus
    outputs: OverviewHealthStatus
  }
  attention: WatchlistsOverviewAttention
  tabBadges: WatchlistsOverviewTabBadges
}

export interface WatchlistsOverviewData {
  fetchedAt: string
  sources: {
    total: number
    healthy: number
    degraded: number
    inactive: number
    unknown: number
  }
  jobs: {
    total: number
    active: number
    nextRunAt: string | null
    attention: number
  }
  items: {
    unread: number
  }
  alerts: {
    unread: number
  }
  runs: {
    running: number
    pending: number
    failed: number
    sourceErrors?: number
    zeroItemSourceErrors?: number
    recentFailed: WatchlistsOverviewFailedRun[]
  }
  outputs: {
    total: number
    expired: number
    deliveryIssues: number
    audioIssues?: number
    attention: number
  }
  health: WatchlistsOverviewHealthModel
  systemHealth: "healthy" | "degraded"
}

export interface FetchWatchlistsOverviewParams {
  watchlist_id?: number
}

const normalizeStatus = (value: string | null | undefined): string =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")

const asFiniteNumber = (value: unknown, fallback = 0): number => {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

const parseEpochMs = (value: string | null | undefined): number | null => {
  if (!value) return null
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : null
}

const DELIVERY_ATTENTION_STATUSES = new Set([
  "failed",
  "error",
  "partial",
  "warning"
])

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return null
}

const hasDeliveryIssue = (metadata: unknown): boolean => {
  const record = asRecord(metadata)
  if (!record) return false
  const deliveries = record.deliveries
  if (Array.isArray(deliveries)) {
    return deliveries.some((entry) => {
      const normalized = normalizeStatus(
        asRecord(entry)?.status as string | null | undefined
      )
      return DELIVERY_ATTENTION_STATUSES.has(normalized)
    })
  }
  if (asRecord(deliveries)) {
    return Object.values(deliveries).some((entry) => {
      if (typeof entry === "string") {
        return DELIVERY_ATTENTION_STATUSES.has(normalizeStatus(entry))
      }
      const normalized = normalizeStatus(
        asRecord(entry)?.status as string | null | undefined
      )
      return DELIVERY_ATTENTION_STATUSES.has(normalized)
    })
  }
  if (deliveries != null) {
    const flattened = JSON.stringify(deliveries).toLowerCase()
    for (const status of DELIVERY_ATTENTION_STATUSES) {
      if (flattened.includes(status)) {
        return true
      }
    }
  }
  return false
}

const hasAudioOutputIssue = (metadata: unknown): boolean => {
  const record = asRecord(metadata)
  if (!record) return false
  const audio = asRecord(record.audio)
  const requested =
    record.audio_briefing_requested === true ||
    record.generate_audio === true ||
    audio?.requested === true ||
    audio?.enabled === true
  const normalized = normalizeStatus(
    (record.audio_briefing_status as string | null | undefined) ||
      (record.audio_status as string | null | undefined) ||
      (audio?.status as string | null | undefined)
  )
  if (
    normalized === "enqueue failed" ||
    normalized === "failed" ||
    normalized === "error"
  ) {
    return true
  }
  return requested && normalized === "skipped"
}

const getSourceErrorCount = (run: Pick<WatchlistRun, "stats">): number => {
  const stats = asRecord(run.stats)
  if (!stats) return 0
  const explicitCount = asFiniteNumber(stats.source_errors, 0)
  if (explicitCount > 0) return explicitCount
  const statuses = stats.source_statuses
  if (!Array.isArray(statuses)) return 0
  return statuses.filter((entry) => {
    const normalized = String(asRecord(entry)?.status || "").trim().toLowerCase()
    return normalized.startsWith("error") || normalized.startsWith("partial:")
  }).length
}

const isZeroItemSourceErrorRun = (run: Pick<WatchlistRun, "stats">): boolean => {
  if (getSourceErrorCount(run) <= 0) return false
  const stats = asRecord(run.stats)
  return asFiniteNumber(stats?.items_ingested, 0) <= 0
}

const fetchAllPages = async <T>(
  fetchPage: (params: { page: number; size: number }) => Promise<PaginatedResponse<T>>
): Promise<{ items: T[]; total: number }> => {
  let page = 1
  let total = 0
  const items: T[] = []

  while (page <= OVERVIEW_MAX_PAGES) {
    const response = await fetchPage({ page, size: OVERVIEW_PAGE_SIZE })
    const pageItems = Array.isArray(response.items) ? response.items : []
    if (page === 1) {
      total = asFiniteNumber(response.total, pageItems.length)
    }
    items.push(...pageItems)

    const reachedTotal = total > 0 && items.length >= total
    const hasMoreByFlag = response.has_more === true
    const hasMoreBySize = pageItems.length >= OVERVIEW_PAGE_SIZE
    if (!pageItems.length || reachedTotal || (!hasMoreByFlag && !hasMoreBySize)) {
      break
    }
    page += 1
  }

  return {
    items,
    total: total > 0 ? total : items.length
  }
}

export const classifySourceHealth = (source: WatchlistSource): SourceHealthBucket => {
  if (!source.active) return "inactive"
  const normalized = normalizeStatus(source.status)
  if (!normalized) return "unknown"
  if (normalized.startsWith("error")) return "degraded"
  if (HEALTHY_SOURCE_STATUSES.has(normalized)) return "healthy"
  if (DEGRADED_SOURCE_STATUSES.has(normalized)) return "degraded"
  return "unknown"
}

export const getEarliestNextRunAt = (
  jobs: Pick<WatchlistJob, "active" | "next_run_at">[]
): string | null => {
  let earliestMs: number | null = null
  let earliestIso: string | null = null

  jobs.forEach((job) => {
    if (!job.active || !job.next_run_at) return
    const epochMs = parseEpochMs(job.next_run_at)
    if (epochMs == null) return
    if (earliestMs == null || epochMs < earliestMs) {
      earliestMs = epochMs
      earliestIso = job.next_run_at
    }
  })

  return earliestIso
}

const classifyOutputsAttention = (
  outputs: Pick<WatchlistOutput, "id" | "expired" | "metadata">[]
): { expired: number; deliveryIssues: number; audioIssues: number; attention: number } => {
  let expired = 0
  let deliveryIssues = 0
  let audioIssues = 0
  const attentionIds = new Set<number>()

  outputs.forEach((output) => {
    if (output.expired) {
      expired += 1
      attentionIds.add(output.id)
    }
    const deliveryIssue =
      hasDeliveryIssue(output.metadata) ||
      (() => {
        const flattened = JSON.stringify(output).toLowerCase()
        if (!flattened.includes("deliver")) return false
        for (const status of DELIVERY_ATTENTION_STATUSES) {
          if (flattened.includes(status)) {
            return true
          }
        }
        return false
      })()

    if (deliveryIssue) {
      deliveryIssues += 1
      attentionIds.add(output.id)
    }
    if (hasAudioOutputIssue(output.metadata)) {
      audioIssues += 1
      attentionIds.add(output.id)
    }
  })

  return {
    expired,
    deliveryIssues,
    audioIssues,
    attention: attentionIds.size
  }
}

export const buildOverviewHealthModel = (params: {
  sources: { total: number; degraded: number; inactive: number }
  jobs: { total: number; active: number; attention: number }
  runs: { running: number; pending: number; failed: number; attention?: number }
  outputs: { total: number; attention: number }
}): WatchlistsOverviewHealthModel => {
  const sourcesStatus: OverviewHealthStatus =
    params.sources.degraded > 0
      ? "attention"
      : params.sources.total > 0 && params.sources.inactive >= params.sources.total
        ? "inactive"
        : params.sources.total > 0
          ? "healthy"
          : "unknown"

  const jobsStatus: OverviewHealthStatus =
    params.jobs.attention > 0
      ? "attention"
      : params.jobs.total > 0 && params.jobs.active <= 0
        ? "inactive"
        : params.jobs.total > 0
          ? "healthy"
          : "unknown"

  const runsStatus: OverviewHealthStatus =
    (params.runs.attention ?? params.runs.failed) > 0
      ? "attention"
      : params.runs.running + params.runs.pending > 0
        ? "healthy"
        : "unknown"

  const outputsStatus: OverviewHealthStatus =
    params.outputs.attention > 0
      ? "attention"
      : params.outputs.total > 0
        ? "healthy"
        : "unknown"

  const attention: WatchlistsOverviewAttention = {
    sources: Math.max(0, params.sources.degraded),
    jobs: Math.max(0, params.jobs.attention),
    runs: Math.max(0, params.runs.attention ?? params.runs.failed),
    outputs: Math.max(0, params.outputs.attention),
    total: 0
  }
  attention.total = attention.sources + attention.jobs + attention.runs + attention.outputs

  return {
    statuses: {
      sources: sourcesStatus,
      jobs: jobsStatus,
      runs: runsStatus,
      outputs: outputsStatus
    },
    attention,
    tabBadges: {
      sources: attention.sources,
      runs: attention.runs,
      outputs: attention.outputs
    }
  }
}

export const getOverviewTabBadges = (
  model: WatchlistsOverviewHealthModel | null | undefined
): WatchlistsOverviewTabBadges => {
  if (!model) {
    return {
      sources: 0,
      runs: 0,
      outputs: 0
    }
  }
  return model.tabBadges
}

export const fetchWatchlistsOverviewData = async (
  params: FetchWatchlistsOverviewParams = {}
): Promise<WatchlistsOverviewData> => {
  const scopedParams = params.watchlist_id != null
    ? { watchlist_id: params.watchlist_id }
    : {}
  const unreadAlertsRequest = params.watchlist_id != null
    ? fetchWatchlistContentAlerts(params.watchlist_id, { status: "unread", page: 1, size: 1 })
    : Promise.resolve({ items: [], total: 0 })
  const [
    sourcesResult,
    jobsResult,
    unreadResult,
    unreadAlertsResult,
    runningResult,
    pendingResult,
    failedResult,
    recentRunsResult,
    outputsResult
  ] = await Promise.all([
    fetchAllPages((pageParams) => fetchWatchlistSources({ ...scopedParams, ...pageParams })),
    fetchAllPages((pageParams) => fetchWatchlistJobs({ ...scopedParams, ...pageParams })),
    fetchScrapedItems({ ...scopedParams, reviewed: false, page: 1, size: 1 }),
    unreadAlertsRequest,
    fetchWatchlistRuns({ ...scopedParams, q: "running", page: 1, size: 1 }),
    fetchWatchlistRuns({ ...scopedParams, q: "pending", page: 1, size: 1 }),
    fetchWatchlistRuns({ ...scopedParams, q: "failed", page: 1, size: 5 }),
    fetchWatchlistRuns({ ...scopedParams, page: 1, size: 10 }),
    fetchWatchlistOutputs({ ...scopedParams, page: 1, size: 100 })
  ])

  let healthy = 0
  let degraded = 0
  let inactive = 0
  let unknown = 0

  sourcesResult.items.forEach((source) => {
    const bucket = classifySourceHealth(source)
    if (bucket === "healthy") healthy += 1
    if (bucket === "degraded") degraded += 1
    if (bucket === "inactive") inactive += 1
    if (bucket === "unknown") unknown += 1
  })

  const jobs = Array.isArray(jobsResult.items) ? jobsResult.items : []
  const jobNameById = new Map<number, string>(
    jobs.map((job) => [job.id, job.name])
  )
  const activeJobs = jobs.filter((job) => job.active).length
  const jobsWithoutSchedule = jobs.filter(
    (job) => job.active && parseEpochMs(job.next_run_at) == null
  ).length
  const recentFailed = (Array.isArray(failedResult.items) ? failedResult.items : [])
    .slice(0, 5)
    .map((run) => ({
      ...run,
      job_name: jobNameById.get(run.job_id)
    }))

  const unreadTotal = asFiniteNumber(unreadResult.total, 0)
  const unreadAlertsTotal = asFiniteNumber(unreadAlertsResult.total, 0)
  const runningTotal = asFiniteNumber(runningResult.total, 0)
  const pendingTotal = asFiniteNumber(pendingResult.total, 0)
  const failedTotal = asFiniteNumber(failedResult.total, recentFailed.length)
  const recentRuns = Array.isArray(recentRunsResult.items) ? recentRunsResult.items : []
  const sourceErrorRuns = recentRuns.filter((run) => getSourceErrorCount(run) > 0)
  const zeroItemSourceErrorRuns = sourceErrorRuns.filter(isZeroItemSourceErrorRun)
  const sourceErrorRunAttention = sourceErrorRuns.filter(
    (run) => normalizeStatus(run.status) !== "failed"
  ).length
  const outputs = Array.isArray(outputsResult.items) ? outputsResult.items : []
  const outputsTotals = classifyOutputsAttention(outputs)
  const outputsTotal = asFiniteNumber(outputsResult.total, outputs.length)

  const health = buildOverviewHealthModel({
    sources: {
      total: asFiniteNumber(sourcesResult.total, sourcesResult.items.length),
      degraded,
      inactive
    },
    jobs: {
      total: asFiniteNumber(jobsResult.total, jobs.length),
      active: activeJobs,
      attention: jobsWithoutSchedule
    },
    runs: {
      running: runningTotal,
      pending: pendingTotal,
      failed: failedTotal,
      attention: failedTotal + sourceErrorRunAttention
    },
    outputs: {
      total: outputsTotal,
      attention: outputsTotals.attention
    }
  })

  const systemHealth: "healthy" | "degraded" =
    health.attention.total > 0 ? "degraded" : "healthy"

  return {
    fetchedAt: new Date().toISOString(),
    sources: {
      total: asFiniteNumber(sourcesResult.total, sourcesResult.items.length),
      healthy,
      degraded,
      inactive,
      unknown
    },
    jobs: {
      total: asFiniteNumber(jobsResult.total, jobs.length),
      active: activeJobs,
      nextRunAt: getEarliestNextRunAt(jobs),
      attention: jobsWithoutSchedule
    },
    items: {
      unread: unreadTotal
    },
    alerts: {
      unread: unreadAlertsTotal
    },
    runs: {
      running: runningTotal,
      pending: pendingTotal,
      failed: failedTotal,
      sourceErrors: sourceErrorRuns.length,
      zeroItemSourceErrors: zeroItemSourceErrorRuns.length,
      recentFailed
    },
    outputs: {
      total: outputsTotal,
      expired: outputsTotals.expired,
      deliveryIssues: outputsTotals.deliveryIssues,
      audioIssues: outputsTotals.audioIssues,
      attention: outputsTotals.attention
    },
    health,
    systemHealth
  }
}
