import type { WatchlistRun } from "@/types/watchlists"
import {
  isWatchlistRunActive,
  isWatchlistRunTerminal,
  normalizeWatchlistRunStatus
} from "../shared/runStatus"

type NotificationKind = "completed" | "failed"
export type RunNotificationKind = NotificationKind | "stalled"
export type RunFailureKind =
  | "auth"
  | "rate_limit"
  | "timeout"
  | "dns"
  | "tls"
  | "network"
  | "unknown"

interface RunNotificationDecision {
  kind: NotificationKind
  hint?: string | null
}

export interface RunNotificationEvent {
  eventKey: string
  kind: RunNotificationKind
  runId: number
  hint?: string | null
}

export interface RunNotificationGroup {
  groupKey: string
  kind: RunNotificationKind
  count: number
  runIds: number[]
  eventKeys: string[]
  deepLinkRunId: number
  hint: string | null
}

type Translator = (...args: any[]) => unknown

const RUN_NOTIFICATION_PRIORITY: Record<RunNotificationKind, number> = {
  failed: 0,
  stalled: 1,
  completed: 2
}
const DEFAULT_RUN_NOTIFICATIONS_PAGE_SIZE = 25
const DEFAULT_RUN_NOTIFICATIONS_REDUCED_PAGE_SIZE = 10
const DEFAULT_RUN_NOTIFICATIONS_BACKGROUND_POLL_MS = 60_000
const DEFAULT_RUN_NOTIFICATIONS_RUNS_TAB_POLL_MS = 30_000

export interface RunNotificationsPollPlanInput {
  isOnline: boolean
  activeTab: string | null | undefined
  runsPollingActive: boolean
  documentVisible: boolean
  baseIntervalMs: number
  minIntervalMs?: number
  defaultPageSize?: number
  reducedPageSize?: number
  backgroundIntervalMs?: number
  runsTabIntervalMs?: number
}

export interface RunNotificationsPollPlan {
  enabled: boolean
  intervalMs: number
  pageSize: number
  suppressCompleted: boolean
}

const normalizeStatus = (status: string | null | undefined): string =>
  normalizeWatchlistRunStatus(status)

const normalizePositiveInt = (value: number, fallback: number): number => {
  if (!Number.isFinite(value)) return fallback
  return Math.max(1, Math.floor(value))
}

const parseEpochMs = (value: string | null | undefined): number | null => {
  if (!value) return null
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : null
}

const resolveHintCopy = (
  t: Translator | undefined,
  key: string,
  fallback: string,
  options?: Record<string, unknown>
): string => {
  if (!t) return fallback
  try {
    const translated = t(key, fallback, options)
    if (typeof translated !== "string" || !translated || translated === key) {
      return fallback
    }
    return translated
  } catch {
    return fallback
  }
}

export const getRunFailureHint = (
  errorMsg: string | null | undefined,
  t?: Translator
): string | null => {
  const kind = classifyRunFailure(errorMsg)
  if (kind === "auth") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.auth",
      "Source may require authentication. Verify credentials or access rules."
    )
  }
  if (kind === "rate_limit") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.rateLimit",
      "Source is rate-limiting requests. Reduce schedule frequency and retry."
    )
  }
  if (kind === "timeout") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.timeout",
      "The source request timed out. Retry, or lower concurrency for this source."
    )
  }
  if (kind === "dns") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.dns",
      "Source host could not be resolved. Verify the source URL and DNS availability."
    )
  }
  if (kind === "tls") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.tls",
      "TLS/SSL validation failed. Verify certificate chain or endpoint URL."
    )
  }
  if (kind === "network") {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.network",
      "Source could not be reached. Check connectivity and endpoint availability."
    )
  }
  const normalized = String(errorMsg || "").toLowerCase()
  if (!normalized) {
    return resolveHintCopy(
      t,
      "watchlists:notifications.failureHints.unknownEmpty",
      "Open run details to inspect logs and retry."
    )
  }
  return resolveHintCopy(
    t,
    "watchlists:notifications.failureHints.unknown",
    "Open run details to inspect logs and adjust source or filter settings."
  )
}

export const classifyRunFailure = (errorMsg: string | null | undefined): RunFailureKind => {
  const normalized = String(errorMsg || "").toLowerCase()

  if (
    normalized.includes("403") ||
    normalized.includes("401") ||
    normalized.includes("forbidden") ||
    normalized.includes("unauthorized")
  ) {
    return "auth"
  }
  if (normalized.includes("429") || normalized.includes("rate limit")) {
    return "rate_limit"
  }
  if (normalized.includes("timeout") || normalized.includes("timed out")) {
    return "timeout"
  }
  if (normalized.includes("dns") || normalized.includes("name resolution")) {
    return "dns"
  }
  if (normalized.includes("ssl") || normalized.includes("certificate")) {
    return "tls"
  }
  if (
    normalized.includes("failed to fetch") ||
    normalized.includes("networkerror") ||
    normalized.includes("econnrefused") ||
    normalized.includes("unreachable")
  ) {
    return "network"
  }
  return "unknown"
}

export const resolveRunTransitionNotification = (
  previousStatus: string | null | undefined,
  run: Pick<WatchlistRun, "status" | "error_msg">,
  t?: Translator
): RunNotificationDecision | null => {
  const previous = normalizeStatus(previousStatus)
  const current = normalizeStatus(run.status)
  if (!previous || !current || previous === current) return null

  if (current === "completed" && isWatchlistRunActive(previous)) {
    return { kind: "completed" }
  }

  if (current === "failed" && (isWatchlistRunActive(previous) || previous === "completed")) {
    return {
      kind: "failed",
      hint: getRunFailureHint(run.error_msg, t)
    }
  }

  return null
}

export const buildRunStateNotificationKey = (
  runId: number,
  status: string | null | undefined
): string => `${runId}:${normalizeStatus(status)}`

export const buildRunStalledNotificationKey = (runId: number): string =>
  `${runId}:stalled`

export const getRunStalledHint = (
  stalledMinutes: number,
  t?: Translator
): string =>
  resolveHintCopy(
    t,
    "watchlists:notifications.failureHints.stalled",
    "Run appears stalled after {{minutes}} minutes. Open Activity to inspect logs, then cancel or retry.",
    { minutes: Math.max(1, stalledMinutes) }
  )

export const resolveStalledRunNotification = (
  run: Pick<WatchlistRun, "id" | "status" | "started_at" | "finished_at">,
  nowMs: number,
  stalledThresholdMs: number,
  t?: Translator
): RunNotificationEvent | null => {
  const status = normalizeStatus(run.status)
  if (!isWatchlistRunActive(status)) return null
  if (run.finished_at) return null
  const startedAtMs = parseEpochMs(run.started_at)
  if (startedAtMs == null) return null
  const elapsedMs = nowMs - startedAtMs
  if (elapsedMs < stalledThresholdMs) return null
  const stalledMinutes = Math.floor(elapsedMs / 60_000)
  return {
    eventKey: buildRunStalledNotificationKey(run.id),
    kind: "stalled",
    runId: run.id,
    hint: getRunStalledHint(stalledMinutes, t)
  }
}

export const dedupeRunNotificationEvents = (
  events: RunNotificationEvent[],
  seenKeys: Set<string>
): RunNotificationEvent[] => {
  const next: RunNotificationEvent[] = []
  events.forEach((event) => {
    if (seenKeys.has(event.eventKey)) return
    seenKeys.add(event.eventKey)
    next.push(event)
  })
  return next
}

export const groupRunNotificationEvents = (
  events: RunNotificationEvent[]
): RunNotificationGroup[] => {
  const byKind = new Map<RunNotificationKind, RunNotificationGroup>()

  events.forEach((event) => {
    const existing = byKind.get(event.kind)
    if (existing) {
      existing.count += 1
      existing.runIds.push(event.runId)
      existing.eventKeys.push(event.eventKey)
      if (event.runId > existing.deepLinkRunId) {
        existing.deepLinkRunId = event.runId
      }
      if (!existing.hint && event.hint) {
        existing.hint = event.hint
      }
      return
    }

    byKind.set(event.kind, {
      groupKey: String(event.kind),
      kind: event.kind,
      count: 1,
      runIds: [event.runId],
      eventKeys: [event.eventKey],
      deepLinkRunId: event.runId,
      hint: event.hint || null
    })
  })

  return Array.from(byKind.values()).sort(
    (a, b) => RUN_NOTIFICATION_PRIORITY[a.kind] - RUN_NOTIFICATION_PRIORITY[b.kind]
  )
}

export const shouldNotifyNewTerminalRun = (
  run: Pick<WatchlistRun, "status" | "finished_at">,
  sessionStartedAtMs: number
): boolean => {
  const current = normalizeStatus(run.status)
  if (!isWatchlistRunTerminal(current) || current === "cancelled") {
    return false
  }
  const finishedAtMs = parseEpochMs(run.finished_at)
  if (finishedAtMs == null) return false
  return finishedAtMs >= sessionStartedAtMs
}

export const resolveRunNotificationsPollPlan = (
  input: RunNotificationsPollPlanInput
): RunNotificationsPollPlan => {
  const minInterval = normalizePositiveInt(input.minIntervalMs ?? 100, 100)
  const baseInterval = Math.max(
    minInterval,
    normalizePositiveInt(input.baseIntervalMs, minInterval)
  )
  const backgroundInterval = Math.max(
    baseInterval,
    normalizePositiveInt(
      input.backgroundIntervalMs ?? DEFAULT_RUN_NOTIFICATIONS_BACKGROUND_POLL_MS,
      DEFAULT_RUN_NOTIFICATIONS_BACKGROUND_POLL_MS
    )
  )
  const runsTabInterval = Math.max(
    baseInterval,
    normalizePositiveInt(
      input.runsTabIntervalMs ?? DEFAULT_RUN_NOTIFICATIONS_RUNS_TAB_POLL_MS,
      DEFAULT_RUN_NOTIFICATIONS_RUNS_TAB_POLL_MS
    )
  )
  const defaultPageSize = normalizePositiveInt(
    input.defaultPageSize ?? DEFAULT_RUN_NOTIFICATIONS_PAGE_SIZE,
    DEFAULT_RUN_NOTIFICATIONS_PAGE_SIZE
  )
  const reducedPageSize = normalizePositiveInt(
    input.reducedPageSize ?? DEFAULT_RUN_NOTIFICATIONS_REDUCED_PAGE_SIZE,
    DEFAULT_RUN_NOTIFICATIONS_REDUCED_PAGE_SIZE
  )
  const activeTab = String(input.activeTab || "")
  const runsTabActive = activeTab === "runs"

  if (!input.isOnline) {
    return {
      enabled: false,
      intervalMs: baseInterval,
      pageSize: defaultPageSize,
      suppressCompleted: false
    }
  }

  if (runsTabActive && input.runsPollingActive) {
    return {
      enabled: false,
      intervalMs: runsTabInterval,
      pageSize: reducedPageSize,
      suppressCompleted: true
    }
  }

  if (!input.documentVisible) {
    return {
      enabled: true,
      intervalMs: backgroundInterval,
      pageSize: reducedPageSize,
      suppressCompleted: true
    }
  }

  if (runsTabActive) {
    return {
      enabled: true,
      intervalMs: runsTabInterval,
      pageSize: reducedPageSize,
      suppressCompleted: true
    }
  }

  return {
    enabled: true,
    intervalMs: baseInterval,
    pageSize: defaultPageSize,
    suppressCompleted: false
  }
}
