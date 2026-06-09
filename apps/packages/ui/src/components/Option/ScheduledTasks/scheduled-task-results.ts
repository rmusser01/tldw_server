import type { CompanionHomeItem } from "@/services/companion-home"
import type { NotificationItem } from "@/services/notifications"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"

import {
  buildScheduledTaskResultDedupeKey,
  buildScheduledTaskResultHref,
  mergeScheduledTaskNotificationTargets,
  normalizeScheduledTaskNotificationTarget,
  type ScheduledTaskResultSignalKind
} from "./scheduled-task-result-links"
import {
  buildWatchlistTaskLinks,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel
} from "./scheduled-task-status"

export type { ScheduledTaskResultSignalKind } from "./scheduled-task-result-links"

export type ScheduledTaskResultsCapabilityMode =
  | "projected_signals"
  | "normalized_results_read"
  | "normalized_results_mutation"

export type ScheduledTaskResultState =
  | "new"
  | "reviewed"
  | "running"
  | "completed_no_results"
  | "failed"
  | "blocked"
  | "paused"

export type ScheduledTaskResultSeverity = "info" | "success" | "warning" | "error"

export type ScheduledTaskResultOwner =
  | "scheduled_tasks"
  | "watchlists"
  | "reminders"
  | "external_workspace"

export interface ScheduledTaskResultItem {
  id: string
  capabilityMode: ScheduledTaskResultsCapabilityMode
  signalKind: ScheduledTaskResultSignalKind
  state: ScheduledTaskResultState
  severity: ScheduledTaskResultSeverity
  taskId: string
  taskTitle: string
  taskTypeLabel: string
  owner: ScheduledTaskResultOwner
  ownerLabel: string
  resultId: string | null
  runId: string | null
  resultCount: number | null
  sourceLabel: string | null
  matchedRuleLabel: string | null
  outputLabel: string | null
  title: string
  summary: string
  occurredAt: string | null
  primaryHref: string
  resultHref: string | null
  runHref: string | null
  sourceHref: string | null
  domainHref: string | null
  dedupeKey: string
  reviewed: boolean
  reviewAvailable: boolean
  retryAvailable: boolean
}

export interface ScheduledTaskResultProjectionOptions {
  capabilityMode?: ScheduledTaskResultsCapabilityMode
  includeCompletedNoResults?: boolean
}

export interface ScheduledTaskResultFilterOptions {
  states?: ScheduledTaskResultState[]
  signalKinds?: ScheduledTaskResultSignalKind[]
  reviewState?: "all" | "reviewed" | "unreviewed"
  owners?: ScheduledTaskResultOwner[]
}

export interface ScheduledTaskAutomationHomeItem {
  id: string
  title: string
  summary: string
  statusLabel: string
  ownerLabel: string
  href: string
  updatedAt: string | null
  severity: ScheduledTaskResultSeverity
  dedupeKey: string
}

export interface ScheduledTaskResultRouteTarget {
  resultId?: string | null
  runId?: string | null
  taskId?: string | null
}

const RUN_ID_KEYS = [
  "run_id",
  "runId",
  "last_run_id",
  "lastRunId",
  "latest_run_id",
  "latestRunId"
] as const

const RESULT_ID_KEYS = [
  "result_id",
  "resultId",
  "output_id",
  "outputId",
  "last_result_id",
  "lastResultId",
  "last_output_id",
  "lastOutputId",
  "latest_result_id",
  "latestResultId",
  "latest_output_id",
  "latestOutputId"
] as const

const RESULT_COUNT_KEYS = [
  "result_count",
  "resultCount",
  "results_count",
  "resultsCount",
  "output_count",
  "outputCount",
  "outputs_count",
  "outputsCount"
] as const

const SOURCE_LABEL_KEYS = [
  "source_label",
  "sourceLabel",
  "source_name",
  "sourceName",
  "origin_label",
  "originLabel",
  "scope_label",
  "scopeLabel"
] as const

const MATCHED_RULE_LABEL_KEYS = [
  "matched_rule_label",
  "matchedRuleLabel",
  "rule_label",
  "ruleLabel",
  "query_label",
  "queryLabel",
  "filter_label",
  "filterLabel"
] as const

const PRIVATE_VALUE_PATTERN =
  /(authorization\s*:|bearer\s+|api[_-]?key\s*=|access[_-]?token\s*=|token\s*=|client[_-]?secret\s*=|secret\s*=|password\s*=)/i

const RESULT_STATUS_TOKENS = [
  "found",
  "match",
  "matched",
  "matches",
  "matching",
  "result",
  "results",
  "output",
  "outputs"
] as const

const FAILURE_STATUS_TOKENS = [
  "fail",
  "failed",
  "failing",
  "failure",
  "error",
  "missed"
] as const

const BLOCKED_STATUS_TOKENS = [
  "blocked",
  "auth",
  "permission",
  "unavailable",
  "dependency"
] as const

const RUNNING_STATUS_TOKENS = ["running", "active", "processing", "in_progress"] as const
const PAUSED_STATUS_TOKENS = ["paused"] as const
const COMPLETED_STATUS_TOKENS = ["complete", "completed", "success", "done", "finished"] as const

const SEVERITY_ORDER: Record<ScheduledTaskResultSeverity, number> = {
  error: 0,
  warning: 1,
  success: 2,
  info: 3
}

const statusIncludes = (status: string, tokens: readonly string[]): boolean => {
  const normalized = status.toLowerCase()
  return tokens.some((token) => {
    const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    return new RegExp(`(^|[^a-z0-9])${escapedToken}($|[^a-z0-9])`).test(
      normalized
    )
  })
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

const normalizeId = (value: unknown): string | null => {
  if (typeof value === "number") {
    return Number.isSafeInteger(value) && value >= 0 ? String(value) : null
  }

  if (typeof value !== "string") {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed || /[\r\n]/.test(trimmed)) {
    return null
  }

  return trimmed
}

const toPositiveInteger = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isSafeInteger(value) && value > 0) {
    return value
  }

  if (typeof value !== "string") {
    return null
  }

  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) {
    return null
  }

  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

const firstId = (
  sourceRef: Record<string, unknown>,
  keys: readonly string[]
): string | null => {
  for (const key of keys) {
    const value = normalizeId(sourceRef[key])
    if (value) {
      return value
    }
  }
  return null
}

const firstPositiveInteger = (
  sourceRef: Record<string, unknown>,
  keys: readonly string[]
): number | null => {
  for (const key of keys) {
    const value = toPositiveInteger(sourceRef[key])
    if (value !== null) {
      return value
    }
  }
  return null
}

const sanitizeProvenanceText = (value: unknown): string | null => {
  if (typeof value !== "string" && typeof value !== "number" && typeof value !== "boolean") {
    return null
  }

  const trimmed = String(value).trim()
  if (!trimmed || /[\r\n]/.test(trimmed) || PRIVATE_VALUE_PATTERN.test(trimmed)) {
    return null
  }

  return trimmed
}

const firstSanitizedText = (
  sourceRef: Record<string, unknown>,
  keys: readonly string[]
): string | null => {
  for (const key of keys) {
    const value = sanitizeProvenanceText(sourceRef[key])
    if (value) {
      return value
    }
  }
  return null
}

const hasPositiveResultSignal = (sourceRef: Record<string, unknown>): boolean =>
  firstPositiveInteger(sourceRef, RESULT_COUNT_KEYS) !== null ||
  firstId(sourceRef, RESULT_ID_KEYS) !== null

const inferOwner = (
  task: ScheduledTask
): { owner: ScheduledTaskResultOwner; ownerLabel: string } => {
  if (task.primitive === "watchlist_job") {
    return { owner: "watchlists", ownerLabel: "Watchlists" }
  }

  if (task.primitive === "reminder_task") {
    return { owner: "reminders", ownerLabel: "Reminders" }
  }

  if (task.edit_mode === "external") {
    return { owner: "external_workspace", ownerLabel: "External automation" }
  }

  return { owner: "scheduled_tasks", ownerLabel: "Scheduled Tasks" }
}

const buildOutputLabel = (
  signalKind: ScheduledTaskResultSignalKind,
  resultCount: number | null,
  resultId: string | null
): string | null => {
  if (signalKind !== "result") {
    return null
  }

  if (resultCount !== null) {
    return `${resultCount} ${resultCount === 1 ? "result" : "results"}`
  }

  if (resultId) {
    return `Output ${resultId}`
  }

  return "Results ready"
}

const buildResultSummary = (
  task: ScheduledTask,
  signalKind: ScheduledTaskResultSignalKind,
  state: ScheduledTaskResultState,
  resultCount: number | null,
  sourceLabel: string | null
): string => {
  if (signalKind === "result") {
    const countText =
      resultCount !== null
        ? `${resultCount} ${resultCount === 1 ? "result" : "results"}`
        : "results"
    return sourceLabel ? `Found ${countText} from ${sourceLabel}.` : `Found ${countText}.`
  }

  if (signalKind === "running") {
    return `${task.title} is running now.`
  }

  if (signalKind === "completed_no_results") {
    return `${task.title} completed its latest run without new results.`
  }

  if (state === "blocked") {
    return `${task.title} is blocked by a required setup or dependency issue.`
  }

  return `${task.title} needs attention. Open details to inspect the latest run.`
}

const buildStatusLabel = (
  signalKind: ScheduledTaskResultSignalKind,
  state: ScheduledTaskResultState
): string => {
  if (signalKind === "result") {
    return state === "reviewed" ? "Reviewed" : "New result"
  }

  if (state === "blocked") {
    return "Blocked"
  }

  if (signalKind === "failure") {
    return "Needs attention"
  }

  if (signalKind === "running") {
    return "Running now"
  }

  return "Completed"
}

const canMutateResults = (mode: ScheduledTaskResultsCapabilityMode): boolean =>
  mode === "normalized_results_mutation"

const createResultItem = (
  task: ScheduledTask,
  signalKind: ScheduledTaskResultSignalKind,
  state: ScheduledTaskResultState,
  severity: ScheduledTaskResultSeverity,
  capabilityMode: ScheduledTaskResultsCapabilityMode
): ScheduledTaskResultItem => {
  const sourceRef = isRecord(task.source_ref) ? task.source_ref : {}
  const runId = firstId(sourceRef, RUN_ID_KEYS)
  const resultId = signalKind === "result" ? firstId(sourceRef, RESULT_ID_KEYS) : null
  const resultCount = signalKind === "result" ? firstPositiveInteger(sourceRef, RESULT_COUNT_KEYS) : null
  const sourceLabel = firstSanitizedText(sourceRef, SOURCE_LABEL_KEYS)
  const matchedRuleLabel = firstSanitizedText(sourceRef, MATCHED_RULE_LABEL_KEYS)
  const occurredAt = task.last_run_at || null
  const watchlistLinks = buildWatchlistTaskLinks(task)
  const { owner, ownerLabel } = inferOwner(task)
  const primaryHref = buildScheduledTaskResultHref({
    resultId,
    runId,
    taskId: task.id
  })
  const resultHref = signalKind === "result" ? watchlistLinks.latestOutputUrl : null
  const runHref = watchlistLinks.latestRunUrl
  const sourceHref = watchlistLinks.settingsUrl
  const domainHref = resultHref ?? runHref ?? sourceHref
  const dedupeKey = buildScheduledTaskResultDedupeKey({
    signalKind,
    taskId: task.id,
    runId,
    resultId,
    state,
    occurredAt
  })
  const outputLabel = buildOutputLabel(signalKind, resultCount, resultId)
  const title = task.title || getScheduledTaskTypeLabel(task)

  return {
    id: dedupeKey,
    capabilityMode,
    signalKind,
    state,
    severity,
    taskId: task.id,
    taskTitle: title,
    taskTypeLabel: getScheduledTaskTypeLabel(task),
    owner,
    ownerLabel,
    resultId,
    runId,
    resultCount,
    sourceLabel,
    matchedRuleLabel,
    outputLabel,
    title,
    summary: buildResultSummary(task, signalKind, state, resultCount, sourceLabel),
    occurredAt,
    primaryHref,
    resultHref,
    runHref,
    sourceHref,
    domainHref,
    dedupeKey,
    reviewed: false,
    reviewAvailable: canMutateResults(capabilityMode) && signalKind === "result",
    retryAvailable: canMutateResults(capabilityMode) && signalKind === "failure"
  }
}

const pathHasMethod = (
  methods: unknown,
  methodName: "get" | "post" | "patch" | "delete"
): boolean => {
  return isRecord(methods) && Object.keys(methods).some(
    (method) => method.toLowerCase() === methodName
  )
}

export const resolveScheduledTaskResultsCapabilityMode = (
  paths: Record<string, unknown> | null | undefined
): ScheduledTaskResultsCapabilityMode => {
  if (!paths) {
    return "projected_signals"
  }

  const entries = Object.entries(paths)
  const hasReadableResults = entries.some(([path, methods]) => {
    const normalizedPath = path.toLowerCase()
    return (
      normalizedPath.endsWith("/scheduled-tasks/results") &&
      pathHasMethod(methods, "get")
    )
  })

  if (!hasReadableResults) {
    return "projected_signals"
  }

  const hasResultMutation = entries.some(([path, methods]) => {
    const normalizedPath = path.toLowerCase()
    return (
      (normalizedPath.includes("/scheduled-tasks/results/") &&
        (normalizedPath.endsWith("/review") || normalizedPath.endsWith("/retry"))) &&
      pathHasMethod(methods, "post")
    )
  })

  return hasResultMutation ? "normalized_results_mutation" : "normalized_results_read"
}

export const projectScheduledTaskResults = (
  tasks: ScheduledTask[],
  options: ScheduledTaskResultProjectionOptions = {}
): ScheduledTaskResultItem[] => {
  const capabilityMode = options.capabilityMode ?? "projected_signals"
  const includeCompletedNoResults = options.includeCompletedNoResults ?? false
  const results: ScheduledTaskResultItem[] = []

  tasks.forEach((task) => {
    if (!task.enabled) {
      return
    }

    const status = task.status || ""
    const sourceRef = isRecord(task.source_ref) ? task.source_ref : {}
    const productStatus = getScheduledTaskProductStatus(task)
    const hasFailure =
      productStatus.key === "needs_attention" || statusIncludes(status, FAILURE_STATUS_TOKENS)
    const hasBlocked =
      productStatus.key === "blocked" || statusIncludes(status, BLOCKED_STATUS_TOKENS)
    const isRunning =
      productStatus.key === "running" || statusIncludes(status, RUNNING_STATUS_TOKENS)
    const isPaused =
      productStatus.key === "paused" || statusIncludes(status, PAUSED_STATUS_TOKENS)
    const isCompleted =
      productStatus.key === "completed" || statusIncludes(status, COMPLETED_STATUS_TOKENS)
    const hasResult =
      productStatus.key === "found_results" ||
      statusIncludes(status, RESULT_STATUS_TOKENS) ||
      hasPositiveResultSignal(sourceRef)

    if (hasFailure) {
      results.push(createResultItem(task, "failure", "failed", "error", capabilityMode))
    }

    if (!hasFailure && hasBlocked) {
      results.push(createResultItem(task, "failure", "blocked", "warning", capabilityMode))
    }

    if (hasResult) {
      results.push(createResultItem(task, "result", "new", "success", capabilityMode))
    }

    if (!hasFailure && !hasBlocked && !hasResult && isRunning) {
      results.push(createResultItem(task, "running", "running", "info", capabilityMode))
    }

    if (!hasFailure && !hasBlocked && !hasResult && !isRunning && isPaused) {
      results.push(createResultItem(task, "running", "paused", "warning", capabilityMode))
    }

    if (
      includeCompletedNoResults &&
      !hasFailure &&
      !hasBlocked &&
      !hasResult &&
      !isRunning &&
      !isPaused &&
      isCompleted
    ) {
      results.push(
        createResultItem(task, "completed_no_results", "completed_no_results", "info", capabilityMode)
      )
    }
  })

  return results.sort((a, b) => {
    const severityDelta = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]
    if (severityDelta !== 0) {
      return severityDelta
    }

    const aTime = a.occurredAt ? Date.parse(a.occurredAt) : 0
    const bTime = b.occurredAt ? Date.parse(b.occurredAt) : 0
    if (aTime !== bTime) {
      return bTime - aTime
    }

    return a.title.localeCompare(b.title)
  })
}

export const filterScheduledTaskResults = (
  results: ScheduledTaskResultItem[],
  options: ScheduledTaskResultFilterOptions
): ScheduledTaskResultItem[] => {
  return results.filter((result) => {
    if (options.states && !options.states.includes(result.state)) {
      return false
    }

    if (options.signalKinds && !options.signalKinds.includes(result.signalKind)) {
      return false
    }

    if (options.owners && !options.owners.includes(result.owner)) {
      return false
    }

    if (options.reviewState === "reviewed" && !result.reviewed) {
      return false
    }

    if (options.reviewState === "unreviewed" && result.reviewed) {
      return false
    }

    return true
  })
}

export const findScheduledTaskResultByRouteState = (
  results: ScheduledTaskResultItem[],
  routeState: ScheduledTaskResultRouteTarget
): ScheduledTaskResultItem | null => {
  const resultId = normalizeId(routeState.resultId)
  if (resultId) {
    return results.find((result) => result.resultId === resultId) ?? null
  }

  const runId = normalizeId(routeState.runId)
  if (runId) {
    return results.find((result) => result.runId === runId) ?? null
  }

  const taskId = normalizeId(routeState.taskId)
  if (taskId) {
    return results.find((result) => result.taskId === taskId) ?? null
  }

  return null
}

export const buildScheduledTaskAutomationHomeItems = (
  results: ScheduledTaskResultItem[]
): ScheduledTaskAutomationHomeItem[] => {
  return results
    .filter((result) => result.signalKind === "result" || result.signalKind === "failure")
    .filter((result) => result.state !== "completed_no_results")
    .map((result) => ({
      id: result.id,
      title: result.title,
      summary: result.summary,
      statusLabel: buildStatusLabel(result.signalKind, result.state),
      ownerLabel: result.ownerLabel,
      href: result.primaryHref,
      updatedAt: result.occurredAt,
      severity: result.severity,
      dedupeKey: result.dedupeKey
    }))
}

const normalizeNotificationHomeSeverity = (
  notification: NotificationItem,
  signalKind: ScheduledTaskResultSignalKind
): ScheduledTaskResultSeverity => {
  if (signalKind === "result") {
    return "success"
  }

  const severity = String(notification.severity || "").toLowerCase()
  if (severity.includes("error") || severity.includes("danger")) {
    return "error"
  }
  if (severity.includes("warn") || severity.includes("blocked")) {
    return "warning"
  }
  return signalKind === "failure" ? "error" : "info"
}

const inferNotificationOwnerLabel = (notification: NotificationItem): string => {
  const taskId = String(notification.source_task_id || "")
  const domain = String(notification.source_domain || "").toLowerCase()
  const jobType = String(notification.source_job_type || "").toLowerCase()

  if (
    taskId.startsWith("watchlist_job:") ||
    domain.includes("watchlist") ||
    jobType.includes("watchlist") ||
    notification.source_job_id != null
  ) {
    return "Watchlists"
  }

  if (taskId.startsWith("reminder_task:")) {
    return "Reminders"
  }

  return "Scheduled Tasks"
}

const stateForNotificationTarget = (
  signalKind: ScheduledTaskResultSignalKind,
  severity: ScheduledTaskResultSeverity
): ScheduledTaskResultState => {
  if (signalKind === "result") {
    return "new"
  }
  if (signalKind === "running") {
    return "running"
  }
  if (signalKind === "completed_no_results") {
    return "completed_no_results"
  }
  return severity === "warning" ? "blocked" : "failed"
}

export const buildScheduledTaskAutomationHomeItemsFromNotifications = (
  notifications: NotificationItem[]
): ScheduledTaskAutomationHomeItem[] => {
  const notificationsById = new Map(
    notifications.map((notification) => [notification.id, notification])
  )

  return mergeScheduledTaskNotificationTargets(
    notifications.map(normalizeScheduledTaskNotificationTarget)
  )
    .filter((target) => target.signalKind === "result" || target.signalKind === "failure")
    .map((target) => {
      const notification = notificationsById.get(target.notificationId)
      const severity = notification
        ? normalizeNotificationHomeSeverity(notification, target.signalKind)
        : target.signalKind === "result"
          ? "success"
          : "error"
      const state = stateForNotificationTarget(target.signalKind, severity)

      return {
        id: `notification:${target.notificationId}`,
        title: notification?.title || "Automation result",
        summary: notification?.message || "Open Scheduled Tasks to inspect this automation signal.",
        statusLabel: buildStatusLabel(target.signalKind, state),
        ownerLabel: notification ? inferNotificationOwnerLabel(notification) : "Scheduled Tasks",
        href: target.href,
        updatedAt: target.createdAt || notification?.created_at || null,
        severity,
        dedupeKey: target.dedupeKey
      }
    })
}

const parseHomeItemTime = (item: ScheduledTaskAutomationHomeItem): number => {
  const timestamp = item.updatedAt ? Date.parse(item.updatedAt) : 0
  return Number.isFinite(timestamp) ? timestamp : 0
}

const shouldPreferHomeItem = (
  candidate: ScheduledTaskAutomationHomeItem,
  current: ScheduledTaskAutomationHomeItem
): boolean => {
  const candidateTime = parseHomeItemTime(candidate)
  const currentTime = parseHomeItemTime(current)
  if (candidateTime !== currentTime) {
    return candidateTime > currentTime
  }

  return SEVERITY_ORDER[candidate.severity] < SEVERITY_ORDER[current.severity]
}

export const mergeScheduledTaskAutomationHomeItems = (
  itemGroups: ScheduledTaskAutomationHomeItem[][]
): ScheduledTaskAutomationHomeItem[] => {
  const merged = new Map<string, ScheduledTaskAutomationHomeItem>()

  itemGroups.flat().forEach((item) => {
    const existing = merged.get(item.dedupeKey)
    if (!existing || shouldPreferHomeItem(item, existing)) {
      merged.set(item.dedupeKey, item)
    }
  })

  return Array.from(merged.values()).sort((a, b) => {
    const severityDelta = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]
    if (severityDelta !== 0) {
      return severityDelta
    }

    const timeDelta = parseHomeItemTime(b) - parseHomeItemTime(a)
    if (timeDelta !== 0) {
      return timeDelta
    }

    return a.title.localeCompare(b.title)
  })
}

export const buildScheduledTaskCompanionHomeItems = (
  items: ScheduledTaskAutomationHomeItem[]
): CompanionHomeItem[] =>
  items.map((item) => ({
    id: `automation:${item.dedupeKey}`,
    entityId: item.dedupeKey,
    entityType: "scheduled_task_result",
    source: "scheduled_task",
    title: item.title,
    summary: `${item.statusLabel} - ${item.ownerLabel}. ${item.summary}`,
    updatedAt: item.updatedAt,
    href: item.href
  }))
