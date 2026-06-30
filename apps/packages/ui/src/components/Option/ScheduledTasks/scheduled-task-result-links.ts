import type { NotificationItem } from "@/services/notifications"

export type ScheduledTaskResultSignalKind =
  | "result"
  | "failure"
  | "running"
  | "completed_no_results"

export interface ScheduledTaskResultHrefInput {
  resultId?: string | number | null
  runId?: string | number | null
  taskId?: string | number | null
}

export interface ScheduledTaskResultDedupeKeyInput {
  signalKind: ScheduledTaskResultSignalKind
  taskId?: string | number | null
  runId?: string | number | null
  resultId?: string | number | null
  state?: string | null
  occurredAt?: string | null
}

export interface ScheduledTaskNotificationTarget {
  notificationId: number
  notificationIds: number[]
  signalKind: ScheduledTaskResultSignalKind
  resultId: string | null
  runId: string | null
  taskId: string | null
  href: string
  dedupeKey: string
  createdAt: string
  severity: string
}

const normalizeRouteId = (value: unknown): string | null => {
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

const readQueryParam = (
  url: string | null | undefined,
  names: readonly string[]
): string | null => {
  const normalizedUrl = normalizeRouteId(url)
  if (!normalizedUrl) {
    return null
  }

  try {
    const parsed = new URL(normalizedUrl, "http://tldw.local")
    for (const name of names) {
      const value = normalizeRouteId(parsed.searchParams.get(name))
      if (value) {
        return value
      }
    }
  } catch {
    return null
  }

  return null
}

const shouldTreatLinkIdAsResult = (linkType: string | null | undefined): boolean => {
  if (!linkType) {
    return false
  }

  const normalized = linkType.toLowerCase()
  return normalized.includes("result") || normalized.includes("output")
}

const shouldTreatLinkIdAsRun = (linkType: string | null | undefined): boolean => {
  if (!linkType) {
    return false
  }

  return linkType.toLowerCase().includes("run")
}

const shouldTreatLinkIdAsTask = (linkType: string | null | undefined): boolean => {
  if (!linkType) {
    return false
  }

  if (shouldTreatLinkIdAsResult(linkType) || shouldTreatLinkIdAsRun(linkType)) {
    return false
  }

  const normalized = linkType.toLowerCase()
  return normalized.includes("task") || normalized.includes("job")
}

const readNotificationField = (
  notification: NotificationItem,
  names: readonly string[]
): string | null => {
  const record = notification as unknown as Record<string, unknown>
  for (const name of names) {
    const value = normalizeRouteId(record[name])
    if (value) {
      return value
    }
  }
  return null
}

const buildTaskIdFromNotification = (notification: NotificationItem): string | null => {
  const sourceTaskId = normalizeRouteId(notification.source_task_id)
  if (sourceTaskId) {
    return sourceTaskId
  }

  const directTaskId = readNotificationField(notification, [
    "task_id",
    "taskId",
    "scheduled_task_id",
    "scheduledTaskId"
  ])
  if (directTaskId) {
    return directTaskId
  }

  const taskIdFromUrl = readQueryParam(notification.link_url, ["task_id", "taskId"])
  if (taskIdFromUrl) {
    return taskIdFromUrl
  }

  if (shouldTreatLinkIdAsTask(notification.link_type)) {
    const linkTaskId = normalizeRouteId(notification.link_id)
    if (linkTaskId) {
      return linkTaskId
    }
  }

  const sourceJobId = normalizeRouteId(notification.source_job_id)
  if (sourceJobId) {
    return `watchlist_job:${sourceJobId}`
  }

  return null
}

const inferNotificationSignalKind = (
  notification: NotificationItem,
  resultId: string | null
): ScheduledTaskResultSignalKind => {
  if (resultId) {
    return "result"
  }

  const normalized = [
    notification.kind,
    notification.severity,
    notification.link_type,
    notification.title,
    notification.message
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()

  if (
    normalized.includes("fail") ||
    normalized.includes("error") ||
    normalized.includes("blocked")
  ) {
    return "failure"
  }

  if (
    normalized.includes("running") ||
    normalized.includes("started") ||
    normalized.includes("processing")
  ) {
    return "running"
  }

  return "completed_no_results"
}

export const buildScheduledTaskResultHref = ({
  resultId,
  runId,
  taskId
}: ScheduledTaskResultHrefInput): string => {
  const params = new URLSearchParams()
  params.set("tab", "results")

  const normalizedResultId = normalizeRouteId(resultId)
  const normalizedRunId = normalizeRouteId(runId)
  const normalizedTaskId = normalizeRouteId(taskId)

  if (normalizedResultId) {
    params.set("result_id", normalizedResultId)
  } else if (normalizedRunId) {
    params.set("run_id", normalizedRunId)
  } else if (normalizedTaskId) {
    params.set("task_id", normalizedTaskId)
  }

  return `/scheduled-tasks?${params.toString()}`
}

export const buildScheduledTaskResultDedupeKey = ({
  signalKind,
  taskId,
  runId,
  resultId,
  state,
  occurredAt
}: ScheduledTaskResultDedupeKeyInput): string => {
  const normalizedResultId = normalizeRouteId(resultId)
  const normalizedRunId = normalizeRouteId(runId)
  const normalizedTaskId = normalizeRouteId(taskId) ?? "unknown"
  const normalizedOccurredAt = normalizeRouteId(occurredAt)
  const normalizedState = normalizeRouteId(state) ?? "unknown"

  if (signalKind === "result" && normalizedResultId) {
    return `result:${normalizedResultId}`
  }

  if (normalizedRunId) {
    return `run:${normalizedRunId}:state:${signalKind}`
  }

  return `task:${normalizedTaskId}:state:${signalKind}:time:${normalizedOccurredAt ?? normalizedState}`
}

export const normalizeScheduledTaskNotificationTarget = (
  notification: NotificationItem
): ScheduledTaskNotificationTarget | null => {
  const resultId =
    readNotificationField(notification, [
      "result_id",
      "resultId",
      "output_id",
      "outputId",
      "source_result_id",
      "sourceResultId",
      "source_output_id",
      "sourceOutputId"
    ]) ??
    (shouldTreatLinkIdAsResult(notification.link_type)
      ? normalizeRouteId(notification.link_id)
      : null) ??
    readQueryParam(notification.link_url, [
      "result_id",
      "resultId",
      "output_id",
      "outputId"
    ])
  const runId =
    normalizeRouteId(notification.source_task_run_id) ??
    readNotificationField(notification, [
      "run_id",
      "runId",
      "scheduled_task_run_id",
      "scheduledTaskRunId",
      "source_run_id",
      "sourceRunId"
    ]) ??
    (shouldTreatLinkIdAsRun(notification.link_type)
      ? normalizeRouteId(notification.link_id)
      : null) ??
    readQueryParam(notification.link_url, ["run_id", "runId"])
  const taskId = buildTaskIdFromNotification(notification)

  if (!resultId && !runId && !taskId) {
    return null
  }

  const href = buildScheduledTaskResultHref({ resultId, runId, taskId })
  const signalKind = inferNotificationSignalKind(notification, resultId)
  return {
    notificationId: notification.id,
    notificationIds: [notification.id],
    signalKind,
    resultId,
    runId,
    taskId,
    href,
    dedupeKey: buildScheduledTaskResultDedupeKey({
      signalKind,
      taskId,
      runId,
      resultId,
      state: notification.severity,
      occurredAt: notification.created_at
    }),
    createdAt: notification.created_at,
    severity: String(notification.severity)
  }
}

const targetIsNewer = (
  candidate: ScheduledTaskNotificationTarget,
  current: ScheduledTaskNotificationTarget
): boolean => {
  const candidateTime = Date.parse(candidate.createdAt)
  const currentTime = Date.parse(current.createdAt)
  if (Number.isFinite(candidateTime) && Number.isFinite(currentTime)) {
    return candidateTime > currentTime
  }
  return candidate.notificationId > current.notificationId
}

const uniqueNotificationIds = (
  first: readonly number[],
  second: readonly number[]
): number[] => Array.from(new Set([...first, ...second])).sort((a, b) => a - b)

export const mergeScheduledTaskNotificationTargets = (
  targets: Array<ScheduledTaskNotificationTarget | null | undefined>
): ScheduledTaskNotificationTarget[] => {
  const merged = new Map<string, ScheduledTaskNotificationTarget>()

  targets.forEach((target) => {
    if (!target) return

    const existing = merged.get(target.dedupeKey)
    if (!existing) {
      merged.set(target.dedupeKey, {
        ...target,
        notificationIds: uniqueNotificationIds(target.notificationIds, [])
      })
      return
    }

    const preferred = targetIsNewer(target, existing) ? target : existing
    merged.set(target.dedupeKey, {
      ...preferred,
      notificationIds: uniqueNotificationIds(existing.notificationIds, target.notificationIds)
    })
  })

  return Array.from(merged.values())
}
