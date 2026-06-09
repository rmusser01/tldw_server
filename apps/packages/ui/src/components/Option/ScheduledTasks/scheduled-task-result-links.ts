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
  resultId: string | null
  runId: string | null
  taskId: string | null
  href: string
  dedupeKey: string
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

const buildTaskIdFromNotification = (notification: NotificationItem): string | null => {
  const sourceTaskId = normalizeRouteId(notification.source_task_id)
  if (sourceTaskId) {
    return sourceTaskId
  }

  const taskIdFromUrl = readQueryParam(notification.link_url, ["task_id", "taskId"])
  if (taskIdFromUrl) {
    return taskIdFromUrl
  }

  const sourceJobId = normalizeRouteId(notification.source_job_id)
  if (sourceJobId) {
    return `watchlist_job:${sourceJobId}`
  }

  return null
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
    readQueryParam(notification.link_url, ["run_id", "runId"])
  const taskId = buildTaskIdFromNotification(notification)

  if (!resultId && !runId && !taskId) {
    return null
  }

  const href = buildScheduledTaskResultHref({ resultId, runId, taskId })
  return {
    notificationId: notification.id,
    resultId,
    runId,
    taskId,
    href,
    dedupeKey: buildScheduledTaskResultDedupeKey({
      signalKind: resultId ? "result" : "failure",
      taskId,
      runId,
      resultId,
      state: notification.severity,
      occurredAt: notification.created_at
    })
  }
}
