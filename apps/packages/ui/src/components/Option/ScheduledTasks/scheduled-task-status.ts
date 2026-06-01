import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"

export type ScheduledTaskStatusTone =
  | "success"
  | "processing"
  | "warning"
  | "error"
  | "default"

export type ScheduledTaskStatusKey =
  | "needs_attention"
  | "blocked"
  | "running"
  | "waiting"
  | "found_results"
  | "paused"
  | "disabled"
  | "draft"
  | "completed"

export interface ScheduledTaskProductStatus {
  key: ScheduledTaskStatusKey
  label: string
  tone: ScheduledTaskStatusTone
  description: string
}

export interface WatchlistTaskLinks {
  settingsUrl: string | null
  activityUrl: string | null
  reportsUrl: string | null
  latestRunUrl: string | null
  latestOutputUrl: string | null
}

type ScheduledTaskStatusInput =
  | ScheduledTask
  | {
      enabled: boolean
      status: string
      source_ref?: Record<string, unknown>
    }

type ScheduledTaskTypeInput = {
  primitive?: unknown
}

type WatchlistTaskLinkInput =
  | ScheduledTask
  | {
      primitive?: unknown
      manage_url?: string | null
      source_ref?: Record<string, unknown>
    }

const RUN_ID_KEYS = [
  "run_id",
  "runId",
  "last_run_id",
  "lastRunId",
  "latest_run_id",
  "latestRunId"
] as const

const OUTPUT_ID_KEYS = [
  "output_id",
  "outputId",
  "last_output_id",
  "lastOutputId",
  "latest_output_id",
  "latestOutputId"
] as const

const RESULT_COUNT_KEYS = [
  "result_count",
  "results_count",
  "output_count",
  "outputs_count"
] as const

export const SCHEDULED_TASK_ATTENTION_STATUS_KEYS: readonly ScheduledTaskStatusKey[] = [
  "needs_attention",
  "blocked"
] as const

export const isNativeReminderTask = (task: ScheduledTask): boolean =>
  task.primitive === "reminder_task" && task.edit_mode === "native"

export const scheduledTaskStatusToneToTagColor = (
  status: ScheduledTaskProductStatus
): string => {
  switch (status.tone) {
    case "success":
      return "green"
    case "processing":
      return "processing"
    case "warning":
      return "gold"
    case "error":
      return "red"
    default:
      return "default"
  }
}

export const formatScheduledTaskTimestamp = (
  timestamp: string | null | undefined,
  fallback: string
): string => {
  if (!timestamp) return fallback

  const parsed = new Date(timestamp)
  if (Number.isNaN(parsed.getTime())) {
    return timestamp
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(parsed)
}

const emptyWatchlistTaskLinks = (settingsUrl: string | null): WatchlistTaskLinks => ({
  settingsUrl,
  activityUrl: null,
  reportsUrl: null,
  latestRunUrl: null,
  latestOutputUrl: null
})

const statusIncludes = (status: string, tokens: readonly string[]): boolean => {
  const normalized = status.toLowerCase()
  return tokens.some((token) => {
    const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    return new RegExp(`(^|[^a-z0-9])${escapedToken}($|[^a-z0-9])`).test(
      normalized
    )
  })
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

const hasPositiveResultSignal = (sourceRef: Record<string, unknown>): boolean =>
  firstPositiveInteger(sourceRef, RESULT_COUNT_KEYS) !== null ||
  firstPositiveInteger(sourceRef, OUTPUT_ID_KEYS) !== null

export const getScheduledTaskProductStatus = (
  task: ScheduledTaskStatusInput
): ScheduledTaskProductStatus => {
  if (!task.enabled) {
    return {
      key: "disabled",
      label: "Disabled",
      tone: "default",
      description: "This task is turned off and will not run until enabled."
    }
  }

  const status = task.status || ""
  const sourceRef = task.source_ref || {}

  if (statusIncludes(status, ["draft"])) {
    return {
      key: "draft",
      label: "Draft",
      tone: "default",
      description: "This task is saved as a draft and is not ready to run."
    }
  }

  if (statusIncludes(status, ["running", "active", "processing", "in_progress"])) {
    return {
      key: "running",
      label: "Running now",
      tone: "processing",
      description: "This task is currently running."
    }
  }

  if (
    statusIncludes(status, ["blocked", "auth", "permission", "unavailable", "dependency"])
  ) {
    return {
      key: "blocked",
      label: "Blocked",
      tone: "warning",
      description: "This task cannot run until a required dependency is fixed."
    }
  }

  if (
    statusIncludes(status, ["fail", "failed", "failing", "failure", "error", "missed"])
  ) {
    return {
      key: "needs_attention",
      label: "Needs attention",
      tone: "error",
      description: "This task hit a problem during its latest run."
    }
  }

  if (
    statusIncludes(status, [
      "found",
      "match",
      "matched",
      "matches",
      "matching",
      "result",
      "results",
      "output",
      "outputs"
    ]) ||
    hasPositiveResultSignal(sourceRef)
  ) {
    return {
      key: "found_results",
      label: "Found results",
      tone: "success",
      description: "This task has produced results that are ready to review."
    }
  }

  if (statusIncludes(status, ["paused"])) {
    return {
      key: "paused",
      label: "Paused",
      tone: "warning",
      description: "This task is paused and will not run again until resumed."
    }
  }

  if (
    statusIncludes(status, ["complete", "completed", "success", "done", "finished"])
  ) {
    return {
      key: "completed",
      label: "Completed last run",
      tone: "success",
      description: "This task completed its latest run successfully."
    }
  }

  return {
    key: "waiting",
    label: "Waiting for next run",
    tone: "processing",
    description: "This task is enabled and waiting for its next scheduled run."
  }
}

export const getScheduledTaskTypeLabel = (task: ScheduledTaskTypeInput): string => {
  switch (task.primitive) {
    case "reminder_task":
      return "Reminder"
    case "watchlist_job":
      return "Watchlist monitor"
    default:
      return "Scheduled task"
  }
}

export const buildWatchlistTaskLinks = (
  task: WatchlistTaskLinkInput
): WatchlistTaskLinks => {
  if (task.primitive !== "watchlist_job") {
    return emptyWatchlistTaskLinks(task.manage_url || null)
  }

  const sourceRef = task.source_ref || {}
  const settingsUrl = task.manage_url || "/watchlists?tab=jobs"
  const jobId = toPositiveInteger(sourceRef.job_id)
  const latestRunId = firstPositiveInteger(sourceRef, RUN_ID_KEYS)
  const latestOutputId = firstPositiveInteger(sourceRef, OUTPUT_ID_KEYS)

  return {
    settingsUrl,
    activityUrl: jobId !== null ? `/watchlists?tab=runs&job_id=${jobId}` : null,
    reportsUrl: jobId !== null ? `/watchlists?tab=outputs&job_id=${jobId}` : null,
    latestRunUrl:
      latestRunId !== null
        ? `/watchlists?tab=runs&run_id=${latestRunId}&open_run=1`
        : null,
    latestOutputUrl:
      latestOutputId !== null
        ? `/watchlists?tab=outputs&output_id=${latestOutputId}&open_output=1`
        : null
  }
}
