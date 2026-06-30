const SUCCESS_RUN_STATUSES = new Set(["completed", "succeeded", "success"])
const ACTIVE_RUN_STATUSES = new Set(["pending", "running", "queued"])
const FAILED_RUN_STATUSES = new Set(["failed"])
const CANCELLED_RUN_STATUSES = new Set(["cancelled", "canceled"])

export const normalizeWatchlistRunStatus = (status: unknown): string => {
  const normalized = String(status ?? "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")

  if (SUCCESS_RUN_STATUSES.has(normalized)) return "completed"
  if (CANCELLED_RUN_STATUSES.has(normalized)) return "cancelled"
  return normalized
}

export const isWatchlistRunActive = (status: unknown): boolean =>
  ACTIVE_RUN_STATUSES.has(normalizeWatchlistRunStatus(status))

export const isWatchlistRunSuccessful = (status: unknown): boolean =>
  normalizeWatchlistRunStatus(status) === "completed"

export const isWatchlistRunTerminal = (status: unknown): boolean => {
  const normalized = normalizeWatchlistRunStatus(status)
  return (
    normalized === "completed" ||
    FAILED_RUN_STATUSES.has(normalized) ||
    normalized === "cancelled"
  )
}
