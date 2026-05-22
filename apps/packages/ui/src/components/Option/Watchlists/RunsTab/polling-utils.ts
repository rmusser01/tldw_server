import type { WatchlistRun } from "@/types/watchlists"
import { isWatchlistRunActive } from "../shared/runStatus"

const MIN_POLL_INTERVAL_MS = 100
const IDLE_POLL_MIN_MS = 30_000
const BACKGROUND_POLL_MIN_MS = 60_000
const DEFAULT_ACTIVE_PAGE_SIZE = 25
const DEFAULT_IDLE_PAGE_SIZE = 10

const normalizePositiveInt = (value: number, fallback: number): number => {
  if (!Number.isFinite(value)) return fallback
  const normalized = Math.floor(value)
  if (normalized <= 0) return fallback
  return normalized
}

export const hasActiveWatchlistRuns = (
  runs: Array<Pick<WatchlistRun, "status">> | null | undefined
): boolean => {
  if (!Array.isArray(runs) || runs.length === 0) return false
  return runs.some((run) => isWatchlistRunActive(run.status))
}

export const resolveAdaptiveRunNotificationsPollMs = (
  basePollMs: number,
  options: {
    documentHidden: boolean
    hasActiveRuns: boolean
  }
): number => {
  const normalizedBase = Math.max(
    MIN_POLL_INTERVAL_MS,
    normalizePositiveInt(basePollMs, IDLE_POLL_MIN_MS / 2)
  )

  if (options.documentHidden) {
    return Math.max(normalizedBase * 4, BACKGROUND_POLL_MIN_MS)
  }

  if (!options.hasActiveRuns) {
    return Math.max(normalizedBase * 2, IDLE_POLL_MIN_MS)
  }

  return normalizedBase
}

export const resolveRunNotificationsPageSize = (options: {
  documentHidden: boolean
  hasActiveRuns: boolean
  activePageSize?: number
  idlePageSize?: number
}): number => {
  const activePageSize = normalizePositiveInt(
    Number(options.activePageSize),
    DEFAULT_ACTIVE_PAGE_SIZE
  )
  const idlePageSize = normalizePositiveInt(
    Number(options.idlePageSize),
    DEFAULT_IDLE_PAGE_SIZE
  )

  if (options.documentHidden || !options.hasActiveRuns) {
    return idlePageSize
  }

  return activePageSize
}
