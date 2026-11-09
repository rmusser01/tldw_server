import {
  recordWatchlistsIaExperimentTelemetry
} from "@/services/watchlists"
import type {
  WatchlistsIaExperimentTelemetryPayload,
  WatchlistsIaExperimentVariant
} from "@/types/watchlists"

export const WATCHLISTS_IA_EXPERIMENT_STORAGE_KEY = "watchlists:ia-experiment:v1"

const MAX_VISITED_TABS = 64

export interface WatchlistsIaExperimentTelemetryState {
  version: 2
  variant: WatchlistsIaExperimentVariant
  session_id: string
  transitions: number
  visited_tabs: string[]
  first_seen_at: string
  last_seen_at: string
}

const createSessionId = (): string =>
  `wl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`

const normalizeTab = (value: string | null | undefined): string | null => {
  const normalized = String(value || "").trim().toLowerCase()
  return normalized.length > 0 ? normalized.slice(0, 64) : null
}

const normalizeVisitedTabs = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const unique = new Set<string>()
  for (const entry of value) {
    const tab = normalizeTab(typeof entry === "string" ? entry : null)
    if (!tab || unique.has(tab)) continue
    unique.add(tab)
    if (unique.size >= MAX_VISITED_TABS) break
  }
  return Array.from(unique)
}

export const readWatchlistsIaExperimentTelemetryState =
  (): WatchlistsIaExperimentTelemetryState | null => {
    if (typeof window === "undefined") return null
    try {
      const raw = localStorage.getItem(WATCHLISTS_IA_EXPERIMENT_STORAGE_KEY)
      if (!raw) return null
      const parsed = JSON.parse(raw) as Partial<WatchlistsIaExperimentTelemetryState>
      const variant =
        parsed.variant === "baseline" || parsed.variant === "experimental"
          ? parsed.variant
          : null
      if (!variant) return null
      const sessionId = String(parsed.session_id || "").trim()
      if (!sessionId) return null
      return {
        version: 2,
        variant,
        session_id: sessionId,
        transitions: Math.max(0, Number(parsed.transitions) || 0),
        visited_tabs: normalizeVisitedTabs(parsed.visited_tabs),
        first_seen_at:
          typeof parsed.first_seen_at === "string" && parsed.first_seen_at
            ? parsed.first_seen_at
            : new Date().toISOString(),
        last_seen_at:
          typeof parsed.last_seen_at === "string" && parsed.last_seen_at
            ? parsed.last_seen_at
            : new Date().toISOString()
      }
    } catch {
      return null
    }
  }

const updateWatchlistsIaExperimentTelemetryState = (
  previousTab: string | null,
  currentTab: string,
  variant: WatchlistsIaExperimentVariant
): WatchlistsIaExperimentTelemetryState | null => {
  if (typeof window === "undefined") return null

  const normalizedCurrentTab = normalizeTab(currentTab) || "unknown"
  const normalizedPreviousTab = normalizeTab(previousTab)
  const nowIso = new Date().toISOString()

  const existingState = readWatchlistsIaExperimentTelemetryState()
  const baseState: WatchlistsIaExperimentTelemetryState =
    existingState && existingState.variant === variant
      ? existingState
      : {
          version: 2,
          variant,
          session_id: createSessionId(),
          transitions: 0,
          visited_tabs: [],
          first_seen_at: nowIso,
          last_seen_at: nowIso
        }

  const visitedSet = new Set<string>(normalizeVisitedTabs(baseState.visited_tabs))
  if (normalizedPreviousTab) visitedSet.add(normalizedPreviousTab)
  visitedSet.add(normalizedCurrentTab)

  const nextState: WatchlistsIaExperimentTelemetryState = {
    version: 2,
    variant,
    session_id: baseState.session_id || createSessionId(),
    transitions:
      baseState.transitions +
      (normalizedPreviousTab && normalizedPreviousTab !== normalizedCurrentTab
        ? 1
        : 0),
    visited_tabs: Array.from(visitedSet).slice(0, MAX_VISITED_TABS),
    first_seen_at: baseState.first_seen_at || nowIso,
    last_seen_at: nowIso
  }

  try {
    localStorage.setItem(
      WATCHLISTS_IA_EXPERIMENT_STORAGE_KEY,
      JSON.stringify(nextState)
    )
  } catch {
    // localStorage may be unavailable.
  }

  return nextState
}

export const trackWatchlistsIaExperimentTransition = (
  previousTab: string | null,
  currentTab: string,
  variant: WatchlistsIaExperimentVariant
): WatchlistsIaExperimentTelemetryState | null => {
  const nextState = updateWatchlistsIaExperimentTelemetryState(previousTab, currentTab, variant)
  if (!nextState) return null
  const normalizedCurrentTab = normalizeTab(currentTab) || "unknown"
  const normalizedPreviousTab = normalizeTab(previousTab)

  const payload: WatchlistsIaExperimentTelemetryPayload = {
    variant: nextState.variant,
    session_id: nextState.session_id,
    previous_tab: normalizedPreviousTab,
    current_tab: normalizedCurrentTab,
    transitions: nextState.transitions,
    visited_tabs: nextState.visited_tabs,
    first_seen_at: nextState.first_seen_at,
    last_seen_at: nextState.last_seen_at
  }

  void recordWatchlistsIaExperimentTelemetry(payload).catch(() => {
    // Telemetry sink failures are non-blocking; local state remains available.
  })

  return nextState
}

export const flushWatchlistsIaExperimentSession = (
  currentTab: string,
  variant: WatchlistsIaExperimentVariant
): WatchlistsIaExperimentTelemetryState | null =>
  // pagehide cannot safely start async network work; keep its final timestamp local.
  updateWatchlistsIaExperimentTelemetryState(currentTab, currentTab, variant)
