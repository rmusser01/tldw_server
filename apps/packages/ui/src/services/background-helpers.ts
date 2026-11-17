import { browser } from "wxt/browser"
import { openSidepanel } from "@/utils/sidepanel"

export const ensureSidepanelOpen = openSidepanel

// WP-60: how many distinct funnels the sidepanel-open guard remembers before
// resetting. Funnel ids are unique per ingest, so this only bounds memory in
// a long-lived worker; after a reset the worst case is one redundant
// sidePanel.setOptions/open call for a still-active funnel.
export const SIDEPANEL_OPEN_GUARD_CAPACITY = 500

// Status broadcasts carry a correlation id (`funnelId` for media ingests,
// `sessionId` for quick ingests). The sidepanel only needs to be OPENED once
// per funnel — the first status emit still opens it when the panel is closed
// (e.g. a context-menu ingest), while every later broadcast for the same
// funnel skips the redundant sidePanel.setOptions/open round trip.
// Uncorrelated broadcasts keep the legacy always-open behavior.
export const createSidepanelOpenOnceGuard = (
  capacity: number = SIDEPANEL_OPEN_GUARD_CAPACITY
): ((correlationId: unknown) => boolean) => {
  let openedFor = new Set<string>()
  return (correlationId: unknown): boolean => {
    const key =
      typeof correlationId === "string" && correlationId.trim()
        ? correlationId
        : null
    if (key === null) return true
    if (openedFor.has(key)) return false
    if (openedFor.size >= capacity) openedFor = new Set()
    openedFor.add(key)
    return true
  }
}

const pickBroadcastCorrelationId = (
  payload: Record<string, unknown> | undefined
): unknown => {
  if (payload && typeof payload === "object") {
    if (typeof payload.funnelId === "string") return payload.funnelId
    if (typeof payload.sessionId === "string") return payload.sessionId
  }
  return undefined
}

export type BackgroundStatusBroadcaster = (
  tabId: number | undefined,
  type: string,
  payload?: Record<string, unknown>
) => Promise<void>

export type BackgroundBroadcasterDependencies = {
  openSidepanel: (tabId?: number) => Promise<void>
  sendMessage: (message: unknown) => Promise<unknown>
  logError: (label: string, error: unknown) => void
}

// Broadcast a status message to the extension's UI contexts. Extracted from
// `entries/background.ts` (WP-60) so the once-per-funnel sidepanel-open guard
// and the 500ms send retry are directly unit-testable. The runtime message
// itself is NEVER suppressed — only the redundant sidepanel open attempts are.
export const createBackgroundStatusBroadcaster = (
  deps: BackgroundBroadcasterDependencies
): BackgroundStatusBroadcaster => {
  const shouldAttemptOpen = createSidepanelOpenOnceGuard()
  return async (tabId, type, payload) => {
    if (shouldAttemptOpen(pickBroadcastCorrelationId(payload))) {
      void deps.openSidepanel(tabId).catch((error) =>
        deps.logError("open sidebar", error)
      )
    }
    const message = { from: "background", type, payload }
    try {
      await deps.sendMessage(message)
    } catch (error) {
      deps.logError(`send ${type}`, error)
      setTimeout(() => {
        Promise.resolve()
          .then(() => deps.sendMessage(message))
          .catch((retryError) =>
            deps.logError(`send ${type} (retry)`, retryError)
          )
      }, 500)
    }
  }
}

export const pickFirstString = (value: any, keys: string[], visited?: WeakSet<object>): string | null => {
  if (!value) return null
  if (typeof value === "string" && value.trim().length > 0) return value.trim()

  if (typeof value === "object") {
    const visitedSet = visited ?? new WeakSet<object>()

    if (visitedSet.has(value as object)) {
      return null
    }
    visitedSet.add(value as object)

    if (Array.isArray(value)) {
      for (const item of value) {
        const found = pickFirstString(item, keys, visitedSet)
        if (found) return found
      }
      return null
    }

    for (const key of keys) {
      const maybe = (value as any)[key]
      if (typeof maybe === "string" && maybe.trim().length > 0) {
        return maybe.trim()
      }
    }
    for (const nestedValue of Object.values(value)) {
      const found = pickFirstString(nestedValue, keys, visitedSet)
      if (found) return found
    }
  }

  return null
}

export const extractTranscriptionPieces = (data: any): { transcript: string | null; summary: string | null } => {
  const transcript = pickFirstString(data, ["transcript", "transcription", "text", "raw_text", "content"])
  const summary = pickFirstString(data, ["summary", "analysis", "overview", "abstract"])
  return { transcript, summary }
}

export const clampText = (value: string | null, max = 8000): string | null => {
  if (!value) return null
  const trimmed = value.trim()
  if (!trimmed) return null
  return trimmed.length > max ? `${trimmed.slice(0, max)}…` : trimmed
}

export const notify = (title: string, message: string) => {
  try {
    if (typeof chrome !== "undefined" && chrome.notifications?.create) {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "/icon.png",
        title,
        message
      })
      return
    }
    browser.notifications?.create?.({
      type: "basic",
      iconUrl: "/icon.png",
      title,
      message
    })
  } catch {
    // ignore notification errors
  }
}
