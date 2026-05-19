import { useState, useCallback, useEffect } from "react"

export type HintId =
  | "keyboard-shortcuts"
  | "template-variables"
  | "gallery-view"
  | "bulk-actions"
  | "command-palette"
  | "filter-presets"

const STORAGE_PREFIX = "tldw-prompt-hint-dismissed-"
const MAX_SHOWS = 3
const MAX_SHOWS_BY_HINT: Partial<Record<HintId, number>> = {
  "filter-presets": 1,
}
const SHOW_COUNT_PREFIX = "tldw-prompt-hint-shown-"

function getMaxShows(id: HintId): number {
  return MAX_SHOWS_BY_HINT[id] ?? MAX_SHOWS
}

function isDismissed(id: HintId): boolean {
  try {
    return localStorage.getItem(`${STORAGE_PREFIX}${id}`) === "true"
  } catch {
    return false
  }
}

function getShowCount(id: HintId): number {
  try {
    return parseInt(localStorage.getItem(`${SHOW_COUNT_PREFIX}${id}`) ?? "0", 10) || 0
  } catch {
    return 0
  }
}

function incrementShowCount(id: HintId): void {
  try {
    const count = getShowCount(id) + 1
    localStorage.setItem(`${SHOW_COUNT_PREFIX}${id}`, String(count))
  } catch {
    // ignore
  }
}

function dismissHint(id: HintId): void {
  try {
    localStorage.setItem(`${STORAGE_PREFIX}${id}`, "true")
  } catch {
    // ignore
  }
}

export function useContextualHints() {
  const [dismissed, setDismissed] = useState<Set<HintId>>(new Set())

  useEffect(() => {
    const allIds: HintId[] = [
      "keyboard-shortcuts",
      "template-variables",
      "gallery-view",
      "bulk-actions",
      "command-palette",
      "filter-presets",
    ]
    const set = new Set<HintId>()
    for (const id of allIds) {
      if (isDismissed(id) || getShowCount(id) >= getMaxShows(id)) {
        set.add(id)
      }
    }
    setDismissed(set)
  }, [])

  const shouldShow = useCallback(
    (id: HintId): boolean => {
      return !dismissed.has(id)
    },
    [dismissed]
  )

  const dismiss = useCallback((id: HintId) => {
    dismissHint(id)
    setDismissed((prev) => new Set(prev).add(id))
  }, [])

  const markShown = useCallback(
    (id: HintId) => {
      const maxShows = getMaxShows(id)
      incrementShowCount(id)
      if (getShowCount(id) >= maxShows) {
        dismissHint(id)
        if (maxShows > 1) {
          setDismissed((prev) => new Set(prev).add(id))
        }
      }
    },
    []
  )

  return { shouldShow, dismiss, markShown }
}
