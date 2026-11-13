import React from 'react'
import type { NoteListItem } from '../notes-manager-types'

/**
 * One bulk action at a time, with progress and a failure list (NL-05, #3101).
 *
 * Bulk delete, add-tags and restore each walk their notes one request at a
 * time. Before this hook nothing showed how far a run had got, and a second
 * action could start while the first was still sending requests. A run now
 * holds the bar: `begin` refuses while another run is active, `start`/`step`
 * report n/N, and `finish` keeps the notes that failed so the bar can list
 * them with Retry.
 */

export type NotesBulkAction = 'delete' | 'addTags' | 'restore'

export type NotesBulkProgress = {
  action: NotesBulkAction
  done: number
  /** 0 until the run knows its size (after any confirm or prompt). */
  total: number
}

export type NotesBulkFailures = {
  action: NotesBulkAction
  notes: NoteListItem[]
  /** The tags an add-tags run was adding, so Retry does not ask again. */
  tags?: string[]
}

export type NotesBulkRunHandle = {
  start: (total: number) => void
  /** One note finished, whatever its outcome. */
  step: () => void
  /** Ends the run. Failed notes stay listed until dismissed or the next run starts. */
  finish: (failed?: NoteListItem[], tags?: string[]) => void
}

export function useNotesBulkRun(authorityScope?: string | null) {
  const activeRunRef = React.useRef<symbol | null>(null)
  const [progress, setProgress] = React.useState<NotesBulkProgress | null>(null)
  const [failures, setFailures] = React.useState<NotesBulkFailures | null>(null)

  // Another account's failed notes are not this account's to retry.
  React.useEffect(() => {
    setFailures(null)
  }, [authorityScope])

  const begin = React.useCallback((action: NotesBulkAction): NotesBulkRunHandle | null => {
    if (activeRunRef.current) return null
    const run = Symbol(action)
    activeRunRef.current = run
    let done = 0
    let total = 0
    const isActive = () => activeRunRef.current === run
    setFailures(null)
    setProgress({ action, done, total })
    return {
      start: (count) => {
        if (!isActive()) return
        total = Math.max(0, count)
        setProgress({ action, done, total })
      },
      step: () => {
        if (!isActive()) return
        done += 1
        setProgress({ action, done, total })
      },
      finish: (failed = [], tags) => {
        if (!isActive()) return
        activeRunRef.current = null
        setProgress(null)
        setFailures(failed.length > 0 ? { action, notes: failed.slice(), ...(tags ? { tags } : {}) } : null)
      }
    }
  }, [])

  const dismissFailures = React.useCallback(() => setFailures(null), [])

  return { progress, failures, isRunning: progress != null, begin, dismissFailures }
}
