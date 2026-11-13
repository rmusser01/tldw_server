import React from 'react'
import type { NoteListItem } from '../notes-manager-types'

/**
 * "Select all N matching" for the notes list (NL-05, #3101).
 *
 * The page selection only holds the visible rows. Selecting every match pages
 * through the same endpoint the list reads (the library, a search or tag
 * filter, or Trash), so the set is exactly what the list would show, and keeps
 * the rows: bulk delete, Add tags, export and restore then run on them as they
 * do on page rows. It stops at SELECT_ALL_MATCHING_CAP notes.
 *
 * The selection belongs to one list: `filterKey` names it (account, mode,
 * search, tags, saved filter, sort; not the page). Once the key changes the
 * selection no longer applies and a gather in flight is abandoned.
 */

/** Most notes one "Select all matching" holds: ten requests of 100. */
export const SELECT_ALL_MATCHING_CAP = 1000
const SELECT_ALL_MATCHING_PAGE_SIZE = 100

export type NotesMatchingPageFetcher = (
  page: number,
  pageSize: number,
  signal: AbortSignal
) => Promise<{ items: NoteListItem[]; total: number }>

export type NotesMatchingSelection = {
  key: string
  notes: NoteListItem[]
  /** Matches the server reported, which can exceed `notes` at the cap. */
  total: number
}

/** What the list header and bulk bar need to offer page and all-matching selection. */
export type NotesBulkSelectionControls = {
  pageCount: number
  total: number
  allOnPageSelected: boolean
  matching: { count: number; total: number } | null
  selectingAllMatching: boolean
  matchingCap: number
  onSelectAllOnPage: (checked: boolean) => void
  onSelectAllMatching: () => void
}

type Options = {
  filterKey: string
  /** Null where the view cannot gather matches (a moodboard). */
  fetchPage: NotesMatchingPageFetcher | null
  onError: (error: unknown) => void
}

export function useNotesMatchingSelection({ filterKey, fetchPage, onError }: Options) {
  const [selection, setSelection] = React.useState<NotesMatchingSelection | null>(null)
  const [selecting, setSelecting] = React.useState(false)
  const filterKeyRef = React.useRef(filterKey)
  filterKeyRef.current = filterKey
  const controllerRef = React.useRef<AbortController | null>(null)

  React.useEffect(() => () => controllerRef.current?.abort(), [])
  React.useEffect(() => {
    controllerRef.current?.abort()
  }, [filterKey])

  const matching = selection && selection.key === filterKey ? selection : null

  const selectAllMatching = React.useCallback(async () => {
    if (!fetchPage) return
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    const key = filterKey
    const isCurrent = () => !controller.signal.aborted && filterKeyRef.current === key
    setSelecting(true)
    try {
      const notes: NoteListItem[] = []
      const seen = new Set<string>()
      let total = 0
      for (let page = 1; notes.length < SELECT_ALL_MATCHING_CAP; page += 1) {
        const result = await fetchPage(page, SELECT_ALL_MATCHING_PAGE_SIZE, controller.signal)
        if (!isCurrent()) return
        total = result.total
        let added = 0
        for (const note of result.items) {
          const id = String(note.id)
          if (seen.has(id)) continue
          seen.add(id)
          notes.push(note)
          added += 1
          if (notes.length >= SELECT_ALL_MATCHING_CAP) break
        }
        // A short or repeated page, or the reported total, ends the list.
        if (result.items.length < SELECT_ALL_MATCHING_PAGE_SIZE || added === 0) break
        if (page * SELECT_ALL_MATCHING_PAGE_SIZE >= total) break
      }
      setSelection({ key, notes, total: Math.max(total, notes.length) })
    } catch (error) {
      if (isCurrent()) onError(error)
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setSelecting(false)
      }
    }
  }, [fetchPage, filterKey, onError])

  const clearMatchingSelection = React.useCallback(() => {
    controllerRef.current?.abort()
    setSelection(null)
  }, [])

  return { matching, selectingAllMatching: selecting, selectAllMatching, clearMatchingSelection }
}
