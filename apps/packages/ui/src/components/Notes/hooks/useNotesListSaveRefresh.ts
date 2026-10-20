import React from 'react'
import type { QueryClient } from '@tanstack/react-query'
import { patchSavedNoteInListCache } from '../notes-list-cache'

/** How long a save waits before refreshing the notes list in the background (NS-04). */
export const NOTES_LIST_REFRESH_DELAY_MS = 1500

type SavedNote = Parameters<typeof patchSavedNoteInListCache>[2]
type SavedNoteFallback = Parameters<typeof patchSavedNoteInListCache>[3]

/**
 * Keep the notes list in step with a saved note without reloading it (NS-04).
 *
 * The returned callback patches the saved row in every cached list right away
 * and then schedules a quiet background refresh to pick up server ordering.
 * The refresh is not awaited, and saves that land close together (autosave
 * plus Ctrl+S, for example) share one refresh.
 */
export function useNotesListSaveRefresh(
  queryClient: QueryClient,
  refetch: () => Promise<unknown>,
  delayMs: number = NOTES_LIST_REFRESH_DELAY_MS
): (noteId: string | number, saved: SavedNote, fallback?: SavedNoteFallback) => void {
  const refetchRef = React.useRef(refetch)
  refetchRef.current = refetch
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(
    () => () => {
      if (timerRef.current != null) clearTimeout(timerRef.current)
      timerRef.current = null
    },
    []
  )

  return React.useCallback(
    (noteId, saved, fallback) => {
      patchSavedNoteInListCache(queryClient, noteId, saved, fallback)
      if (timerRef.current != null) clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => {
        timerRef.current = null
        void Promise.resolve()
          .then(() => refetchRef.current())
          .catch((error) => {
            console.debug('[NotesManagerPage] Background list refresh failed:', error)
          })
      }, delayMs)
    },
    [delayMs, queryClient]
  )
}
