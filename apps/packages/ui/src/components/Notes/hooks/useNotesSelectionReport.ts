import React from 'react'
import { getSetting } from '@/services/settings/registry'
import { LAST_NOTE_ID_SETTING, NOTES_RECENT_OPENED_SETTING } from '@/services/settings/ui-settings'
import { notesAuthoritySetting } from '../notes-authority-storage'

export type NotesSelectionReport = {
  /** The open note, or null for an empty editor or an unsaved draft. */
  noteId: string | null
  /** The open note's title as shown in the editor. */
  title: string
  /**
   * The note this account opened last, offered while nothing has been opened
   * on the page yet and no draft is in progress (XP-05).
   */
  lastOpenedNoteId: string | null
}

export type NotesSelectionListener = (report: NotesSelectionReport) => void

/**
 * Tell the Notes route which note is open, so it can keep the address and
 * page title in step (XP-05). Does nothing without a listener.
 */
export const useNotesSelectionReport = (
  listener: NotesSelectionListener | undefined,
  {
    selectedId,
    title,
    authorityScope,
    hasDraft
  }: {
    selectedId: string | number | null
    title: string
    authorityScope: string | null | undefined
    hasDraft: boolean
  }
) => {
  const listening = typeof listener === 'function'
  const [lastOpened, setLastOpened] = React.useState<{ scope: string; noteId: string | null } | null>(null)
  const openedOnPageRef = React.useRef(false)

  React.useEffect(() => {
    if (!listening || !authorityScope) return
    let cancelled = false
    void (async () => {
      // An older "open this note" hint is handled by the page itself.
      const hinted = await getSetting(LAST_NOTE_ID_SETTING).catch(() => null)
      const recentSetting = notesAuthoritySetting(NOTES_RECENT_OPENED_SETTING, authorityScope)
      const recent = hinted || !recentSetting ? null : await getSetting(recentSetting).catch(() => null)
      const first = Array.isArray(recent) ? recent[0] : null
      const noteId = first && typeof first === 'object' ? String((first as { id?: unknown }).id ?? '').trim() : ''
      if (!cancelled) setLastOpened({ scope: authorityScope, noteId: noteId || null })
    })()
    return () => {
      cancelled = true
    }
  }, [authorityScope, listening])

  const noteId = selectedId == null ? null : String(selectedId)
  if (noteId) openedOnPageRef.current = true
  const lastOpenedNoteId =
    !openedOnPageRef.current && !hasDraft && lastOpened?.scope === authorityScope
      ? lastOpened.noteId
      : null

  React.useEffect(() => {
    listener?.({ noteId, title, lastOpenedNoteId })
  }, [listener, noteId, title, lastOpenedNoteId])
}
