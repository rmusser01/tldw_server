import React from 'react'
import { isEditableTarget } from '@/utils/editable-target'
import {
  NOTES_LIST_SELECTOR,
  NOTES_LIST_ROW_SELECTOR,
  focusActiveNotesListRow,
  focusNotesEditorBody,
  isEscapeOwnedByPopup
} from '../notes-list-keyboard'

type NoteRef = { id: string | number }

type UseNotesKeyboardNavigationOptions = {
  visibleNotes: readonly NoteRef[]
  selectedId: string | number | null
  selectNote: (id: string | number) => Promise<boolean | void>
  richEditorRef?: React.RefObject<HTMLElement | null>
  contentTextareaRef?: React.RefObject<HTMLTextAreaElement | null>
  editorRegionId: string
}

const scheduleFrame = (callback: () => void) => {
  if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
    window.requestAnimationFrame(callback)
    return
  }
  setTimeout(callback, 0)
}

const focusListRowById = (id: string | number): boolean => {
  if (typeof document === 'undefined') return false
  const list = document.querySelector(NOTES_LIST_SELECTOR)
  const rows = Array.from(list?.querySelectorAll<HTMLElement>(NOTES_LIST_ROW_SELECTOR) ?? [])
  const row = rows.find((element) => element.dataset.noteRowId === String(id))
  if (!row) return false
  row.focus()
  return true
}

/**
 * AX-03: keyboard paths between the notes list and the editor.
 *
 * - Enter on a row opens the note and moves focus into the editor body.
 * - Escape in the editor returns focus to the active list row.
 * - Alt+ArrowUp / Alt+ArrowDown open the previous / next note (outside text fields).
 */
export const useNotesKeyboardNavigation = ({
  visibleNotes,
  selectedId,
  selectNote,
  richEditorRef,
  contentTextareaRef,
  editorRegionId
}: UseNotesKeyboardNavigationOptions) => {
  const latestRef = React.useRef({ visibleNotes, selectedId, selectNote })
  latestRef.current = { visibleNotes, selectedId, selectNote }

  const focusEditorBody = React.useCallback(
    () => focusNotesEditorBody({ richEditorRef, contentTextareaRef, editorRegionId }),
    [contentTextareaRef, editorRegionId, richEditorRef]
  )

  const openNoteAndFocusEditor = React.useCallback(
    async (id: string | number) => {
      const opened = await latestRef.current.selectNote(id)
      if (opened === false) return
      // Wait for the loaded note to render (the editor is disabled while loading).
      scheduleFrame(() => {
        if (!focusEditorBody()) scheduleFrame(() => void focusEditorBody())
      })
    },
    [focusEditorBody]
  )

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return

      if (event.key === 'Escape' && !event.altKey && !event.ctrlKey && !event.metaKey) {
        const target = event.target
        if (!(target instanceof Element)) return
        const editorRegion = document.getElementById(editorRegionId)
        if (!editorRegion || !editorRegion.contains(target)) return
        if (isEscapeOwnedByPopup(target)) return
        if (focusActiveNotesListRow()) event.preventDefault()
        return
      }

      if (
        event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        (event.key === 'ArrowDown' || event.key === 'ArrowUp')
      ) {
        if (isEditableTarget(event.target)) return
        const { visibleNotes: notes, selectedId: currentId } = latestRef.current
        if (notes.length === 0) return
        const currentIndex =
          currentId == null ? -1 : notes.findIndex((note) => String(note.id) === String(currentId))
        const nextIndex =
          event.key === 'ArrowDown'
            ? Math.min(notes.length - 1, currentIndex + 1)
            : currentIndex < 0
              ? 0
              : Math.max(0, currentIndex - 1)
        const next = notes[nextIndex]
        if (!next || String(next.id) === String(currentId)) return
        event.preventDefault()
        const focusWasInList =
          event.target instanceof Element && Boolean(event.target.closest(NOTES_LIST_SELECTOR))
        void latestRef.current.selectNote(next.id).then((opened) => {
          if (opened === false || !focusWasInList) return
          scheduleFrame(() => void focusListRowById(next.id))
        })
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [editorRegionId])

  return { openNoteAndFocusEditor, focusEditorBody }
}
