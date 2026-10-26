import React from 'react'
import type { MarkdownToolbarAction } from '../notes-manager-utils'

type FormattingKeyEvent = Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey'>

/**
 * Ctrl/Cmd+B and Ctrl/Cmd+I, as the Notes shortcut sheet documents (XP-07).
 * Shift or Alt chords stay with the page's other shortcuts.
 */
export const notesFormattingActionForKey = (
  event: FormattingKeyEvent
): Extract<MarkdownToolbarAction, 'bold' | 'italic'> | null => {
  if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return null
  const key = event.key.toLowerCase()
  if (key === 'b') return 'bold'
  if (key === 'i') return 'italic'
  return null
}

/**
 * Run the editor's bold/italic toolbar actions from the keyboard, then hand
 * every other key to the editor's own keydown handler.
 */
export const useNotesFormattingKeyDown = <E extends HTMLElement>(
  handleKeyDown: (event: React.KeyboardEvent<E>) => void,
  applyMarkdownToolbarAction: (action: MarkdownToolbarAction) => void
) =>
  React.useCallback(
    (event: React.KeyboardEvent<E>) => {
      const target = event.currentTarget as E & { readOnly?: boolean }
      const action = event.nativeEvent.isComposing || target.readOnly
        ? null
        : notesFormattingActionForKey(event)
      if (action) {
        event.preventDefault()
        applyMarkdownToolbarAction(action)
        return
      }
      handleKeyDown(event)
    },
    [applyMarkdownToolbarAction, handleKeyDown]
  )
