/**
 * Keyboard shortcuts hook for Characters Playground
 *
 * Shortcuts:
 * - N: New character (when no modal open)
 * - E: Edit selected/hovered character
 * - /: Focus search input
 * - Escape: Close modal / Clear selection
 * - G T: Switch to table view
 * - G G: Switch to gallery view
 */

import { useEffect, useRef, useCallback } from 'react'
import { isEditableTarget } from "@/utils/editable-target"

interface UseCharacterShortcutsOptions {
  /** Whether any modal is currently open */
  modalOpen: boolean
  /** Callback to open the new character modal */
  onNewCharacter: () => void
  /** Callback to focus the search input */
  onFocusSearch: () => void
  /** Callback to close current modal */
  onCloseModal: () => void
  /** Callback to switch to table view */
  onTableView: () => void
  /** Callback to switch to gallery view */
  onGalleryView: () => void
  /** Callback to edit currently selected/hovered character */
  onEditSelected?: () => void
  /** Whether shortcuts are enabled */
  enabled?: boolean
}

interface ShortcutInfo {
  key: string
  description: string
  chord?: string
}

export const CHARACTER_SHORTCUTS: ShortcutInfo[] = [
  { key: 'N', description: 'New character' },
  { key: 'E', description: 'Edit selected character' },
  { key: '/', description: 'Focus search' },
  { key: 'Escape', description: 'Close modal' },
  { key: 'G T', description: 'Table view', chord: 'g' },
  { key: 'G G', description: 'Gallery view', chord: 'g' }
]

export function useCharacterShortcuts(options: UseCharacterShortcutsOptions) {
  const {
    modalOpen,
    onNewCharacter,
    onFocusSearch,
    onCloseModal,
    onTableView,
    onGalleryView,
    onEditSelected,
    enabled = true
  } = options

  // For chord shortcuts (like G T, G G)
  const chordKeyRef = useRef<string | null>(null)
  const chordTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearChord = useCallback(() => {
    chordKeyRef.current = null
    if (chordTimeoutRef.current) {
      clearTimeout(chordTimeoutRef.current)
      chordTimeoutRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in an input/textarea
      const target = e.target as HTMLElement
      const isTyping = isEditableTarget(target)

      // Always allow Escape
      if (e.key === 'Escape') {
        if (modalOpen) {
          e.preventDefault()
          onCloseModal()
        }
        clearChord()
        return
      }

      // Skip other shortcuts if typing or modal open
      if (isTyping || modalOpen) {
        clearChord()
        return
      }

      // Handle chord sequences (G + T, G + G)
      if (chordKeyRef.current === 'g') {
        clearChord()
        if (e.key === 't' || e.key === 'T') {
          e.preventDefault()
          onTableView()
          return
        }
        if (e.key === 'g' || e.key === 'G') {
          e.preventDefault()
          onGalleryView()
          return
        }
      }

      // Start chord with G
      if (e.key === 'g' || e.key === 'G') {
        chordKeyRef.current = 'g'
        // Clear chord after 1 second if no follow-up
        chordTimeoutRef.current = setTimeout(clearChord, 1000)
        return
      }

      // Single key shortcuts
      switch (e.key) {
        case 'n':
        case 'N':
          e.preventDefault()
          onNewCharacter()
          break
        case 'e':
        case 'E':
          if (onEditSelected) {
            e.preventDefault()
            onEditSelected()
          }
          break
        case '/':
          e.preventDefault()
          onFocusSearch()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      clearChord()
    }
  }, [
    enabled,
    modalOpen,
    onNewCharacter,
    onFocusSearch,
    onCloseModal,
    onTableView,
    onGalleryView,
    onEditSelected,
    clearChord
  ])
}

/**
 * Helper component to display keyboard shortcut hints
 */
export function formatShortcutKey(key: string): string {
  // Handle special keys
  const keyMap: Record<string, string> = {
    Escape: 'Esc',
    ' ': 'Space',
    ArrowUp: '↑',
    ArrowDown: '↓',
    ArrowLeft: '←',
    ArrowRight: '→'
  }
  return keyMap[key] || key.toUpperCase()
}
