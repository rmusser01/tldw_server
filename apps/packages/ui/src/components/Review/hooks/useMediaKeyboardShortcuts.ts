import { useState, useEffect } from 'react'
import type { MediaResultItem } from '@/components/Media/types'
import { isEditableTarget } from "@/utils/editable-target"

export interface UseMediaKeyboardShortcutsDeps {
  hasNext: boolean
  hasPrevious: boolean
  page: number
  totalPages: number
  displayResults: MediaResultItem[]
  selectedIndex: number
  searchCollapsed: boolean
  setSearchCollapsed: React.Dispatch<React.SetStateAction<boolean>>
  searchInputRef: React.RefObject<HTMLInputElement | null>
  setSelected: React.Dispatch<React.SetStateAction<MediaResultItem | null>>
  setPage: React.Dispatch<React.SetStateAction<number>>
}

export function useMediaKeyboardShortcuts(deps: UseMediaKeyboardShortcutsDeps) {
  const {
    hasNext,
    hasPrevious,
    page,
    totalPages,
    displayResults,
    selectedIndex,
    searchCollapsed,
    setSearchCollapsed,
    searchInputRef,
    setSelected,
    setPage
  } = deps

  const [shortcutsOverlayOpen, setShortcutsOverlayOpen] = useState(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (isEditableTarget(e.target)) {
        return
      }

      switch (e.key) {
        case '?':
          e.preventDefault()
          setShortcutsOverlayOpen((prev) => !prev)
          break
        case '/':
          if (e.ctrlKey || e.metaKey || e.altKey) break
          e.preventDefault()
          if (searchCollapsed) {
            setSearchCollapsed(false)
            window.setTimeout(() => {
              const input = searchInputRef.current
              if (!input) return
              input.focus()
              input.select()
            }, 0)
          } else {
            const input = searchInputRef.current
            if (input) {
              input.focus()
              input.select()
            }
          }
          break
        case 'j':
          if (hasNext) {
            e.preventDefault()
            setSelected(displayResults[selectedIndex + 1])
          }
          break
        case 'k':
          if (hasPrevious) {
            e.preventDefault()
            setSelected(displayResults[selectedIndex - 1])
          }
          break
        case 'ArrowLeft':
          if (page > 1) {
            e.preventDefault()
            setPage((p) => p - 1)
          }
          break
        case 'ArrowRight':
          if (page < totalPages) {
            e.preventDefault()
            setPage((p) => p + 1)
          }
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    displayResults,
    hasNext,
    hasPrevious,
    page,
    searchCollapsed,
    searchInputRef,
    selectedIndex,
    setPage,
    setSearchCollapsed,
    setSelected,
    totalPages,
  ])

  return {
    shortcutsOverlayOpen,
    setShortcutsOverlayOpen,
  }
}
