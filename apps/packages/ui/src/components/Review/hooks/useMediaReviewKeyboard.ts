import React from "react"
import { shouldHandleGlobalShortcut } from "@/components/Review/interaction-context"
import type { MediaReviewState } from "@/components/Review/media-review-types"
import type { MediaReviewActions } from "@/components/Review/media-review-types"

export function useMediaReviewKeyboard(
  s: MediaReviewState,
  actions: MediaReviewActions
): void {
  const {
    t, message,
    focusedId, previewedId, readingActive, setHelpModalOpen,
    selectedIds,
    helpModalOpen,
    compareDiffOpen,
    selectedItemsDrawerOpen,
    searchInputRef,
    lastEscapePressRef,
    setContentExpandedIds,
    isMobileViewport,
    autoViewMode,
    manualViewModePinned,
    setAutoModeInlineNotice,
    setViewModeState,
    prevAutoViewModeRef,
    allResults
  } = s

  const { goRelative, clearSelectionWithGuard, addVisibleToSelection, previewItem, toggleSelect } = actions

  // Global keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!shouldHandleGlobalShortcut(e.target)) return
      if (helpModalOpen || compareDiffOpen || selectedItemsDrawerOpen) return

      switch (e.key) {
        case 'a':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            addVisibleToSelection()
          }
          break
        case '/':
          if (e.ctrlKey || e.metaKey || e.altKey) break
          e.preventDefault()
          {
            const searchTarget =
              searchInputRef.current ??
              document.querySelector<HTMLInputElement>(
                'input[aria-label="Search media (title/content)"]'
              )
            if (searchTarget) {
              searchTarget.focus()
              searchTarget.select()
            }
          }
          break
        case 'ArrowDown':
          e.preventDefault()
          goRelative(1)
          break
        case 'ArrowUp':
          e.preventDefault()
          goRelative(-1)
          break
        case 'j':
          e.preventDefault()
          goRelative(1)
          break
        case 'k':
          e.preventDefault()
          goRelative(-1)
          break
        case 'x':
        case ' ':
          // x or Space: toggle selection on previewed/focused item (Gmail-style)
          if (e.key === ' ' && (e.ctrlKey || e.metaKey || e.altKey)) break
          {
            const targetId = readingActive ? focusedId : previewedId
            if (targetId != null) {
              e.preventDefault()
              void toggleSelect(targetId)
            }
          }
          break
        case 'o':
          e.preventDefault()
          {
            const targetId = readingActive ? focusedId : previewedId
            if (targetId == null) break
            const key = String(targetId)
            setContentExpandedIds(prev => {
              const next = new Set(prev)
              if (next.has(key)) next.delete(key)
              else next.add(key)
              return next
            })
          }
          break
        case '?':
          if (e.shiftKey) {
            e.preventDefault()
            setHelpModalOpen(true)
          }
          break
        case 'Escape':
          e.preventDefault()
          if (selectedIds.length > 5) {
            const now = Date.now()
            if (now - lastEscapePressRef.current < 500) {
              clearSelectionWithGuard()
              lastEscapePressRef.current = 0
            } else {
              lastEscapePressRef.current = now
              message.info(t('mediaPage.escapeDoubleTapHint', 'Press Escape again to clear {{count}} items', { count: selectedIds.length }), 2)
            }
          } else {
            clearSelectionWithGuard()
          }
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [
    goRelative,
    readingActive, setHelpModalOpen,
    focusedId,
    previewedId,
    selectedIds.length,
    clearSelectionWithGuard,
    t,
    addVisibleToSelection,
    previewItem,
    toggleSelect,
    helpModalOpen,
    compareDiffOpen,
    selectedItemsDrawerOpen,
    searchInputRef,
    lastEscapePressRef,
    setContentExpandedIds,
    message,
    allResults
  ])

  // Auto-select view mode by item count with notification
  React.useEffect(() => {
    if (isMobileViewport) {
      prevAutoViewModeRef.current = "list"
      setAutoModeInlineNotice(null)
      return
    }
    if (!autoViewMode) {
      setAutoModeInlineNotice(null)
      return
    }
    if (manualViewModePinned) return
    const count = selectedIds.length
    if (count === 0) {
      setAutoModeInlineNotice(null)
      return
    }

    let newMode: "spread" | "list" | "all"
    if (count === 1) newMode = "list"
    else if (count <= 4) newMode = "spread"
    else newMode = "all"

    if (prevAutoViewModeRef.current !== null && prevAutoViewModeRef.current !== newMode) {
      const modeNames = { spread: t('mediaPage.spreadMode', 'Side-by-side'), list: t('mediaPage.listMode', 'Focus'), all: t('mediaPage.allMode', 'Stack') }
      const notice = t('mediaPage.autoViewModeSwitched', 'Auto-switched to {{mode}} view ({{count}} items)', {
        mode: modeNames[newMode],
        count
      })
      setAutoModeInlineNotice(notice)
      message.info(notice, 3)
    }

    prevAutoViewModeRef.current = newMode
    setViewModeState(newMode)
  }, [isMobileViewport, selectedIds.length, autoViewMode, t, manualViewModePinned, message, prevAutoViewModeRef, setAutoModeInlineNotice, setViewModeState])
}
