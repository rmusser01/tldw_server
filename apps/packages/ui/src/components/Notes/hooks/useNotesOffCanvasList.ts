import React from 'react'
import {
  getFocusableElements,
  isEscapeOwnedByPopup,
  trapTabKey
} from '../notes-list-keyboard'

export const NOTES_SEARCH_INPUT_ATTR = 'data-notes-search-input'

type UseNotesOffCanvasListOptions = {
  isMobileViewport: boolean
  mobileSidebarOpen: boolean
  sidebarCollapsed: boolean
  setMobileSidebarOpen: (open: boolean) => void
}

/**
 * AX-05: the notes list is off-canvas on mobile and collapsible on desktop.
 * While hidden it is inert (no Tab stops, hidden from assistive tech). While
 * open on mobile it acts as a modal dialog: focus moves to search, Tab stays
 * inside, Escape closes it and focus returns to whatever opened it.
 */
export const useNotesOffCanvasList = ({
  isMobileViewport,
  mobileSidebarOpen,
  sidebarCollapsed,
  setMobileSidebarOpen
}: UseNotesOffCanvasListOptions) => {
  const asideRef = React.useRef<HTMLElement | null>(null)
  const returnFocusRef = React.useRef<HTMLElement | null>(null)
  const wasModalRef = React.useRef(false)
  const hidden = isMobileViewport ? !mobileSidebarOpen : sidebarCollapsed
  const isModal = isMobileViewport && mobileSidebarOpen

  // React 18 has no `inert` prop, so set the attribute directly.
  React.useLayoutEffect(() => {
    asideRef.current?.toggleAttribute('inert', hidden)
  }, [hidden])

  React.useEffect(() => {
    if (typeof document === 'undefined') return
    const aside = asideRef.current
    if (isModal && !wasModalRef.current) {
      const active = document.activeElement
      returnFocusRef.current =
        active instanceof HTMLElement && active !== document.body && !aside?.contains(active)
          ? active
          : null
      const searchInput = aside?.querySelector<HTMLElement>(`[${NOTES_SEARCH_INPUT_ATTR}]`)
      const target = searchInput || getFocusableElements(aside)[0] || aside
      target?.focus()
    } else if (!isModal && wasModalRef.current) {
      const returnTarget = returnFocusRef.current
      returnFocusRef.current = null
      const active = document.activeElement
      const focusLost =
        !active || active === document.body || Boolean(aside && aside.contains(active))
      if (returnTarget?.isConnected && focusLost) returnTarget.focus()
    }
    wasModalRef.current = isModal
  }, [isModal])

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLElement>) => {
      if (!isModal) return
      // React bubbles keys from portaled popovers (Search tips, tag menus) up
      // to the list; those keys belong to the popover, not the dialog.
      if (!(event.target instanceof Node) || !event.currentTarget.contains(event.target)) return
      if (event.key === 'Escape') {
        if (event.isDefaultPrevented() || isEscapeOwnedByPopup(event.target)) return
        event.preventDefault()
        event.stopPropagation()
        setMobileSidebarOpen(false)
        return
      }
      if (event.key === 'Tab') {
        trapTabKey(event.currentTarget, event)
      }
    },
    [isModal, setMobileSidebarOpen]
  )

  return { asideRef, hidden, isModal, handleKeyDown }
}

type RevealNotesListSearchOptions = {
  isMobileViewport: boolean
  mobileSidebarOpen: boolean
  sidebarCollapsed: boolean
  setMobileSidebarOpen: (open: boolean) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  focusSearch: () => void
}

/**
 * The "/" and Ctrl/Cmd+K search shortcuts (AX-05). A hidden list is inert, so
 * focusing its search box would do nothing: show the list first, then focus
 * search once it has rendered.
 */
export const revealNotesListAndFocusSearch = ({
  isMobileViewport,
  mobileSidebarOpen,
  sidebarCollapsed,
  setMobileSidebarOpen,
  setSidebarCollapsed,
  focusSearch
}: RevealNotesListSearchOptions) => {
  const hidden = isMobileViewport ? !mobileSidebarOpen : sidebarCollapsed
  if (!hidden) {
    focusSearch()
    return
  }
  if (isMobileViewport) setMobileSidebarOpen(true)
  else setSidebarCollapsed(false)
  if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
    window.requestAnimationFrame(() => focusSearch())
  } else {
    setTimeout(focusSearch, 0)
  }
}
