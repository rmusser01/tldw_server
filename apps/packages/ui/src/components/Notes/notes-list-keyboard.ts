/**
 * Keyboard and focus helpers for the notes list (AX-03, AX-05).
 *
 * The list is a single Tab stop: one row's open button carries tabIndex=0 and
 * the arrow, Home/End and Page keys move focus between rows. These helpers are
 * shared by the list panel, the sidebar and the page so that skip links, Esc
 * from the editor and the off-canvas dialog all agree on which row is active.
 */
import type React from 'react'

export const NOTES_LIST_SELECTOR = '[data-notes-list]'
export const NOTES_LIST_ROW_SELECTOR = '[data-notes-list-item]'
export const NOTES_LIST_PAGE_STEP = 10

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]'
].join(',')

/** Keyboard-reachable elements inside `container`, in DOM order. */
export const getFocusableElements = (container: ParentNode | null | undefined): HTMLElement[] => {
  if (!container) return []
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) =>
      element.getAttribute('tabindex') !== '-1' &&
      !element.hasAttribute('hidden') &&
      !element.closest('[inert]') &&
      !element.closest('[aria-hidden="true"]')
  )
}

/**
 * Keep Tab and Shift+Tab inside `container` (a modal surface). Returns true
 * when focus was wrapped.
 */
export const trapTabKey = (
  container: HTMLElement,
  event: Pick<KeyboardEvent, 'key' | 'shiftKey' | 'preventDefault'>
): boolean => {
  if (event.key !== 'Tab') return false
  const focusable = getFocusableElements(container)
  if (focusable.length === 0) {
    event.preventDefault()
    container.focus()
    return true
  }
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  const active = typeof document !== 'undefined' ? document.activeElement : null
  const activeInside = active instanceof HTMLElement && container.contains(active)
  if (event.shiftKey) {
    if (!activeInside || active === first || active === container) {
      event.preventDefault()
      last.focus()
      return true
    }
    return false
  }
  if (!activeInside || active === last) {
    event.preventDefault()
    first.focus()
    return true
  }
  return false
}

/** The row that currently holds the list's single Tab stop. */
export const getActiveNotesListRow = (
  root: ParentNode | null | undefined = typeof document !== 'undefined' ? document : null
): HTMLElement | null => {
  if (!root) return null
  const list = root.querySelector<HTMLElement>(NOTES_LIST_SELECTOR)
  if (!list || list.closest('[inert]')) return null
  return (
    list.querySelector<HTMLElement>(`${NOTES_LIST_ROW_SELECTOR}[tabindex="0"]`) ||
    list.querySelector<HTMLElement>(NOTES_LIST_ROW_SELECTOR)
  )
}

/** Move focus to the active list row. Returns false when no row is reachable. */
export const focusActiveNotesListRow = (
  root: ParentNode | null | undefined = typeof document !== 'undefined' ? document : null
): boolean => {
  const row = getActiveNotesListRow(root)
  if (!row) return false
  row.focus()
  return true
}

/**
 * Resolve the row a navigation key should move to, or null when the key is
 * not a list navigation key.
 */
export const resolveNotesListNavigationIndex = (
  key: string,
  currentIndex: number,
  rowCount: number
): number | null => {
  if (rowCount <= 0) return null
  const last = rowCount - 1
  switch (key) {
    case 'ArrowDown':
      return Math.min(last, currentIndex + 1)
    case 'ArrowUp':
      return Math.max(0, currentIndex - 1)
    case 'Home':
      return 0
    case 'End':
      return last
    case 'PageDown':
      return Math.min(last, currentIndex + NOTES_LIST_PAGE_STEP)
    case 'PageUp':
      return Math.max(0, currentIndex - NOTES_LIST_PAGE_STEP)
    default:
      return null
  }
}

/** Arrow/Home/End/Page navigation between row buttons inside the list. */
export const moveNotesListRowFocus = (event: React.KeyboardEvent<HTMLElement>): boolean => {
  const rows = Array.from(
    event.currentTarget
      .closest(NOTES_LIST_SELECTOR)
      ?.querySelectorAll<HTMLElement>(NOTES_LIST_ROW_SELECTOR) ?? []
  )
  const currentIndex = rows.indexOf(event.currentTarget)
  if (currentIndex < 0) return false
  const nextIndex = resolveNotesListNavigationIndex(event.key, currentIndex, rows.length)
  if (nextIndex == null) return false
  event.preventDefault()
  rows[nextIndex]?.focus()
  return true
}

/** Focus the editor body (WYSIWYG root, else the Markdown textarea, else the region). */
export const focusNotesEditorBody = (refs: {
  richEditorRef?: React.RefObject<HTMLElement | null>
  contentTextareaRef?: React.RefObject<HTMLTextAreaElement | null>
  editorRegionId: string
}): boolean => {
  if (typeof document === 'undefined') return false
  const isUsable = (element: HTMLElement | null | undefined): element is HTMLElement =>
    Boolean(
      element &&
        element.isConnected &&
        !(element as HTMLTextAreaElement).disabled &&
        !element.closest('[inert]')
    )
  const rich = refs.richEditorRef?.current
  if (isUsable(rich)) {
    rich.focus({ preventScroll: true })
    return true
  }
  const textarea = refs.contentTextareaRef?.current
  if (isUsable(textarea)) {
    textarea.focus({ preventScroll: true })
    return true
  }
  const region = document.getElementById(refs.editorRegionId)
  if (isUsable(region)) {
    region.focus({ preventScroll: true })
    return true
  }
  return false
}

/** True when a keydown comes from an open combobox popup that owns Escape. */
export const isEscapeOwnedByPopup = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false
  if (target.getAttribute('aria-expanded') === 'true') return true
  return Boolean(target.closest('[role="dialog"][aria-modal="true"]:not([data-notes-list-dialog])'))
}
