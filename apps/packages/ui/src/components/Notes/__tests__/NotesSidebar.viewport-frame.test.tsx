import React from 'react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import {
  buildNotesSidebarProps,
  translateWithDefaults
} from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

/** jsdom has no layout, so give the controls scroller explicit geometry. */
const setScrollGeometry = (
  element: HTMLElement,
  geometry: { scrollHeight: number; clientHeight: number; scrollTop: number }
) => {
  Object.defineProperty(element, 'scrollHeight', {
    configurable: true,
    value: geometry.scrollHeight
  })
  Object.defineProperty(element, 'clientHeight', {
    configurable: true,
    value: geometry.clientHeight
  })
  element.scrollTop = geometry.scrollTop
}

describe('NotesSidebar viewport frame (NO-01)', () => {
  beforeEach(() => {
    localStorage.clear()
  })
  afterEach(cleanup)

  it('fills the page frame instead of pinning a window-derived pixel height', () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)
    const region = screen.getByTestId('notes-list-region')
    // A fixed innerHeight-120px height left a void under the sidebar and
    // pushed the document past the viewport.
    expect(region.style.height).toBe('')
    expect(region.style.minHeight).toBe('')
    expect(region.className).toContain('h-full')
    expect(region.className).toContain('min-h-0')
  })

  it('says when sidebar controls are cut off and scrolls them into view on request', () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)
    const controls = screen.getByTestId('notes-sidebar-controls')
    expect(screen.queryByTestId('notes-sidebar-controls-more')).toBeNull()

    setScrollGeometry(controls, {
      scrollHeight: 900,
      clientHeight: 400,
      scrollTop: 0
    })
    act(() => {
      fireEvent.scroll(controls)
    })
    // A pointer-only cue: keyboard focus already scrolls the controls, and a
    // focused cue would drop focus when it unmounts.
    const more = screen.getByText('More filters below').closest('button')!
    expect(more).toHaveAttribute('tabindex', '-1')
    expect(more).toHaveAttribute('aria-hidden', 'true')
    expect(screen.getByTestId('notes-sidebar-controls-more')).toBeInTheDocument()

    fireEvent.click(more)
    expect(controls.scrollTop).toBeGreaterThan(0)

    setScrollGeometry(controls, {
      scrollHeight: 900,
      clientHeight: 400,
      scrollTop: 500
    })
    act(() => {
      fireEvent.scroll(controls)
    })
    expect(screen.queryByTestId('notes-sidebar-controls-more')).toBeNull()
  })
})
