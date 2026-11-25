import React from 'react'
import userEvent from '@testing-library/user-event'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within
} from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import { buildNotesSidebarProps } from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === 'string')
        return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return String(defaultValueOrOptions.defaultValue).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(defaultValueOrOptions[token] ?? '')
        )
      }
      return key
    }
  })
}))

const handleSelectNote = vi.fn(async () => {})

const notes = Array.from({ length: 5 }, (_, index) => ({
  id: String(index + 1),
  title: 'Note ' + (index + 1),
  content: 'Saved body',
  version: 1
}))
const expandedProps = buildNotesSidebarProps({
  total: 5,
  visibleNotes: notes,
  filteredCount: 5,
  recentNotes: notes.slice(0, 3).map(({ id, title }) => ({ id, title })),
  handleSelectNote
})

describe('Notes expanded sidebar result access', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      value: 1280
    })
    Object.defineProperty(window, 'innerHeight', {
      configurable: true,
      value: 720
    })
  })
  afterEach(cleanup)
  it('bounds expanded controls and reserves a scrollable result area with five notes and three recents', async () => {
    const user = userEvent.setup()
    render(<NotesSidebar {...expandedProps} />)
    const filters = screen.getByTestId('notes-section-filters-toggle')
    if (filters.getAttribute('aria-expanded') !== 'true')
      await user.click(filters)
    expect(
      screen.getByTestId('notes-section-view-organize-toggle')
    ).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getAllByTestId(/^notes-recent-item-/)).toHaveLength(3)
    const controls = screen.getByTestId('notes-sidebar-controls')
    expect(controls.style.maxHeight).toBe('50%')
    expect(controls.className).toContain('overflow-y-auto')
    const results = screen.getByTestId('notes-sidebar-results')
    expect(results.className).toContain('min-h-[240px]')
    expect(screen.getByTestId('notes-results-scroll').className).toContain(
      'overflow-auto'
    )
    const fifth = within(results).getByTestId('notes-open-button-5')
    await user.click(fifth)
    expect(handleSelectNote).toHaveBeenCalledWith('5')
    // The results are one Tab stop that remembers the last row used (AX-03).
    filters.focus()
    for (
      let step = 0;
      step < 80 && !document.activeElement?.hasAttribute('data-notes-list-item');
      step += 1
    ) {
      await user.tab()
    }
    expect(fifth).toHaveFocus()
    await user.keyboard('{Home}')
    const first = within(results).getByTestId('notes-open-button-1')
    expect(first).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(handleSelectNote).toHaveBeenCalledWith('1')
  })
  it('retains expanded controls after reload and keeps results available on resize and mobile', async () => {
    const user = userEvent.setup()
    const view = render(<NotesSidebar {...expandedProps} />)
    const filters = screen.getByTestId('notes-section-filters-toggle')
    if (filters.getAttribute('aria-expanded') !== 'true')
      await user.click(filters)
    view.unmount()
    const restored = render(<NotesSidebar {...expandedProps} />)
    expect(screen.getByTestId('notes-section-filters-toggle')).toHaveAttribute(
      'aria-expanded',
      'true'
    )
    fireEvent(window, new Event('resize'))
    // The desktop sidebar fills the viewport-framed page; it no longer
    // tracks window.innerHeight (NO-01).
    expect(screen.getByTestId('notes-list-region').className).toContain(
      'h-full'
    )
    restored.rerender(
      <NotesSidebar {...expandedProps} isMobileViewport mobileSidebarOpen />
    )
    expect(screen.getByTestId('notes-list-region')).toHaveStyle({
      height: '100%'
    })
    await user.click(screen.getByTestId('notes-open-button-5'))
    expect(handleSelectNote).toHaveBeenCalledWith('5')
  })
})
