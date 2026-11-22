/**
 * NL-07: the sort menu offers "Best match" while a text search is active,
 * and only then (the plain list has no relevance to sort by).
 */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import { buildNotesSidebarProps, translateWithDefaults } from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

const openSortMenu = async (): Promise<string[]> => {
  const select = screen.getByTestId('notes-sort-select')
  fireEvent.mouseDown(select.querySelector('.ant-select-content') || select)
  return waitFor(() => {
    const options = Array.from(document.querySelectorAll('.ant-select-item-option')).map(
      (option) => option.textContent || ''
    )
    if (options.length === 0) throw new Error('sort menu did not open')
    return options
  })
}

describe('NotesSidebar Best match sort (NL-07)', () => {
  afterEach(cleanup)

  it('shows and offers Best match during a text search', async () => {
    const setSortOption = vi.fn()
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          query: 'sourdough',
          sortOption: 'relevance',
          sortBestMatchAvailable: true,
          setSortOption
        })}
      />
    )

    expect(screen.getByTestId('notes-sort-select')).toHaveTextContent('Best match')
    const options = await openSortMenu()
    expect(options[0]).toBe('Best match')

    const dateOption = Array.from(document.querySelectorAll('.ant-select-item-option')).find(
      (option) => option.textContent === 'Date modified (newest first)'
    )
    fireEvent.click(dateOption as Element)
    expect(setSortOption).toHaveBeenCalledWith('modified_desc')
  })

  it('does not offer Best match without a text search', async () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)

    const options = await openSortMenu()
    expect(options).not.toContain('Best match')
    expect(options).toContain('Date modified (newest first)')
  })
})
