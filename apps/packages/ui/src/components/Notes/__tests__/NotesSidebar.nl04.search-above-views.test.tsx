/**
 * Notes search sits above Views (NL-04, paired with NL-12 and decision 16):
 * it is the sidebar's first control after the header, so the reader reaches
 * it first by eye and by Tab, in Notes and in Trash alike.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import { buildNotesSidebarProps, translateWithDefaults } from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

const precedes = (first: Element, second: Element): boolean =>
  Boolean(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING)

describe('NotesSidebar search placement', () => {
  afterEach(cleanup)

  it('puts the search and sort row above Views, after New note', () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)

    const newNote = screen.getByTestId('notes-new-note-button')
    const search = screen.getByRole('textbox', { name: 'Search notes' })
    const sort = screen.getByTestId('notes-sort-select')
    const views = screen.getByTestId('notes-section-view-organize')

    expect(precedes(newNote, search)).toBe(true)
    expect(precedes(search, sort)).toBe(true)
    expect(precedes(sort, views)).toBe(true)
  })

  it('keeps the filters below Views', () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)

    expect(
      precedes(
        screen.getByTestId('notes-section-view-organize'),
        screen.getByTestId('notes-section-filters')
      )
    ).toBe(true)
  })

  it('puts the Trash search above Views in Trash', () => {
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          listMode: 'trash',
          trashQuery: '',
          setTrashQuery: vi.fn()
        })}
      />
    )

    expect(
      precedes(
        screen.getByTestId('notes-trash-search'),
        screen.getByTestId('notes-section-view-organize')
      )
    ).toBe(true)
  })
})
