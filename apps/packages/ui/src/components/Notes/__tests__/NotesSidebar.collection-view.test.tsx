import React from 'react'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import {
  buildNotesSidebarProps,
  translateWithDefaults
} from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

const collection = { id: 7, name: 'Inspiration', description: null, version: 1 }

describe('NotesSidebar Collection view (NL-13)', () => {
  beforeEach(() => {
    localStorage.clear()
  })
  afterEach(cleanup)

  it('shows the collection picker in the Collection view without opening Organize', () => {
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          listViewMode: 'moodboard',
          moodboards: [collection],
          selectedMoodboardId: 7,
          selectedMoodboard: collection
        })}
      />
    )
    const view = screen.getByTestId('notes-moodboard-view')
    // Organize stays collapsed; the controls must not depend on it.
    expect(screen.getByTestId('notes-section-organize-toggle')).toHaveAttribute(
      'aria-expanded',
      'false'
    )
    expect(within(view).getByTestId('notes-moodboard-controls')).toBeVisible()
    expect(within(view).getByTestId('notes-moodboard-select')).toBeInTheDocument()
    expect(within(view).getByTestId('notes-moodboard-create')).toBeInTheDocument()
  })

  it('gives the no-collections empty state a New collection action', () => {
    const createMoodboard = vi.fn(async () => {})
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({ listViewMode: 'moodboard', createMoodboard })}
      />
    )
    const empty = screen.getByTestId('notes-moodboard-empty-selection')
    const create = within(empty).getByRole('button', { name: 'New collection' })
    fireEvent.click(create)
    expect(createMoodboard).toHaveBeenCalledTimes(1)
  })

  it('adds the open note to an empty collection', () => {
    const addNoteToMoodboard = vi.fn(async () => {})
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          listViewMode: 'moodboard',
          moodboards: [collection],
          selectedMoodboardId: 7,
          selectedMoodboard: collection,
          selectedId: 'note-42',
          addNoteToMoodboard
        })}
      />
    )
    const empty = screen.getByTestId('notes-moodboard-empty')
    expect(empty).toHaveTextContent('Open a note, then choose Add open note to put it here.')
    fireEvent.click(screen.getByRole('button', { name: 'Add open note' }))
    expect(addNoteToMoodboard).toHaveBeenCalledWith('note-42')
  })

  it('keeps Add open note disabled when no note is open or it is already here', () => {
    const { rerender } = render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          listViewMode: 'moodboard',
          moodboards: [collection],
          selectedMoodboardId: 7,
          selectedMoodboard: collection
        })}
      />
    )
    expect(screen.getByRole('button', { name: 'Add open note' })).toBeDisabled()

    rerender(
      <NotesSidebar
        {...buildNotesSidebarProps({
          listViewMode: 'moodboard',
          moodboards: [collection],
          selectedMoodboardId: 7,
          selectedMoodboard: collection,
          selectedId: 3,
          total: 1,
          filteredCount: 1,
          visibleNotes: [{ id: 3, title: 'Already pinned', membership_source: 'manual' }]
        })}
      />
    )
    expect(screen.getByRole('button', { name: 'Add open note' })).toBeDisabled()
  })
})
