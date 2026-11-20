/**
 * Decision 19 (#3101): a saved filter always matches all of its tags, and
 * its description says so.
 */
import React from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import { buildNotesSidebarProps, translateWithDefaults } from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

// The section remembers being open, so open it only when it is closed.
const openOrganize = () => {
  const toggle = screen.getByTestId('notes-section-organize-toggle')
  if (toggle.getAttribute('aria-expanded') !== 'true') fireEvent.click(toggle)
}

describe('NotesSidebar saved filter description', () => {
  afterEach(cleanup)

  it('says a saved filter shows notes with all of its tags, and names them', () => {
    const reading = { id: 7, name: 'Reading', keywords: ['rag', 'baking'] }
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          notebookOptions: [reading],
          selectedNotebookId: 7,
          selectedNotebook: reading
        })}
      />
    )
    openOrganize()

    expect(screen.getByTestId('notes-notebook-helper-text')).toHaveTextContent(
      'Shows notes that have all of its tags: rag, baking.'
    )
  })

  it('says what a saved filter with no tags shows', () => {
    const empty = { id: 8, name: 'Empty', keywords: [] as string[] }
    render(
      <NotesSidebar
        {...buildNotesSidebarProps({
          notebookOptions: [empty],
          selectedNotebookId: 8,
          selectedNotebook: empty
        })}
      />
    )
    openOrganize()

    expect(screen.getByTestId('notes-notebook-helper-text')).toHaveTextContent(
      'This saved filter has no tags, so it does not narrow the list.'
    )
  })
})
