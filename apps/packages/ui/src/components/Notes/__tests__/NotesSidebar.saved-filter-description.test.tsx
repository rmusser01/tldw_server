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
    fireEvent.click(screen.getByTestId('notes-section-organize-toggle'))

    expect(screen.getByTestId('notes-notebook-helper-text')).toHaveTextContent(
      'Shows notes that have all of its tags: rag, baking.'
    )
  })
})
