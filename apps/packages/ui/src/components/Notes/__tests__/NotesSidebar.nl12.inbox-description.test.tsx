/**
 * NL-12 review: the Inbox view's description ("Notes saved from browser
 * capture.") was in a hover-only tooltip. It is now the button's accessible
 * description, so keyboard, screen-reader and touch users get it too.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar from '../NotesSidebar'
import { buildNotesSidebarProps, translateWithDefaults } from './notes-sidebar-test-props'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: translateWithDefaults })
}))

describe('NotesSidebar Inbox description', () => {
  afterEach(cleanup)

  it('describes the Inbox view without needing a hover', () => {
    render(<NotesSidebar {...buildNotesSidebarProps()} />)

    expect(screen.getByTestId('notes-view-mode-inbox')).toHaveAccessibleDescription(
      'Notes saved from browser capture.'
    )
  })
})
