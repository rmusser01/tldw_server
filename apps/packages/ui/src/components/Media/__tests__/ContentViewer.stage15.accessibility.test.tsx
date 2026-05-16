import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'

const registryLabels = vi.hoisted(() => ({
  loading: 'Loading via registry'
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || _key
    }
  })
}))

vi.mock('@/design-system', async (importActual) => {
  const actual = await importActual<typeof import('@/design-system')>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === 'loading' ? registryLabels.loading : state.label
        }
      }
    )
  }
})

vi.mock('@/hooks/useSetting', async () => {
  const React = await import('react')
  return {
    useSetting: (setting: { defaultValue: unknown }) => {
      const [value, setValue] = React.useState(setting.defaultValue)
      const setAsync = async (next: unknown | ((prev: unknown) => unknown)) => {
        setValue((prev) =>
          typeof next === 'function' ? (next as (prev: unknown) => unknown)(prev) : next
        )
      }
      return [value, setAsync, { isLoading: false }] as const
    }
  }
})

vi.mock('@/hooks/useMediaReadingProgress', () => ({
  useMediaReadingProgress: vi.fn()
}))

vi.mock('../AnalysisModal', () => ({
  AnalysisModal: () => null
}))

vi.mock('../AnalysisEditModal', () => ({
  AnalysisEditModal: () => null
}))

vi.mock('../VersionHistoryPanel', () => ({
  VersionHistoryPanel: () => null
}))

vi.mock('../DeveloperToolsSection', () => ({
  DeveloperToolsSection: () => null
}))

vi.mock('../DiffViewModal', () => ({
  DiffViewModal: () => null
}))

vi.mock('@/components/Common/MarkdownPreview', () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))

const mediaOne = {
  kind: 'media' as const,
  id: 1101,
  title: 'Accessibility Item One',
  raw: {},
  meta: { type: 'document' }
}

const mediaTwo = {
  kind: 'media' as const,
  id: 1102,
  title: 'Accessibility Item Two',
  raw: {},
  meta: { type: 'document' }
}

describe('ContentViewer stage 15 content announcements', () => {
  it('announces loading and ready status when content state changes', async () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={mediaOne}
        content=""
        mediaDetail={{ type: 'document' }}
        isDetailLoading
      />
    )

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Loading via registry Accessibility Item One')
    )

    rerender(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body"
        mediaDetail={{ type: 'document' }}
        isDetailLoading={false}
      />
    )
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
  })

  it('updates announcements for pointer, keyboard, and programmatic selection changes', async () => {
    const user = userEvent.setup()

    function Harness() {
      const [selected, setSelected] = React.useState(mediaOne)
      return (
        <div>
          <button type="button" onClick={() => setSelected(mediaOne)}>
            Select One
          </button>
          <button type="button" onClick={() => setSelected(mediaTwo)}>
            Select Two
          </button>
          <ContentViewer
            selectedMedia={selected}
            content="Body"
            mediaDetail={{ type: 'document' }}
          />
        </div>
      )
    }

    render(<Harness />)

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )

    await user.click(screen.getByRole('button', { name: 'Select Two' }))
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item Two')
    )

    const selectOneButton = screen.getByRole('button', { name: 'Select One' })
    selectOneButton.focus()
    await user.keyboard('{Enter}')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
  })

  it('does not re-announce on unrelated rerenders for the same item state', async () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body"
        mediaDetail={{ type: 'document' }}
      />
    )

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
    const baselineAnnouncement = liveRegion.textContent

    rerender(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body updated"
        mediaDetail={{ type: 'document' }}
      />
    )

    await waitFor(() => {
      expect(liveRegion.textContent).toBe(baselineAnnouncement)
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    })
  })
})
