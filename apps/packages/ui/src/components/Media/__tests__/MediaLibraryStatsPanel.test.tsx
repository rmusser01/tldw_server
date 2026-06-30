import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { MediaLibraryStatsPanel } from '../MediaLibraryStatsPanel'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; percent?: number }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue.replace(
          '{{percent}}',
          String(fallbackOrOptions.percent ?? '')
        )
      }
      return key
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
          label:
            key === 'unavailable' ? 'Unavailable via registry' : state.label
        }
      }
    )
  }
})

describe('MediaLibraryStatsPanel', () => {
  it('renders visible totals, type distribution, words, and storage usage', () => {
    render(
      <MediaLibraryStatsPanel
        results={[
          {
            kind: 'media',
            id: 1,
            title: 'Paper',
            meta: { type: 'pdf' },
            raw: { word_count: 1000 }
          },
          {
            kind: 'media',
            id: 2,
            title: 'Video',
            meta: { type: 'video' },
            raw: { metadata: { word_count: 400 } }
          },
          {
            kind: 'note',
            id: 3,
            title: 'Note',
            meta: { type: 'note' },
            raw: { safe_metadata: { word_count: 120 } }
          }
        ]}
        totalCount={25}
        storageUsage={{
          loading: false,
          error: null,
          totalMb: 256.4,
          quotaMb: 1024,
          usagePercentage: 25,
          warning: null
        }}
      />
    )

    const toggle = screen.getByTestId('media-library-stats-toggle')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(toggle)

    expect(screen.getByTestId('media-library-stats-visible')).toHaveTextContent('3')
    expect(screen.getByTestId('media-library-stats-total')).toHaveTextContent('25')
    expect(screen.getByTestId('media-library-stats-words').textContent || '').toMatch(
      /1[, ]?520/
    )
    expect(screen.getByTestId('media-library-stats-types')).toHaveTextContent('3')
    expect(screen.getAllByTestId('media-library-stats-type-row')).toHaveLength(3)
    expect(screen.getByTestId('media-library-stats-storage-value')).toHaveTextContent(
      '256.4 MB / 1024.0 MB'
    )
    expect(screen.getByTestId('media-library-stats-storage-value')).toHaveTextContent(
      '25% used'
    )
  })

  it('renders loading and error storage states', () => {
    const { rerender } = render(
      <MediaLibraryStatsPanel
        results={[]}
        totalCount={0}
        storageUsage={{
          loading: true,
          error: null,
          totalMb: null,
          quotaMb: null,
          usagePercentage: null,
          warning: null
        }}
      />
    )

    fireEvent.click(screen.getByTestId('media-library-stats-toggle'))
    expect(screen.getByTestId('media-library-stats-storage-loading')).toBeInTheDocument()

    rerender(
      <MediaLibraryStatsPanel
        results={[]}
        totalCount={0}
        storageUsage={{
          loading: false,
          error: 'Failed to load',
          totalMb: null,
          quotaMb: null,
          usagePercentage: null,
          warning: null
        }}
      />
    )

    expect(screen.getByTestId('media-library-stats-storage-error')).toHaveTextContent(
      'Failed to load'
    )
  })

  it('collapses and expands stats content from the header toggle', () => {
    render(
      <MediaLibraryStatsPanel
        results={[
          {
            kind: 'media',
            id: 1,
            title: 'Paper',
            meta: { type: 'pdf' },
            raw: { word_count: 1000 }
          }
        ]}
        totalCount={1}
        storageUsage={{
          loading: false,
          error: null,
          totalMb: 10,
          quotaMb: 100,
          usagePercentage: 10,
          warning: null
        }}
      />
    )

    const toggle = screen.getByTestId('media-library-stats-toggle')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByTestId('media-library-stats-visible')).not.toBeInTheDocument()

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByTestId('media-library-stats-visible')).toBeInTheDocument()

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByTestId('media-library-stats-visible')).not.toBeInTheDocument()
  })

  it('renders unavailable storage state from the design-system registry', () => {
    render(
      <MediaLibraryStatsPanel
        results={[]}
        totalCount={0}
        storageUsage={{
          loading: false,
          error: null,
          totalMb: null,
          quotaMb: null,
          usagePercentage: null,
          warning: null
        }}
      />
    )

    fireEvent.click(screen.getByTestId('media-library-stats-toggle'))
    expect(screen.getByTestId('media-library-stats-storage-empty')).toHaveTextContent(
      'Unavailable via registry'
    )
  })
})
