import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import axe from 'axe-core'
import ViewMediaPage from '../ViewMediaPage'

const mocks = vi.hoisted(() => ({
  queryClient: { removeQueries: vi.fn() },
  queryData: [] as Array<any>,
  detailSequencesById: {} as Record<string, Array<{ ok: boolean; value: any }>>,
  refetch: vi.fn(),
  bgRequest: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  clearSetting: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  messageWarning: vi.fn(),
  showUndoNotification: vi.fn(),
  setChatMode: vi.fn(),
  setSelectedKnowledge: vi.fn(),
  setRagMediaIds: vi.fn()
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => mocks.queryClient,
  useQuery: () => ({
    data: mocks.queryData,
    refetch: mocks.refetch,
    isLoading: false,
    isFetching: false
  })
}))

vi.mock('@plasmohq/storage', () => ({
  Storage: class {
    async get() {
      return null
    }
    async set() {
      return undefined
    }
    async remove() {
      return undefined
    }
  }
}))

vi.mock('@plasmohq/storage/hook', async () => {
  const React = await import('react')
  return {
    useStorage: (_key: string, initialValue: unknown) => {
      const [value, setValue] = React.useState(initialValue)
      return [value, setValue] as const
    }
  }
})

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock('@/services/settings/registry', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/settings/registry')>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting,
    clearSetting: mocks.clearSetting
  }
})

vi.mock('@/hooks/useDebounce', () => ({
  useDebounce: (value: string) => value
}))

vi.mock('@/hooks/useServerOnline', () => ({
  useServerOnline: () => true
}))

vi.mock('@/hooks/useServerCapabilities', () => ({
  useServerCapabilities: () => ({ capabilities: { hasMedia: true }, loading: false })
}))

vi.mock('@/context/demo-mode', () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock('@/hooks/useConnectionState', () => ({
  useConnectionState: () => ({ serverUrl: 'http://localhost:8000' }),
  useConnectionUxState: () => ({
    uxState: 'connected_ok',
    hasCompletedFirstRun: true
  }),
  useConnectionActions: () => ({
    checkOnce: vi.fn()
  })
}))

vi.mock('@/hooks/useMessageOption', () => ({
  useMessageOption: () => ({
    setChatMode: mocks.setChatMode,
    setSelectedKnowledge: mocks.setSelectedKnowledge,
    setRagMediaIds: mocks.setRagMediaIds
  })
}))

vi.mock('@/hooks/useAntdMessage', () => ({
  useAntdMessage: () => ({
    success: mocks.messageSuccess,
    error: mocks.messageError,
    warning: mocks.messageWarning
  })
}))

vi.mock('@/hooks/useUndoNotification', () => ({
  useUndoNotification: () => ({
    showUndoNotification: mocks.showUndoNotification
  })
}))

vi.mock('@/hooks/useFeatureFlags', () => ({
  useMediaNavigationPanel: () => [false],
  useMediaNavigationGeneratedFallbackDefault: () => [false],
  useMediaRichRendering: () => [false],
  useMediaAnalysisDisplayModeSelector: () => [false]
}))

vi.mock('@/hooks/useMediaNavigation', () => ({
  useMediaNavigation: () => ({
    data: { nodes: [] },
    isLoading: false,
    error: null,
    refetch: vi.fn()
  })
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getConfig: vi.fn().mockResolvedValue({})
  }
}))

vi.mock('@/components/Common/FeatureEmptyState', () => ({
  default: () => <div data-testid="feature-empty" />
}))

vi.mock('@/components/Media/SearchBar', () => ({
  SearchBar: ({ value, onChange, inputRef }: any) => (
    <input
      data-testid="search-bar"
      ref={inputRef}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  )
}))

vi.mock('@/components/Media/FilterPanel', () => ({
  FilterPanel: () => <div data-testid="filter-panel" />
}))

vi.mock('@/components/Media/JumpToNavigator', () => ({
  JumpToNavigator: () => <div data-testid="jump-to-navigator" />
}))

vi.mock('@/components/Media/KeyboardShortcutsOverlay', () => ({
  KeyboardShortcutsOverlay: () => null
}))

vi.mock('@/components/Media/FilterChips', () => ({
  FilterChips: () => <div data-testid="filter-chips" />
}))

vi.mock('@/components/Media/Pagination', () => ({
  Pagination: () => <div data-testid="pagination" />
}))

vi.mock('@/components/Media/MediaSectionNavigator', () => ({
  MediaSectionNavigator: () => <div data-testid="media-section-navigator" />
}))

vi.mock('@/components/Media/ResultsList', () => ({
  ResultsList: ({ results, selectedId, onSelect }: any) => (
    <div data-testid="results-list">
      {results.map((result: any) => (
        <button
          key={String(result.id)}
          type="button"
          data-testid={`result-${String(result.id)}`}
          aria-pressed={selectedId === result.id}
          onClick={() => onSelect(result.id)}
        >
          {result.title}
        </button>
      ))}
    </div>
  )
}))

vi.mock('@/components/Media/ContentViewer', () => ({
  ContentViewer: ({ selectedMedia, content, isDetailLoading }: any) => (
    <div data-testid="mock-content-viewer">
      <div data-testid="selected-media-id">
        {selectedMedia?.id != null ? String(selectedMedia.id) : 'none'}
      </div>
      <div data-testid="selected-content">{String(content || '')}</div>
      <div data-testid="selected-loading">{String(Boolean(isDetailLoading))}</div>
    </div>
  )
}))

const renderMediaPage = () => {
  return render(
    <MemoryRouter initialEntries={['/media']}>
      <Routes>
        <Route path="/media" element={<ViewMediaPage />} />
      </Routes>
    </MemoryRouter>
  )
}

describe('ViewMediaPage stage 13 error handling', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  beforeEach(() => {
    mocks.queryData = []
    mocks.detailSequencesById = {}
    mocks.refetch.mockReset()
    mocks.refetch.mockResolvedValue({ data: mocks.queryData })
    mocks.getSetting.mockReset()
    mocks.getSetting.mockResolvedValue(undefined)
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    mocks.showUndoNotification.mockReset()
    mocks.setChatMode.mockReset()
    mocks.setSelectedKnowledge.mockReset()
    mocks.setRagMediaIds.mockReset()
    mocks.bgRequest.mockReset()

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      if (path.startsWith('/api/v1/media/?page=')) {
        return { items: [], pagination: { total_pages: 1, total_items: 0 } }
      }
      if (path.startsWith('/api/v1/media/keywords')) {
        return { keywords: [] }
      }
      if (path.startsWith('/api/v1/media/')) {
        const id = path.replace('/api/v1/media/', '').split('?')[0]
        const sequence = mocks.detailSequencesById[id]
        if (sequence && sequence.length > 0) {
          const next = sequence.shift()!
          if (next.ok) return next.value
          throw next.value
        }
        return {
          id,
          title: `Media ${id}`,
          type: 'document',
          content: { text: `Content for ${id}` }
        }
      }
      if (path.startsWith('/api/v1/notes')) {
        return { items: [], pagination: { total_items: 0 } }
      }
      return {}
    })
  })

  it('shows inline fetch failure with retry and recovers on successful retry', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 11,
        title: 'Retry candidate',
        snippet: '',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailSequencesById['11'] = [
      {
        ok: false,
        value: Object.assign(new Error('temporary failure'), { status: 500 })
      },
      {
        ok: true,
        value: {
          id: 11,
          title: 'Retry candidate',
          type: 'document',
          content: { text: 'Recovered detail content' }
        }
      }
    ]

    renderMediaPage()
    fireEvent.click(await screen.findByTestId('result-11'))

    await waitFor(() => {
      expect(screen.getByTestId('media-detail-fetch-error')).toBeInTheDocument()
    })
    expect(screen.getByTestId('selected-content')).toHaveTextContent('')
    expect(screen.getByTestId('media-detail-fetch-error')).toHaveTextContent(
      'Unable to load this item. Please try again.'
    )
    expect(screen.getByTestId('media-detail-fetch-error')).not.toHaveTextContent(
      'temporary failure'
    )

    fireEvent.click(screen.getByTestId('media-detail-fetch-retry'))

    await waitFor(() => {
      expect(screen.queryByTestId('media-detail-fetch-error')).not.toBeInTheDocument()
    })
    expect(screen.getByTestId('selected-content')).toHaveTextContent(
      'Recovered detail content'
    )
    expect(
      mocks.bgRequest.mock.calls.filter((entry) =>
        String(entry?.[0]?.path || '').startsWith('/api/v1/media/11')
      ).length
    ).toBe(2)
  })

  it('has no axe violations for core aria naming and region rules', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 77,
        title: 'Accessibility media',
        snippet: 'Accessible snippet',
        keywords: ['a11y'],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    const { container } = renderMediaPage()
    await screen.findByTestId('result-77')

    const results = await axe.run(container, {
      runOnly: {
        type: 'rule',
        values: [
          'aria-required-attr',
          'aria-valid-attr',
          'aria-valid-attr-value',
          'button-name',
          'region'
        ]
      }
    })

    expect(results.violations).toEqual([])
  })

  it('clears previous content when newly selected item detail fetch fails', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 1,
        title: 'First item',
        snippet: '',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      {
        kind: 'media',
        id: 2,
        title: 'Second item',
        snippet: '',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailSequencesById['1'] = [
      {
        ok: true,
        value: {
          id: 1,
          title: 'First item',
          type: 'document',
          content: { text: 'First detail content' }
        }
      }
    ]
    mocks.detailSequencesById['2'] = [
      {
        ok: false,
        value: Object.assign(new Error('missing'), { status: 404 })
      }
    ]

    renderMediaPage()
    fireEvent.click(await screen.findByTestId('result-1'))

    await waitFor(() => {
      expect(screen.getByTestId('selected-content')).toHaveTextContent('First detail content')
    })

    fireEvent.click(screen.getByTestId('result-2'))

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('2')
      expect(screen.getByTestId('media-detail-fetch-error')).toBeInTheDocument()
    })
    expect(screen.getByTestId('selected-content')).toHaveTextContent('')
    expect(screen.getByTestId('media-detail-fetch-error')).toHaveTextContent(
      'This item is no longer available. It may have been deleted.'
    )
    expect(screen.queryByText('First detail content')).not.toBeInTheDocument()
  })

  it('detects stale selected media and auto-switches to the next available item', async () => {
    let stalePollCallback: (() => void) | null = null
    let applyStaleRefetch = false
    vi.spyOn(window, 'setInterval').mockImplementation((handler: TimerHandler) => {
      stalePollCallback =
        typeof handler === 'function'
          ? () => {
              handler()
            }
          : null
      return 1 as unknown as ReturnType<typeof window.setInterval>
    })

    const secondItem = {
      kind: 'media',
      id: 2,
      title: 'Second item',
      snippet: '',
      keywords: [],
      meta: { type: 'document' },
      raw: {}
    } as const

    mocks.queryData = [
      {
        kind: 'media',
        id: 1,
        title: 'First item',
        snippet: '',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      secondItem
    ]
    mocks.detailSequencesById['1'] = [
      {
        ok: true,
        value: {
          id: 1,
          title: 'First item',
          type: 'document',
          content: { text: 'First detail content' }
        }
      },
      {
        ok: false,
        value: Object.assign(new Error('gone'), { status: 404 })
      }
    ]
    mocks.detailSequencesById['2'] = [
      {
        ok: true,
        value: {
          id: 2,
          title: 'Second item',
          type: 'document',
          content: { text: 'Second detail content' }
        }
      }
    ]
    mocks.refetch.mockImplementation(async () => {
      if (applyStaleRefetch) {
        mocks.queryData = [secondItem]
      }
      return { data: mocks.queryData }
    })

    renderMediaPage()
    fireEvent.click(await screen.findByTestId('result-1'))

    await waitFor(() => {
      expect(screen.getByTestId('selected-content')).toHaveTextContent('First detail content')
    })
    applyStaleRefetch = true
    stalePollCallback?.()

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('2')
      expect(screen.getByTestId('selected-content')).toHaveTextContent('Second detail content')
    })
    expect(screen.getByTestId('media-stale-selection-notice')).toHaveTextContent(
      'The selected item is no longer available. Your selection was updated.'
    )
    expect(mocks.messageWarning).toHaveBeenCalled()
  })

  it('clears selection when stale detection finds no replacement items', async () => {
    let stalePollCallback: (() => void) | null = null
    let applyStaleRefetch = false
    vi.spyOn(window, 'setInterval').mockImplementation((handler: TimerHandler) => {
      stalePollCallback =
        typeof handler === 'function'
          ? () => {
              handler()
            }
          : null
      return 1 as unknown as ReturnType<typeof window.setInterval>
    })

    mocks.queryData = [
      {
        kind: 'media',
        id: 5,
        title: 'Only item',
        snippet: '',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailSequencesById['5'] = [
      {
        ok: true,
        value: {
          id: 5,
          title: 'Only item',
          type: 'document',
          content: { text: 'Only detail content' }
        }
      },
      {
        ok: false,
        value: Object.assign(new Error('gone'), { status: 404 })
      }
    ]
    mocks.refetch.mockImplementation(async () => {
      if (applyStaleRefetch) {
        mocks.queryData = []
      }
      return { data: mocks.queryData }
    })

    renderMediaPage()
    fireEvent.click(await screen.findByTestId('result-5'))

    await waitFor(() => {
      expect(screen.getByTestId('selected-content')).toHaveTextContent('Only detail content')
    })
    applyStaleRefetch = true
    stalePollCallback?.()

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('none')
      expect(screen.getByTestId('selected-content')).toHaveTextContent('')
    })
    expect(screen.getByTestId('media-stale-selection-notice')).toBeInTheDocument()
  })
})
