import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ViewMediaPage from '../ViewMediaPage'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const mocks = vi.hoisted(() => ({
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
  setRagMediaIds: vi.fn(),
  metadataPaths: [] as string[],
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
  SearchBar: ({
    value,
    onChange,
    placeholder = 'Search media (title/content)',
    inputRef
  }: {
    value: string
    onChange: (next: string) => void
    placeholder?: string
    inputRef?: React.Ref<HTMLInputElement>
  }) => (
    <input
      data-testid="search-bar"
      ref={inputRef}
      value={value}
      placeholder={placeholder}
      aria-label={placeholder}
      onChange={(event) => onChange(event.target.value)}
    />
  )
}))

vi.mock('@/components/Media/FilterPanel', () => ({
  FilterPanel: ({
    onSearchModeChange,
    onMetadataFiltersChange,
    onMediaTypesChange,
    onKeywordsChange,
    onExcludedKeywordsChange,
    onSortByChange,
    onDateRangeChange,
  }: any) => (
    <div data-testid="filter-panel">
      <button
        type="button"
        data-testid="set-metadata-mode"
        onClick={() => onSearchModeChange?.('metadata')}
      >
        metadata-mode
      </button>
      <button
        type="button"
        data-testid="set-metadata-filter"
        onClick={() =>
          onMetadataFiltersChange?.([
            { id: 'meta-1', field: 'doi', op: 'eq', value: '10.1000/xyz' }
          ])
        }
      >
        metadata-filter
      </button>
      <button
        type="button"
        data-testid="set-media-types"
        onClick={() => onMediaTypesChange?.(['pdf'])}
      >
        media-types
      </button>
      <button
        type="button"
        data-testid="set-keywords"
        onClick={() => onKeywordsChange?.(['biology'])}
      >
        include-keywords
      </button>
      <button
        type="button"
        data-testid="set-excluded-keywords"
        onClick={() => onExcludedKeywordsChange?.(['private'])}
      >
        exclude-keywords
      </button>
      <button
        type="button"
        data-testid="set-sort-date-desc"
        onClick={() => onSortByChange?.('date_desc')}
      >
        sort-date-desc
      </button>
      <button
        type="button"
        data-testid="set-date-range"
        onClick={() =>
          onDateRangeChange?.({
            startDate: '2026-01-01T00:00:00.000Z',
            endDate: '2026-01-31T23:59:59.999Z'
          })
        }
      >
        date-range
      </button>
    </div>
  )
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
  Pagination: ({
    currentPage,
    totalItems
  }: {
    currentPage: number
    totalItems: number
  }) => (
    <div data-testid="pagination">
      <div data-testid="pagination-current-page">{currentPage}</div>
      <div data-testid="pagination-total-items">{totalItems}</div>
    </div>
  )
}))

vi.mock('@/components/Media/MediaSectionNavigator', () => ({
  MediaSectionNavigator: () => <div data-testid="media-section-navigator" />
}))

vi.mock('@/components/Media/ResultsList', () => ({
  ResultsList: ({
    results,
    onClearSearch,
    onClearFilters,
    onOpenQuickIngest
  }: any) => (
    <div data-testid="results-list">
      {results.length === 0 ? (
        <div data-testid="results-empty">
          <button
            type="button"
            data-testid="result-clear-search"
            onClick={() => onClearSearch?.()}
          >
            clear-search
          </button>
          <button
            type="button"
            data-testid="result-clear-filters"
            onClick={() => onClearFilters?.()}
          >
            clear-filters
          </button>
          <button
            type="button"
            data-testid="result-open-quick-ingest"
            onClick={() => onOpenQuickIngest?.()}
          >
            open-quick-ingest
          </button>
        </div>
      ) : (
        results.map((result: any) => (
          <div key={String(result.id)} data-testid={`result-title-${String(result.id)}`}>
            {result.title}
            <div data-testid={`result-snippet-${String(result.id)}`}>
              {result.snippet || ''}
            </div>
          </div>
        ))
      )}
    </div>
  )
}))

vi.mock('@/components/Media/ContentViewer', () => ({
  ContentViewer: ({ selectedMedia }: any) => (
    <div data-testid="mock-content-viewer">
      <div data-testid="selected-media-id">
        {selectedMedia?.id != null ? String(selectedMedia.id) : 'none'}
      </div>
    </div>
  )
}))

vi.mock('@/components/Media/MediaIngestJobsPanel', () => ({
  MediaIngestJobsPanel: () => <div data-testid="media-ingest-jobs-panel" />
}))

vi.mock('@/components/Media/MediaLibraryStatsPanel', () => ({
  MediaLibraryStatsPanel: () => <div data-testid="media-library-stats-panel" />
}))

const renderMediaPage = (initialEntry: string) => {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/" element={<div data-testid="root-route" />} />
        <Route path="/media" element={<ViewMediaPage />} />
      </Routes>
    </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ViewMediaPage metadata search integration', () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.metadataPaths.length = 0
    mocks.getSetting.mockReset()
    mocks.getSetting.mockResolvedValue(undefined)
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.showUndoNotification.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      if (path.startsWith('/api/v1/media/metadata-search')) {
        mocks.metadataPaths.push(path)
        return {
          results: [
            {
              media_id: 99,
              title: 'Server Metadata Hit',
              type: 'pdf',
              created_at: '2026-01-15T12:00:00.000Z',
              safe_metadata: {
                doi: '10.1000/xyz',
                journal: 'Nature Medicine'
              }
            }
          ],
          pagination: {
            page: 1,
            per_page: 20,
            total: 37,
            total_pages: 2
          }
        }
      }
      if (path === '/api/v1/media/?page=1&results_per_page=50') {
        return { items: [], pagination: { total_pages: 1, total_items: 0 } }
      }
      if (path.startsWith('/api/v1/media/?page=')) {
        return { items: [], pagination: { total_pages: 1, total_items: 0 } }
      }
      if (path.startsWith('/api/v1/media/keywords')) {
        return { keywords: [] }
      }
      if (path.startsWith('/api/v1/media/')) {
        const id = path.replace('/api/v1/media/', '').split('?')[0]
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

  it('serializes metadata-mode constraints and preserves server-authoritative totals/results', async () => {
    renderMediaPage('/media')

    fireEvent.click(screen.getByTestId('set-metadata-mode'))
    fireEvent.click(screen.getByTestId('set-metadata-filter'))
    fireEvent.click(screen.getByTestId('set-media-types'))
    fireEvent.click(screen.getByTestId('set-keywords'))
    fireEvent.click(screen.getByTestId('set-excluded-keywords'))
    fireEvent.click(screen.getByTestId('set-sort-date-desc'))
    fireEvent.click(screen.getByTestId('set-date-range'))
    fireEvent.change(screen.getByTestId('search-bar'), {
      target: { value: 'nature medicine' }
    })

    await waitFor(() => {
      expect(mocks.metadataPaths.length).toBeGreaterThan(0)
    })

    const latestMetadataPath = mocks.metadataPaths[mocks.metadataPaths.length - 1]
    expect(latestMetadataPath).toContain('q=nature+medicine')
    expect(latestMetadataPath).toContain('media_types=pdf')
    expect(latestMetadataPath).toContain('must_have=biology')
    expect(latestMetadataPath).toContain('must_not_have=private')
    expect(latestMetadataPath).toContain('date_start=2026-01-01T00%3A00%3A00.000Z')
    expect(latestMetadataPath).toContain('date_end=2026-01-31T23%3A59%3A59.999Z')
    expect(latestMetadataPath).toContain('sort_by=date_desc')

    await waitFor(() => {
      expect(screen.getByTestId('result-title-99')).toHaveTextContent('Server Metadata Hit')
    })
    expect(screen.getByTestId('result-snippet-99')).toHaveTextContent('doi: 10.1000/xyz')
    expect(screen.getByTestId('result-snippet-99')).toHaveTextContent('journal: Nature Medicine')
    expect(screen.getByTestId('pagination-total-items')).toHaveTextContent('37')
  })

  it('wires no-results recovery actions from ViewMediaPage', async () => {
    const metadataPaths: string[] = []
    const fullTextPaths: string[] = []
    const fullTextBodies: Array<Record<string, unknown> | undefined> = []

    mocks.bgRequest.mockImplementation(async (request: { path?: string; body?: any }) => {
      const path = String(request?.path || '')
      if (path.startsWith('/api/v1/media/metadata-search')) {
        metadataPaths.push(path)
        return {
          results: [],
          pagination: {
            page: 1,
            per_page: 20,
            total: 0,
            total_pages: 0
          }
        }
      }
      if (path.startsWith('/api/v1/media/search')) {
        fullTextPaths.push(path)
        fullTextBodies.push(request?.body)
        return {
          items: [],
          pagination: {
            page: 1,
            results_per_page: 20,
            total_pages: 0,
            total_items: 0
          }
        }
      }
      if (path === '/api/v1/media/?page=1&results_per_page=50') {
        return { items: [], pagination: { total_pages: 1, total_items: 0 } }
      }
      if (path.startsWith('/api/v1/media/?page=')) {
        return { items: [], pagination: { total_pages: 1, total_items: 0 } }
      }
      if (path.startsWith('/api/v1/media/keywords')) {
        return { keywords: [] }
      }
      if (path.startsWith('/api/v1/media/')) {
        const id = path.replace('/api/v1/media/', '').split('?')[0]
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

    const quickIngestListener = vi.fn()
    window.addEventListener(
      'tldw:open-quick-ingest',
      quickIngestListener as EventListener
    )

    renderMediaPage('/media')

    fireEvent.click(screen.getByTestId('set-metadata-mode'))
    fireEvent.click(screen.getByTestId('set-metadata-filter'))

    await waitFor(() => {
      expect(metadataPaths.length).toBeGreaterThan(0)
    })

    fireEvent.change(screen.getByTestId('search-bar'), {
      target: { value: 'rare phrase' }
    })
    expect(screen.getByTestId('search-bar')).toHaveValue('rare phrase')

    fireEvent.click(screen.getByTestId('result-clear-filters'))
    fireEvent.change(screen.getByTestId('search-bar'), {
      target: { value: 'after reset' }
    })

    await waitFor(() => {
      expect(fullTextPaths.length).toBeGreaterThan(0)
    })
    expect(fullTextBodies[fullTextBodies.length - 1]).toMatchObject({
      query: 'after reset',
      fields: ['title', 'content'],
      sort_by: 'relevance'
    })

    fireEvent.click(screen.getByTestId('result-clear-search'))
    expect(screen.getByTestId('search-bar')).toHaveValue('')

    fireEvent.click(screen.getByTestId('result-open-quick-ingest'))
    expect(quickIngestListener).toHaveBeenCalledTimes(1)

    window.removeEventListener(
      'tldw:open-quick-ingest',
      quickIngestListener as EventListener
    )
  })
})
