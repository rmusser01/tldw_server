import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import ViewMediaPage from '../ViewMediaPage'
import { MEDIA_REVIEW_SELECTION_SETTING } from '@/services/settings/ui-settings'

const mocks = vi.hoisted(() => ({
  queryData: [] as Array<any>,
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
  setRagMediaIds: vi.fn(),
  downloadBlob: vi.fn(),
  owner: 'alice',
  transferStudyPack: vi.fn()
}))

vi.mock('@/hooks/useHomeMilestoneScope', () => ({ useHomeMilestoneScope: () => mocks.owner }))
vi.mock('@/hooks/useFlashcardsGenerateTransfer', () => ({
  useStudyPackTransfer: () => mocks.transferStudyPack,
  useFlashcardsGenerateTransfer: () => vi.fn()
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

vi.mock('@/utils/download-blob', () => ({
  downloadBlob: mocks.downloadBlob
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

vi.mock('@/components/Media/MediaIngestJobsPanel', () => ({
  MediaIngestJobsPanel: () => <div data-testid="media-ingest-jobs-panel" />
}))

vi.mock('@/components/Media/ResultsList', () => ({
  ResultsList: ({
    results,
    selectedId,
    onSelect,
    selectionMode,
    selectedIds,
    onToggleSelected
  }: any) => (
    <div data-testid="results-list">
      {results.map((result: any) => {
        const key = String(result.id)
        const checked = selectedIds?.has(key) === true
        return (
          <div key={key}>
            <button
              type="button"
              data-testid={`result-${key}`}
              aria-pressed={selectedId === result.id}
              onClick={() => {
                if (selectionMode) {
                  onToggleSelected?.(result.id)
                  return
                }
                onSelect(result.id)
              }}
            >
              {result.title}
            </button>
            {selectionMode ? (
              <button
                type="button"
                data-testid={`toggle-selected-${key}`}
                aria-pressed={checked}
                onClick={() => onToggleSelected?.(result.id)}
              >
                {checked ? 'selected' : 'select'}
              </button>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}))

vi.mock('@/components/Media/ContentViewer', () => ({
  ContentViewer: ({ selectedMedia }: any) => (
    <div data-testid="mock-content-viewer">
      {selectedMedia?.id != null ? String(selectedMedia.id) : 'none'}
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

const enableBulkMode = async () => {
  fireEvent.click(screen.getByTestId('media-bulk-mode-toggle'))
  await screen.findByTestId('media-bulk-toolbar')
}

const toggleBulkSelection = (id: string | number) => {
  fireEvent.click(screen.getByTestId(`result-${id}`))
}

const selectBulkItems = async (...ids: Array<string | number>) => {
  ids.forEach((id) => {
    toggleBulkSelection(id)
  })
  await waitFor(() => {
    expect(screen.getByTestId('media-bulk-delete')).not.toBeDisabled()
  })
}

describe('ViewMediaPage stage 14 bulk actions baseline', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  beforeEach(() => {
    mocks.owner = 'alice'
    mocks.transferStudyPack.mockReset()
    mocks.queryData = [
      {
        kind: 'media',
        id: 1,
        title: 'Doc 1',
        keywords: ['alpha'],
        raw: {},
        meta: { type: 'pdf' }
      },
      {
        kind: 'media',
        id: 2,
        title: 'Doc 2',
        keywords: [],
        raw: {},
        meta: { type: 'audio' }
      }
    ]
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
    mocks.downloadBlob.mockReset()
    mocks.bgRequest.mockReset()
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request?.path || '')
      if (path === '/api/v1/media/capabilities') {
        return { can_delete: true }
      }
      if (request?.method === 'GET' && path.startsWith('/api/v1/media/?')) {
        return {
          items: mocks.queryData,
          pagination: {
            total_items: mocks.queryData.length
          }
        }
      }
      if (path.startsWith('/api/v1/media/keywords')) {
        return { keywords: [] }
      }
      if (request?.method === 'DELETE' && path.startsWith('/api/v1/media/')) {
        return { ok: true }
      }
      if (request?.method === 'PUT' && path.startsWith('/api/v1/media/')) {
        return { ok: true }
      }
      if (request?.method === 'GET' && path.startsWith('/api/v1/media/')) {
        const id = Number(path.split('/').pop() || 0)
        return { media_id: id, content: { text: `content-${id}` }, source: { title: `Doc ${id}` } }
      }
      return {}
    })
  })

  it.each(['same-media', 'other-media', 'other-owner'])('validates the Study Pack source after detail hydration: %s', async change => {
    let finishDetails!: (value: unknown) => void
    const details = new Promise(resolve => { finishDetails = resolve })
    let finishAuthority!: () => void
    const authority = new Promise<void>(resolve => { finishAuthority = resolve })
    const delivered = vi.fn()
    mocks.transferStudyPack.mockImplementation(async (acquire: () => unknown) => {
      await authority
      delivered(acquire())
    })
    const originalRequest = mocks.bgRequest.getMockImplementation()!
    mocks.bgRequest.mockImplementation((request: { path?: string; method?: string }) =>
      request.method === 'GET' && request.path === '/api/v1/media/2'
        ? details
        : originalRequest(request)
    )
    const view = renderMediaPage()
    fireEvent.click(screen.getByTestId('result-2'))
    await waitFor(() => expect(screen.getByTestId('mock-content-viewer')).toHaveTextContent('2'))
    fireEvent.click(screen.getByRole('button', { name: 'Create study pack' }))
    await waitFor(() => expect(mocks.transferStudyPack).toHaveBeenCalledOnce())
    await act(async () => { finishDetails({ media_id: 2, content: 'Hydrated text', keywords: ['hydrated'] }) })
    if (change === 'other-media') fireEvent.click(screen.getByTestId('result-1'))
    if (change === 'other-owner') {
      mocks.owner = 'bob'
      view.rerender(<MemoryRouter initialEntries={['/media']}><Routes><Route path='/media' element={<ViewMediaPage />} /></Routes></MemoryRouter>)
    }
    await act(async () => { finishAuthority() })
    if (change === 'same-media') {
      expect(delivered).toHaveBeenCalledWith({ title: 'Doc 2', sourceItems: [{ sourceType: 'media', sourceId: '2', sourceTitle: 'Doc 2' }] })
      expect(mocks.messageError).not.toHaveBeenCalled()
    } else {
      expect(delivered).not.toHaveBeenCalled()
      expect(mocks.messageError).toHaveBeenCalledWith(expect.stringContaining('source account or selection changed'))
    }
  })

  it('deletes selected items in bulk mode', async () => {
    renderMediaPage()

    await enableBulkMode()
    await selectBulkItems(1, 2)
    fireEvent.click(screen.getByTestId('media-bulk-delete'))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({ path: '/api/v1/media/1', method: 'DELETE' })
      )
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({ path: '/api/v1/media/2', method: 'DELETE' })
      )
    })

    expect(mocks.messageSuccess).toHaveBeenCalledWith(
      expect.stringContaining('Deleted')
    )
  })

  it('places ingest jobs and library stats after the results flow in the sidebar', () => {
    renderMediaPage()

    const resultsList = screen.getByTestId('results-list')
    const bottomUtilities = screen.getByTestId('media-sidebar-bottom-utilities')
    fireEvent.click(screen.getByTestId('media-library-tools-toggle'))
    const ingestJobsPanel = screen.findByTestId('media-ingest-jobs-panel')
    const libraryStatsPanel = screen.findByTestId('media-library-stats-panel')

    return Promise.all([ingestJobsPanel, libraryStatsPanel]).then(
      ([resolvedIngestJobsPanel, resolvedLibraryStatsPanel]) => {
        expect(
          resultsList.compareDocumentPosition(bottomUtilities) & Node.DOCUMENT_POSITION_FOLLOWING
        ).toBeTruthy()
        expect(
          bottomUtilities.compareDocumentPosition(resolvedIngestJobsPanel) &
            Node.DOCUMENT_POSITION_CONTAINED_BY
        ).toBeTruthy()
        expect(
          resultsList.compareDocumentPosition(resolvedIngestJobsPanel) &
            Node.DOCUMENT_POSITION_FOLLOWING
        ).toBeTruthy()
        expect(
          resolvedIngestJobsPanel.compareDocumentPosition(resolvedLibraryStatsPanel) &
            Node.DOCUMENT_POSITION_FOLLOWING
        ).toBeTruthy()
      }
    )
  })

  it('adds keywords in bulk mode for selected media items', async () => {
    renderMediaPage()

    await enableBulkMode()
    await selectBulkItems(1)
    fireEvent.change(screen.getByTestId('media-bulk-keywords-input'), {
      target: { value: 'beta, gamma' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-tag'))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/1',
          method: 'PUT',
          body: { keywords: ['alpha', 'beta', 'gamma'] }
        })
      )
    })
  })

  it('exports selected items via download blob', async () => {
    renderMediaPage()

    await enableBulkMode()
    await selectBulkItems(1)
    fireEvent.click(screen.getByTestId('media-bulk-export'))

    await waitFor(() => {
      expect(mocks.downloadBlob).toHaveBeenCalledTimes(1)
    })

    const [blob, filename] = mocks.downloadBlob.mock.calls[0] as [Blob, string]
    expect(filename).toContain('media-bulk-export-')
    expect(filename.endsWith('.json')).toBe(true)
    await expect(blob.text()).resolves.toContain('"id": 1')
  })

  it('creates a collection from bulk selection and opens it in multi-review', async () => {
    renderMediaPage()

    await enableBulkMode()
    await selectBulkItems(1)
    fireEvent.change(screen.getByTestId('media-bulk-collection-name'), {
      target: { value: 'Reading sprint' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-add-collection'))

    await waitFor(() => {
      expect(screen.getByTestId('media-collection-filter')).toBeInTheDocument()
    })

    // Active collection filter should keep only selected collection members visible.
    expect(screen.getByTestId('result-1')).toBeInTheDocument()
    expect(screen.queryByTestId('result-2')).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId('media-collection-open-multi'))

    await waitFor(() => {
      expect(mocks.setSetting).toHaveBeenCalledWith(
        MEDIA_REVIEW_SELECTION_SETTING,
        ['1']
      )
    })
  })

  it('merges selection into an existing collection and filters by combined members', async () => {
    renderMediaPage()

    await enableBulkMode()

    await selectBulkItems(1)
    fireEvent.change(screen.getByTestId('media-bulk-collection-name'), {
      target: { value: 'Reading sprint' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-add-collection'))

    await waitFor(() => {
      expect(screen.queryByTestId('result-2')).not.toBeInTheDocument()
    })

    fireEvent.change(screen.getByTestId('media-collection-filter'), {
      target: { value: '' }
    })

    await selectBulkItems(2)
    fireEvent.change(screen.getByTestId('media-bulk-collection-name'), {
      target: { value: 'Reading sprint' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-add-collection'))

    await waitFor(() => {
      expect(screen.getByTestId('result-1')).toBeInTheDocument()
      expect(screen.getByTestId('result-2')).toBeInTheDocument()
    })
  })

  it('keeps bulk keyword tagging scoped to the active collection filter', async () => {
    renderMediaPage()

    await enableBulkMode()

    await selectBulkItems(1)
    fireEvent.change(screen.getByTestId('media-bulk-collection-name'), {
      target: { value: 'Tag scope' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-add-collection'))

    await waitFor(() => {
      expect(screen.queryByTestId('result-2')).not.toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('media-bulk-select-all'))
    fireEvent.change(screen.getByTestId('media-bulk-keywords-input'), {
      target: { value: 'only-active' }
    })
    fireEvent.click(screen.getByTestId('media-bulk-tag'))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/1',
          method: 'PUT',
          body: { keywords: ['alpha', 'only-active'] }
        })
      )
    })

    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/media/2',
        method: 'PUT',
        body: expect.any(Object)
      })
    )
  })
})
