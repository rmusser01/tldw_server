import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import ViewMediaPage from '../ViewMediaPage'

const mocks = vi.hoisted(() => ({
  canDelete: true,
  queryData: [] as Array<any>,
  detailById: {} as Record<string, any>,
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

vi.mock('@/hooks/useMediaCapabilities', () => ({
  useMediaCapabilities: () => ({ canDelete: mocks.canDelete, loading: false })
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
    useStorage: (key: string, initialValue: unknown) => {
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
  FilterPanel: () => <div data-testid="filter-panel" />
}))

vi.mock('@/components/Media/JumpToNavigator', () => ({
  JumpToNavigator: () => <div data-testid="jump-to-navigator" />
}))

vi.mock('@/components/Media/KeyboardShortcutsOverlay', () => ({
  KeyboardShortcutsOverlay: ({ open }: { open: boolean }) =>
    open ? <div data-testid="keyboard-shortcuts-overlay">Shortcuts overlay</div> : null
}))

vi.mock('@/components/Media/FilterChips', () => ({
  FilterChips: () => <div data-testid="filter-chips" />
}))

vi.mock('@/components/Media/Pagination', () => ({
  Pagination: ({
    currentPage,
    onPageChange
  }: {
    currentPage: number
    onPageChange?: (page: number) => void
  }) => (
    <div data-testid="pagination">
      <div data-testid="pagination-current-page">{currentPage}</div>
      <button
        type="button"
        data-testid="pagination-next-page"
        onClick={() => onPageChange?.(currentPage + 1)}
      >
        next-page
      </button>
    </div>
  )
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
  ContentViewer: ({
    selectedMedia,
    onNext,
    onPrevious,
    hasNext,
    hasPrevious,
    onDeleteItem,
    onChatWithMedia,
    onChatAboutMedia
  }: any) => (
    <div data-testid="mock-content-viewer">
      <div data-testid="selected-media-id">
        {selectedMedia?.id != null ? String(selectedMedia.id) : 'none'}
      </div>
      <button
        type="button"
        onClick={onPrevious}
        disabled={!hasPrevious}
        aria-label="Previous item"
      >
        Previous item
      </button>
      <button
        type="button"
        onClick={onNext}
        disabled={!hasNext}
        aria-label="Next item"
      >
        Next item
      </button>
      <button
        type="button"
        onClick={() => {
          if (selectedMedia && onDeleteItem) {
            void onDeleteItem(selectedMedia, null)
          }
        }}
        disabled={!selectedMedia || !onDeleteItem}
        aria-label="Delete selected item"
      >
        Delete selected item
      </button>
      <button
        type="button"
        onClick={() => onChatWithMedia?.()}
        disabled={!selectedMedia || !onChatWithMedia}
        aria-label="Chat with media action"
      >
        Chat with media action
      </button>
      <button
        type="button"
        onClick={() => onChatAboutMedia?.()}
        disabled={!selectedMedia || !onChatAboutMedia}
        aria-label="Chat about media action"
      >
        Chat about media action
      </button>
    </div>
  )
}))

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-search">{location.search}</div>
}

const renderMediaPage = (initialEntry: string) => {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/" element={<div data-testid="root-route" />} />
        <Route path="/chat" element={<div data-testid="chat-route">Chat composer</div>} />
        <Route path="/media-trash" element={<div>Media Trash destination</div>} />
        <Route
          path="/media"
          element={
            <>
              <LocationProbe />
              <ViewMediaPage />
            </>
          }
        />
      </Routes>
    </MemoryRouter>
  )
}

describe('ViewMediaPage Stage 3 permalinks', () => {
  beforeEach(() => {
    mocks.canDelete = true
    mocks.queryData = []
    mocks.detailById = {}
    mocks.refetch.mockReset()
    mocks.refetch.mockResolvedValue({ data: mocks.queryData })
    mocks.getSetting.mockReset()
    mocks.getSetting.mockResolvedValue(undefined)
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
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
        return (
          mocks.detailById[id] || {
            id,
            title: `Media ${id}`,
            type: 'document',
            content: { text: `Content for ${id}` }
          }
        )
      }
      if (path.startsWith('/api/v1/notes')) {
        return { items: [], pagination: { total_items: 0 } }
      }
      return {}
    })
  })

  it('shows a newly ingested deep link even when the cached library is empty', async () => {
    mocks.queryData = []
    mocks.detailById['1'] = { media_id: 1, source: { title: 'First source' }, content: { text: 'Cedar source' } }
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
  })

  it('clears a deleted deep link without hydrating its cached row again', async () => {
    mocks.queryData = [{ kind: 'media', id: 1, title: 'Aster', raw: {}, meta: { type: 'document' } }]
    mocks.getSetting.mockResolvedValue('1')
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
    fireEvent.click(screen.getByRole('button', { name: 'Delete selected item' }))
    await waitFor(() => expect(mocks.showUndoNotification).toHaveBeenCalled())
    await waitFor(() => expect(screen.getByTestId('location-search')).toHaveTextContent(/^$/))
    expect(screen.getByTestId('selected-media-id')).toHaveTextContent('none')
  })

  it('retains Trash navigation after deleting the sole item and refreshing to an empty library', async () => {
    mocks.queryData = [{ kind: 'media', id: 1, title: 'Only source', raw: {}, meta: { type: 'document' } }]
    const originalRequest = mocks.bgRequest.getMockImplementation()!
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === '/api/v1/media/1' && request.method === 'DELETE') {
        mocks.queryData = []
        return {}
      }
      return originalRequest(request)
    })
    mocks.refetch.mockImplementation(async () => ({ data: mocks.queryData }))
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
    fireEvent.click(screen.getByRole('button', { name: 'Delete selected item' }))

    await waitFor(() => expect(screen.getByTestId('location-search')).toHaveTextContent(/^$/))
    await waitFor(() => expect(screen.queryByTestId('mock-content-viewer')).not.toBeInTheDocument())
    expect(screen.getByTestId('results-list')).toBeEmptyDOMElement()
    fireEvent.click(screen.getByRole('button', { name: 'Trash', exact: true }))
    expect(await screen.findByText('Media Trash destination')).toBeInTheDocument()
  })

  it('disables Delete before confirmation when the caller lacks media.delete', async () => {
    mocks.canDelete = false
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
    expect(screen.getByRole('button', { name: 'Delete selected item' })).toBeDisabled()
    mocks.canDelete = true
  })

  it('opens the Chat composer directly with media context', async () => {
    mocks.queryData = [{ kind: 'media', id: 1, title: 'Aster', raw: {}, meta: { type: 'document' } }]
    renderMediaPage('/media?id=1')
    fireEvent.click(await screen.findByRole('button', { name: 'Chat with media action' }))
    expect(await screen.findByText('Chat composer')).toBeInTheDocument()
  })

  it('hydrates and selects permalink media id even when not in current results', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 1,
        title: 'First item',
        snippet: 'one',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailById['900'] = {
      id: 900,
      title: 'Deep linked item',
      type: 'document',
      content: { text: 'Deep-linked content' }
    }

    renderMediaPage('/media?id=900')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('900')
    })
    expect(screen.getByTestId('location-search')).toHaveTextContent('?id=900')
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/v1/media/900', method: 'GET' })
    )
  })

  it('falls back to LAST_MEDIA_ID_SETTING when URL has no id and clears legacy setting', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 42,
        title: 'Stored selection',
        snippet: 'stored',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      {
        kind: 'media',
        id: 43,
        title: 'Neighbor selection',
        snippet: 'next',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.getSetting.mockResolvedValue('42')

    renderMediaPage('/media')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('42')
    })
    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('?id=42')
    })
    expect(mocks.clearSetting).toHaveBeenCalledWith(
      expect.objectContaining({ key: 'tldw:lastMediaId' })
    )
  })

  it('keeps permalink id synchronized when navigating previous/next from a deep link', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      {
        kind: 'media',
        id: 200,
        title: 'Item 200',
        snippet: '200',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })
    expect(screen.getByTestId('location-search')).toHaveTextContent('?id=100')

    fireEvent.click(screen.getByRole('button', { name: 'Next item' }))

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('200')
    })
    expect(screen.getByTestId('location-search')).toHaveTextContent('?id=200')

    fireEvent.click(screen.getByRole('button', { name: 'Previous item' }))

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })
    expect(screen.getByTestId('location-search')).toHaveTextContent('?id=100')
  })

  it('focuses the media search input when slash shortcut is pressed', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media')

    const searchInput = await screen.findByRole('textbox', {
      name: 'Search media (title/content)'
    })
    expect(searchInput).not.toHaveFocus()

    fireEvent.keyDown(window, { key: '/' })

    await waitFor(() => {
      expect(searchInput).toHaveFocus()
    })
  })

  it('keeps sidebar collapse/expand toggle behavior functional', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media?id=100')

    const collapseButton = await screen.findByRole('button', { name: 'Collapse sidebar' })
    fireEvent.click(collapseButton)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument()
    })
  })

  it('preserves j/k keyboard navigation between selected items', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      {
        kind: 'media',
        id: 200,
        title: 'Item 200',
        snippet: '200',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })

    fireEvent.keyDown(window, { key: 'j' })
    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('200')
    })

    fireEvent.keyDown(window, { key: 'k' })
    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })
  })

  it('preserves arrow-key pagination behavior', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      if (path.startsWith('/api/v1/media/?page=1')) {
        return { items: [], pagination: { total_pages: 2, total_items: 40 } }
      }
      if (path.startsWith('/api/v1/media/?page=2')) {
        return { items: [], pagination: { total_pages: 2, total_items: 40 } }
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

    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(
        mocks.bgRequest.mock.calls.some((call) =>
          String(call?.[0]?.path || '').startsWith('/api/v1/media/?page=1')
        )
      ).toBe(true)
    })

    fireEvent.keyDown(window, { key: 'ArrowRight' })

    await waitFor(() => {
      expect(
        mocks.bgRequest.mock.calls.some((call) =>
          String(call?.[0]?.path || '').startsWith('/api/v1/media/?page=2')
        )
      ).toBe(true)
    })
    expect(
      mocks.bgRequest.mock.calls.some((call) =>
        String(call?.[0]?.path || '').startsWith('/api/v1/media/?page=0')
      )
    ).toBe(false)
  })

  it('does not reset pagination after an explicit page change', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('pagination-current-page')).toHaveTextContent('1')
    })

    fireEvent.click(screen.getByTestId('pagination-next-page'))

    await waitFor(() => {
      expect(screen.getByTestId('pagination-current-page')).toHaveTextContent('2')
    })
    await waitFor(() => {
      expect(screen.getByTestId('pagination-current-page')).toHaveTextContent('2')
    })
  })

  it('toggles keyboard shortcuts overlay with ? key', async () => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Item 100',
        snippet: '100',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]

    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })
    expect(screen.queryByTestId('keyboard-shortcuts-overlay')).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: '?' })
    await waitFor(() => {
      expect(screen.getByTestId('keyboard-shortcuts-overlay')).toBeInTheDocument()
    })

    fireEvent.keyDown(window, { key: '?' })
    await waitFor(() => {
      expect(screen.queryByTestId('keyboard-shortcuts-overlay')).not.toBeInTheDocument()
    })
  })
})

describe('ViewMediaPage Stage 1 trash undo flow', () => {
  beforeEach(() => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 1,
        title: 'Item 1',
        snippet: 'one',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      },
      {
        kind: 'media',
        id: 2,
        title: 'Item 2',
        snippet: 'two',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailById = {
      '1': {
        id: 1,
        title: 'Item 1',
        type: 'document',
        content: { text: 'Content 1' }
      },
      '2': {
        id: 2,
        title: 'Item 2',
        type: 'document',
        content: { text: 'Content 2' }
      }
    }
    mocks.refetch.mockReset()
    mocks.refetch.mockResolvedValue({ data: mocks.queryData })
    mocks.getSetting.mockReset()
    mocks.getSetting.mockResolvedValue(undefined)
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.showUndoNotification.mockReset()
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
        return (
          mocks.detailById[id] || {
            id,
            title: `Media ${id}`,
            type: 'document',
            content: { text: `Content for ${id}` }
          }
        )
      }
      if (path.startsWith('/api/v1/notes')) {
        return { items: [], pagination: { total_items: 0 } }
      }
      return {}
    })
  })

  it('shows undo notification after soft-delete from /media', async () => {
    renderMediaPage('/media?id=1')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete selected item' }))

    await waitFor(() => {
      expect(mocks.showUndoNotification).toHaveBeenCalledTimes(1)
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/media/1',
        method: 'DELETE'
      })
    )
  })

  it('restores deleted media when undo action is invoked from toast', async () => {
    renderMediaPage('/media?id=1')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete selected item' }))

    await waitFor(() => {
      expect(mocks.showUndoNotification).toHaveBeenCalledTimes(1)
    })

    const undoPayload = mocks.showUndoNotification.mock.calls[0]?.[0]
    expect(undoPayload).toBeTruthy()
    expect(typeof undoPayload.onUndo).toBe('function')

    await undoPayload.onUndo()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/media/1/restore',
        method: 'POST'
      })
    )
    expect(mocks.refetch).toHaveBeenCalled()
  })

  it('does not call restore API unless undo action is triggered', async () => {
    renderMediaPage('/media?id=1')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete selected item' }))

    await waitFor(() => {
      expect(mocks.showUndoNotification).toHaveBeenCalledTimes(1)
    })

    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/media/1/restore',
        method: 'POST'
      })
    )
  })
})

describe('ViewMediaPage Stage 1 chat action semantics', () => {
  beforeEach(() => {
    mocks.queryData = [
      {
        kind: 'media',
        id: 100,
        title: 'Chat target',
        snippet: 'snippet',
        keywords: [],
        meta: { type: 'document' },
        raw: {}
      }
    ]
    mocks.detailById = {
      '100': {
        id: 100,
        title: 'Chat target',
        type: 'document',
        content: { text: 'Full content for chat handoff' }
      }
    }
    mocks.refetch.mockReset()
    mocks.refetch.mockResolvedValue({ data: mocks.queryData })
    mocks.getSetting.mockReset()
    mocks.getSetting.mockResolvedValue(undefined)
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
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
        return (
          mocks.detailById[id] || {
            id,
            title: `Media ${id}`,
            type: 'document',
            content: { text: `Content for ${id}` }
          }
        )
      }
      if (path.startsWith('/api/v1/notes')) {
        return { items: [], pagination: { total_items: 0 } }
      }
      return {}
    })
  })

  it('keeps "chat with media" flow on normal mode with discuss-media payload', async () => {
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent')
    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))

    await waitFor(() => {
      expect(mocks.setChatMode).toHaveBeenCalledWith('normal')
      expect(mocks.setRagMediaIds).toHaveBeenCalledWith(null)
    })

    const discussEvent = dispatchSpy.mock.calls
      .map((call) => call[0])
      .find((event) => event.type === 'tldw:discuss-media') as CustomEvent | undefined
    expect(discussEvent).toBeDefined()
    expect(discussEvent?.detail).toEqual(
      expect.objectContaining({
        mediaId: '100',
        mode: 'normal'
      })
    )
    expect(
      mocks.setSetting.mock.calls.some(
        (call) =>
          typeof call?.[1] === 'object' &&
          call?.[1] !== null &&
          (call[1] as Record<string, unknown>).mode === 'normal' &&
          (call[1] as Record<string, unknown>).mediaId === '100'
      )
    ).toBe(true)
    dispatchSpy.mockRestore()
  })

  it('keeps "chat about media" flow on rag mode with media-scoped payload', async () => {
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent')
    renderMediaPage('/media?id=100')

    await waitFor(() => {
      expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Chat about media action' }))

    await waitFor(() => {
      expect(mocks.setChatMode).toHaveBeenCalledWith('rag')
      expect(mocks.setRagMediaIds).toHaveBeenCalledWith([100])
    })

    const discussEvent = dispatchSpy.mock.calls
      .map((call) => call[0])
      .find((event) => event.type === 'tldw:discuss-media') as CustomEvent | undefined
    expect(discussEvent).toBeDefined()
    expect(discussEvent?.detail).toEqual(
      expect.objectContaining({
        mediaId: '100',
        mode: 'rag_media'
      })
    )
    expect(
      mocks.setSetting.mock.calls.some(
        (call) =>
          typeof call?.[1] === 'object' &&
          call?.[1] !== null &&
          (call[1] as Record<string, unknown>).mode === 'rag_media' &&
          (call[1] as Record<string, unknown>).mediaId === '100'
      )
    ).toBe(true)
    dispatchSpy.mockRestore()
  })
})
