import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import ViewMediaPage, { MEDIA_STALE_CHECK_INTERVAL_MS } from '../ViewMediaPage'
import { useMediaNavigationState } from '../hooks/useMediaNavigationState'
import { useConnectionStore } from '@/store/connection'
import type { MediaResultItem } from '@/components/Media/types'
import { DISCUSS_MEDIA_PROMPT_SETTING } from '@/services/settings/ui-settings'
import * as mediaHandoff from '@/services/tldw/media-chat-handoff'
import { getFlashcardSourceMeta } from '@/components/Flashcards/utils/source-reference'
import { useReadingProgress, type UseReadingProgressDeps } from '@/components/Media/hooks/useReadingProgress'

const mocks = vi.hoisted(() => ({
  queryClient: { removeQueries: vi.fn() },
  ownerScope: 'alice-scope' as string | null,
  canDelete: true,
  readingProbe: false,
  navigationData: { nodes: [] as Array<{ id: string; title: string; level: number; target_type: 'char_range'; target_start: number; target_end: number }> },
  getNavigationResume: vi.fn(),
  getReadingProgress: vi.fn(),
  updateReadingProgress: vi.fn(),
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

vi.mock('@/hooks/useHomeMilestoneScope', () => ({ useHomeMilestoneScope: () => mocks.ownerScope }))

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
  useMediaNavigationPanel: () => [mocks.readingProbe],
  useMediaNavigationGeneratedFallbackDefault: () => [false],
  useMediaRichRendering: () => [false],
  useMediaAnalysisDisplayModeSelector: () => [false]
}))

vi.mock('@/hooks/useMediaNavigation', () => ({
  useMediaNavigation: () => ({
    data: mocks.navigationData,
    isLoading: false,
    error: null,
    refetch: vi.fn()
  })
}))

vi.mock('@/utils/media-navigation-resume', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/utils/media-navigation-resume')>(),
  getMediaNavigationResumeEntry: mocks.getNavigationResume,
  saveMediaNavigationResumeSelection: vi.fn().mockResolvedValue(null)
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getConfig: vi.fn().mockResolvedValue({}),
    getReadingProgress: mocks.getReadingProgress,
    updateReadingProgress: mocks.updateReadingProgress
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
  MediaSectionNavigator: ({ nodes, onSelectNode }: { nodes: Array<{ id: string; title: string }>; onSelectNode: (node: { id: string; title: string }) => void }) => <div data-testid="media-section-navigator">
    {nodes.map(node => <button key={node.id} onClick={() => onSelectNode(node)}>Jump to {node.title}</button>)}
  </div>
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
    onChatAboutMedia,
    ...readingProps
  }: any) => (
    <div data-testid="mock-content-viewer">
      {mocks.readingProbe && <ReadingProgressProbe selectedMedia={selectedMedia} {...readingProps} />}
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

// Real reading/navigation hooks run beneath the real page owner. Only browser
// dimensions/scroll events and external storage are controlled by this fixture.
function ReadingProgressProbe(props: Omit<UseReadingProgressDeps, 'contentScrollContainerRef' | 'contentBodyRef' | 'selectedMediaId' | 't'>) {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const bodyRef = React.useRef<HTMLDivElement | null>(null)
  const attachContainer = React.useCallback((el: HTMLDivElement | null) => {
    containerRef.current = el
    if (!el) return
    let top = 0
    Object.defineProperties(el, {
      scrollHeight: { configurable: true, get: () => 835 },
      clientHeight: { configurable: true, get: () => 540 },
      scrollTop: { configurable: true, get: () => top, set: (next: number) => {
        if (top === next) return
        top = next
        queueMicrotask(() => el.dispatchEvent(new Event('scroll')))
      } }
    })
    el.scrollTo = ((options: ScrollToOptions) => { el.scrollTop = options.top ?? 0 }) as typeof el.scrollTo
  }, [])
  useReadingProgress({
    ...props,
    content: props.content ?? '',
    contentScrollContainerRef: containerRef,
    contentBodyRef: bodyRef,
    selectedMediaId: props.selectedMedia ? String(props.selectedMedia.id) : null,
    t: (_key: string, options?: Record<string, unknown>) => typeof options?.defaultValue === 'string' ? options.defaultValue : _key
  })
  return <div ref={attachContainer} data-testid="reading-scroller"><div ref={bodyRef}>{props.content}</div></div>
}

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-search">{location.search}</div>
}

const renderMediaPage = (initialEntry: string) => {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/" element={<div data-testid="root-route" />} />
        <Route path="/chat" element={<div data-testid="chat-route">Chat composer<LocationProbe /></div>} />
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
  afterEach(() => vi.useRealTimers())
  beforeEach(() => {
    mocks.ownerScope = 'alice-scope'
    mocks.canDelete = true
    mocks.readingProbe = false
    mocks.navigationData = { nodes: [] }
    mocks.getNavigationResume.mockReset().mockResolvedValue(null)
    mocks.getReadingProgress.mockReset().mockResolvedValue({ media_id: 1, percent_complete: 54.2, cfi: 'scroll:54.24' })
    mocks.updateReadingProgress.mockReset().mockResolvedValue({})
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

  it('retains the incoming source permalink throughout initial hydration', async () => {
    const observedSearches: string[] = []
    const HydratingMediaPage = () => {
      const location = useLocation()
      React.useEffect(() => { observedSearches.push(location.search) }, [location.search])
      return <ViewMediaPage />
    }
    render(<MemoryRouter initialEntries={['/media?id=1']}><HydratingMediaPage /></MemoryRouter>)
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
    expect(observedSearches.length).toBeGreaterThan(0)
    expect(observedSearches.every(search => new URLSearchParams(search).get('id') === '1')).toBe(true)
  })

  it('shows a newly ingested deep link even when the cached library is empty', async () => {
    mocks.queryData = []
    mocks.detailById['1'] = { media_id: 1, source: { title: 'First source' }, content: { text: 'Cedar source' } }
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('1'))
  })

  it.each([false, true])('keeps saved reading position when passive chapter selection resolves after progress restoration (remembered=%s)', async remembered => {
    mocks.readingProbe = true
    mocks.navigationData = { nodes: [{ id: 'chapter-1', title: 'Opening', level: 1, target_type: 'char_range', target_start: 0, target_end: 12 }] }
    mocks.detailById['1'] = { media_id: 1, source: { title: 'Aster' }, content: { text: 'Original Aster source' } }
    let release!: (value: unknown) => void
    mocks.getNavigationResume.mockReturnValue(new Promise(resolve => { release = resolve }))
    renderMediaPage('/media?id=1')
    await waitFor(() => expect(screen.getByTestId('reading-scroller').scrollTop).toBe(160))
    await waitFor(() => expect(mocks.getNavigationResume).toHaveBeenCalled())

    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    await act(async () => {
      release(remembered ? { media_id: '1', node_id: 'chapter-1', title: 'Opening', level: 1 } : null)
      await vi.advanceTimersByTimeAsync(1100)
    })

    expect(screen.getByTestId('reading-scroller').scrollTop).toBe(160)
    expect(mocks.updateReadingProgress.mock.calls.some(([, payload]) => payload.percentage === 0)).toBe(false)
  })

  it('allows genuine scrolling and an explicit chapter jump to save their current positions', async () => {
    mocks.readingProbe = true
    mocks.navigationData = { nodes: [{ id: 'chapter-1', title: 'Opening', level: 1, target_type: 'char_range', target_start: 0, target_end: 12 }] }
    mocks.detailById['1'] = { media_id: 1, source: { title: 'Aster' }, content: { text: 'Original Aster source' } }
    renderMediaPage('/media?id=1')
    await screen.findByRole('button', { name: 'Jump to Opening' })
    const scroller = screen.getByTestId('reading-scroller')
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    await act(async () => { scroller.scrollTop = 90; await vi.advanceTimersByTimeAsync(1100) })
    expect(mocks.updateReadingProgress).toHaveBeenCalledWith('1', expect.objectContaining({ percentage: 30.51, zoom_level: 100 }))

    fireEvent.click(screen.getByRole('button', { name: 'Jump to Opening' }))
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(scroller.scrollTop).toBe(0)
    await act(async () => { await vi.advanceTimersByTimeAsync(1100) })
    expect(mocks.updateReadingProgress).toHaveBeenLastCalledWith('1', expect.objectContaining({ percentage: 0, zoom_level: 100 }))
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
    fireEvent.click(screen.getByRole('button', { name: /^Trash$/ }))
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

  it('opens the saved Flashcard media source through its actual permalink consumer', async () => {
    mocks.queryData = []
    mocks.detailById['900'] = { id: 900, title: 'Flashcard source', type: 'document', content: { text: 'Exact original source' } }
    const source = getFlashcardSourceMeta({ source_ref_type: 'media', source_ref_id: '900' })!
    renderMediaPage(source.href!)
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('900'))
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/media/900', method: 'GET' }))
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
    mocks.ownerScope = 'alice-scope'
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
  const readDestination = async () => {
    const token = new URLSearchParams(screen.getByTestId('location-search').textContent || '').get(mediaHandoff.MEDIA_CHAT_HANDOFF_PARAM)!
    expect(token).toMatch(/^tab-/)
    return mediaHandoff.readMediaChatHandoff(token, 'alice-scope')
  }

  beforeEach(() => {
    sessionStorage.clear()
    mocks.ownerScope = 'alice-scope'
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

  it.each(['Chat with media action', 'Chat about media action'])('does not expose a pending %s handoff after account replacement', async action => {
    let finishWrite!: () => void
    const create = mediaHandoff.createMediaChatHandoff
    let savedToken = ''
    vi.spyOn(mediaHandoff, 'createMediaChatHandoff').mockImplementationOnce(async payload => {
      savedToken = await create(payload)
      await new Promise<void>(resolve => { finishWrite = resolve })
      return savedToken
    })
    const events: CustomEvent[] = []
    const observe = (event: Event) => events.push(event as CustomEvent)
    window.addEventListener('tldw:discuss-media', observe)
    const view = renderMediaPage('/media?id=100')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100'))
    fireEvent.click(screen.getByRole('button', { name: action }))
    await waitFor(() => expect(finishWrite).toBeDefined())
    expect(await mediaHandoff.readMediaChatHandoff(savedToken, 'alice-scope')).toEqual(expect.objectContaining({ ownerScope: 'alice-scope' }))
    expect(screen.queryByText('Chat composer')).not.toBeInTheDocument()
    mocks.ownerScope = 'bob-scope'
    view.unmount()
    renderMediaPage('/media?id=100')
    await act(async () => { finishWrite() })
    expect(events).toHaveLength(0)
    expect(await mediaHandoff.readMediaChatHandoff(savedToken, 'alice-scope')).toBeNull()
    expect(mocks.setChatMode).not.toHaveBeenCalled()
    expect(screen.queryByText('Chat composer')).not.toBeInTheDocument()
    window.removeEventListener('tldw:discuss-media', observe)
  })

  it('requires a resolved owner before preparing a source handoff', async () => {
    mocks.ownerScope = null
    renderMediaPage('/media?id=100')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100'))
    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))
    fireEvent.click(screen.getByRole('button', { name: 'Chat about media action' }))
    expect(mocks.setSetting.mock.calls.filter(([setting]) => setting === DISCUSS_MEDIA_PROMPT_SETTING)).toHaveLength(0)
    expect(screen.queryByText('Chat composer')).not.toBeInTheDocument()
  })

  it('does not hand off partial content while selected detail is pending, then sends the full source', async () => {
    mocks.queryData = [{ kind: 'media', id: 100, title: 'Pending source', meta: { type: 'document' }, raw: {} }]
    let resolveDetail!: (value: unknown) => void
    const request = mocks.bgRequest.getMockImplementation()!
    mocks.bgRequest.mockImplementation((input: { path?: string }) => input.path === '/api/v1/media/100'
      ? new Promise(resolve => { resolveDetail = resolve }) : request(input))
    renderMediaPage('/media?id=100')
    await waitFor(() => expect(resolveDetail).toBeDefined())
    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))
    expect(screen.queryByText('Chat composer')).not.toBeInTheDocument()
    expect(mocks.setChatMode).not.toHaveBeenCalled()
    const content = 'Complete Cedar source. '.repeat(90).trim()
    await act(async () => { resolveDetail({ id: 100, title: 'Pending source', content: { text: content } }) })
    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))
    expect(await screen.findByText('Chat composer')).toBeInTheDocument()
    expect(await readDestination()).toEqual(expect.objectContaining({ mediaId: '100', mode: 'normal', content }))
  })

  it.each(['empty', 'failure'])('does not hand off %s selected detail', async state => {
    mocks.queryData = [{ kind: 'media', id: 100, title: 'No source', meta: { type: 'document' }, raw: {} }]
    const request = mocks.bgRequest.getMockImplementation()!
    mocks.bgRequest.mockImplementation(async (input: { path?: string }) => {
      if (input.path !== '/api/v1/media/100') return request(input)
      if (state === 'failure') throw Object.assign(new Error('Not found'), { status: 404 })
      return { id: 100, content: '' }
    })
    renderMediaPage('/media?id=100')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100'))
    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))
    expect(screen.queryByText('Chat composer')).not.toBeInTheDocument()
    expect(mocks.setChatMode).not.toHaveBeenCalled()
  })

  it('ignores late source A after selecting and loading source B', async () => {
    mocks.queryData = [100, 101].map(id => ({ kind: 'media', id, title: `Source ${id}`, meta: { type: 'document' }, raw: {} }))
    let resolveA!: (value: unknown) => void
    const request = mocks.bgRequest.getMockImplementation()!
    mocks.bgRequest.mockImplementation((input: { path?: string }) => input.path === '/api/v1/media/100'
      ? new Promise(resolve => { resolveA = resolve }) : request(input))
    renderMediaPage('/media?id=100')
    await waitFor(() => expect(resolveA).toBeDefined())
    fireEvent.click(screen.getByRole('button', { name: 'Next item' }))
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('101'))
    await act(async () => { resolveA({ id: 100, content: { text: 'Wrong old source A' } }) })
    fireEvent.click(screen.getByRole('button', { name: 'Chat with media action' }))
    expect(await screen.findByText('Chat composer')).toBeInTheDocument()
    expect(await readDestination()).toEqual(expect.objectContaining({ mediaId: '101', content: 'Content for 101' }))
    expect(mocks.bgRequest.mock.calls.filter(([input]) => input.path === '/api/v1/media/101')).toHaveLength(1)
  })

  it.each([
    ['Chat with media action', 'normal'],
    ['Chat about media action', 'rag_media']
  ])('addresses %s to the initiating tab with %s mode', async (action, mode) => {
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent')
    renderMediaPage('/media?id=100')
    await waitFor(() => expect(screen.getByTestId('selected-media-id')).toHaveTextContent('100'))
    fireEvent.click(screen.getByRole('button', { name: action }))
    expect(await screen.findByText('Chat composer')).toBeInTheDocument()
    expect(await readDestination()).toEqual(expect.objectContaining({ ownerScope: 'alice-scope', mediaId: '100', mode }))
    expect(mocks.setChatMode).not.toHaveBeenCalled()
    expect(mocks.setRagMediaIds).not.toHaveBeenCalled()
    expect(dispatchSpy.mock.calls.some(([event]) => event.type === 'tldw:discuss-media')).toBe(false)
    expect(mocks.setSetting.mock.calls.some(([setting]) => setting === DISCUSS_MEDIA_PROMPT_SETTING)).toBe(false)
  })

})


describe('Media stale-selection callback lifetime', () => {
  const items: MediaResultItem[] = [1, 2, 3].map(id => ({ kind: 'media', id, title: 'Source ' + id, snippet: '', keywords: [], meta: {}, raw: {} }))
  const t = (_key: string, options?: Record<string, unknown>) => String(options?.defaultValue || _key)
  beforeEach(() => {
    mocks.getSetting.mockReset().mockResolvedValue(undefined)
    mocks.setSetting.mockReset().mockResolvedValue(undefined)
    mocks.bgRequest.mockReset().mockResolvedValue({ content: 'Current source' })
    mocks.messageWarning.mockReset()
    useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: 'http://localhost:8000' } }))
  })
  const mount = () => {
    let tick: (() => void) | undefined
    const original = window.setInterval.bind(window)
    const timer = vi.spyOn(window, 'setInterval').mockImplementation((handler, delay, ...args) => {
      if (delay === MEDIA_STALE_CHECK_INTERVAL_MS) {
        tick = handler as () => void
        return 123 as unknown as ReturnType<typeof window.setInterval>
      }
      return original(handler, delay, ...args)
    })
    const refetch = vi.fn().mockResolvedValue({ data: items.slice(1) })
    const message = { error: mocks.messageError, warning: mocks.messageWarning, success: mocks.messageSuccess }
    const hook = renderHook(() => useMediaNavigationState({ t, message, displayResults: items, refetch }), {
      wrapper: ({ children }: React.PropsWithChildren) => <MemoryRouter initialEntries={['/media']}>{children}</MemoryRouter>
    })
    return { ...hook, refetch, timer, tick: () => tick?.() }
  }
  it('ignores a late deletion response after synchronous authority loss and recovery', async () => {
    const hook = mount()
    try {
      await act(async () => { hook.result.current.setSelected(items[0]) })
      await waitFor(() => expect(hook.result.current.detailLoading).toBe(false))
      let reject!: (error: unknown) => void
      mocks.bgRequest.mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail }))
      act(() => hook.tick())
      expect(reject).toBeDefined()
      await act(async () => {
        useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: false } }))
        useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true } }))
      })
      expect(hook.result.current.selected).toBeNull()
      await act(async () => { reject({ status: 404 }) })
      expect(mocks.messageWarning).not.toHaveBeenCalled()
      expect(hook.refetch).not.toHaveBeenCalled()
      expect(hook.result.current.selected).toBeNull()
    } finally { hook.unmount(); hook.timer.mockRestore() }
  })
  it('keeps a newer selected source when deletion recovery refetch resolves late', async () => {
    const hook = mount()
    try {
      await act(async () => { hook.result.current.setSelected(items[0]) })
      await waitFor(() => expect(hook.result.current.detailLoading).toBe(false))
      let finish!: (value: { data: MediaResultItem[] }) => void
      hook.refetch.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
      mocks.bgRequest.mockRejectedValueOnce({ status: 404 })
      await act(async () => { hook.tick() })
      expect(hook.refetch).toHaveBeenCalledTimes(1)
      await act(async () => { hook.result.current.setSelected(items[1]) })
      await act(async () => { finish({ data: [items[2]] }) })
      expect(hook.result.current.selected?.id).toBe(2)
    } finally { hook.unmount(); hook.timer.mockRestore() }
  })
})
