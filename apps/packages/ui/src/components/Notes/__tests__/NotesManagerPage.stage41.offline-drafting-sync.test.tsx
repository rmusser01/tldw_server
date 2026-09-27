import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import NotesManagerPage from '../NotesManagerPage'

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  mockOnlineState,
  mockRemoteVersion,
  mockAuthority
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockMessageInfo: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockClearSetting: vi.fn(),
    mockOnlineState: { value: true },
    mockRemoteVersion: { value: 1 },
    mockAuthority: { value: 'alice' as string | null }
  }
})

vi.mock('@/components/Notes/hooks/useNotesGraphAuthorityScope', () => ({
  useNotesGraphAuthorityScope: () => mockAuthority.value,
  createNotesGraphAuthorityScope: (_origin: string, principal: string) => principal
}))

vi.mock('@/hooks/useCanonicalConnectionConfig', () => ({
  useCanonicalConnectionConfig: () => ({
    config: { serverUrl: 'https://notes.example.test', authMode: 'single-user', apiKey: 'test-key' },
    loading: false
  })
}))

vi.mock('@/services/tldw/TldwAuth', () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: mockAuthority.value, is_active: true })) }
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === 'string') return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mockBgRequest
}))

vi.mock('@/hooks/useServerOnline', () => ({
  useServerOnline: () => mockOnlineState.value
}))

vi.mock('@/context/demo-mode', () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock('@/hooks/useServerCapabilities', () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock('@/components/Common/confirm-danger', () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock('@/hooks/useAntdMessage', () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: mockMessageInfo
  })
}))

vi.mock('@/services/note-keywords', () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => [])
}))

vi.mock('@/store/option', () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
}))

vi.mock('@/services/settings/registry', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/settings/registry')>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock('@/components/Common/MarkdownPreview', () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid='markdown-preview-content'>{content}</div>
  )
}))

vi.mock('@/components/Notes/NotesListPanel', () => ({
  default: () => <div data-testid='notes-list-panel' />
}))

const OFFLINE_QUEUE_KEY = 'tldw:notesOfflineDraftQueue:v1:alice'

const ensureLocalStorage = () => {
  if (typeof window.localStorage !== 'undefined') return
  const storage = new Map<string, string>()
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      clear: () => storage.clear(),
      getItem: (key: string) => storage.get(key) ?? null,
      key: (index: number) => Array.from(storage.keys())[index] ?? null,
      removeItem: (key: string) => {
        storage.delete(key)
      },
      setItem: (key: string, value: string) => {
        storage.set(key, String(value))
      },
      get length() {
        return storage.size
      }
    }
  })
}

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(<NotesManagerPage />, {
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  })
}

const postCreateCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || '')
    const method = String(request?.method || 'GET').toUpperCase()
    return path === '/api/v1/notes/' && method === 'POST'
  })

const putUpdateCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || '')
    const method = String(request?.method || 'GET').toUpperCase()
    return path === '/api/v1/notes/11' && method === 'PUT'
  })

describe('NotesManagerPage stage 41 offline drafting and sync', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    ensureLocalStorage()
    window.localStorage.clear()
    mockOnlineState.value = true
    mockAuthority.value = 'alice'
    mockRemoteVersion.value = 1
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()

      if (path.startsWith('/api/v1/notes/?')) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }

      if (path === '/api/v1/notes/' && method === 'POST') {
        return {
          id: 11,
          version: 1,
          last_modified: '2026-02-18T11:00:00.000Z'
        }
      }

      if (path === '/api/v1/notes/11' && method === 'GET') {
        return {
          id: 11,
          title: 'Existing note',
          content: 'Server body',
          metadata: { keywords: [] },
          version: mockRemoteVersion.value,
          last_modified: '2026-02-18T11:00:00.000Z'
        }
      }

      if (path === '/api/v1/notes/11' && method === 'PUT') {
        return {
          id: 11,
          version: mockRemoteVersion.value + 1,
          last_modified: '2026-02-18T11:10:00.000Z'
        }
      }

      return {}
    })
  })

  it('queues an offline save locally without hitting create/update endpoints', async () => {
    mockOnlineState.value = false
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Offline draft title' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Offline draft content' }
    })

    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-offline-sync-status')).toHaveTextContent(
        'Offline: changes stored locally and queued for sync.'
      )
    })
    expect(screen.getByTestId('notes-save-status')).not.toHaveAttribute('data-state', 'saved')
    expect(screen.queryByText('All changes saved')).not.toBeInTheDocument()
    expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Not saved yet')

    expect(mockMessageInfo).toHaveBeenCalledWith(
      'Saved locally. Sync will resume when connection returns.'
    )
    expect(postCreateCalls()).toHaveLength(0)
    expect(putUpdateCalls()).toHaveLength(0)

    const rawQueue = window.localStorage.getItem(OFFLINE_QUEUE_KEY)
    expect(rawQueue).toBeTruthy()
    const parsedQueue = JSON.parse(String(rawQueue)) as Record<string, { content?: string }>
    expect(parsedQueue['draft:new']?.content).toBe('Offline draft content')
  })

  it('recovers a persisted queued draft and syncs it once online', async () => {
    window.localStorage.setItem(
      OFFLINE_QUEUE_KEY,
      JSON.stringify({
        'draft:new': {
          key: 'draft:new',
          noteId: null,
          baseVersion: null,
          title: 'Recovered queue draft',
          content: 'Recovered queue content',
          keywords: [],
          metadata: null,
          backlinkConversationId: null,
          backlinkMessageId: null,
          updatedAt: '2026-02-18T11:00:00.000Z',
          syncState: 'queued',
          lastError: null
        }
      })
    )

    renderPage()

    await waitFor(() => {
      expect(postCreateCalls()).toHaveLength(1)
    })

    await waitFor(() => {
      const rawQueue = window.localStorage.getItem(OFFLINE_QUEUE_KEY)
      expect(rawQueue).toBe('{}')
    })

    expect(mockMessageSuccess).toHaveBeenCalledWith('Synced {{count}} queued offline draft(s).')
  })

  it('keeps queued updates when reconnect detects newer server versions', async () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Versioned note' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Original online content' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(postCreateCalls()).toHaveLength(1)
    })

    mockOnlineState.value = false
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Offline update to conflict later' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-offline-sync-status')).toHaveTextContent(
        'Offline: changes stored locally and queued for sync.'
      )
    })

    mockRemoteVersion.value = 5
    mockOnlineState.value = true
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Offline update to conflict later (online tick)' }
    })

    await waitFor(() => {
      expect(screen.getByTestId('notes-offline-sync-status')).toHaveTextContent(
        'Offline sync conflict: server has a newer version. Your local draft is still saved; copy it before reloading.'
      )
    })

    expect(putUpdateCalls()).toHaveLength(0)

    const rawQueue = window.localStorage.getItem(OFFLINE_QUEUE_KEY)
    const parsedQueue = JSON.parse(String(rawQueue)) as Record<string, { syncState?: string }>
    expect(parsedQueue['note:11']?.syncState).toBe('conflict')
  })

  it('never syncs unowned legacy browser drafts as the signed-in user', async () => {
    window.localStorage.setItem('tldw:notesOfflineDraftQueue:v1', JSON.stringify({
      'draft:new': {
        key: 'draft:new', noteId: null, baseVersion: null,
        title: 'Another account private draft', content: 'Private content',
        keywords: [], metadata: null, backlinkConversationId: null, backlinkMessageId: null,
        updatedAt: '2026-02-18T11:00:00.000Z', syncState: 'queued', lastError: null
      }
    }))
    renderPage()
    await act(async () => {})
    expect(postCreateCalls()).toHaveLength(0)
    expect(screen.getByPlaceholderText('Title')).not.toHaveValue('Another account private draft')
  })

  it('leaves Alice offline draft in her storage when Bob signs in online', async () => {
    mockOnlineState.value = false
    const view = renderPage()
    fireEvent.change(screen.getByPlaceholderText('Title'), { target: { value: 'Alice private draft' } })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Alice private content' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))
    await waitFor(() => expect(window.localStorage.getItem(OFFLINE_QUEUE_KEY)).toContain('Alice private content'))
    mockAuthority.value = 'bob'
    mockOnlineState.value = true
    view.rerender(<NotesManagerPage />)
    await act(async () => {})
    expect(postCreateCalls()).toHaveLength(0)
    expect(screen.getByPlaceholderText('Title')).not.toHaveValue('Alice private draft')
    expect(window.localStorage.getItem(OFFLINE_QUEUE_KEY)).toContain('Alice private content')
    expect(window.localStorage.getItem('tldw:notesOfflineDraftQueue:v1:bob')).not.toContain('Alice private content')
  })

  it('binds and aborts an in-flight offline save when the authority changes', async () => {
    let resolveSave!: (value: unknown) => void
    const save = new Promise((resolve) => { resolveSave = resolve })
    const original = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation((request) => request.path === '/api/v1/notes/' && request.method === 'POST'
      ? save : original(request))
    mockOnlineState.value = false
    const view = renderPage()
    fireEvent.change(screen.getByPlaceholderText('Title'), { target: { value: 'Alice private draft' } })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Alice private content' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))
    await waitFor(() => expect(window.localStorage.getItem(OFFLINE_QUEUE_KEY)).toContain('Alice private content'))
    mockOnlineState.value = true
    view.rerender(<NotesManagerPage />)
    await waitFor(() => expect(postCreateCalls()).toHaveLength(1))
    const request = postCreateCalls()[0][0]
    expect(request).toMatchObject({
      servicePromptConfig: { serverUrl: 'https://notes.example.test', authMode: 'single-user', expectedUserId: 'alice' },
      headers: { 'X-TLDW-Expected-User-ID': 'alice' }
    })
    mockAuthority.value = 'bob'
    view.rerender(<NotesManagerPage />)
    expect(request.abortSignal.aborted).toBe(true)
    await act(async () => { resolveSave({ id: 11, version: 1 }) })
    expect(window.localStorage.getItem(OFFLINE_QUEUE_KEY)).toContain('Alice private content')
    expect(mockMessageSuccess).not.toHaveBeenCalledWith('Synced {{count}} queued offline draft(s).')
  })
})
