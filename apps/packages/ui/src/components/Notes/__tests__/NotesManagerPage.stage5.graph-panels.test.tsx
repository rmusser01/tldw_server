import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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
  mockClearSetting
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
    mockClearSetting: vi.fn()
  }
})

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
  useServerOnline: () => true
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

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const seedAndSaveNote = async () => {
  fireEvent.change(screen.getByPlaceholderText('Title'), {
    target: { value: 'Graph seed note' }
  })
  fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
    target: { value: 'Seed content' }
  })
  fireEvent.click(screen.getByTestId('notes-save-button'))

  await waitFor(() => {
    expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Version 1')
  })
}

describe('NotesManagerPage graph stage 1 related/backlinks panels', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it('renders related notes and backlinks, and opens clicked related note', async () => {
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
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Graph seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Graph seed note' },
            { id: 'note-b', type: 'note', label: 'Linked note' },
            { id: 'note-c', type: 'note', label: 'Referencing note' }
          ],
          edges: [
            { id: 'e1', source: 'note-a', target: 'note-b', type: 'manual', directed: false },
            { id: 'e2', source: 'note-c', target: 'note-a', type: 'wikilink', directed: true }
          ]
        }
      }
      if (path === '/api/v1/notes/note-b' && method === 'GET') {
        return {
          id: 'note-b',
          title: 'Linked note',
          content: 'linked',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:05:00.000Z'
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const relatedList = await screen.findByTestId('notes-related-list')
    const backlinksList = await screen.findByTestId('notes-backlinks-list')
    expect(within(relatedList).getByRole('button', { name: 'Linked note' })).toBeInTheDocument()
    expect(within(backlinksList).getByRole('button', { name: 'Referencing note' })).toBeInTheDocument()

    const backlinksPanel = screen.getByTestId('notes-backlinks-heading').closest('div')
    expect(backlinksPanel).toHaveTextContent('Referencing note')
    expect(backlinksPanel).not.toHaveTextContent('Linked note')

    fireEvent.click(within(relatedList).getByRole('button', { name: 'Linked note' }))

    await waitFor(() => {
      const called = mockBgRequest.mock.calls.some(([request]) => String(request?.path || '') === '/api/v1/notes/note-b')
      expect(called).toBe(true)
    })
  })

  it('auto-saves unsaved changes before opening a related note', async () => {
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
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Graph seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a') && method === 'PUT') {
        return { id: 'note-a', version: 2, last_modified: '2026-02-18T10:01:00.000Z' }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Graph seed note' },
            { id: 'note-b', type: 'note', label: 'Linked note' }
          ],
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'manual', directed: false }]
        }
      }
      if (path === '/api/v1/notes/note-b' && method === 'GET') {
        return {
          id: 'note-b',
          title: 'Linked note',
          content: 'linked',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:05:00.000Z'
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Unsaved changes present' }
    })

    const relatedList = await screen.findByTestId('notes-related-list')
    fireEvent.click(within(relatedList).getByRole('button', { name: 'Linked note' }))

    // Auto-save triggers a PUT instead of showing a confirm dialog
    await waitFor(() => {
      const putCalled = mockBgRequest.mock.calls.some(
        ([request]) =>
          String(request?.path || '').startsWith('/api/v1/notes/note-a') &&
          String(request?.method || '').toUpperCase() === 'PUT'
      )
      expect(putCalled).toBe(true)
    })
  })

  it('shows loading and empty states for relation panels', async () => {
    let resolveNeighbors: ((value: any) => void) | null = null
    const neighborsPromise = new Promise((resolve) => {
      resolveNeighbors = resolve
    })

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
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Graph seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return neighborsPromise
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    expect(await screen.findByText('Loading related notes...')).toBeInTheDocument()
    expect(await screen.findByText('Loading backlinks...')).toBeInTheDocument()

    resolveNeighbors?.({
      nodes: [{ id: 'note-a', type: 'note', label: 'Graph seed note' }],
      edges: []
    })

    expect(await screen.findByTestId('notes-related-empty')).toHaveTextContent('No related notes yet.')
    expect(await screen.findByTestId('notes-backlinks-empty')).toHaveTextContent('No backlinks yet.')
  })

  it('shows error states when neighbors request fails', async () => {
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
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Graph seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        throw new Error('graph unavailable')
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    expect(await screen.findByTestId('notes-related-error')).toHaveTextContent(
      'Could not load related notes.'
    )
    expect(await screen.findByTestId('notes-backlinks-error')).toHaveTextContent(
      'Could not load backlinks.'
    )
  })
})
