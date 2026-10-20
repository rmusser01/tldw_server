import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getDesignSystemState } from '@/design-system'
import type { NotesGraphEdge, NotesGraphNode } from '@/services/note-graph-suggestions'
import NotesManagerPage from '../NotesManagerPage'

const notesConnectionConfig = {
  serverUrl: 'https://notes.example.test',
  authMode: 'multi-user' as const,
  accessToken: 'test-access-token'
}

vi.mock('@/hooks/useCanonicalConnectionConfig', () => ({
  useCanonicalConnectionConfig: () => ({
    config: notesConnectionConfig,
    loading: false,
    authorityLoading: false
  })
}))

vi.mock('@/services/tldw/TldwAuth', () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: 1, is_active: true }))
  }
}))

vi.mock('@/components/Notes/hooks/useNotesGraphAuthorityScope', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/Notes/hooks/useNotesGraphAuthorityScope')>()
  return {
    ...actual,
    useNotesGraphAuthorityScope: () =>
      actual.createNotesGraphAuthorityScope(notesConnectionConfig.serverUrl, 1)
  }
})

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

const registryLabels = vi.hoisted(() => ({
  unavailable: 'Registry Unavailable'
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
          label: key === 'unavailable' ? registryLabels.unavailable : state.label
        }
      }
    )
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

// The Graph view replaces the editor pane; a stub is enough to unmount it.
vi.mock('@/components/Notes/NotesGraphWorkspace', () => ({
  default: () => <div data-testid='notes-graph-workspace' />
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
  fireEvent.click(screen.getByTestId('notes-section-connections-toggle'))
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

    const manualList = await screen.findByTestId('notes-manual-links-list')
    const backlinksList = await screen.findByTestId('notes-backlinks-list')
    expect(within(manualList).getByRole('button', { name: 'Linked note' })).toBeInTheDocument()
    expect(within(backlinksList).getByRole('button', { name: 'Referencing note' })).toBeInTheDocument()

    const backlinksPanel = screen.getByTestId('notes-backlinks-heading').closest('div')
    expect(backlinksPanel).toHaveTextContent('Referencing note')
    expect(backlinksPanel).not.toHaveTextContent('Linked note')

    fireEvent.click(within(manualList).getByRole('button', { name: 'Linked note' }))

    await waitFor(() => {
      const called = mockBgRequest.mock.calls.some(([request]) => String(request?.path || '') === '/api/v1/notes/note-b')
      expect(called).toBe(true)
    })
  })

  it('labels connection chips by relationship type and keeps source nodes out of related notes', async () => {
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
            { id: 'note-c', type: 'note', label: 'Referencing note' },
            { id: 'source:web:media-77', type: 'source', label: 'web: media-77' }
          ],
          edges: [
            { id: 'e1', source: 'note-a', target: 'note-b', type: 'manual', directed: false },
            { id: 'e2', source: 'note-c', target: 'note-a', type: 'wikilink', directed: true },
            {
              id: 'e3',
              source: 'note-a',
              target: 'source:web:media-77',
              type: 'source_membership',
              directed: false
            }
          ]
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const linkedNoteChip = await screen.findByTestId('notes-manual-link-e1')
    expect(linkedNoteChip).toHaveTextContent('Linked note')
    expect(linkedNoteChip).toHaveTextContent('Manual link')
    const relatedList = await screen.findByTestId('notes-related-list')
    expect(relatedList).not.toHaveTextContent('web: media-77')
    expect(relatedList).not.toHaveTextContent('Note source:web:media-77')

    const backlinksList = await screen.findByTestId('notes-backlinks-list')
    const backlinkChip = within(backlinksList).getByTestId('notes-backlink-note-note-c')
    expect(backlinkChip).toHaveTextContent('Referencing note')
    expect(backlinkChip).toHaveTextContent('Backlink')

    const sourceChip = await screen.findByTestId('notes-source-link-source_web_media-77')
    expect(sourceChip).toHaveTextContent('Web source: media-77')
  })

  it('marks deleted linked-note targets as unavailable instead of opening them', async () => {
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
            { id: 'note-deleted', type: 'note', label: 'Archived target', deleted: true }
          ],
          edges: [
            { id: 'e:deleted', source: 'note-a', target: 'note-deleted', type: 'manual', directed: false }
          ]
        }
      }
      throw new Error(`Unexpected request ${method} ${path}`)
    })

    renderPage()
    await seedAndSaveNote()

    const manualLinkChip = await screen.findByTestId('notes-manual-link-e_deleted')
    // Listed once, as a manual link (NE-10).
    expect(screen.queryByTestId('notes-related-note-note-deleted')).not.toBeInTheDocument()
    expect(manualLinkChip).toHaveTextContent('Archived target')
    expect(manualLinkChip).toHaveTextContent(registryLabels.unavailable)
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith('unavailable')
    const manualOpenButton = within(manualLinkChip)
      .getAllByRole('button')
      .find((button) => button.getAttribute('aria-label')?.startsWith('Archived target'))
    expect(manualOpenButton).toBeDisabled()

    fireEvent.click(manualOpenButton!)
    await waitFor(() => {
      const openedDeletedTarget = mockBgRequest.mock.calls.some(
        ([request]) => String(request?.path || '') === '/api/v1/notes/note-deleted'
      )
      expect(openedDeletedTarget).toBe(false)
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
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'wikilink', directed: true }]
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

  it('lists a manual link once in Connections (NE-10)', async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()
      if (path.startsWith('/api/v1/notes/?')) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
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
            { id: 'note-b', type: 'note', label: 'My first note' }
          ],
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'manual', directed: false }]
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const panels = await screen.findByTestId('notes-graph-relation-panels')
    await within(panels).findByTestId('notes-manual-link-e1')
    expect(within(panels).getAllByRole('button', { name: 'My first note' })).toHaveLength(1)
    // No "No related notes yet." under a link that is right there.
    expect(screen.queryByTestId('notes-related-empty')).not.toBeInTheDocument()
  })

  it('lists an incoming manual link once and keeps the [[link]] to a manually linked note (NE-10)', async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()
      if (path.startsWith('/api/v1/notes/?')) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
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
            { id: 'note-b', type: 'note', label: 'Incoming note' },
            { id: 'note-c', type: 'note', label: 'Both ways note' }
          ],
          edges: [
            // Someone linked note-b to this note by hand, one way.
            { id: 'e-in', source: 'note-b', target: 'note-a', type: 'manual', directed: true },
            // note-c is linked by hand and by a [[link]] in this note.
            { id: 'e-manual-c', source: 'note-a', target: 'note-c', type: 'manual', directed: false },
            { id: 'e-wiki-c', source: 'note-a', target: 'note-c', type: 'wikilink', directed: true }
          ]
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const panels = await screen.findByTestId('notes-graph-relation-panels')
    const incoming = await within(panels).findByTestId('notes-manual-link-e-in')
    // Once, on the row that can remove it, and still marked as incoming.
    expect(within(panels).getAllByRole('button', { name: 'Incoming note' })).toHaveLength(1)
    expect(incoming).toHaveTextContent('Manual backlink')
    expect(screen.queryByTestId('notes-backlinks-list')).not.toBeInTheDocument()

    // The [[link]] to note-c still shows under Related, beside its manual link.
    expect(within(panels).getByTestId('notes-manual-link-e-manual-c')).toHaveTextContent('Both ways note')
    const related = within(panels).getByTestId('notes-related-note-note-c')
    expect(related).toHaveTextContent('Note link')
  })

  it('keeps Connections open when opening another note from it (NE-10)', async () => {
    const noteRecord = (id: string, title: string) => ({
      id,
      title,
      content: `${title} body`,
      metadata: { keywords: [] },
      version: 1,
      last_modified: '2026-02-18T10:00:00.000Z'
    })
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()
      if (path.startsWith('/api/v1/notes/?')) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === '/api/v1/notes/' && method === 'POST') {
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') return noteRecord('note-a', 'Graph seed note')
      if (path === '/api/v1/notes/note-b' && method === 'GET') return noteRecord('note-b', 'Linked note')
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Graph seed note' },
            { id: 'note-b', type: 'note', label: 'Linked note' }
          ],
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'wikilink', directed: true }]
        }
      }
      if (path.startsWith('/api/v1/notes/note-b/neighbors')) {
        return {
          nodes: [
            { id: 'note-b', type: 'note', label: 'Linked note' },
            { id: 'note-a', type: 'note', label: 'Graph seed note' }
          ],
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'wikilink', directed: true }]
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const relatedList = await screen.findByTestId('notes-related-list')
    fireEvent.click(within(relatedList).getByRole('button', { name: 'Linked note' }))

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Title')).toHaveValue('Linked note')
    })
    await waitFor(() => {
      expect(screen.getByTestId('notes-section-connections-toggle')).toHaveAttribute(
        'aria-expanded',
        'true'
      )
    })
    const backlinksList = await screen.findByTestId('notes-backlinks-list')
    expect(within(backlinksList).getByRole('button', { name: 'Graph seed note' })).toBeInTheDocument()
  })

  it('keeps Connections open after a visit to the Graph view (NE-10)', async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()
      if (path.startsWith('/api/v1/notes/?')) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
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
            { id: 'note-b', type: 'note', label: 'Linked note' }
          ],
          edges: [{ id: 'e1', source: 'note-a', target: 'note-b', type: 'wikilink', directed: true }]
        }
      }
      return {}
    })

    renderPage()
    // Opens Connections on the saved note.
    await seedAndSaveNote()
    await screen.findByTestId('notes-related-list')

    fireEvent.click(screen.getByTestId('notes-view-mode-graph'))
    await screen.findByTestId('notes-graph-workspace')
    fireEvent.click(screen.getByTestId('notes-view-mode-list'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-section-connections-toggle')).toHaveAttribute(
        'aria-expanded',
        'true'
      )
    })
    expect(await screen.findByTestId('notes-related-list')).toHaveTextContent('Linked note')
  })

  it('shows loading and empty states for relation panels', async () => {
    let resolveNeighbors!: (value: {
      nodes: Pick<NotesGraphNode, 'id' | 'type' | 'label'>[]
      edges: NotesGraphEdge[]
    }) => void
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

    expect(await screen.findByText('Loading note connections...')).toHaveAttribute('role', 'status')

    resolveNeighbors({
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
      'Could not load note connections.'
    )
  })

  it('retries neighbor loading after an error', async () => {
    let neighborAttempts = 0
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
        neighborAttempts += 1
        if (neighborAttempts === 1) throw new Error('graph unavailable')
        return {
          nodes: [{ id: 'note-a', type: 'note', label: 'Graph seed note' }],
          edges: []
        }
      }
      return {}
    })

    renderPage()
    await seedAndSaveNote()

    const retryButton = await screen.findByTestId('notes-related-retry')
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(neighborAttempts).toBe(2)
    })
    expect(await screen.findByTestId('notes-related-empty')).toHaveTextContent(
      'No related notes yet.'
    )
  })
})
