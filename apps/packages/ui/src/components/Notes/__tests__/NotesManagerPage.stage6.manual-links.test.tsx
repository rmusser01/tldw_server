import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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

const saveSeedNote = async () => {
  fireEvent.change(screen.getByPlaceholderText('Title'), {
    target: { value: 'Seed note' }
  })
  fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
    target: { value: 'Seed content' }
  })
  fireEvent.click(screen.getByTestId('notes-save-button'))
  await waitFor(() => {
    expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Version 1')
  })
}

describe('NotesManagerPage stage 6 manual link management', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it('creates and removes manual links with graph refresh', async () => {
    let relationMode: 'empty' | 'linked' = 'empty'

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()

      if (path.startsWith('/api/v1/notes/?')) {
        return {
          items: [
            { id: 'note-a', title: 'Seed note', content: 'Seed content', version: 1 },
            { id: 'note-b', title: 'Target note', content: 'Target content', version: 1 }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }
      if (path === '/api/v1/notes/' && method === 'POST') {
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        const edges =
          relationMode === 'linked'
            ? [{ id: 'e:link1', source: 'note-a', target: 'note-b', type: 'manual', directed: false }]
            : []
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Seed note' },
            { id: 'note-b', type: 'note', label: 'Target note' }
          ],
          edges
        }
      }
      if (path === '/api/v1/notes/note-a/links' && method === 'POST') {
        relationMode = 'linked'
        return {
          status: 'created',
          edge: { edge_id: 'link1', from_note_id: 'note-a', to_note_id: 'note-b' }
        }
      }
      if (path === '/api/v1/notes/links/e%3Alink1' && method === 'DELETE') {
        relationMode = 'empty'
        return { deleted: true, edge_id: 'link1' }
      }
      return {}
    })

    renderPage()
    await saveSeedNote()

    const targetContainer = screen.getByTestId('notes-manual-link-target-select')
    const targetContent = targetContainer.querySelector('.ant-select-content') || targetContainer
    fireEvent.mouseDown(targetContent)
    await waitFor(() => {
      const options = document.querySelectorAll('.ant-select-item-option')
      const match = Array.from(options).find(el => el.textContent === 'Target note')
      if (!match) throw new Error('Option "Target note" not found')
      fireEvent.click(match)
    })
    fireEvent.click(screen.getByTestId('notes-manual-link-add'))

    await waitFor(() => {
      const called = mockBgRequest.mock.calls.some(([request]) => {
        const path = String(request?.path || '')
        const method = String(request?.method || '').toUpperCase()
        return path === '/api/v1/notes/note-a/links' && method === 'POST'
      })
      expect(called).toBe(true)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith('Manual link created')

    expect(await screen.findByTestId('notes-manual-links-list')).toHaveTextContent('Target note')

    fireEvent.click(screen.getByTestId('notes-manual-link-remove-e_link1'))

    await waitFor(() => {
      const called = mockBgRequest.mock.calls.some(([request]) => {
        const path = String(request?.path || '')
        const method = String(request?.method || '').toUpperCase()
        return path === '/api/v1/notes/links/e%3Alink1' && method === 'DELETE'
      })
      expect(called).toBe(true)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith('Manual link removed')
    expect(await screen.findByTestId('notes-manual-links-empty')).toHaveTextContent(
      'No manual links yet.'
    )
  })

  it('surfaces duplicate-link conflicts without mutating local state', async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()

      if (path.startsWith('/api/v1/notes/?')) {
        return {
          items: [
            { id: 'note-a', title: 'Seed note', content: 'Seed content', version: 1 },
            { id: 'note-b', title: 'Target note', content: 'Target content', version: 1 }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }
      if (path === '/api/v1/notes/' && method === 'POST') {
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Seed note' },
            { id: 'note-b', type: 'note', label: 'Target note' }
          ],
          edges: []
        }
      }
      if (path === '/api/v1/notes/note-a/links' && method === 'POST') {
        throw { status: 409, message: 'duplicate manual link' }
      }
      return {}
    })

    renderPage()
    await saveSeedNote()

    const targetContainer2 = screen.getByTestId('notes-manual-link-target-select')
    const targetContent2 = targetContainer2.querySelector('.ant-select-content') || targetContainer2
    fireEvent.mouseDown(targetContent2)
    await waitFor(() => {
      const options = document.querySelectorAll('.ant-select-item-option')
      const match = Array.from(options).find(el => el.textContent === 'Target note')
      if (!match) throw new Error('Option "Target note" not found')
      fireEvent.click(match)
    })
    fireEvent.click(screen.getByTestId('notes-manual-link-add'))

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith('Manual link already exists')
    })
    expect(screen.getByTestId('notes-manual-links-empty')).toHaveTextContent('No manual links yet.')
  })

  it('marks manual links with unavailable targets instead of opening placeholder notes', async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || '')
      const method = String(request.method || 'GET').toUpperCase()

      if (path.startsWith('/api/v1/notes/?')) {
        return {
          items: [
            { id: 'note-a', title: 'Seed note', content: 'Seed content', version: 1 }
          ],
          pagination: { total_items: 1, total_pages: 1 }
        }
      }
      if (path === '/api/v1/notes/' && method === 'POST') {
        return { id: 'note-a', version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }
      if (path === '/api/v1/notes/note-a' && method === 'GET') {
        return {
          id: 'note-a',
          title: 'Seed note',
          content: 'Seed content',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }
      if (path.startsWith('/api/v1/notes/note-a/neighbors')) {
        return {
          nodes: [
            { id: 'note-a', type: 'note', label: 'Seed note' }
          ],
          edges: [
            {
              id: 'e:missing-link',
              source: 'note-a',
              target: 'note-missing',
              type: 'manual',
              directed: false
            }
          ]
        }
      }
      return {}
    })

    renderPage()
    await saveSeedNote()

    const unavailableLink = await screen.findByTestId('notes-manual-link-e_missing-link')
    expect(unavailableLink).toHaveTextContent('Unavailable note')
    expect(screen.queryByRole('button', { name: 'Note note-missing' })).not.toBeInTheDocument()
    expect(screen.getByTestId('notes-manual-link-target-select')).toHaveClass('ant-select-disabled')
  })
})
