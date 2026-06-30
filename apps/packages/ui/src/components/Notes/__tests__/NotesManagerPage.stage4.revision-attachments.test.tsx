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

vi.mock('@/store/tutorials', () => ({
  useTutorialStore: {
    getState: () => ({
      startTutorial: vi.fn()
    })
  }
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

const createCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || '')
    const method = String(request?.method || 'GET').toUpperCase()
    return path === '/api/v1/notes/' && method === 'POST'
  })

describe('NotesManagerPage stage 4 revision and attachments groundwork', () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
        return { id: 22, version: 1, last_modified: '2026-02-18T10:00:00.000Z' }
      }

      if (path === '/api/v1/notes/22' && method === 'GET') {
        return {
          id: 22,
          title: 'Stage 4 note',
          content: 'Existing body',
          metadata: { keywords: [] },
          version: 1,
          last_modified: '2026-02-18T10:00:00.000Z'
        }
      }

      if (path === '/api/v1/notes/22' && method === 'PUT') {
        return {
          id: 22,
          version: 2,
          last_modified: '2026-02-18T10:05:00.000Z'
        }
      }

      if (path === '/api/v1/notes/22/attachments' && method === 'POST') {
        return {
          file_name: 'diagram.png',
          original_file_name: 'diagram.png',
          content_type: 'image/png',
          size_bytes: 11,
          uploaded_at: '2026-02-18T10:06:00.000Z',
          url: '/api/v1/notes/22/attachments/diagram.png'
        }
      }

      return {}
    })
  })

  it('shows version and last-saved metadata in the editor footer', async () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Stage 4 note' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Existing body' }
    })

    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Version 1')
    })
    expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Last saved')
  })

  it('uploads attachments and inserts markdown links for selected notes', async () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Attachment note' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Body' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Version 1')
    })

    const textarea = screen.getByPlaceholderText(
      'Write your note here... (Markdown supported)'
    ) as HTMLTextAreaElement
    textarea.setSelectionRange(textarea.value.length, textarea.value.length)

    const input = screen.getByTestId('notes-attachment-input')
    const image = new File(['image-bytes'], 'diagram.png', { type: 'image/png' })
    fireEvent.change(input, { target: { files: [image] } })

    await waitFor(() => {
      expect(textarea.value).toContain(
        '![diagram.png](/api/v1/notes/22/attachments/diagram.png)'
      )
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith(
      expect.stringContaining('Uploaded')
    )
    expect(mockMessageInfo).not.toHaveBeenCalledWith(
      expect.stringContaining('POST /api/v1/notes/{id}/attachments')
    )
    const uploadCalls = mockBgRequest.mock.calls.filter(([request]) => {
      const path = String(request?.path || '')
      const method = String(request?.method || 'GET').toUpperCase()
      return path === '/api/v1/notes/22/attachments' && method === 'POST'
    })
    expect(uploadCalls.length).toBeGreaterThan(0)
    expect(uploadCalls[0][0]?.body).toBeInstanceOf(FormData)
  })

  it('saves inserted image attachment markdown with the backend version header', async () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Attachment save note' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Body' }
    })
    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      expect(screen.getByTestId('notes-editor-revision-meta')).toHaveTextContent('Version 1')
    })

    const textarea = screen.getByPlaceholderText(
      'Write your note here... (Markdown supported)'
    ) as HTMLTextAreaElement
    textarea.setSelectionRange(textarea.value.length, textarea.value.length)

    const input = screen.getByTestId('notes-attachment-input')
    const image = new File(['image-bytes'], 'diagram.png', { type: 'image/png' })
    fireEvent.change(input, { target: { files: [image] } })

    await waitFor(() => {
      expect(textarea.value).toContain(
        '![diagram.png](/api/v1/notes/22/attachments/diagram.png)'
      )
    })

    fireEvent.click(screen.getByTestId('notes-save-button'))

    await waitFor(() => {
      const updateCall = mockBgRequest.mock.calls.find(([request]) => {
        const path = String(request?.path || '')
        const method = String(request?.method || 'GET').toUpperCase()
        return path === '/api/v1/notes/22' && method === 'PUT'
      })
      expect(updateCall?.[0]?.headers).toMatchObject({
        'Content-Type': 'application/json',
        'expected-version': '1'
      })
      expect(updateCall?.[0]?.body?.content).toContain(
        '![diagram.png](/api/v1/notes/22/attachments/diagram.png)'
      )
    })
  })

  it('does not intercept native undo shortcuts', () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText('Title'), {
      target: { value: 'Undo test' }
    })
    fireEvent.change(screen.getByPlaceholderText('Write your note here... (Markdown supported)'), {
      target: { value: 'Undo content' }
    })

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'z', metaKey: true })

    expect(createCalls()).toHaveLength(0)
  })
})
