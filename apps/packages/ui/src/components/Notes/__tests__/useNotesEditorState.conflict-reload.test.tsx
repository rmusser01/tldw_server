import React from 'react'
import { act, cleanup, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'

// NS-N1 (#3102): reloading after a save conflict must never move the editor's
// base version without also loading the content that version belongs to.

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  refetch: vi.fn(),
  messages: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }
}))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
vi.mock('@/services/notes-tasks', () => ({
  listNoteTasks: vi.fn(async () => ({ tasks: [], reconciliation: null })),
  listTaskActivity: vi.fn(async () => ({ events: [] })),
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn()
}))
vi.mock('@/services/tldw/TldwAuth', () => ({ tldwAuth: { getCurrentUser: vi.fn(async () => null) } }))
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/settings/registry')>(),
  getSetting: vi.fn(async () => null), setSetting: vi.fn(async () => undefined), clearSetting: vi.fn()
}))

const notes: Record<string, Record<string, unknown>> = {}

const renderEditor = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: 'alice', isOnline: true, isMobileViewport: false,
    message: mocks.messages as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: vi.fn(async () => true), queryClient,
    t: (key, options) => (options as { defaultValue?: string } | undefined)?.defaultValue ?? key,
    listMode: 'active', setListMode: vi.fn(), data: [], refetch: mocks.refetch,
    setPage: vi.fn(), setQuery: vi.fn(), setQueryInput: vi.fn(), setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(), setMobileSidebarOpen: vi.fn(), editorKeywords: [],
    setEditorKeywords: vi.fn(), keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(), setKeywordSuggestionSelection: vi.fn(), editorDisabled: false
  }
  return renderHook(() => {
    const [keywords, setKeywords] = React.useState<string[]>([])
    const editor = useNotesEditorState({ ...deps, editorKeywords: keywords, setEditorKeywords: setKeywords })
    return { ...editor, editorKeywords: keywords }
  }, {
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  })
}

describe('Notes editor conflict reload (NS-N1)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.refetch.mockResolvedValue(undefined)
    notes.one = {
      id: 'one', title: 'Shared note', content: 'Original body',
      metadata: { keywords: ['alpha'] }, version: 2, last_modified: '2026-10-03T10:00:00Z'
    }
    notes.other = {
      id: 'other', title: 'Other note', content: 'Other body',
      metadata: { keywords: [] }, version: 9, last_modified: '2026-10-03T10:00:00Z'
    }
    mocks.request.mockImplementation(async ({ path }: { path: string }) => {
      const id = path.split('/').at(-1) || ''
      return notes[id] ? { ...notes[id] } : {}
    })
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn(async () => undefined) }
    })
  })

  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(navigator, 'clipboard')
  })

  it('replaces the selected note’s title, content, tags and version together and copies the local text', async () => {
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Stale local text') })
    notes.one = {
      ...notes.one, title: 'Server title', content: 'Server body',
      metadata: { keywords: ['beta'] }, version: 3
    }

    await act(async () => { await view.result.current.reloadNotes('one') })

    expect({
      title: view.result.current.title,
      content: view.result.current.content,
      keywords: view.result.current.editorKeywords,
      version: view.result.current.selectedVersion,
      isDirty: view.result.current.isDirty
    }).toEqual({ title: 'Server title', content: 'Server body', keywords: ['beta'], version: 3, isDirty: false })
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('Stale local text'))
    expect(mocks.refetch).toHaveBeenCalled()
  })

  it('leaves the editor and its base version alone when another note is reloaded', async () => {
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Unsaved edit to note one') })

    await act(async () => { await view.result.current.reloadNotes('other') })

    expect({
      selectedId: view.result.current.selectedId,
      version: view.result.current.selectedVersion,
      content: view.result.current.content,
      isDirty: view.result.current.isDirty
    }).toEqual({ selectedId: 'one', version: 2, content: 'Unsaved edit to note one', isDirty: true })
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
    expect(mocks.refetch).toHaveBeenCalled()
  })
})
