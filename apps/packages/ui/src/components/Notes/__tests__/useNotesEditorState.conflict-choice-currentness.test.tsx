import React from 'react'
import { act, cleanup, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'
import { createNotesGraphAuthorityScope } from '../hooks/useNotesGraphAuthorityScope'

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  refetch: vi.fn(),
  confirmDanger: vi.fn(),
  messages: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }
}))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
vi.mock('@/services/notes-tasks', () => ({
  listNoteTasks: vi.fn(async () => ({ tasks: [], reconciliation: null })),
  listTaskActivity: vi.fn(async () => ({ events: [] })),
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn()
}))
vi.mock('@/services/tldw/TldwAuth', () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 7, is_active: true })) }
}))
vi.mock('@/hooks/useCallerCapabilities', () => {
  const capabilities = { monitoringAlerts: 'denied', userId: 7, refreshAfterForbidden: vi.fn(async () => undefined) }
  return { useCallerCapabilities: () => capabilities }
})
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/settings/registry')>(),
  getSetting: vi.fn(async () => null), setSetting: vi.fn(async () => undefined), clearSetting: vi.fn()
}))

type Note = { id: string; title: string; content: string; version: number; last_modified: string; metadata: Record<string, unknown> }
type Request = { path: string; method?: string; headers?: Record<string, string>; body?: Partial<Note> }
const notes = new Map<string, Note>()
const SERVER_URL = 'https://notes.test'
const AUTHORITY = createNotesGraphAuthorityScope(SERVER_URL, 7)
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((complete) => { resolve = complete })
  return { promise, resolve }
}
const respond = async (request: Request) => {
  const id = decodeURIComponent(request.path.split('/').at(-1) || '')
  const note = notes.get(id)
  if (!note) throw { status: 404 }
  if (request.method === 'PUT') {
    if (Number(request.headers?.['expected-version']) !== note.version) throw { status: 409 }
    const saved = { ...note, ...request.body, version: note.version + 1 }
    notes.set(id, saved)
    return { ...saved }
  }
  return { ...note }
}
const requests = (method: 'GET' | 'PUT') => mocks.request.mock.calls
  .map(([request]) => request as Request)
  .filter((request) => (request.method || 'GET') === method)

const renderEditor = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: AUTHORITY,
    connectionConfig: {
      serverUrl: SERVER_URL, authMode: 'multi-user',
      accessToken: `test.${btoa(JSON.stringify({ sub: '7' }))}.signature`
    } as UseNotesEditorStateDeps['connectionConfig'],
    isOnline: true, isMobileViewport: false,
    message: mocks.messages as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: mocks.confirmDanger, queryClient,
    t: (key, options) => (options as { defaultValue?: string } | undefined)?.defaultValue ?? key,
    listMode: 'active', setListMode: vi.fn(), data: [], refetch: mocks.refetch,
    setPage: vi.fn(), setQuery: vi.fn(), setQueryInput: vi.fn(), setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(), setMobileSidebarOpen: vi.fn(), editorKeywords: [],
    setEditorKeywords: vi.fn(), keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(), setKeywordSuggestionSelection: vi.fn(), editorDisabled: false
  }
  return renderHook(({ authorityScope }) => {
    const [keywords, setKeywords] = React.useState<string[]>([])
    return useNotesEditorState({ ...deps, authorityScope, editorKeywords: keywords, setEditorKeywords: setKeywords })
  }, {
    initialProps: { authorityScope: AUTHORITY },
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  })
}
type EditorView = ReturnType<typeof renderEditor>
const open = async (view: EditorView, id = 'one') => {
  await act(async () => { await view.result.current.loadDetail(id) })
}
const edit = (view: EditorView, text: string) => {
  act(() => { view.result.current.setContentDirty(text) })
}
const conflict = async (view: EditorView) => {
  await open(view)
  edit(view, 'My original conflict text')
  notes.set('one', { ...notes.get('one')!, version: 3, content: 'Their body' })
  await act(async () => { await view.result.current.saveNote({ showSuccessMessage: false }) })
  expect(view.result.current.saveIndicator).toBe('conflict')
  mocks.request.mockClear()
}
const delayVersionRead = () => {
  const gate = deferred<Note>()
  mocks.request.mockImplementationOnce(() => gate.promise)
  return gate
}

describe('Notes conflict choices retain their captured owner and edit', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    localStorage.clear()
    notes.clear()
    for (const [id, version] of [['one', 2], ['other', 3]] as const) {
      notes.set(id, { id, title: `Note ${id}`, content: `${id} body`, version,
        last_modified: '2026-10-07T04:00:00Z', metadata: { keywords: [] } })
    }
    mocks.request.mockImplementation(respond)
    mocks.refetch.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true, value: { writeText: vi.fn(async () => undefined) }
    })
  })
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    Reflect.deleteProperty(navigator, 'clipboard')
  })

  it('does not replace text typed while the clipboard copy is pending', async () => {
    const view = renderEditor()
    await conflict(view)
    const clipboard = deferred<void>()
    vi.mocked(navigator.clipboard.writeText).mockImplementationOnce(() => clipboard.promise)
    let taking!: Promise<boolean>
    act(() => { taking = view.result.current.takeTheirVersion() })
    edit(view, 'Newer typing not in the clipboard')
    let accepted!: boolean
    await act(async () => { clipboard.resolve(); accepted = await taking })

    expect({ accepted, content: view.result.current.content, dirty: view.result.current.isDirty,
      reads: requests('GET').length }).toEqual({ accepted: false, content: 'Newer typing not in the clipboard', dirty: true, reads: 0 })
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('My original conflict text'))
  })

  it('does not reselect the old note after a delayed clipboard copy', async () => {
    const view = renderEditor()
    await conflict(view)
    const clipboard = deferred<void>()
    vi.mocked(navigator.clipboard.writeText).mockImplementationOnce(() => clipboard.promise)
    let taking!: Promise<boolean>
    act(() => { taking = view.result.current.takeTheirVersion() })
    await open(view, 'other')
    edit(view, 'Other note draft')
    let accepted!: boolean
    await act(async () => { clipboard.resolve(); accepted = await taking })

    expect({ accepted, id: view.result.current.selectedId, content: view.result.current.content }).toEqual({
      accepted: false, id: 'other', content: 'Other note draft'
    })
  })

  it('does not discard edits made while confirming a failed clipboard copy', async () => {
    const view = renderEditor()
    await conflict(view)
    vi.mocked(navigator.clipboard.writeText).mockRejectedValueOnce(new Error('Clipboard denied'))
    const confirmation = deferred<boolean>()
    mocks.confirmDanger.mockImplementationOnce(() => confirmation.promise)
    let taking!: Promise<boolean>
    act(() => { taking = view.result.current.takeTheirVersion() })
    await act(async () => {})
    expect(mocks.confirmDanger).toHaveBeenCalledTimes(1)
    edit(view, 'Typed during confirmation')
    let accepted!: boolean
    await act(async () => { confirmation.resolve(true); accepted = await taking })

    expect({ accepted, content: view.result.current.content, dirty: view.result.current.isDirty,
      reads: requests('GET').length }).toEqual({ accepted: false, content: 'Typed during confirmation', dirty: true, reads: 0 })
  })

  it('does not force-save B using the version fetched for A', async () => {
    const view = renderEditor()
    await conflict(view)
    const version = delayVersionRead()
    let keeping!: Promise<boolean>
    act(() => { keeping = view.result.current.keepMyVersion() })
    await open(view, 'other')
    edit(view, 'Other note draft')
    let accepted!: boolean
    await act(async () => { version.resolve({ ...notes.get('one')! }); accepted = await keeping })

    expect({ accepted, writes: requests('PUT').length, stored: notes.get('other')?.content,
      content: view.result.current.content, dirty: view.result.current.isDirty }).toEqual({
      accepted: false, writes: 0, stored: 'other body', content: 'Other note draft', dirty: true
    })
  })

  it('does not reuse an earlier keep decision after an A-B-A selection', async () => {
    const view = renderEditor()
    await conflict(view)
    const version = delayVersionRead()
    let keeping!: Promise<boolean>
    act(() => { keeping = view.result.current.keepMyVersion() })
    await open(view, 'other')
    await open(view, 'one')
    edit(view, 'Later draft for the same note')
    let accepted!: boolean
    await act(async () => { version.resolve({ ...notes.get('one')! }); accepted = await keeping })

    expect({ accepted, writes: requests('PUT').length, content: view.result.current.content }).toEqual({
      accepted: false, writes: 0, content: 'Later draft for the same note'
    })
  })

  it('does not apply keep permission to text typed during the version read', async () => {
    const view = renderEditor()
    await conflict(view)
    const version = delayVersionRead()
    let keeping!: Promise<boolean>
    act(() => { keeping = view.result.current.keepMyVersion() })
    edit(view, 'Newer unapproved draft')
    let accepted!: boolean
    await act(async () => { version.resolve({ ...notes.get('one')! }); accepted = await keeping })

    expect({ accepted, writes: requests('PUT').length, content: view.result.current.content, dirty: view.result.current.isDirty }).toEqual({
      accepted: false, writes: 0, content: 'Newer unapproved draft', dirty: true
    })
  })

  it.each(['take', 'keep'] as const)('retires a pending %s choice after an authority change', async (choice) => {
    const view = renderEditor()
    await conflict(view)
    const gate = choice === 'take' ? deferred<void>() : delayVersionRead()
    if (choice === 'take') vi.mocked(navigator.clipboard.writeText).mockImplementationOnce(() => gate.promise as Promise<void>)
    let choosing!: Promise<boolean>
    act(() => { choosing = choice === 'take' ? view.result.current.takeTheirVersion() : view.result.current.keepMyVersion() })
    view.rerender({ authorityScope: createNotesGraphAuthorityScope(SERVER_URL, 8) })
    const readsBefore = requests('GET').length
    let accepted!: boolean
    await act(async () => {
      if (choice === 'take') (gate as ReturnType<typeof deferred<void>>).resolve()
      else (gate as ReturnType<typeof deferred<Note>>).resolve({ ...notes.get('one')! })
      accepted = await choosing
    })
    expect({ accepted, reads: requests('GET').length, writes: requests('PUT').length }).toEqual({ accepted: false, reads: readsBefore, writes: 0 })
  })

  it.each(['take', 'keep'] as const)('retires a pending %s choice after unmount', async (choice) => {
    const view = renderEditor()
    await conflict(view)
    const gate = choice === 'take' ? deferred<void>() : delayVersionRead()
    if (choice === 'take') vi.mocked(navigator.clipboard.writeText).mockImplementationOnce(() => gate.promise as Promise<void>)
    let choosing!: Promise<boolean>
    act(() => { choosing = choice === 'take' ? view.result.current.takeTheirVersion() : view.result.current.keepMyVersion() })
    view.unmount()
    const readsBefore = requests('GET').length
    let accepted!: boolean
    await act(async () => {
      if (choice === 'take') (gate as ReturnType<typeof deferred<void>>).resolve()
      else (gate as ReturnType<typeof deferred<Note>>).resolve({ ...notes.get('one')! })
      accepted = await choosing
    })
    expect({ accepted, reads: requests('GET').length, writes: requests('PUT').length }).toEqual({ accepted: false, reads: readsBefore, writes: 0 })
  })

  it('still loads the copied server version for an unchanged conflict choice', async () => {
    const view = renderEditor()
    await conflict(view)
    let accepted!: boolean
    await act(async () => { accepted = await view.result.current.takeTheirVersion() })
    expect({ accepted, content: view.result.current.content, version: view.result.current.selectedVersion,
      dirty: view.result.current.isDirty }).toEqual({ accepted: true, content: 'Their body', version: 3, dirty: false })
  })

  it('still keeps the original draft with the current server version', async () => {
    const view = renderEditor()
    await conflict(view)
    let accepted!: boolean
    await act(async () => { accepted = await view.result.current.keepMyVersion() })
    expect({ accepted, content: notes.get('one')?.content, version: notes.get('one')?.version,
      expectedVersion: requests('PUT')[0]?.headers?.['expected-version'] }).toEqual({
      accepted: true, content: 'My original conflict text', version: 4, expectedVersion: '3'
    })
  })
})
