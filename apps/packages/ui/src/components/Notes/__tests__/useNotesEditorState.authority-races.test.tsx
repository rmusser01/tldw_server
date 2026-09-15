import React from 'react'
import { act, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'

const mocks = vi.hoisted(() => ({ request: vi.fn(), tasks: vi.fn(), activity: vi.fn(), error: vi.fn() }))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
vi.mock('@/services/notes-tasks', () => ({
  listNoteTasks: mocks.tasks, listTaskActivity: mocks.activity,
  markTaskActivityRead: vi.fn(), setNoteTaskStatus: vi.fn()
}))
vi.mock('@/services/tldw/TldwAuth', () => ({ tldwAuth: { getCurrentUser: vi.fn(async () => null) } }))
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/settings/registry')>(),
  getSetting: vi.fn(async () => null), setSetting: vi.fn(async () => undefined), clearSetting: vi.fn()
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

const renderEditor = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: 'alice', isOnline: false, isMobileViewport: false,
    message: { error: mocks.error } as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: vi.fn(async () => true), queryClient, t: (key) => key,
    listMode: 'active', setListMode: vi.fn(), data: [], refetch: vi.fn(async () => undefined),
    setPage: vi.fn(), setQuery: vi.fn(), setQueryInput: vi.fn(), setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(), setMobileSidebarOpen: vi.fn(), editorKeywords: [],
    setEditorKeywords: vi.fn(), keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(), setKeywordSuggestionSelection: vi.fn(), editorDisabled: false
  }
  return renderHook(({ scope }) => useNotesEditorState({ ...deps, authorityScope: scope }), {
    initialProps: { scope: 'alice' },
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  })
}

describe('Notes editor authority races', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.tasks.mockResolvedValue({ tasks: [], reconciliation: null })
    mocks.activity.mockResolvedValue({ events: [] })
  })

  it('ignores an old detail response after Alice → Bob → Alice', async () => {
    const old = deferred<unknown>()
    mocks.request.mockImplementation(({ path }: { path: string }) => path.endsWith('/old')
      ? old.promise : Promise.resolve({ id: 'new', title: 'Current note', content: 'Current content' }))
    const view = renderEditor()
    let loading!: Promise<boolean>
    act(() => { loading = view.result.current.loadDetail('old') })
    view.rerender({ scope: 'bob' })
    view.rerender({ scope: 'alice' })
    await act(async () => { await view.result.current.loadDetail('new') })
    await act(async () => {
      old.resolve({ id: 'old', title: 'Stale note', content: 'Stale content' })
      expect(await loading).toBe(false)
    })
    expect(view.result.current.content).toBe('Current content')
  })

  it('starts the next authority with no stale loading indicator', async () => {
    const old = deferred<unknown>()
    mocks.request.mockReturnValue(old.promise)
    const view = renderEditor()
    act(() => { void view.result.current.loadDetail('old') })
    expect(view.result.current.loadingDetail).toBe(true)
    view.rerender({ scope: 'bob' })
    expect(view.result.current.loadingDetail).toBe(false)
    await act(async () => { old.resolve({ id: 'old' }) })
  })

  it('does not show a stale error or end the next authority’s loading state', async () => {
    const old = deferred<unknown>()
    const current = deferred<unknown>()
    mocks.request.mockImplementation(({ path }: { path: string }) => path.endsWith('/old') ? old.promise : current.promise)
    const view = renderEditor()
    let loading!: Promise<boolean>
    act(() => { loading = view.result.current.loadDetail('old') })
    view.rerender({ scope: 'bob' })
    act(() => { void view.result.current.loadDetail('new') })
    await act(async () => { old.reject(new Error('Old failure')); await loading })
    expect({ errors: mocks.error.mock.calls, loading: view.result.current.loadingDetail }).toEqual({ errors: [], loading: true })
    await act(async () => { current.resolve({ id: 'new', content: 'Current' }) })
  })

  it('ignores delayed task state after Alice → Bob → Alice', async () => {
    const old = deferred<unknown>()
    mocks.tasks.mockImplementation((id: string) => id === 'old'
      ? old.promise : Promise.resolve({ tasks: [{ id: 'current-task' }], reconciliation: null }))
    mocks.request.mockResolvedValue({ title: 'Note', content: 'Body' })
    const view = renderEditor()
    let loading!: Promise<boolean>
    await act(async () => { loading = view.result.current.loadDetail('old') })
    view.rerender({ scope: 'bob' })
    view.rerender({ scope: 'alice' })
    await act(async () => { await view.result.current.loadDetail('new') })
    await act(async () => { old.resolve({ tasks: [{ id: 'stale-task' }], reconciliation: null }); await loading })
    expect(view.result.current.noteTasks.map((task) => task.id)).toEqual(['current-task'])
    expect(mocks.activity).not.toHaveBeenCalledWith({ note_id: 'old', limit: 50 })
  })

  it.each(['resolve', 'reject'] as const)('ignores delayed activity %s after Alice → Bob → Alice', async (settle) => {
    const old = deferred<unknown>()
    mocks.activity.mockImplementation(({ note_id }: { note_id: string }) => note_id === 'old'
      ? old.promise : Promise.resolve({ events: [{ id: 'current-event', note_id: 'new' }] }))
    mocks.request.mockResolvedValue({ title: 'Note', content: 'Body' })
    const view = renderEditor()
    let loading!: Promise<boolean>
    await act(async () => { loading = view.result.current.loadDetail('old') })
    view.rerender({ scope: 'bob' })
    view.rerender({ scope: 'alice' })
    await act(async () => { await view.result.current.loadDetail('new') })
    await act(async () => {
      if (settle === 'resolve') old.resolve({ events: [{ id: 'stale-event', note_id: 'old' }] })
      else old.reject(new Error('Old activity failure'))
      await loading
    })
    expect(view.result.current.taskActivityEvents.map((event) => event.id)).toEqual(['current-event'])
  })
})
