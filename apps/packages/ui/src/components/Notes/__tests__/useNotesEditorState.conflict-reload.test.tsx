import React from 'react'
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { BgRequestInit } from '@/services/background-proxy'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'
import { createNotesGraphAuthorityScope } from '../hooks/useNotesGraphAuthorityScope'
import { NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY } from '../notes-manager-utils'

// NS-N1 (#3102): reloading after a save conflict must never move the editor's
// base version without also loading the content that version belongs to.

const mocks = vi.hoisted(() => ({
  userId: 1,
  serverUrl: 'https://notes.example.test',
  currentUser: vi.fn<() => Promise<{ id: number; is_active: boolean }>>(),
  confirmDanger: vi.fn<() => Promise<boolean>>(),
  request: vi.fn<(request: Pick<BgRequestInit, 'servicePromptConfig' | 'abortSignal'> & { path: string; method?: string; headers?: Record<string, string> }) => Promise<unknown>>(),
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
vi.mock('@/services/tldw/TldwAuth', () => ({ tldwAuth: { getCurrentUser: mocks.currentUser } }))
vi.mock('@/hooks/useCallerCapabilities', () => ({
  useCallerCapabilities: () => ({ monitoringAlerts: 'denied', userId: mocks.userId, refreshAfterForbidden: vi.fn(async () => undefined) })
}))
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/settings/registry')>(),
  getSetting: vi.fn(async () => null), setSetting: vi.fn(async () => undefined), clearSetting: vi.fn()
}))

const notes: Record<string, Record<string, unknown>> = {}
const serverResponse = async ({ path }: { path: string }) => {
  const id = path.split('/').at(-1) || ''
  return notes[id] ? { ...notes[id] } : {}
}

const renderEditor = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const deps: UseNotesEditorStateDeps = {
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
  return renderHook(() => {
    const [keywords, setKeywords] = React.useState<string[]>([])
    const editor = useNotesEditorState({
      ...deps, editorKeywords: keywords, setEditorKeywords: setKeywords,
      authorityScope: createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
      connectionConfig: { serverUrl: mocks.serverUrl, authMode: 'multi-user', accessToken: 'test-access-token' }
    })
    return { ...editor, editorKeywords: keywords }
  }, {
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  })
}

describe('Notes editor conflict reload (NS-N1)', () => {
  let user: ReturnType<typeof userEvent.setup>
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.userId = 1
    mocks.serverUrl = 'https://notes.example.test'
    mocks.currentUser.mockImplementation(async () => ({ id: mocks.userId, is_active: true }))
    mocks.confirmDanger.mockResolvedValue(true)
    user = userEvent.setup()
    mocks.refetch.mockResolvedValue(undefined)
    notes.one = {
      id: 'one', title: 'Shared note', content: 'Original body',
      metadata: { keywords: ['alpha'] }, version: 2, last_modified: '2026-10-03T10:00:00Z'
    }
    notes.other = {
      id: 'other', title: 'Other note', content: 'Other body',
      metadata: { keywords: [] }, version: 9, last_modified: '2026-10-03T10:00:00Z'
    }
    mocks.request.mockImplementation(serverResponse)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn(async () => undefined) }
    })
  })

  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(navigator, 'clipboard')
    Reflect.deleteProperty(navigator, 'onLine')
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

  describe('Notes conflict Reload original ownership', () => {
    const deferred = <T,>() => {
      let resolve!: (value: T) => void
      const promise = new Promise<T>(done => { resolve = done })
      return { promise, resolve }
    }
    const queueKey = () => `${NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY}:${createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId)}`
    const readQueue = (key: string) => JSON.parse(localStorage.getItem(key) || '{}')
    const reads = () => mocks.request.mock.calls.map(([request]) => request)
      .filter(request => String(request.method || 'GET').toUpperCase() === 'GET' && request.path.startsWith('/api/v1/notes/'))
    const state = (view: ReturnType<typeof renderEditor>) => ({
      selectedId: view.result.current.selectedId, title: view.result.current.title,
      content: view.result.current.content, version: view.result.current.selectedVersion,
      keywords: view.result.current.editorKeywords, isDirty: view.result.current.isDirty,
      originalMetadata: view.result.current.originalMetadata
    })
    const prepare = async (dirty = true) => {
      const view = renderEditor()
      await act(async () => { await view.result.current.loadDetail('one') })
      await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
      if (dirty) act(() => { view.result.current.setContentDirty('Original local draft') })
      notes.one = { ...notes.one, title: 'Server title', content: 'Server body', metadata: { keywords: ['beta'] }, version: 3 }
      mocks.request.mockClear()
      mocks.currentUser.mockClear()
      mocks.refetch.mockClear()
      return view
    }
    const queueNewer = async (view: ReturnType<typeof renderEditor>, text: string) => {
      Object.defineProperty(navigator, 'onLine', { configurable: true, value: false })
      act(() => { view.result.current.setContentDirty(text) })
      let queued = false
      await act(async () => { queued = await view.result.current.saveNote({ showSuccessMessage: false, trigger: 'manual' }) })
      expect(queued).toBe(true)
      const entryKey = `note:${String(view.result.current.selectedId)}`
      const entry = view.result.current.offlineDraftQueue[entryKey]
      expect(entry).toMatchObject({ content: text, syncState: 'queued' })
      const storageKey = queueKey()
      expect(readQueue(storageKey)[entryKey]).toEqual(entry)
      return { entry, entryKey, storageKey }
    }
    const changeOwner = async (view: ReturnType<typeof renderEditor>, change: string) => {
      if (change === 'unmount') { view.unmount(); return }
      if (change === 'selection-return') {
        await act(async () => { await view.result.current.loadDetail('other') })
        await act(async () => { await view.result.current.loadDetail('one') })
        return
      }
      const changeOnce = async (away: boolean) => {
        await act(async () => {
          if (change.startsWith('account')) mocks.userId = away ? 2 : 1
          else mocks.serverUrl = away ? 'https://other-notes.example.test' : 'https://notes.example.test'
          view.rerender()
          window.dispatchEvent(change.startsWith('account') ? new Event('tldw:auth-principal-changed')
            : new CustomEvent('tldw:config-updated', { detail: { authorityChanged: true } }))
        })
      }
      await changeOnce(true)
      if (change.endsWith('return')) await changeOnce(false)
    }
    const toast = (view: ReturnType<typeof renderEditor>) => {
      act(() => { view.result.current.handleVersionConflict('one') })
      const payload = mocks.messages.error.mock.calls.at(-1)?.[0]
      expect(payload?.content).toBeTruthy()
      const rendered = render(<>{payload.content}</>)
      mocks.messages.error.mockClear()
      return { button: rendered.getByRole('button', { name: 'Reload notes' }), alert: rendered.getByRole('alert') }
    }
    const expectOwned = (request: ReturnType<typeof reads>[number]) => {
      expect(request.servicePromptConfig).toMatchObject({ serverUrl: 'https://notes.example.test', expectedUserId: 1 })
      expect(request.headers?.['X-TLDW-Expected-User-ID']).toBe('1')
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
    }

    it.each([
      ['selection-return', 'button'], ['account-return', 'button'], ['server-return', 'button'],
      ['unmount', 'button'], ['selection-return', 'keyboard']
    ])('refuses a retained Reload callback after %s using %s before clipboard, auth, reads or queue removal', async (change, input) => {
      const view = await prepare()
      const queued = await queueNewer(view, 'Original owner queued draft')
      const { button, alert } = toast(view)
      await changeOwner(view, change)
      const replacement = state(view)
      const confirmedQueue = readQueue(queued.storageKey)[queued.entryKey]
      expect(confirmedQueue).toBeDefined()
      mocks.request.mockClear()
      mocks.currentUser.mockClear()
      mocks.refetch.mockClear()
      vi.mocked(navigator.clipboard.writeText).mockClear()
      mocks.confirmDanger.mockClear()
      if (input === 'keyboard') {
        alert.focus()
        await user.keyboard('{Enter}')
      } else await user.click(button)
      await act(async () => {})
      expect(reads()).toHaveLength(0)
      expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
      expect(mocks.currentUser).not.toHaveBeenCalled()
      expect(mocks.confirmDanger).not.toHaveBeenCalled()
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(readQueue(queued.storageKey)[queued.entryKey]).toEqual(confirmedQueue)
      if (change !== 'unmount') {
        expect(state(view)).toEqual(replacement)
        expect(view.result.current.offlineDraftQueue[queued.entryKey]).toEqual(confirmedQueue)
      }
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it('copies edits made on the same note after the toast appeared and then loads the scoped complete server head', async () => {
      const view = await prepare()
      const { button } = toast(view)
      act(() => { view.result.current.setContentDirty('Newer local text before Reload click') })
      await user.click(button)
      await waitFor(() => expect(view.result.current.content).toBe('Server body'))
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('Newer local text before Reload click'))
      expect(state(view)).toMatchObject({ selectedId: 'one', title: 'Server title', content: 'Server body', version: 3, keywords: ['beta'], isDirty: false })
      expectOwned(reads()[0])
      expect(mocks.currentUser).toHaveBeenCalledTimes(1)
      expect(mocks.refetch).toHaveBeenCalledTimes(1)
    })

    it.each(['clipboard', 'dialog'])('retains a confirmed newer queue and draft while conflict Reload waits for %s', async boundary => {
      const view = await prepare()
      const gate = deferred<void>()
      if (boundary === 'clipboard') vi.mocked(navigator.clipboard.writeText).mockImplementation(async () => { await gate.promise })
      else {
        vi.mocked(navigator.clipboard.writeText).mockRejectedValue(new Error('Clipboard unavailable'))
        mocks.confirmDanger.mockImplementation(async () => { await gate.promise; return true })
      }
      let pending!: Promise<void>
      act(() => { pending = view.result.current.reloadNotes('one') })
      await waitFor(() => expect(boundary === 'clipboard' ? navigator.clipboard.writeText : mocks.confirmDanger).toHaveBeenCalledTimes(1))
      const queued = await queueNewer(view, 'Newer queued text during Reload')
      const replacement = state(view)
      await act(async () => { gate.resolve(undefined); await pending })
      expect(readQueue(queued.storageKey)[queued.entryKey]).toEqual(queued.entry)
      expect(view.result.current.offlineDraftQueue[queued.entryKey]).toEqual(queued.entry)
      expect(state(view)).toEqual(replacement)
      expect(reads()).toHaveLength(0)
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it('cancels conflict Reload when newer text and a confirmed queue replace the draft during fresh auth', async () => {
      const view = await prepare()
      const auth = deferred<{ id: number; is_active: boolean }>()
      mocks.currentUser.mockReturnValue(auth.promise)
      let pending!: Promise<void>
      act(() => { pending = view.result.current.reloadNotes('one') })
      await act(async () => {})
      // Missing baseline auth is an assertion failure, never an awaited setup gate.
      expect(mocks.currentUser).toHaveBeenCalledTimes(1)
      const queued = await queueNewer(view, 'Newer queued text during auth')
      const replacement = state(view)
      await act(async () => { auth.resolve({ id: 1, is_active: true }); await pending })
      expect(reads()).toHaveLength(0)
      expect(state(view)).toEqual(replacement)
      expect(readQueue(queued.storageKey)[queued.entryKey]).toEqual(queued.entry)
      expect(view.result.current.offlineDraftQueue[queued.entryKey]).toEqual(queued.entry)
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
    })

    it.each(['success', 'failure'])('keeps a newer Beta draft and queue after a delayed conflict detail %s', async outcome => {
      const view = await prepare()
      const gate = deferred<void>()
      mocks.request.mockImplementation(async request => {
        if (request.path === '/api/v1/notes/one') {
          await gate.promise
          if (outcome === 'failure') throw new Error('Old detail unavailable')
        }
        return serverResponse(request)
      })
      let pending!: Promise<void>
      act(() => { pending = view.result.current.reloadNotes('one') })
      await waitFor(() => expect(reads().filter(request => request.path === '/api/v1/notes/one')).toHaveLength(1))
      await act(async () => { await view.result.current.loadDetail('other') })
      const queued = await queueNewer(view, 'Newer Beta queued draft')
      const replacement = state(view)
      mocks.refetch.mockClear()
      await act(async () => { gate.resolve(undefined); await pending })
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(state(view)).toEqual(replacement)
      expect(view.result.current.offlineDraftQueue[queued.entryKey]).toEqual(queued.entry)
      expect(readQueue(queued.storageKey)[queued.entryKey]).toEqual(queued.entry)
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it('loads a clean scoped server head quietly without copying or confirming', async () => {
      const view = await prepare(false)
      let loaded = false
      await act(async () => { loaded = await view.result.current.loadLatestVersion() })
      expect(loaded).toBe(true)
      expect(state(view)).toMatchObject({ selectedId: 'one', title: 'Server title', content: 'Server body', version: 3, keywords: ['beta'], isDirty: false })
      expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
      expect(mocks.confirmDanger).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expectOwned(reads()[0])
      expect(mocks.currentUser).toHaveBeenCalledTimes(1)
    })

    it('cancels clean Reload before reading when auth outlives an account away-and-return', async () => {
      const view = await prepare(false)
      const auth = deferred<{ id: number; is_active: boolean }>()
      mocks.currentUser.mockReturnValue(auth.promise)
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.loadLatestVersion() })
      await act(async () => {})
      expect(mocks.currentUser).toHaveBeenCalledTimes(1)
      await changeOwner(view, 'account-return')
      const replacement = state(view)
      let loaded = true
      await act(async () => { auth.resolve({ id: 1, is_active: true }); loaded = await pending })
      expect(loaded).toBe(false)
      expect(reads()).toHaveLength(0)
      expect(state(view)).toEqual(replacement)
      expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it.each(['selection-return', 'newer-edit'])('keeps the newer editor after a clean detail read and %s', async change => {
      const view = await prepare(false)
      const gate = deferred<void>()
      const detailReads: ReturnType<typeof reads> = []
      mocks.request.mockImplementation(async request => {
        if (request.path === '/api/v1/notes/one' && detailReads.length === 0) {
          detailReads.push(request)
          await gate.promise
        }
        return serverResponse(request)
      })
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.loadLatestVersion() })
      await waitFor(() => expect(detailReads).toHaveLength(1))
      if (change === 'selection-return') await changeOwner(view, change)
      else act(() => { view.result.current.setContentDirty('Newer text during clean Reload') })
      const replacement = state(view)
      let loaded = true
      await act(async () => { gate.resolve(undefined); loaded = await pending })
      expect(loaded).toBe(false)
      expect(state(view)).toEqual(replacement)
      expectOwned(detailReads[0])
      expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })
  })
})
