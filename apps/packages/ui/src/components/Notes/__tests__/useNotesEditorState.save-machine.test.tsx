import React from 'react'
import type { BgRequestInit } from '@/services/background-proxy'
import { act, cleanup, fireEvent, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The editor hook drives one save state machine (#3102: NS-01, NS-03, NS-N2,
// NS-05, NS-02, NS-06). These tests run it against a small versioned server.

const mocks = vi.hoisted(() => ({
  userId: 7,
  serverUrl: 'https://notes.test',
  currentUser: vi.fn(),
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
  tldwAuth: { getCurrentUser: mocks.currentUser }
}))
vi.mock('@/hooks/useCallerCapabilities', () => {
  const capabilities = { monitoringAlerts: 'denied', get userId() { return mocks.userId }, refreshAfterForbidden: vi.fn(async () => undefined) }
  return { useCallerCapabilities: () => capabilities }
})
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  getSetting: vi.fn(async () => null),
  setSetting: vi.fn(async () => undefined),
  clearSetting: vi.fn()
}))

import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'
import { createNotesGraphAuthorityScope } from '../hooks/useNotesGraphAuthorityScope'
import { NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY } from '../notes-manager-utils'

type StoredNote = {
  id: string
  title: string
  content: string
  version: number
  last_modified: string
  metadata: Record<string, unknown>
}
type RequestInit = Pick<BgRequestInit, 'servicePromptConfig' | 'abortSignal'> & {
  path: string
  method?: string
  headers?: Record<string, string>
  body?: { title?: string; content?: string; auto_title?: boolean; metadata?: Record<string, unknown>; [key: string]: unknown }
}

const SERVER_URL = 'https://notes.test'
const AUTHORITY = createNotesGraphAuthorityScope(SERVER_URL, 7)
const QUEUE_KEY = `${NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY}:${AUTHORITY}`

const server = {
  notes: new Map<string, StoredNote>(),
  failures: [] as Array<{ method: 'PUT' | 'POST'; error: unknown }>,
  created: 0
}

const writes = (method: 'PUT' | 'POST') =>
  mocks.request.mock.calls
    .map(([init]) => init as RequestInit)
    .filter((init) => String(init.method || 'GET').toUpperCase() === method && init.path.startsWith('/api/v1/notes/'))

const respond = async (init: RequestInit) => {
  const method = String(init.method || 'GET').toUpperCase()
  const scripted = server.failures.findIndex((failure) => failure.method === method)
  if (scripted >= 0) {
    const [failure] = server.failures.splice(scripted, 1)
    throw failure.error
  }
  if (init.path.startsWith('/api/v1/monitoring/')) return { items: [] }
  if (method === 'POST' && init.path === '/api/v1/notes/') {
    const body = init.body || {}
    if (!body.title && !body.auto_title) {
      throw { status: 400, message: 'Title is required unless auto_title=true (POST /api/v1/notes/)' }
    }
    server.created += 1
    const id = `n${server.created}`
    const note: StoredNote = {
      id,
      title: body.title || `Auto: ${String(body.content).split('\n')[0]}`,
      content: String(body.content),
      version: 1,
      last_modified: '2026-10-03T12:00:00Z',
      metadata: body.metadata || {}
    }
    server.notes.set(id, note)
    return { ...note }
  }
  const id = decodeURIComponent(init.path.split('/').at(-1) || '')
  const stored = server.notes.get(id)
  if (!stored) throw { status: 404, message: `Not found (${method} ${init.path})` }
  if (method === 'PUT') {
    if (Number(init.headers?.['expected-version']) !== stored.version) {
      throw { status: 409, message: `Version conflict (PUT ${init.path})` }
    }
    const next = {
      ...stored,
      title: init.body?.title ?? stored.title,
      content: String(init.body?.content ?? stored.content),
      version: stored.version + 1,
      last_modified: '2026-10-03T12:30:00Z'
    }
    server.notes.set(id, next)
    return { ...next }
  }
  return { ...stored }
}

const editElsewhere = (id: string, content: string) => {
  const stored = server.notes.get(id)!
  server.notes.set(id, { ...stored, content, version: stored.version + 1, last_modified: '2026-10-03T12:25:00Z' })
}

const translate: UseNotesEditorStateDeps['t'] = (key, options) =>
  (options as { defaultValue?: string } | undefined)?.defaultValue ?? key

const renderEditor = ({ unstableT = false }: { unstableT?: boolean } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: AUTHORITY,
    connectionConfig: {
      serverUrl: SERVER_URL,
      authMode: 'multi-user',
      accessToken: `test.${btoa(JSON.stringify({ sub: '7' }))}.signature`
    } as UseNotesEditorStateDeps['connectionConfig'],
    isOnline: true,
    isMobileViewport: false,
    message: mocks.messages as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: mocks.confirmDanger,
    queryClient,
    t: (key, options) => (options as { defaultValue?: string } | undefined)?.defaultValue ?? key,
    listMode: 'active',
    setListMode: vi.fn(),
    data: [],
    refetch: mocks.refetch,
    setPage: vi.fn(),
    setQuery: vi.fn(),
    setQueryInput: vi.fn(),
    setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(),
    setMobileSidebarOpen: vi.fn(),
    editorKeywords: [],
    setEditorKeywords: vi.fn(),
    keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(),
    setKeywordSuggestionSelection: vi.fn(),
    editorDisabled: false
  }
  return renderHook(
    () => {
      const [keywords, setKeywords] = React.useState<string[]>([])
      // Some page suites mock `t` as a new function on every render.
      const t: UseNotesEditorStateDeps['t'] = unstableT ? (key, options) => translate(key, options) : deps.t
      return useNotesEditorState({
        ...deps, t, editorKeywords: keywords, setEditorKeywords: setKeywords,
        authorityScope: createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
        connectionConfig: {
          ...deps.connectionConfig,
          serverUrl: mocks.serverUrl,
          accessToken: `test.${btoa(JSON.stringify({ sub: String(mocks.userId) }))}.signature`
        } as UseNotesEditorStateDeps['connectionConfig']
      })
    },
    { wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider> }
  )
}

type EditorView = ReturnType<typeof renderEditor>

const openNote = async (view: EditorView, id = 'one') => {
  await act(async () => {
    await view.result.current.loadDetail(id)
  })
}

const edit = (view: EditorView, content: string) => {
  act(() => view.result.current.setContentDirty(content))
}

const advance = async (ms: number) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

const readQueue = () => JSON.parse(localStorage.getItem(QUEUE_KEY) || '{}') as Record<string, { content?: string; syncState?: string }>

describe('Notes editor save state machine', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.userId = 7
    mocks.serverUrl = SERVER_URL
    mocks.currentUser.mockImplementation(async () => ({ id: mocks.userId, is_active: true }))
    localStorage.clear()
    server.notes.clear()
    server.failures = []
    server.created = 0
    server.notes.set('one', {
      id: 'one',
      title: 'Shared note',
      content: 'Original body',
      version: 2,
      last_modified: '2026-10-03T10:00:00Z',
      metadata: { keywords: [] }
    })
    mocks.request.mockImplementation(respond)
    mocks.refetch.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn(async () => undefined) }
    })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    Reflect.deleteProperty(navigator, 'clipboard')
    document.body.innerHTML = ''
  })

  it('characterizes first-save identity assignment resetting Split to Edit', async () => {
    const view = renderEditor()
    act(() => {
      view.result.current.setTitle('New note')
      view.result.current.setContentDirty('New body')
      view.result.current.setEditorMode('split')
    })
    expect(view.result.current.selectedId).toBeNull()
    expect(view.result.current.editorMode).toBe('split')
    await act(async () => { await view.result.current.saveNote({ showSuccessMessage: false }) })
    expect(view.result.current.selectedId).not.toBeNull()
    expect(view.result.current.editorMode).toBe('edit')
  })

  it('says "saved" only after the server acknowledges the save', async () => {
    const view = renderEditor()
    await openNote(view)
    expect(view.result.current.saveIndicator).toBe('saved')

    edit(view, 'Edited body')
    expect(view.result.current.saveIndicator).toBe('dirty')

    let acknowledge!: () => void
    mocks.request.mockImplementation(async (init: RequestInit) => {
      if (String(init.method).toUpperCase() === 'PUT') {
        await new Promise<void>((resolve) => {
          acknowledge = resolve
        })
      }
      return respond(init)
    })
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote({ showSuccessMessage: false })
    })
    await waitFor(() => expect(writes('PUT')).toHaveLength(1))
    expect(view.result.current.saveIndicator).toBe('saving')

    await act(async () => {
      acknowledge()
      await saving
    })
    expect(view.result.current.saveIndicator).toBe('saved')
    expect(view.result.current.selectedVersion).toBe(3)
    expect(server.notes.get('one')?.content).toBe('Edited body')
  })

  it('never shows "saved" for a failed save', async () => {
    const view = renderEditor()
    await openNote(view)
    edit(view, 'Edited body')
    server.failures.push({ method: 'PUT', error: { status: 500, message: 'Boom (PUT /api/v1/notes/one)' } })

    await act(async () => {
      await view.result.current.saveNote({ showSuccessMessage: false })
    })

    expect(view.result.current.saveIndicator).toBe('retrying')
    expect(view.result.current.isDirty).toBe(true)
  })

  it('does not retry a 400 until the note changes again', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    server.failures.push({ method: 'POST', error: { status: 400, message: 'Content is invalid (POST /api/v1/notes/)' } })
    edit(view, 'A note that the server rejects')

    await advance(5_000)
    expect(writes('POST')).toHaveLength(1)
    expect(view.result.current.saveIndicator).toBe('error')
    expect(view.result.current.saveIssue).toMatchObject({ kind: 'error' })
    expect(view.result.current.saveIssue?.message).toContain('Content is invalid')
    expect(view.result.current.saveIssue?.message).not.toContain('POST /api')

    await advance(120_000)
    expect(writes('POST')).toHaveLength(1)

    edit(view, 'A note that the server now accepts')
    await advance(5_000)
    expect(writes('POST')).toHaveLength(2)
    expect(view.result.current.saveIndicator).toBe('saved')
  })

  it('enters a conflict on 409, asks nothing on its own and stops autosaving', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')

    edit(view, 'My body')
    await advance(5_000)
    expect(writes('PUT')).toHaveLength(1)
    expect(view.result.current.saveIndicator).toBe('conflict')
    expect(view.result.current.saveIssue).toMatchObject({ kind: 'conflict' })

    await advance(5 * 60_000)
    edit(view, 'My body, still typing')
    await advance(60_000)
    expect(writes('PUT')).toHaveLength(1)

    let saved = true
    await act(async () => {
      saved = await view.result.current.saveNote()
    })
    expect(saved).toBe(false)
    expect(writes('PUT')).toHaveLength(1)
    expect(mocks.confirmDanger).not.toHaveBeenCalled()
    expect(document.querySelector('[role="dialog"]')).toBeNull()
    expect(view.result.current.content).toBe('My body, still typing')
  })

  it('turns a newer server version seen while editing into a conflict without a modal', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')

    await advance(30_000)
    expect(view.result.current.saveIssue).toMatchObject({ kind: 'remote-newer' })

    edit(view, 'My body')
    await advance(0)
    expect(view.result.current.saveIndicator).toBe('conflict')
    await advance(20_000)
    expect(writes('PUT')).toHaveLength(0)
    expect(mocks.confirmDanger).not.toHaveBeenCalled()
  })

  it('keeps my version by saving over the server’s latest version, never the stale one', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')
    edit(view, 'My body')
    await advance(5_000)
    expect(view.result.current.saveIndicator).toBe('conflict')

    await act(async () => {
      await view.result.current.keepMyVersion()
    })

    const puts = writes('PUT')
    expect(puts).toHaveLength(2)
    expect(puts[0].headers?.['expected-version']).toBe('2')
    expect(puts[1].headers?.['expected-version']).toBe('3')
    expect(server.notes.get('one')).toMatchObject({ content: 'My body', version: 4 })
    expect(view.result.current.saveIndicator).toBe('saved')
    expect(view.result.current.isDirty).toBe(false)
  })

  it('takes their version: loads the server copy and copies my text first', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')
    edit(view, 'My body')
    await advance(5_000)

    await act(async () => {
      await view.result.current.takeTheirVersion()
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('My body'))
    expect(view.result.current.content).toBe('Their body')
    expect(view.result.current.isDirty).toBe(false)
    expect(view.result.current.saveIndicator).toBe('saved')
    expect(view.result.current.selectedVersion).toBe(3)
    await advance(60_000)
    expect(writes('PUT')).toHaveLength(1)
  })

  it('copies my text without leaving the conflict', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')
    edit(view, 'My body')
    await advance(5_000)

    await act(async () => {
      await view.result.current.copyMyText()
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('My body'))
    expect(view.result.current.saveIndicator).toBe('conflict')
    expect(view.result.current.content).toBe('My body')
  })

  it('retries network failures with backoff, then hands the draft to the offline queue', async () => {
    vi.useFakeTimers()
    const view = renderEditor()
    await openNote(view)
    for (let i = 0; i < 3; i += 1) {
      server.failures.push({ method: 'PUT', error: new TypeError('Failed to fetch') })
    }

    edit(view, 'Typed while the server was unreachable')
    await advance(5_000)
    expect(writes('PUT')).toHaveLength(1)
    expect(view.result.current.saveIndicator).toBe('retrying')
    expect(readQueue()['note:one']?.content).toBe('Typed while the server was unreachable')

    await advance(4_999)
    expect(writes('PUT')).toHaveLength(1)
    await advance(1)
    expect(writes('PUT')).toHaveLength(2)

    await advance(15_000)
    expect(writes('PUT')).toHaveLength(3)
    expect(view.result.current.saveIndicator).toBe('offline')
    expect(view.result.current.isDirty).toBe(false)

    // The queue now owns the draft and syncs it once the server answers again.
    await advance(60_000)
    expect(server.notes.get('one')?.content).toBe('Typed while the server was unreachable')
    expect(readQueue()['note:one']).toBeUndefined()
    expect(view.result.current.saveIndicator).toBe('saved')
  })

  it('flushes pending edits before leaving and lets the navigation continue', async () => {
    const view = renderEditor()
    await openNote(view)
    edit(view, 'Typed just before leaving')
    expect(view.result.current.leaveGuard.when).toBe(true)

    let leave = false
    await act(async () => {
      leave = await view.result.current.leaveGuard.onLeave()
    })

    expect(leave).toBe(true)
    expect(server.notes.get('one')?.content).toBe('Typed just before leaving')
    expect(view.result.current.leaveGuard.when).toBe(false)
    expect(mocks.refetch).not.toHaveBeenCalled()
  })

  it('leaves without a request when nothing is pending', async () => {
    const view = renderEditor()
    await openNote(view)
    const before = mocks.request.mock.calls.length

    let leave = false
    await act(async () => {
      leave = await view.result.current.leaveGuard.onLeave()
    })

    expect(leave).toBe(true)
    expect(mocks.request.mock.calls.length).toBe(before)
  })

  it('keeps a copy on this device when the server is unreachable, then lets the user go', async () => {
    const view = renderEditor()
    await openNote(view)
    edit(view, 'Typed with the server down')
    server.failures.push({ method: 'PUT', error: new TypeError('Failed to fetch') })

    let leaving!: Promise<boolean>
    act(() => {
      leaving = view.result.current.leaveGuard.onLeave()
    })
    const dialog = await screen.findByRole('dialog')

    // The copy is on this device before the user decides.
    expect(readQueue()['note:one']?.content).toBe('Typed with the server down')
    expect(dialog).toHaveTextContent(/kept on this device/i)
    fireEvent.click(screen.getByRole('button', { name: /continue, sync later/i }))
    await expect(leaving).resolves.toBe(true)
  })

  it('asks before leaving when the save cannot succeed, and stays unless the user discards', async () => {
    const view = renderEditor()
    await openNote(view)
    editElsewhere('one', 'Their body')
    edit(view, 'My body')
    await act(async () => {
      await view.result.current.saveNote({ showSuccessMessage: false })
    })
    expect(view.result.current.saveIndicator).toBe('conflict')

    let first!: Promise<boolean>
    act(() => {
      first = view.result.current.leaveGuard.onLeave()
    })
    const stayDialog = await screen.findByRole('dialog')
    expect(stayDialog).toHaveTextContent(/changed in another tab or device/i)
    expect(stayDialog).not.toHaveTextContent(/try saving again/i)
    fireEvent.click(screen.getByRole('button', { name: /stay/i }))
    await expect(first).resolves.toBe(false)
    expect(view.result.current.content).toBe('My body')

    let second!: Promise<boolean>
    act(() => {
      second = view.result.current.leaveGuard.onLeave()
    })
    // The first dialog may still be closing; act on the newest one.
    await waitFor(() => expect(screen.getAllByRole('button', { name: /discard/i }).length).toBeGreaterThan(0))
    const discardButtons = screen.getAllByRole('button', { name: /discard/i })
    fireEvent.click(discardButtons[discardButtons.length - 1])
    await expect(second).resolves.toBe(true)
    expect(writes('PUT')).toHaveLength(1)
  })

  it('guards a page unload while there are unsaved edits', async () => {
    const view = renderEditor()
    await openNote(view)
    const clean = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(clean)
    expect(clean.defaultPrevented).toBe(false)

    edit(view, 'Not saved yet')
    const dirty = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(dirty)
    expect(dirty.defaultPrevented).toBe(true)
  })

  it('lets the server name an untitled note', async () => {
    const view = renderEditor()
    edit(view, 'First line becomes the title\nmore text')

    let saved = false
    await act(async () => {
      saved = await view.result.current.saveNote({ showSuccessMessage: false })
    })

    expect(saved).toBe(true)
    const [post] = writes('POST')
    expect(post.body).toMatchObject({ auto_title: true })
    expect(post.body?.title).toBeUndefined()
    await waitFor(() => expect(view.result.current.title).toBe('Auto: First line becomes the title'))
    expect(view.result.current.saveIndicator).toBe('saved')
  })

  it('does not resync a failing offline queue on every render', async () => {
    localStorage.setItem(QUEUE_KEY, JSON.stringify({
      'note:one': {
        key: 'note:one', noteId: 'one', baseVersion: 2, title: 'Shared note', content: 'Queued text',
        keywords: [], metadata: null, backlinkConversationId: null, backlinkMessageId: null,
        updatedAt: '2026-10-03T12:00:00Z', syncState: 'queued', lastError: null
      }
    }))
    const { tldwAuth } = await import('@/services/tldw/TldwAuth')
    const getCurrentUser = vi.mocked(tldwAuth.getCurrentUser)
    getCurrentUser.mockRejectedValue(new TypeError('Failed to fetch'))
    const view = renderEditor({ unstableT: true })
    await act(async () => {})
    await act(async () => {})
    const attemptsAfterMount = getCurrentUser.mock.calls.length
    expect(attemptsAfterMount).toBeGreaterThan(0)

    // Re-renders with new callback identities must not start new syncs.
    for (let i = 0; i < 5; i += 1) {
      await act(async () => {
        view.rerender()
      })
    }

    expect(getCurrentUser.mock.calls.length).toBe(attemptsAfterMount)
    getCurrentUser.mockImplementation(async () => ({ id: 7, is_active: true }) as never)
  })

  it('keeps unsaved edits on this device when the editor goes away', async () => {
    const view = renderEditor()
    await openNote(view)
    edit(view, 'Unsaved when the page unmounted')

    view.unmount()

    expect(readQueue()['note:one']).toMatchObject({ content: 'Unsaved when the page unmounted', syncState: 'queued' })
  })

  describe('Notes conflict choice ownership', () => {
    function deferred() {
      let resolve!: () => void
      const promise = new Promise<void>((done) => { resolve = done })
      return { promise, resolve }
    }
    const changes = [
      'selection', 'selection-return', 'account', 'account-return',
      'server', 'server-return', 'edit', 'unmount'
    ] as const
    const state = (view: EditorView) => ({
      selectedId: view.result.current.selectedId,
      title: view.result.current.title,
      content: view.result.current.content,
      selectedVersion: view.result.current.selectedVersion,
      originalMetadata: view.result.current.originalMetadata,
      isDirty: view.result.current.isDirty,
      saveIndicator: view.result.current.saveIndicator
    })
    const prepareConflict = async () => {
      server.notes.set('two', {
        id: 'two', title: 'Other note', content: 'Other body', version: 3,
        last_modified: '2026-10-03T11:00:00Z', metadata: { keywords: [] }
      })
      const view = renderEditor()
      await openNote(view)
      edit(view, 'My conflict draft')
      editElsewhere('one', 'Their concurrent body')
      await act(async () => { await view.result.current.saveNote({ showSuccessMessage: false }) })
      expect(view.result.current.saveIndicator).toBe('conflict')
      mocks.refetch.mockClear()
      for (const message of Object.values(mocks.messages)) message.mockClear()
      return view
    }
    const changeContext = async (view: EditorView, change: typeof changes[number]) => {
      if (change.startsWith('selection')) {
        await openNote(view, 'two')
        if (change === 'selection-return') await openNote(view)
        edit(view, 'Replacement selected draft')
      } else if (change.startsWith('account') || change.startsWith('server')) {
        const changeOwner = (away: boolean) => {
          act(() => {
            if (change.startsWith('account')) mocks.userId = away ? 8 : 7
            else mocks.serverUrl = away ? 'https://other-notes.test' : SERVER_URL
            view.rerender()
            window.dispatchEvent(change.startsWith('account')
              ? new Event('tldw:auth-principal-changed')
              : new CustomEvent('tldw:config-updated', { detail: { authorityChanged: true } }))
          })
        }
        changeOwner(true)
        if (change.endsWith('-return')) changeOwner(false)
        await openNote(view)
        edit(view, 'Replacement owner draft')
      } else if (change === 'edit') edit(view, 'Newer human edit')
      else view.unmount()
      await act(async () => {})
    }
    const gateDetail = (fail = false) => {
      const gate = deferred()
      const reads: RequestInit[] = []
      const remote = { ...server.notes.get('one')! }
      mocks.request.mockImplementation(async (request: RequestInit) => {
        if (String(request.method || 'GET').toUpperCase() === 'GET' && request.path === '/api/v1/notes/one' && reads.length === 0) {
          reads.push(request)
          await gate.promise
          if (fail) throw new Error('Original detail read failed')
          return remote
        }
        return respond(request)
      })
      return { gate, reads }
    }
    const finish = async (gate: ReturnType<typeof deferred>, pending: Promise<boolean>) => {
      let result!: boolean
      await act(async () => { gate.resolve(); result = await pending })
      return result
    }
    const expectOwned = (request: RequestInit) => {
      expect(request).toMatchObject({
        servicePromptConfig: { serverUrl: SERVER_URL, expectedUserId: 7 },
        headers: { 'X-TLDW-Expected-User-ID': '7' },
        abortSignal: expect.any(AbortSignal)
      })
    }

    it.each(changes)('cancels Keep mine after %s during its fresh-version read', async (change) => {
      const view = await prepareConflict()
      const read = gateDetail()
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.keepMyVersion() })
      await waitFor(() => expect(read.reads).toHaveLength(1))
      await changeContext(view, change)
      const replacement = state(view)
      const putsBefore = writes('PUT').length
      const succeeded = await finish(read.gate, pending)
      expect(succeeded).toBe(false)
      expect(writes('PUT')).toHaveLength(putsBefore)
      const current = state(view)
      expect(current.selectedId).toBe(replacement.selectedId)
      expect(current.title).toBe(replacement.title)
      expect(current.content).toBe(replacement.content)
      expect(current.selectedVersion).toBe(replacement.selectedVersion)
      expect(current.originalMetadata).toEqual(replacement.originalMetadata)
      expect(current.isDirty).toBe(replacement.isDirty)
      if (change === 'edit') {
        await waitFor(() => expect(view.result.current.remoteVersionInfo?.version).toBe(3))
        expect(view.result.current.saveIndicator).toBe('conflict')
        expect(view.result.current.saving).toBe(false)
      } else expect(current.saveIndicator).toBe(replacement.saveIndicator)
      expect(mocks.messages.success).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it('uses the original scoped fresh version for a legitimate Keep mine overwrite', async () => {
      const view = await prepareConflict()
      editElsewhere('one', 'Their even newer body')
      const read = gateDetail()
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.keepMyVersion() })
      await waitFor(() => expect(read.reads).toHaveLength(1))
      expect(await finish(read.gate, pending)).toBe(true)
      expectOwned(read.reads[0])
      expect(writes('PUT').at(-1)).toMatchObject({ path: '/api/v1/notes/one', headers: { 'expected-version': '4' } })
      expectOwned(writes('PUT').at(-1)!)
      expect(server.notes.get('one')).toMatchObject({ content: 'My conflict draft', version: 5 })
      expect(view.result.current.saveIndicator).toBe('saved')
      expect(view.result.current.isDirty).toBe(false)
    })

    it('keeps the local draft when clipboard copying fails and discard is declined', async () => {
      const view = await prepareConflict()
      const replacement = state(view)
      const before = mocks.request.mock.calls.length
      vi.mocked(navigator.clipboard.writeText).mockRejectedValue(new Error('Clipboard unavailable'))
      mocks.confirmDanger.mockResolvedValue(false)
      let result!: boolean
      await act(async () => { result = await view.result.current.takeTheirVersion() })
      expect(result).toBe(false)
      expect(mocks.confirmDanger).toHaveBeenCalledTimes(1)
      expect(mocks.request.mock.calls.slice(before)).toEqual([])
      expect(state(view)).toEqual(replacement)
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
    })

    it.each(['clipboard', 'dialog'].flatMap((boundary) => changes.map((change) => [boundary, change] as const)))('cancels Use theirs after %s waits and %s replaces its owner/draft', async (boundary, change) => {
      const view = await prepareConflict()
      const gate = deferred()
      if (boundary === 'clipboard') vi.mocked(navigator.clipboard.writeText).mockImplementation(async () => { await gate.promise })
      else {
        vi.mocked(navigator.clipboard.writeText).mockRejectedValue(new Error('Clipboard unavailable'))
        mocks.confirmDanger.mockImplementation(async () => { await gate.promise; return true })
      }
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.takeTheirVersion() })
      await waitFor(() => expect(boundary === 'clipboard' ? navigator.clipboard.writeText : mocks.confirmDanger).toHaveBeenCalledTimes(1))
      await changeContext(view, change)
      const replacement = state(view)
      const before = mocks.request.mock.calls.length
      const succeeded = await finish(gate, pending)
      expect(succeeded).toBe(false)
      expect(mocks.request.mock.calls.slice(before).filter(([request]) => request.path === '/api/v1/notes/one')).toEqual([])
      expect(state(view)).toEqual(replacement)
      expect(mocks.refetch).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.messages.error).not.toHaveBeenCalled()
    })

    it.each(['success', 'failure'].flatMap((outcome) => changes.map((change) => [outcome, change] as const)))('discards Use theirs detail %s after %s without a stale toast', async (outcome, change) => {
      const view = await prepareConflict()
      const read = gateDetail(outcome === 'failure')
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.takeTheirVersion() })
      await waitFor(() => expect(read.reads).toHaveLength(1))
      await changeContext(view, change)
      const replacement = state(view)
      const succeeded = await finish(read.gate, pending)
      expect(succeeded).toBe(false)
      expect(state(view)).toEqual(replacement)
      expect(mocks.messages.error).not.toHaveBeenCalled()
      expect(mocks.messages.info).not.toHaveBeenCalled()
      expect(mocks.refetch).not.toHaveBeenCalled()
    })

    it('loads their complete scoped version and reports success after its own selection increment', async () => {
      const view = await prepareConflict()
      const read = gateDetail()
      let pending!: Promise<boolean>
      act(() => { pending = view.result.current.takeTheirVersion() })
      await waitFor(() => expect(read.reads).toHaveLength(1))
      expect(await finish(read.gate, pending)).toBe(true)
      expectOwned(read.reads[0])
      expect(state(view)).toMatchObject({ selectedId: 'one', content: 'Their concurrent body', selectedVersion: 3, isDirty: false, saveIndicator: 'saved' })
      expect(mocks.refetch).toHaveBeenCalledTimes(1)
      expect(mocks.messages.info).toHaveBeenCalledWith(expect.objectContaining({ content: expect.stringContaining('copied to the clipboard') }))
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining('My conflict draft'))
    })

    describe('Notes conflict choice queue and receipt ownership', () => {
      it.each(['clipboard', 'dialog'] as const)('preserves a confirmed newer offline draft while Use theirs waits for %s', async (boundary) => {
        const view = await prepareConflict()
        const gate = deferred()
        if (boundary === 'clipboard') vi.mocked(navigator.clipboard.writeText).mockImplementation(async () => { await gate.promise })
        else {
          vi.mocked(navigator.clipboard.writeText).mockRejectedValue(new Error('Clipboard unavailable'))
          mocks.confirmDanger.mockImplementation(async () => { await gate.promise; return true })
        }
        let pending!: Promise<boolean>
        act(() => { pending = view.result.current.takeTheirVersion() })
        await waitFor(() => expect(boundary === 'clipboard' ? navigator.clipboard.writeText : mocks.confirmDanger).toHaveBeenCalledTimes(1))
        Object.defineProperty(navigator, 'onLine', { configurable: true, value: false })
        try {
          edit(view, 'Newer queued body')
          let queued = false
          await act(async () => { queued = await view.result.current.saveNote({ showSuccessMessage: false, trigger: 'keep-mine' }) })
          expect(queued).toBe(true)
          const confirmed = view.result.current.offlineDraftQueue['note:one']
          expect(confirmed).toMatchObject({ content: 'Newer queued body', syncState: 'queued' })
          expect(readQueue()['note:one']).toEqual(confirmed)
          const replacement = state(view)
          const result = await finish(gate, pending)
          expect(Object.hasOwn(view.result.current.offlineDraftQueue, 'note:one')).toBe(true)
          expect(view.result.current.offlineDraftQueue['note:one']).toEqual(confirmed)
          expect(readQueue()['note:one']).toEqual(confirmed)
          expect(result).toBe(false)
          expect(state(view)).toEqual(replacement)
          expect(mocks.refetch).not.toHaveBeenCalled()
          expect(mocks.messages.info).not.toHaveBeenCalled()
        } finally {
          view.unmount()
          Reflect.deleteProperty(navigator, 'onLine')
        }
      })

      it('replays the uncertain Keep mine receipt without changing its key, body, or version base', async () => {
        const view = await prepareConflict()
        const receipts = new Map<string, { note: StoredNote; body: string; version: string | undefined }>()
        let loseAcknowledgment = true
        mocks.request.mockImplementation(async (request: RequestInit) => {
          if (request.method !== 'PUT') return respond(request)
          const key = request.headers?.['Idempotency-Key'] || ''
          const receipt = receipts.get(key)
          if (receipt) {
            if (receipt.body !== JSON.stringify(request.body) || receipt.version !== request.headers?.['expected-version'])
              throw Object.assign(new Error('Idempotency receipt payload mismatch'), { status: 409 })
            return { ...receipt.note }
          }
          const acknowledged = await respond(request) as StoredNote
          receipts.set(key, { note: acknowledged, body: JSON.stringify(request.body), version: request.headers?.['expected-version'] })
          if (loseAcknowledgment) {
            loseAcknowledgment = false
            throw new TypeError('Response lost after the server committed Keep mine')
          }
          return acknowledged
        })
        let first = true
        await act(async () => { first = await view.result.current.keepMyVersion() })
        expect(first).toBe(false)
        const original = writes('PUT').at(-1)!
        expect(original.headers?.['Idempotency-Key']).toBeTruthy()
        expect(original.headers?.['expected-version']).toBe('3')
        expect(server.notes.get('one')).toMatchObject({ content: 'My conflict draft', version: 4 })
        edit(view, 'Newer text while the receipt is uncertain')
        let replayed = false
        await act(async () => { replayed = await view.result.current.keepMyVersion() })
        expect(replayed).toBe(true)
        const replay = writes('PUT').at(-1)!
        expect(replay.headers?.['Idempotency-Key']).toBe(original.headers?.['Idempotency-Key'])
        expect(replay.headers?.['expected-version']).toBe(original.headers?.['expected-version'])
        expect(replay.body).toEqual(original.body)
        expect(server.notes.get('one')).toMatchObject({ content: 'My conflict draft', version: 4 })
        expect(state(view)).toMatchObject({ content: 'Newer text while the receipt is uncertain', selectedVersion: 4, isDirty: true, saveIndicator: 'dirty' })
      })

      it.each(['edit', 'account-return', 'unmount'] as const)('rejects a captured Use theirs auth result after %s', async (change) => {
        const view = await prepareConflict()
        let releaseUser!: () => void
        const user = new Promise<{ id: number; is_active: boolean }>((resolve) => {
          releaseUser = () => resolve({ id: 7, is_active: true })
        })
        mocks.currentUser.mockClear()
        mocks.currentUser.mockImplementationOnce(() => user)
        let pending!: Promise<boolean>
        await act(async () => {
          pending = view.result.current.takeTheirVersion()
          await Promise.resolve()
          await Promise.resolve()
        })
        // Do not wait for an auth call the unchanged implementation never makes.
        const authCallsBeforeChange = mocks.currentUser.mock.calls.length
        await changeContext(view, change)
        const replacement = state(view)
        let result!: boolean
        await act(async () => { releaseUser(); result = await pending })
        expect(authCallsBeforeChange).toBe(1)
        expect(result).toBe(false)
        expect(state(view)).toEqual(replacement)
        expect(mocks.refetch).not.toHaveBeenCalled()
        expect(mocks.messages.info).not.toHaveBeenCalled()
        expect(mocks.messages.error).not.toHaveBeenCalled()
      })
    })

  })

})
