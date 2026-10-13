import React from 'react'
import { act, cleanup, fireEvent, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The editor hook drives one save state machine (#3102: NS-01, NS-03, NS-N2,
// NS-05, NS-02, NS-06). These tests run it against a small versioned server.

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
type RequestInit = {
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
      return useNotesEditorState({ ...deps, t, editorKeywords: keywords, setEditorKeywords: setKeywords })
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
})
