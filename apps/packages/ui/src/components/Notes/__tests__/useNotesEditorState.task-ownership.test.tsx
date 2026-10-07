import React from 'react'
import { message } from 'antd'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BgRequestInit } from '@/services/background-proxy'
import type { NoteTask } from '@/services/notes-tasks'
import TaskChecklistPreview from '../TaskChecklistPreview'
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState'
import { createNotesGraphAuthorityScope } from '../hooks/useNotesGraphAuthorityScope'

const mocks = vi.hoisted(() => ({
  userId: 1,
  serverUrl: 'https://notes.example.test',
  request: vi.fn<(request: BgRequestInit) => Promise<unknown>>(),
  currentUser: vi.fn<() => Promise<{ id: number; is_active: boolean }>>(),
  error: vi.fn(),
  refetch: vi.fn(async () => undefined)
}))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
vi.mock('@/services/tldw/TldwAuth', () => ({ tldwAuth: { getCurrentUser: mocks.currentUser } }))
vi.mock('@/hooks/useCallerCapabilities', () => ({
  useCallerCapabilities: () => ({ monitoringAlerts: 'denied', userId: mocks.userId, refreshAfterForbidden: vi.fn(async () => undefined) })
}))
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/services/settings/registry')>(),
  getSetting: vi.fn(async () => null), setSetting: vi.fn(async () => undefined), clearSetting: vi.fn()
}))

const pending = new Set<() => void>()
const deferred = <T,>(fallback: T) => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  const drain = () => resolve(fallback)
  pending.add(drain)
  return { promise, resolve: (value: T) => { pending.delete(drain); resolve(value) } }
}
const actions = new Set<Promise<unknown>>()
const track = (action: Promise<unknown>) => {
  actions.add(action)
  void action.finally(() => actions.delete(action))
}
type Note = { id: string; title: string; content: string; version: number; metadata: { keywords: string[] }; last_modified: string }
const notes = new Map<string, Note>()
let taskVersion = 5
const taskFor = (note: Note): NoteTask => ({
  id: `task-${note.id}`, note_id: note.id, text: note.id === 'one' ? 'Alpha' : 'Beta',
  status: note.content.startsWith('- [x]') ? 'done' : 'open', metadata: {}, projection_status: 'live', version: taskVersion,
  projection: { note_id: note.id, note_version: note.version, line_number: 1, start_offset: 0,
    end_offset: note.content.length, raw_line: note.content, has_child_content: false, projection_status: 'live' }
})
const respond = async (request: BgRequestInit): Promise<unknown> => {
  const path = String(request.path)
  if (path === '/api/v1/notes/tasks/status') {
    const updates: unknown = request.body?.updates
    if (!Array.isArray(updates) || updates.length !== 1) throw new Error('one task update required')
    const update: unknown = updates[0]
    if (!update || typeof update !== 'object' || !('task_id' in update) || !('status' in update) ||
        !('expected_task_version' in update) || !('expected_note_version' in update)) throw new Error('task CAS required')
    const note = notes.get(String(update.task_id).replace('task-', ''))
    if (!note || update.expected_task_version !== taskVersion || update.expected_note_version !== note.version) {
      throw { status: 409, message: 'Task version conflict' }
    }
    note.content = note.content.replace(/^- \[[ x]\]/, update.status === 'done' ? '- [x]' : '- [ ]')
    note.version += 1
    taskVersion += 1
    return { tasks: [taskFor(note)] }
  }
  if (path.startsWith('/api/v1/notes/tasks/activity')) return { events: [] }
  const id = path.split('/')[4]
  const note = notes.get(id)
  if (!note) throw { status: 404, message: 'Note unavailable' }
  if (path.includes('/tasks')) return { tasks: [taskFor(note)], reconciliation: { status: 'clean', note_id: id, note_version: note.version } }
  return { ...note, metadata: { ...note.metadata } }
}
let editor: ReturnType<typeof useNotesEditorState> | undefined
const current = () => {
  if (!editor) throw new Error('Editor not mounted')
  return editor
}
const Harness = () => {
  const [api, holder] = message.useMessage()
  const [keywords, setKeywords] = React.useState<string[]>([])
  const ownedMessage = React.useMemo(() => ({ ...api, error: (...args: Parameters<typeof api.error>) => {
    mocks.error(...args)
    return api.error(...args)
  } }), [api])
  const deps: UseNotesEditorStateDeps = {
    authorityScope: createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
    connectionConfig: { serverUrl: mocks.serverUrl, authMode: 'multi-user', accessToken: 'fixture-token' },
    isOnline: true, isMobileViewport: false, message: ownedMessage,
    confirmDanger: async () => true, queryClient: client,
    t: (key, options) => typeof options?.defaultValue === 'string' ? options.defaultValue : key,
    listMode: 'active', setListMode: vi.fn(), data: [], refetch: mocks.refetch,
    setPage: vi.fn(), setQuery: vi.fn(), setQueryInput: vi.fn(), setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(), setMobileSidebarOpen: vi.fn(), editorKeywords: keywords,
    setEditorKeywords: setKeywords, keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(), setKeywordSuggestionSelection: vi.fn(), editorDisabled: false
  }
  const value = useNotesEditorState(deps)
  React.useLayoutEffect(() => { editor = value }, [value])
  return <>{holder}
    <button onClick={() => track(current().handleSelectNote('one'))}>Open Alpha</button>
    <button onClick={() => track(current().handleSelectNote('other'))}>Open Beta</button>
    <textarea aria-label="Note text" value={value.content} onChange={(event) => current().setContentDirty(event.target.value)} />
    <TaskChecklistPreview content={value.content} tasks={value.noteTasks} isDirty={value.isDirty}
      onToggleLocal={value.toggleTaskCheckboxLocal} onToggleTaskStatus={(payload) => track(current().toggleTaskCheckboxStatus(payload))} />
  </>
}
let client: QueryClient
const mount = () => render(<QueryClientProvider client={client}><Harness /></QueryClientProvider>)
const requests = () => mocks.request.mock.calls.map(([request]) => request)
const posts = () => requests().filter((request) => String(request.path) === '/api/v1/notes/tasks/status')
const openAlpha = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole('button', { name: 'Open Alpha' }))
  await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Task: Alpha' })).toBeEnabled())
  expect(current().selectedVersion).toBe(2)
}
const changeOwner = (view: ReturnType<typeof mount>, kind: 'account' | 'server') => {
  const original = kind === 'account' ? mocks.userId : mocks.serverUrl
  if (kind === 'account') mocks.userId = 2
  else mocks.serverUrl = 'https://other.example.test'
  view.rerender(<QueryClientProvider client={client}><Harness /></QueryClientProvider>)
  act(() => window.dispatchEvent(new Event(kind === 'account' ? 'tldw:auth-principal-changed' : 'tldw:config-updated')))
  if (kind === 'account') mocks.userId = Number(original)
  else mocks.serverUrl = String(original)
  view.rerender(<QueryClientProvider client={client}><Harness /></QueryClientProvider>)
}

describe('Notes checkbox original ownership', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.userId = 1
    mocks.serverUrl = 'https://notes.example.test'
    mocks.currentUser.mockImplementation(async () => ({ id: mocks.userId, is_active: true }))
    mocks.request.mockImplementation(respond)
    taskVersion = 5
    notes.clear()
    for (const [id, title, text] of [['one', 'Alpha note', 'Alpha'], ['other', 'Beta note', 'Beta']]) {
      notes.set(id, { id, title, content: `- [ ] ${text}\n`, version: 2, metadata: { keywords: [text] }, last_modified: '2026-10-07T10:00:00Z' })
    }
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  })
  afterEach(async () => {
    for (const drain of pending) drain()
    pending.clear()
    await act(async () => { await Promise.allSettled([...actions]) })
    cleanup()
    client.clear()
    editor = undefined
  })

  it('uses the original principal, signal and both CAS versions through the accepted reload', async () => {
    const user = userEvent.setup()
    mount()
    await openAlpha(user)
    await act(async () => { await Promise.allSettled([...actions]) })
    mocks.request.mockClear()
    await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
    await act(async () => { await Promise.allSettled([...actions]) })
    await waitFor(() => expect(current().selectedVersion).toBe(3))
    const write = posts()[0]
    expect(write).toMatchObject({ servicePromptConfig: { serverUrl: mocks.serverUrl, expectedUserId: 1 },
      headers: { 'X-TLDW-Expected-User-ID': '1' }, body: { updates: [{ task_id: 'task-one', status: 'done', expected_task_version: 5, expected_note_version: 2 }] } })
    expect(write.abortSignal).toBeInstanceOf(AbortSignal)
    const inventory = requests().map((r, index) => ({
      index, path: String(r.path), method: r.method,
      hasConfig: r.servicePromptConfig !== undefined, hasSignal: r.abortSignal !== undefined,
      sameSignal: r.abortSignal === write.abortSignal
    }))
    for (const [index, request] of requests().entries()) {
      const diagnostic = JSON.stringify({ inventory, index, path: String(request.path), method: request.method })
      expect(request.servicePromptConfig, diagnostic).toEqual(write.servicePromptConfig)
      expect(request.abortSignal, diagnostic).toBe(write.abortSignal)
      expect(request.headers?.['X-TLDW-Expected-User-ID'], diagnostic).toBe('1')
    }
    expect(current().content).toBe('- [x] Alpha\n')
    expect(current().isDirty).toBe(false)
  })

  it('cancels a fresh principal lookup after selection away and back before any status write', async () => {
    const user = userEvent.setup()
    mount()
    await openAlpha(user)
    const gate = deferred({ id: 1, is_active: true })
    mocks.currentUser.mockClear()
    mocks.currentUser.mockImplementation(() => gate.promise)
    mocks.request.mockClear()
    await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
    // Missing lookup is a causal assertion, rather than an unreachable setup wait.
    expect(mocks.currentUser).toHaveBeenCalledTimes(1)
    await user.click(screen.getByRole('button', { name: 'Open Beta' }))
    await waitFor(() => expect(current().selectedId).toBe('other'))
    await user.click(screen.getByRole('button', { name: 'Open Alpha' }))
    await waitFor(() => expect(current().selectedId).toBe('one'))
    await act(async () => gate.resolve({ id: 1, is_active: true }))
    await act(async () => { await Promise.allSettled([...actions]) })
    expect(posts()).toHaveLength(0)
    expect(current().selectedVersion).toBe(2)
  })

  it.each(['account', 'server', 'selection', 'edit', 'unmount'] as const)(
    'does not reload or publish an already applied status ACK after %s invalidation', async (change) => {
      const user = userEvent.setup()
      const view = mount()
      await openAlpha(user)
      const gate = deferred(undefined)
      mocks.request.mockImplementation(async (request) => {
        const response = await respond(request)
        if (String(request.path) === '/api/v1/notes/tasks/status') await gate.promise
        return response
      })
      await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
      await waitFor(() => expect(notes.get('one')?.version).toBe(3))
      if (change === 'account' || change === 'server') changeOwner(view, change)
      if (change === 'selection') {
        await user.click(screen.getByRole('button', { name: 'Open Beta' }))
        await waitFor(() => expect(current().selectedId).toBe('other'))
        await user.click(screen.getByRole('button', { name: 'Open Alpha' }))
        await waitFor(() => expect(current().selectedId).toBe('one'))
      }
      if (change === 'edit') await user.type(screen.getByRole('textbox', { name: 'Note text' }), 'New local text')
      const expected = { id: current().selectedId, content: current().content, version: current().selectedVersion, dirty: current().isDirty }
      if (change === 'unmount') view.unmount()
      mocks.request.mockClear()
      mocks.error.mockClear()
      await act(async () => gate.resolve(undefined))
      await act(async () => { await Promise.allSettled([...actions]) })
      expect(requests()).toHaveLength(0)
      expect(mocks.error).not.toHaveBeenCalled()
      if (change !== 'unmount') expect({ id: current().selectedId, content: current().content, version: current().selectedVersion, dirty: current().isDirty }).toEqual(expected)
    }
  )

  it('preserves a newer draft and its version while the owned post-status detail is delayed', async () => {
    const user = userEvent.setup()
    mount()
    await openAlpha(user)
    const gate = deferred(undefined)
    let detailStarted = false
    mocks.request.mockImplementation(async (request) => {
      const response = await respond(request)
      if (String(request.path) === '/api/v1/notes/one') { detailStarted = true; await gate.promise }
      return response
    })
    await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
    await waitFor(() => expect(detailStarted).toBe(true))
    await user.type(screen.getByRole('textbox', { name: 'Note text' }), 'New draft')
    const text = current().content
    await act(async () => gate.resolve(undefined))
    await act(async () => { await Promise.allSettled([...actions]) })
    expect(current().content).toBe(text)
    expect(current().selectedVersion).toBe(2)
    expect(current().isDirty).toBe(true)
    expect(mocks.error).not.toHaveBeenCalled()
  })

  it.each([409, 503])('keeps a late %s failure quiet after its editor or owner changed', async (status) => {
    const user = userEvent.setup()
    const view = mount()
    await openAlpha(user)
    const gate = deferred(undefined)
    mocks.request.mockImplementation(async (request) => {
      if (String(request.path) === '/api/v1/notes/tasks/status') { await gate.promise; throw { status, message: 'Task update failed' } }
      return respond(request)
    })
    await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
    await waitFor(() => expect(posts()).toHaveLength(1))
    if (status === 409) {
      await user.click(screen.getByRole('button', { name: 'Open Beta' }))
      await waitFor(() => expect(current().selectedId).toBe('other'))
    } else changeOwner(view, 'server')
    mocks.error.mockClear()
    await act(async () => gate.resolve(undefined))
    await act(async () => { await Promise.allSettled([...actions]) })
    expect(mocks.error).not.toHaveBeenCalled()
    expect(current().taskConflictNotice).toBeNull()
  })

  it('retains the current draft and head while reporting a current-owner task conflict', async () => {
    const user = userEvent.setup()
    mount()
    await openAlpha(user)
    mocks.request.mockImplementation(async (request) => {
      if (String(request.path) === '/api/v1/notes/tasks/status') throw { status: 409, message: 'Task version conflict' }
      return respond(request)
    })
    await user.click(screen.getByRole('checkbox', { name: 'Task: Alpha' }))
    await waitFor(() => expect(current().taskConflictNotice).toBeTruthy())
    expect(current().selectedId).toBe('one')
    expect(current().content).toBe('- [ ] Alpha\n')
    expect(current().selectedVersion).toBe(2)
    expect(mocks.error).toHaveBeenCalledTimes(1)
  })

  describe('scheduled note freshness ownership', () => {
    let poll: (() => void) | undefined
    let restoreInterval: () => void
    beforeEach(() => {
      poll = undefined
      const setInterval = window.setInterval.bind(window)
      const spy = vi.spyOn(window, 'setInterval').mockImplementation((handler, timeout, ...args) => {
        if (timeout === 30_000 && typeof handler === 'function') poll = () => handler(...args)
        return setInterval(handler, timeout, ...args)
      })
      restoreInterval = () => spy.mockRestore()
    })
    afterEach(() => restoreInterval())

    it('scopes the scheduled detail read to the original owner and reports a newer server version', async () => {
      const user = userEvent.setup()
      mount()
      await openAlpha(user)
      await act(async () => { await Promise.allSettled([...actions]) })
      mocks.request.mockClear()
      notes.get('one')!.version = 3
      expect(poll).toBeDefined()
      await act(async () => { poll!() })
      await waitFor(() => expect(current().saveIssue?.kind).toBe('remote-newer'))
      const detail = requests().find((request) => String(request.path) === '/api/v1/notes/one')
      expect(detail).toMatchObject({ method: 'GET',
        servicePromptConfig: { serverUrl: 'https://notes.example.test', expectedUserId: 1 },
        headers: { 'X-TLDW-Expected-User-ID': '1' } })
      expect(detail?.abortSignal).toBeInstanceOf(AbortSignal)
      expect(current().selectedVersion).toBe(2)
      expect(current().content).toBe('- [ ] Alpha\n')
    })

    it.each(['selection', 'account', 'server', 'edit'] as const)(
      'rejects a delayed freshness reply after %s invalidates the captured editor', async (change) => {
        const user = userEvent.setup()
        const view = mount()
        await openAlpha(user)
        await act(async () => { await Promise.allSettled([...actions]) })
        const gate = deferred<unknown>(undefined)
        let detailStarted = false
        mocks.request.mockImplementation(async (request) => {
          if (String(request.path) === '/api/v1/notes/one' && !detailStarted) {
            detailStarted = true
            return gate.promise
          }
          return respond(request)
        })
        expect(poll).toBeDefined()
        await act(async () => { poll!() })
        await waitFor(() => expect(detailStarted).toBe(true))
        if (change === 'selection') {
          await user.click(screen.getByRole('button', { name: 'Open Beta' }))
          await waitFor(() => expect(current().selectedId).toBe('other'))
          await user.click(screen.getByRole('button', { name: 'Open Alpha' }))
          await waitFor(() => expect(current().selectedId).toBe('one'))
        } else if (change === 'edit') {
          await user.type(screen.getByRole('textbox', { name: 'Note text' }), 'New draft')
        } else changeOwner(view, change)
        await act(async () => { await Promise.allSettled([...actions]) })
        const expected = { id: current().selectedId, content: current().content,
          version: current().selectedVersion, dirty: current().isDirty, issue: current().saveIssue, remote: current().remoteVersionInfo }
        await act(async () => gate.resolve({ ...notes.get('one')!, version: 9 }))
        expect({ id: current().selectedId, content: current().content,
          version: current().selectedVersion, dirty: current().isDirty, issue: current().saveIssue, remote: current().remoteVersionInfo }).toEqual(expected)
      }
    )
  })
})
