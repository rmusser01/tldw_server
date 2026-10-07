import type { BgRequestInit } from "@/services/background-proxy"
import React from 'react'
import { Modal } from 'antd'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  discovery: vi.fn(),
  noteTasks: vi.fn(),
  user: vi.fn(),
  messages: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  },
  config: {} as Record<string, unknown>,
  userId: 7,
  watchers: new Set<Record<string, () => void>>()
}))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    ensureConfigForRequest: vi.fn(async () => ({ ...mocks.config })),
    requestWithCurrentConfig: async (factory: (config: unknown) => unknown) =>
      mocks.discovery(factory(mocks.config))
  }
}))
vi.mock('@/services/tldw/TldwAuth', () => ({
  tldwAuth: { getCurrentUser: mocks.user }
}))
vi.mock('@/services/notes-tasks', () => ({
  listNoteTasks: mocks.noteTasks,
  listTaskActivity: vi.fn(async () => ({ events: [] })),
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn()
}))
vi.mock('@/utils/safe-storage', () => ({
  createSafeStorage: () => ({
    watch: (watchers: Record<string, () => void>) =>
      mocks.watchers.add(watchers),
    unwatch: (watchers: Record<string, () => void>) =>
      mocks.watchers.delete(watchers)
  })
}))
vi.mock('@/store/connection', async () => {
  const { create } = await import('zustand')
  return {
    useConnectionStore: create(() => ({
      state: { isConnected: true, mode: 'normal' }
    }))
  }
})
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  getSetting: vi.fn(async () => null),
  setSetting: vi.fn(async () => undefined),
  clearSetting: vi.fn()
}))

import {
  useNotesEditorState,
  type UseNotesEditorStateDeps
} from '../hooks/useNotesEditorState'
import { createNotesGraphAuthorityScope } from '../hooks/useNotesGraphAuthorityScope'
import { useConnectionStore } from '@/store/connection'

const note = (id = 'one', overrides = {}) => ({
  id,
  title: `Note ${id}`,
  content: `Body ${id}`,
  version: 2,
  last_modified: '2026-09-15T12:00:00Z',
  ...overrides
})
const caps = (allowed = true) => ({
  user_id: mocks.userId,
  can_read_scheduled_tasks: false,
  can_read_notifications: false,
  can_read_monitoring_alerts: allowed
})
const alert = (id = 'created', owner = 7) => ({
  user_id: String(owner),
  source_id: id,
  rule_severity: 'warning',
  created_at: new Date().toISOString()
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const authority = () =>
  createNotesGraphAuthorityScope(String(mocks.config.serverUrl), mocks.userId)
const renderEditor = (isOnline = true, refetch = async () => undefined, onNoteRenamed?: UseNotesEditorStateDeps['onNoteRenamed']) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: authority(),
    onNoteRenamed,
    connectionConfig:
      mocks.config as UseNotesEditorStateDeps['connectionConfig'],
    isOnline,
    isMobileViewport: false,
    message: mocks.messages as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: vi.fn(async () => true),
    queryClient,
    t: (key, options) => options?.defaultValue ?? key,
    listMode: 'active',
    setListMode: vi.fn(),
    data: [],
    refetch: vi.fn(refetch),
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
    ({ scope, online }) => {
      const [keywords, setKeywords] = React.useState<string[]>([])
      return useNotesEditorState({
        ...deps,
        authorityScope: scope,
        connectionConfig:
          mocks.config as UseNotesEditorStateDeps['connectionConfig'],
        isOnline: online,
        editorKeywords: keywords,
        setEditorKeywords: setKeywords
      })
    },
    {
      initialProps: { scope: authority(), online: isOnline },
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      )
    }
  )
}
const editAndSave = async (view: ReturnType<typeof renderEditor>) => {
  act(() => view.result.current.setContentDirty('New note content'))
  let saved = false
  await act(async () => {
    saved = await view.result.current.saveNote()
  })
  return saved
}

describe('Notes saved-state hydration and optional monitoring', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.noteTasks.mockResolvedValue({ tasks: [], reconciliation: null })
    localStorage.clear()
    mocks.userId = 7
    mocks.config = {
      serverUrl: 'https://notes.test',
      authMode: 'multi-user',
      accessToken: `test.${btoa(JSON.stringify({ sub: '7' }))}.signature`,
      refreshToken: 'synthetic-refresh'
    }
    mocks.user.mockImplementation(async () => ({
      id: mocks.userId,
      is_active: true
    }))
    mocks.discovery.mockImplementation(async () => caps())
    mocks.request.mockImplementation(async ({ path, method = 'GET' }) => {
      if (path.startsWith('/api/v1/monitoring/alerts?'))
        return { items: [alert()] }
      if (path === '/api/v1/notes/' && method === 'POST') return note('created')
      if (/^\/api\/v1\/notes\/[^/?]+$/.test(path))
        return note(path.split('/').at(-1))
      return {}
    })
    useConnectionStore.setState(({ state }) => ({
      state: { ...state, isConnected: true }
    }))
  })
  afterEach(() => { vi.useRealTimers(); cleanup() })

  it('announces saved after a versioned server load, retaining its saved timestamp', async () => {
    const view = renderEditor()
    await act(async () => {
      await view.result.current.loadDetail('one')
    })
    expect(view.result.current.saveIndicator).toBe('saved')
    // The pill is the only status now (NS-06); no second "All changes saved" line.
    expect(view.result.current.saveIssue).toBeNull()
    expect(view.result.current.selectedLastSavedAt).toBe(
      '2026-09-15T12:00:00.000Z'
    )
    expect(view.result.current.selectedVersion).toBe(2)
  })

  it('keeps unversioned and offline loads distinct from confirmed saved state', async () => {
    mocks.request.mockResolvedValue(
      note('one', { version: undefined, last_modified: undefined })
    )
    const view = renderEditor()
    await act(async () => {
      await view.result.current.loadDetail('one')
    })
    expect(view.result.current.saveIndicator).toBe('idle')
    view.rerender({ scope: authority(), online: false })
    mocks.request.mockResolvedValue(note())
    await act(async () => {
      await view.result.current.loadDetail('one')
    })
    expect(view.result.current.saveIndicator).not.toBe('saved')
  })

  it('does not overwrite an edit made while detail hydration is pending', async () => {
    const pending = deferred<unknown>()
    mocks.request.mockReturnValue(pending.promise)
    const view = renderEditor()
    let loading!: Promise<boolean>
    act(() => {
      loading = view.result.current.loadDetail('one')
    })
    act(() => {
      view.result.current.setContentDirty('Typed during load')
      view.result.current.setTitle('Title typed during load')
    })
    await act(async () => {
      pending.resolve(note())
      await loading
    })
    expect(view.result.current.content).toBe('Typed during load')
    expect(view.result.current.title).toBe('Title typed during load')
    expect(view.result.current.saveIndicator).toBe('dirty')
  })

  it('rejects an older selected-note load in the same account', async () => {
    const old = deferred<unknown>()
    mocks.request.mockImplementation(async ({ path }) =>
      path.endsWith('/one') ? old.promise : note('two')
    )
    const view = renderEditor()
    let loading!: Promise<boolean>
    act(() => {
      loading = view.result.current.loadDetail('one')
    })
    await act(async () => {
      await view.result.current.loadDetail('two')
    })
    await act(async () => {
      old.resolve(note())
      await loading
    })
    expect(view.result.current.selectedId).toBe('two')
    expect(view.result.current.content).toBe('Body two')
  })

  it.each(['denied', 'unknown', 'unsupported'])(
    'saves successfully without optional monitoring when discovery is %s',
    async (state) => {
      if (state === 'denied')
        mocks.discovery.mockImplementation(async () => caps(false))
      else
        mocks.discovery.mockRejectedValue(
          Object.assign(new Error('Discovery unavailable'), {
            status: state === 'unknown' ? 403 : 404
          })
        )
      const view = renderEditor()
      await act(async () => {})
      expect(await editAndSave(view)).toBe(true)
      expect(
        mocks.request.mock.calls.some(([request]) =>
          request.path.startsWith('/api/v1/monitoring/alerts')
        )
      ).toBe(false)
      expect(view.result.current.saveIndicator).toBe('saved')
    }
  )

  it('retains authorized custom-role feedback with verified owner filter and guarded transport', async () => {
    const view = renderEditor()
    await act(async () => {})
    expect(await editAndSave(view)).toBe(true)
    await waitFor(() =>
      expect(view.result.current.monitoringNotice?.severity).toBe('warning')
    )
    const request = mocks.request.mock.calls.find(([r]) =>
      r.path.startsWith('/api/v1/monitoring/alerts')
    )?.[0]
    expect(
      new URL(request.path, 'https://notes.test').searchParams.get('user_id')
    ).toBe('7')
    expect(request.servicePromptConfig).toMatchObject({
      serverUrl: 'https://notes.test',
      expectedUserId: 7
    })
    expect(request.headers['X-TLDW-Expected-User-ID']).toBe('7')
    expect(request.abortSignal).toBeInstanceOf(AbortSignal)
  })

  it('keeps a successful save when protected monitoring403 refreshes discovery', async () => {
    mocks.request.mockImplementation(async ({ path, method }) => {
      if (path.startsWith('/api/v1/monitoring/alerts'))
        throw Object.assign(new Error('Permission revoked'), { status: 403 })
      return note(method === 'POST' ? 'created' : path.split('/').at(-1))
    })
    const view = renderEditor()
    await act(async () => {})
    expect(await editAndSave(view)).toBe(true)
    await waitFor(() =>
      expect(mocks.discovery.mock.calls.length).toBeGreaterThan(1)
    )
    expect(mocks.messages.error).not.toHaveBeenCalled()
    expect(view.result.current.saveIndicator).toBe('saved')
  })

  it('retains a truthful failed save and never checks monitoring after the write fails', async () => {
    mocks.request.mockRejectedValue(new Error('Write failed'))
    const view = renderEditor()
    await act(async () => {})
    expect(await editAndSave(view)).toBe(false)
    // No HTTP status: a transient failure that will retry, never "saved" (#3102).
    expect(view.result.current.saveIndicator).toBe('retrying')
    expect(view.result.current.content).toBe('New note content')
    expect(
      mocks.request.mock.calls.some(([r]) =>
        r.path.startsWith('/api/v1/monitoring/alerts')
      )
    ).toBe(false)
  })

  it('does not dispatch a save after the account changes during owner verification', async () => {
    const pending = deferred<unknown>()
    mocks.user.mockReturnValue(pending.promise)
    const view = renderEditor()
    await act(async () => {})
    act(() => view.result.current.setContentDirty('Old draft'))
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote()
    })
    await waitFor(() => expect(mocks.user).toHaveBeenCalled())
    view.rerender({ scope: 'bob', online: true })
    view.rerender({ scope: authority(), online: true })
    await act(async () => {
      pending.resolve({ id: 7, is_active: true })
      await saving
    })
    expect(mocks.request).not.toHaveBeenCalled()
    expect(view.result.current.saving).toBe(false)
  })

  it('does not display another owner’s alert even if the response ignores the user filter', async () => {
    mocks.request.mockImplementation(async ({ path, method }) => {
      if (path.startsWith('/api/v1/monitoring/alerts'))
        return { items: [alert('created', 8)] }
      return note(method === 'POST' ? 'created' : path.split('/').at(-1))
    })
    const view = renderEditor()
    await act(async () => {})
    expect(await editAndSave(view)).toBe(true)
    expect(view.result.current.monitoringNotice).toBeNull()
  })

  it('does not refresh a replacement capability generation for a late protected403', async () => {
    const old = deferred<unknown>()
    mocks.request.mockImplementation(async ({ path, method }) => {
      if (path.startsWith('/api/v1/monitoring/alerts')) return old.promise
      return note(method === 'POST' ? 'created' : path.split('/').at(-1))
    })
    const view = renderEditor()
    await act(async () => {})
    await editAndSave(view)
    act(() => window.dispatchEvent(new CustomEvent('tldw:config-updated')))
    await waitFor(() => expect(mocks.discovery).toHaveBeenCalledTimes(2))
    await act(async () => {})
    await act(async () => {
      old.reject(
        Object.assign(new Error('Old permission failure'), { status: 403 })
      )
    })
    expect(mocks.discovery).toHaveBeenCalledTimes(2)
    expect(view.result.current.monitoringNotice).toBeNull()
  })

  it('does not publish a delayed monitoring notice into a replacement Note', async () => {
    const old = deferred<unknown>()
    mocks.request.mockImplementation(async ({ path, method }) => {
      if (path.startsWith('/api/v1/monitoring/alerts')) return old.promise
      return note(method === 'POST' ? 'created' : path.split('/').at(-1))
    })
    const view = renderEditor()
    await act(async () => {})
    await editAndSave(view)
    await act(async () => {
      await view.result.current.loadDetail('two')
    })
    await act(async () => {
      old.resolve({ items: [alert()] })
    })
    expect(view.result.current.monitoringNotice).toBeNull()
  })

  it('does not publish an older save’s alert after a newer save of the same Note', async () => {
    const old = deferred<unknown>()
    let monitorReads = 0
    mocks.request.mockImplementation(async ({ path, method }) => {
      if (path.startsWith('/api/v1/monitoring/alerts'))
        return ++monitorReads === 1 ? old.promise : { items: [] }
      return note(method === 'POST' ? 'created' : path.split('/').at(-1))
    })
    const view = renderEditor()
    await act(async () => {})
    await editAndSave(view)
    act(() => view.result.current.setContentDirty('Second save'))
    await act(async () => {
      await view.result.current.saveNote()
    })
    expect(monitorReads).toBe(2)
    await act(async () => {
      old.resolve({ items: [alert()] })
    })
    expect(view.result.current.monitoringNotice).toBeNull()
  })

  it.each(['note', 'account'])(
    'releases a pending save after a %s change without clearing the replacement save',
    async (boundary) => {
      const old = deferred<unknown>()
      const replacement = deferred<unknown>()
      mocks.request.mockImplementation(async ({ path, method }) => {
        if (method === 'POST') return old.promise
        if (method === 'PUT') return replacement.promise
        return note(path.split('/').at(-1))
      })
      const view = renderEditor()
      await act(async () => {})
      act(() => view.result.current.setContentDirty('Old draft'))
      let saving!: Promise<boolean>
      act(() => {
        saving = view.result.current.saveNote()
      })
      await waitFor(() => expect(mocks.request).toHaveBeenCalled())
      if (boundary === 'account') {
        view.rerender({ scope: 'bob', online: true })
        view.rerender({ scope: authority(), online: true })
      }
      await act(async () => {
        await view.result.current.loadDetail('two')
      })
      expect(view.result.current.saving).toBe(false)
      act(() => view.result.current.setContentDirty('Replacement draft'))
      let replacementSaving!: Promise<boolean>
      act(() => {
        replacementSaving = view.result.current.saveNote()
      })
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(([request]) => request.method === 'PUT')
        ).toBe(true)
      )
      await act(async () => {
        old.resolve(note('created'))
        await saving
      })
      expect(view.result.current.saving).toBe(true)
      await act(async () => {
        replacement.resolve(note('two'))
        await replacementSaving
      })
      expect(view.result.current.saving).toBe(false)
      expect(view.result.current.selectedId).toBe('two')
      expect(view.result.current.saveIndicator).toBe('saved')
    }
  )

  it.each(['save', 'alert'])(
    'rejects delayed %s completion across A-to-B-to-A',
    async (boundary) => {
      const old = deferred<unknown>()
      mocks.request.mockImplementation(async ({ path, method }) => {
        if (boundary === 'save' && method === 'POST') return old.promise
        if (path.startsWith('/api/v1/monitoring/alerts'))
          return boundary === 'alert' ? old.promise : { items: [alert()] }
        return note(method === 'POST' ? 'created' : path.split('/').at(-1))
      })
      const view = renderEditor()
      await act(async () => {})
      act(() => view.result.current.setContentDirty('Old account draft'))
      let saved!: Promise<boolean>
      act(() => {
        saved = view.result.current.saveNote()
      })
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(([request]) =>
            boundary === 'save'
              ? request.method === 'POST'
              : request.path.startsWith('/api/v1/monitoring/alerts')
          )
        ).toBe(true)
      )
      act(() =>
        window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed'))
      )
      view.rerender({ scope: 'bob', online: true })
      view.rerender({ scope: authority(), online: true })
      await act(async () => {
        await view.result.current.loadDetail('two')
      })
      await act(async () => {
        old.resolve(
          boundary === 'save' ? note('created') : { items: [alert()] }
        )
        await saved
      })
      expect(view.result.current.selectedId).toBe('two')
      expect(view.result.current.monitoringNotice).toBeNull()
      if (boundary === 'save')
        // An independently acknowledged owner-scoped queue may report its own sync.
        expect(mocks.messages.success).not.toHaveBeenCalledWith('Note created')
    }
  )

  it('preserves a committed create across a canonical benign config event and updates its acknowledged ID', async () => {
    const pending = deferred<unknown>()
    mocks.request.mockImplementation(async ({ path, method }) =>
      method === 'POST' ? pending.promise : note(path.split('/').at(-1))
    )
    const view = renderEditor()
    await act(async () => {})
    act(() => view.result.current.setContentDirty('Own saved content'))
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote()
    })
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(([request]) => request.method === 'POST')
      ).toBe(true)
    )
    act(() =>
      window.dispatchEvent(
        new CustomEvent('tldw:config-updated', {
          detail: { authorityChanged: false }
        })
      )
    )
    await act(async () => {
      pending.resolve(note('created'))
      expect(await saving).toBe(true)
    })
    expect(view.result.current.selectedId).toBe('created')
    expect(view.result.current.saveIndicator).toBe('saved')
    act(() => view.result.current.setContentDirty('Update the same Note'))
    await act(async () => {
      expect(await view.result.current.saveNote()).toBe(true)
    })
    expect(
      mocks.request.mock.calls.filter(([request]) => request.method === 'POST')
    ).toHaveLength(1)
    expect(
      mocks.request.mock.calls.find(
        ([request]) => request.method === 'PUT'
      )?.[0].path
    ).toBe('/api/v1/notes/created')
  })

  it.each(['target', 'session'])(
    'still cancels on a real %s change accompanying a benign-labelled event',
    async (change) => {
      const pending = deferred<unknown>()
      mocks.request.mockImplementation(async ({ path, method }) =>
        method === 'POST' ? pending.promise : note(path.split('/').at(-1))
      )
      const view = renderEditor()
      await act(async () => {})
      act(() => view.result.current.setContentDirty('Old account content'))
      let saving!: Promise<boolean>
      act(() => {
        saving = view.result.current.saveNote()
      })
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(
            ([request]) => request.method === 'POST'
          )
        ).toBe(true)
      )
      if (change === 'target') {
        const previousScope = authority()
        mocks.config = { ...mocks.config, serverUrl: 'https://other.test' }
        view.rerender({ scope: previousScope, online: true })
      }
      act(() =>
        window.dispatchEvent(
          new CustomEvent('tldw:config-updated', {
            detail: {
              authorityChanged: false,
              refreshSessionInvalidated: change === 'session'
            }
          })
        )
      )
      await act(async () => {
        pending.resolve(note('created'))
        expect(await saving).toBe(false)
      })
      expect(view.result.current.selectedId).toBeNull()
    }
  )

  it.each(['create', 'update'])(
    'retains later body, title and metadata edits when an earlier %s snapshot is acknowledged',
    async (operation) => {
      const pending = deferred<unknown>()
      mocks.request.mockImplementation(async ({ path, method }) =>
        method === (operation === 'create' ? 'POST' : 'PUT')
          ? pending.promise
          : note(path.split('/').at(-1), { content: 'Original saved content' })
      )
      const view = renderEditor()
      await act(async () => {})
      if (operation === 'update')
        await act(async () => {
          await view.result.current.loadDetail('created')
        })
      act(() => view.result.current.setContentDirty('Original saved content'))
      let saving!: Promise<boolean>
      act(() => {
        saving = view.result.current.saveNote()
      })
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(
            ([request]) =>
              request.method === (operation === 'create' ? 'POST' : 'PUT')
          )
        ).toBe(true)
      )
      act(() => {
        view.result.current.setContentDirty('Later unsaved own draft')
        view.result.current.setTitle('Later title')
        view.result.current.setBacklinkMessageId('later-metadata')
      })
      await act(async () => {
        pending.resolve(
          note('created', { version: 3, content: 'Original saved content' })
        )
        await saving
      })
      expect(view.result.current.content).toBe('Later unsaved own draft')
      expect(view.result.current.title).toBe('Later title')
      expect(view.result.current.backlinkMessageId).toBe('later-metadata')
      expect(view.result.current.isDirty).toBe(true)
      expect(view.result.current.saveIndicator).toBe('dirty')
      expect(view.result.current.selectedId).toBe('created')
      expect(view.result.current.selectedVersion).toBe(3)
      mocks.request.mockImplementation(async ({ path }) =>
        note(path.split('/').at(-1), { version: 4 })
      )
      await act(async () => {
        expect(await view.result.current.saveNote()).toBe(true)
      })
      const lastWrite = mocks.request.mock.calls
        .filter(([request]) => request.method === 'PUT')
        .at(-1)?.[0]
      expect(lastWrite.path).toBe('/api/v1/notes/created')
      expect(lastWrite.headers['expected-version']).toBe('3')
      expect(lastWrite.body).toMatchObject({
        title: 'Later title',
        content: 'Later unsaved own draft',
        metadata: { message_id: 'later-metadata' }
      })
      expect(
        mocks.request.mock.calls.filter(
          ([request]) => request.method === 'POST'
        )
      ).toHaveLength(operation === 'create' ? 1 : 0)
    }
  )

  it('preserves edits made during owner verification before the submitted create is dispatched', async () => {
    const owner = deferred<unknown>()
    mocks.user.mockReturnValue(owner.promise)
    const view = renderEditor()
    await act(async () => {})
    act(() =>
      view.result.current.setContentDirty('Submitted before verification')
    )
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote()
    })
    act(() => view.result.current.setContentDirty('Typed during verification'))
    await act(async () => {
      owner.resolve({ id: 7, is_active: true })
      await saving
    })
    expect(
      mocks.request.mock.calls.find(
        ([request]) => request.method === 'POST'
      )?.[0].body.content
    ).toBe('Submitted before verification')
    expect(view.result.current.content).toBe('Typed during verification')
    expect(view.result.current.selectedId).toBe('created')
    expect(view.result.current.saveIndicator).toBe('dirty')
  })

  it('retains a later draft queued offline under the acknowledged create ID and version', async () => {
    const pending = deferred<unknown>()
    mocks.request.mockImplementation(async ({ path, method }) =>
      method === 'POST' ? pending.promise : note(path.split('/').at(-1))
    )
    const view = renderEditor()
    await act(async () => {})
    act(() => view.result.current.setContentDirty('Original online draft'))
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote()
    })
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(([request]) => request.method === 'POST')
      ).toBe(true)
    )
    view.rerender({ scope: authority(), online: false })
    act(() => view.result.current.setContentDirty('Later offline draft'))
    await waitFor(() =>
      expect(view.result.current.queuedOfflineDraftCount).toBe(1)
    )
    await act(async () => {
      pending.resolve(note('created', { version: 3 }))
      await saving
    })
    expect(Object.keys(view.result.current.offlineDraftQueue)).toEqual([
      'note:created'
    ])
    expect(view.result.current.offlineDraftQueue['note:created']).toMatchObject(
      {
        noteId: 'created',
        baseVersion: 3,
        content: 'Later offline draft'
      }
    )
    expect(view.result.current.selectedId).toBe('created')
    expect(view.result.current.saveIndicator).toBe('dirty')
  })

  it('retains the acknowledged ID when a later edit prevents delayed post-create hydration', async () => {
    const hydration = deferred<unknown>()
    mocks.request.mockImplementation(async ({ method }) =>
      method === 'POST' ? note('created', { version: 3 }) : hydration.promise
    )
    const view = renderEditor()
    await act(async () => {})
    act(() => view.result.current.setContentDirty('Submitted draft'))
    let saving!: Promise<boolean>
    act(() => {
      saving = view.result.current.saveNote()
    })
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(
          ([request]) =>
            request.path === '/api/v1/notes/created' && request.method === 'GET'
        )
      ).toBe(true)
    )
    act(() =>
      view.result.current.setContentDirty('Typed during detail refresh')
    )
    await act(async () => {
      hydration.resolve(note('created', { version: 3 }))
      await saving
    })
    expect(view.result.current.content).toBe('Typed during detail refresh')
    expect(view.result.current.selectedId).toBe('created')
    expect(view.result.current.selectedVersion).toBe(3)
    expect(view.result.current.saveIndicator).toBe('dirty')
  })
  it.each([true, false])(
    'retains canonical content provenance without API metadata through body edits (online=%s)',
    async (online) => {
      const provenance = {
        origin: 'knowledge_qa',
        trust_state: 'uncited_degraded_answer',
        evidence_origin: 'local_library',
        thread_id: 'owned-thread',
      }
      const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(provenance))} -->`
      let saved = {
        ...note('canonical-uuid', {
          content: `Original cited excerpt\n\n${marker}`,
        }),
        conversation_id: 'owned-thread',
      }
      mocks.request.mockImplementation(async (request) => {
        if (request.method === 'PUT')
          saved = {
            ...saved,
            title: request.body.title,
            content: request.body.content,
            version: saved.version + 1,
          }
        return saved // Canonical NoteResponse has no metadata field.
      })
      const view = renderEditor()
      await act(async () => {
        await view.result.current.loadDetail('canonical-uuid')
      })
      expect(view.result.current.provenanceSummaryText).toContain(
        'Knowledge QA',
      )
      expect(view.result.current.content).toBe('Original cited excerpt')
      expect(view.result.current.metricSummaryText).toContain('3 words')
      view.rerender({ scope: authority(), online })
      act(() => {
        view.result.current.setTitle('Edited title')
        // Rich-text conversion and body replacement can omit HTML comments.
        view.result.current.setContentDirty(
          'Edited body retaining the source quotation',
        )
      })
      await act(async () => {
        await view.result.current.saveNote()
      })
      if (!online) {
        const draft =
          view.result.current.offlineDraftQueue['note:canonical-uuid']
        expect(draft.content).toContain(marker)
        expect(draft.content).toContain('Edited body')
        view.unmount()
        return
      }
      expect(saved).not.toHaveProperty('metadata')
      expect(saved.content).toContain(marker)
      expect(saved.title).toBe('Edited title')
      view.unmount()
      const reopened = renderEditor()
      await act(async () => {
        await reopened.result.current.loadDetail('canonical-uuid')
      })
      expect(reopened.result.current.content).toContain('Edited body')
      expect(reopened.result.current.provenanceSummaryText).toContain(
        'Knowledge QA',
      )
      expect(reopened.result.current.provenanceSummaryText).toContain(
        'uncited_degraded_answer',
      )
      expect(reopened.result.current.provenanceSummaryText).toContain(
        'owned-thread',
      )
      reopened.unmount()
    },
  )
  it('retains Knowledge QA origin and qualifications after reopening and editing the exported title', async () => {
    const metadata = {
      origin: 'knowledge_qa',
      thread_id: 'thread-export',
      trust_state: 'uncited_degraded_answer',
      evidence_origin: 'local_library',
    }
    let saved: ReturnType<typeof note> & { metadata: typeof metadata } = {
      ...note('exported-uuid', {
        content: 'Exact exported draft with source excerpt',
        metadata,
      }),
      metadata,
    }
    mocks.request.mockImplementation(async (request) => {
      if (request.method === 'PUT') saved = { ...saved, ...request.body }
      return saved
    })
    const view = renderEditor()
    await act(async () => {
      await view.result.current.loadDetail('exported-uuid')
    })
    expect(view.result.current.content).toBe(
      'Exact exported draft with source excerpt',
    )
    expect(view.result.current.provenanceSummaryText).toContain('Knowledge QA')
    act(() => view.result.current.setTitle('Edited title'))
    expect(view.result.current.provenanceSummaryText).toContain(
      'uncited_degraded_answer',
    )
    expect(view.result.current.provenanceSummaryText).toContain('thread-export')
    await act(async () => {
      await view.result.current.saveNote()
    })
    expect(saved.metadata).toMatchObject(metadata)
    expect(saved.title).toBe('Edited title')
    view.unmount()
    const reopened = renderEditor()
    await act(async () => {
      await reopened.result.current.loadDetail('exported-uuid')
    })
    expect(reopened.result.current.content).toBe(
      'Exact exported draft with source excerpt',
    )
    expect(reopened.result.current.provenanceSummaryText).toContain(
      'Knowledge QA',
    )
    expect(reopened.result.current.provenanceSummaryText).toContain(
      'thread-export',
    )
  })
  it.each(['online', 'queue'])('replays the original sourced save after a lost acknowledgment without clearing later edits (%s)', async via => {
    const history = { origin: 'knowledge_qa', question: 'Original question' }
    let stored = note('one', { content: `Body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`, knowledge_provenance_state: 'absent', knowledge_provenance_version: 0 })
    const writes: BgRequestInit[] = []
    mocks.request.mockImplementation(async request => {
      if (request.method === 'PUT') {
        writes.push(request)
        stored = { ...stored, ...request.body, version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 1, knowledge_provenance_hash: `sha256:${'a'.repeat(64)}` }
        if (writes.length === 1) throw new Error('Lost acknowledgment')
      }
      return stored
    })
    const view = renderEditor()
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => view.result.current.setContentDirty('First edit'))
    await act(async () => { expect(await view.result.current.saveNote()).toBe(false) })
    act(() => view.result.current.setContentDirty('Later unsaved edit'))
    if (via === 'queue') {
      view.rerender({ scope: authority(), online: false })
      view.rerender({ scope: authority(), online: true })
      await waitFor(() => expect(writes).toHaveLength(2))
      await waitFor(() => expect(view.result.current.saving).toBe(false))
    } else await act(async () => { expect(await view.result.current.saveNote()).toBe(true) })
    expect(writes[0].body).toMatchObject({ expected_provenance_version: 0, knowledge_provenance: history })
    expect(writes[1].body).toEqual(writes[0].body)
    expect(writes[0].headers['Idempotency-Key']).toBeTruthy()
    expect(writes[1].headers).toEqual(writes[0].headers)
    expect(view.result.current.content).toBe('Later unsaved edit')
    expect(view.result.current.isDirty).toBe(true)
    expect(view.result.current.originalMetadata).toMatchObject({ knowledge_provenance_state: 'active', knowledge_provenance_version: 1 })
    expect(view.result.current.offlineDraftQueue['note:one']?.pendingWrite).toBeUndefined()
    if (via === 'online') expect(view.result.current.offlineDraftQueue['note:one']?.metadata).toMatchObject({ knowledge_provenance_state: 'active', knowledge_provenance_version: 1 })
    act(() => view.result.current.setContentDirty('A fresh edit after acknowledgment'))
    await act(async () => { expect(await view.result.current.saveNote()).toBe(true) })
    expect(writes[2].body.content).toMatch(/^A fresh edit after acknowledgment(?:\n|$)/)
    expect(writes[2].headers['Idempotency-Key']).not.toBe(writes[0].headers['Idempotency-Key'])
    expect(writes[2].headers['expected-version']).toBe('3')
    expect(writes[2].body).not.toHaveProperty('expected_provenance_version')
    view.unmount()
  })
  it('restores retained history explicitly while preserving edits and advancing both heads', async () => {
    const gate = deferred<ReturnType<typeof note>>()
    const history = { origin: 'knowledge_qa', question: 'Original question' }
    const removed = note('one', { knowledge_provenance_state: 'deleted', knowledge_provenance_version: 4, knowledge_provenance_hash: `sha256:${'a'.repeat(64)}`, knowledge_provenance: null })
    mocks.request.mockImplementation(request => request.path.endsWith('/provenance/restore') ? gate.promise : Promise.resolve(removed))
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    let restoring!: Promise<boolean>
    act(() => { restoring = view.result.current.saveNote({ restoreProvenance: true }) })
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/notes/one/provenance/restore', method: 'POST', headers: expect.objectContaining({ 'expected-version': '2' }), body: { expected_provenance_version: 4, expected_provenance_hash: removed.knowledge_provenance_hash } })))
    act(() => view.result.current.setContentDirty('Dirty while restoring'))
    await act(async () => { gate.resolve({ ...removed, version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 5, knowledge_provenance: history }); await restoring })
    expect(view.result.current.content).toBe('Dirty while restoring')
    expect(view.result.current.selectedVersion).toBe(3)
    expect(view.result.current.originalMetadata).toMatchObject({ knowledge_provenance_version: 5, knowledge_provenance: history })
    expect(view.result.current.isDirty).toBe(true)
  })

  it.each([false, true])('hands an online sourced create to the offline queue with its original identity after remount (inflight=%s)', async inflight => {
    const history = { origin: 'knowledge_qa', question: 'Online question' }
    const original = `Original body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
    const replay = deferred<ReturnType<typeof note>>()
    const interrupted = deferred<ReturnType<typeof note>>()
    const authoredSave = deferred<ReturnType<typeof note>>()
    const updates: BgRequestInit[] = []
    const creates: BgRequestInit[] = []
    mocks.request.mockImplementation(async request => {
      if (request.method === 'POST') {
        creates.push(request)
        if (creates.length === 1) {
          if (inflight) return interrupted.promise
          throw new Error('Server committed but its response was lost')
        }
        return replay.promise
      }
      if (request.method === 'PUT') {
        updates.push(request)
        return authoredSave.promise
      }
      return note('online-created', { ...creates[0]?.body, version: 1 })
    })
    const view = renderEditor(true)
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => view.result.current.setContentDirty(original))
    let saving!: Promise<boolean>
    act(() => { saving = view.result.current.saveNote() })
    await waitFor(() => expect(creates).toHaveLength(1))
    if (!inflight) await act(async () => { expect(await saving).toBe(false) })
    act(() => view.result.current.setContentDirty('Later authored edits'))
    view.unmount()
    const reopened = renderEditor(true)
    await waitFor(() => expect(creates).toHaveLength(2))
    expect(creates[1].headers['Idempotency-Key']).toBe(creates[0].headers['Idempotency-Key'])
    expect(creates[1].body).toEqual(creates[0].body)
    expect(creates[1].body).toMatchObject({ knowledge_provenance: history, expected_provenance_version: 0 })
    expect(reopened.result.current.content).toBe('Later authored edits')
    await act(async () => { replay.resolve(note('online-created', { ...creates[0].body, version: 1, knowledge_provenance_state: 'active', knowledge_provenance_version: 1 })); await replay.promise })
    await waitFor(() => expect(reopened.result.current.selectedId).toBe('online-created'))
    expect(reopened.result.current.content).toBe('Later authored edits')
    expect(reopened.result.current.saveIndicator).not.toBe('saved')
    expect(reopened.result.current.isDirty || reopened.result.current.offlineDraftQueue['note:online-created']?.content === 'Later authored edits').toBe(true)
    if (updates.length) {
      expect(updates[0].body.content).toMatch(/^Later authored edits(?:\n|$)/)
      await act(async () => { authoredSave.resolve(note('online-created', { ...updates[0].body, version: 2 })); await authoredSave.promise })
      await waitFor(() => expect(reopened.result.current.saveIndicator).toBe('saved'))
    }
    if (inflight) await act(async () => { interrupted.resolve(note('online-created')); expect(await saving).toBe(false) })
    reopened.unmount()
  })

  it.each([
    ['create', 'manual'], ['create', 'auto'], ['create', 'timer'],
    ['restore', 'manual'], ['restore', 'auto'], ['restore', 'timer']
  ] as const)('settles the durable %s request after a remounted queue retry fails before a newer %s save', async (operation, trigger) => {
    const hash = `sha256:${'a'.repeat(64)}`
    let stored = note('one', { knowledge_provenance_state: 'deleted', knowledge_provenance_version: 4, knowledge_provenance_hash: hash })
    const writes: BgRequestInit[] = []
    const ack = deferred<ReturnType<typeof note>>()
    const nextSave = deferred<ReturnType<typeof note>>()
    mocks.request.mockImplementation(async request => {
      if (request.method === 'GET') return stored
      writes.push(request)
      if (writes.length <= 2) throw Object.assign(new Error('Server committed but its response was lost'), { status: 503 })
      if (writes.length === 3) return ack.promise
      return nextSave.promise
    })
    const view = renderEditor()
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    if (operation === 'restore') await act(async () => { await view.result.current.loadDetail('one') })
    else act(() => view.result.current.setContentDirty('First submitted body'))
    await act(async () => { expect(await view.result.current.saveNote({ restoreProvenance: operation === 'restore' })).toBe(false) })
    view.unmount()
    const reopened = renderEditor()
    await waitFor(() => expect(writes).toHaveLength(2))
    await waitFor(() => expect(Object.values(reopened.result.current.offlineDraftQueue)[0]?.syncState).toBe('error'))
    if (operation === 'restore') await act(async () => { await reopened.result.current.loadDetail('one') })
    if (trigger === 'timer') vi.useFakeTimers()
    act(() => reopened.result.current.setContentDirty('Newer authored body'))
    let retry!: Promise<boolean>
    if (trigger === 'timer') await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    else act(() => { retry = reopened.result.current.saveNote({ trigger, showSuccessMessage: false }) })
    if (trigger !== 'timer') await waitFor(() => expect(writes).toHaveLength(3))
    expect(writes).toHaveLength(3)
    expect(writes[2].path).toBe(writes[0].path)
    expect(writes[2].method).toBe(writes[0].method)
    expect(writes[2].body).toEqual(writes[0].body)
    expect(writes[2].headers['Idempotency-Key']).toBe(writes[0].headers['Idempotency-Key'])
    if (trigger === 'timer') {
      await act(async () => { ack.reject(Object.assign(new Error('Still unavailable'), { status: 503 })) })
      await act(async () => { await vi.advanceTimersByTimeAsync(4_000) })
      expect(writes).toHaveLength(3)
      expect(reopened.result.current.content).toBe('Newer authored body')
      expect(Object.values(reopened.result.current.offlineDraftQueue)[0]?.content).toMatch(/^Newer authored body(?:\n|$)/)
      expect(Object.values(reopened.result.current.offlineDraftQueue)[0]?.pendingWrite?.key).toBe(writes[0].headers['Idempotency-Key'])
      reopened.unmount()
      return
    }
    stored = note(operation === 'create' ? 'created' : 'one', {
      ...(operation === 'create' ? writes[0].body : {}), version: operation === 'create' ? 1 : 3,
      knowledge_provenance_state: 'active', knowledge_provenance_version: operation === 'create' ? 1 : 5
    })
    await act(async () => { ack.resolve(stored); expect(await retry).toBe(true) })
    expect(reopened.result.current.content).toBe('Newer authored body')
    expect(reopened.result.current.saveIndicator).not.toBe('saved')
    await waitFor(() => expect(Object.values(reopened.result.current.offlineDraftQueue).every(draft => !draft.pendingWrite)).toBe(true))
    reopened.unmount()
  })

  it('announces a rename of a still-open queued create from its acknowledged automatic title without overwriting newer typing', async () => {
    const events: unknown[] = []
    const ack = deferred<ReturnType<typeof note>>()
    let stored = note('queued-created', { title: 'Server-generated heading', content: 'First queued body', version: 1 })
    mocks.request.mockImplementation(async request => {
      if (request.method === 'POST') return ack.promise
      if (request.method === 'PUT') stored = { ...stored, ...request.body, version: 2 }
      return stored
    })
    const view = renderEditor(false, undefined, event => events.push(event))
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => view.result.current.setContentDirty('First queued body'))
    await act(async () => { await view.result.current.saveNote() })
    view.rerender({ scope: authority(), online: true })
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ method: 'POST', body: expect.objectContaining({ auto_title: true }) })))
    act(() => { view.result.current.setTitle('Newer authored title'); view.result.current.setContentDirty('Newer authored body') })
    await act(async () => { ack.resolve(stored) })
    await waitFor(() => expect(view.result.current.selectedId).toBe('queued-created'))
    expect(view.result.current.title).toBe('Newer authored title')
    expect(view.result.current.content).toBe('Newer authored body')
    await act(async () => { expect(await view.result.current.saveNote()).toBe(true) })
    expect(events).toEqual([{ noteId: 'queued-created', oldTitle: 'Server-generated heading', newTitle: 'Newer authored title', authorityScope: authority() }])
    view.unmount()
  })

  it.each(['retry', 'leave', 'navigation', 'owner'] as const)('returns the owned durable-create result after canonical readback (%s)', async boundary => {
    const original = 'Original submitted body'
    const ack = deferred<ReturnType<typeof note>>()
    const writes: BgRequestInit[] = []
    const modal = vi.spyOn(Modal, 'confirm').mockImplementation(options => {
      options.onCancel?.()
      return { destroy: vi.fn(), update: vi.fn() }
    })
    mocks.request.mockImplementation(async request => {
      if (request.method !== 'POST') return note(request.path.includes('other') ? 'other' : 'retry-clean', { title: 'Canonical retry heading', content: original, version: 1 })
      writes.push(request)
      if (writes.length <= 2) throw Object.assign(new Error('Committed response lost'), { status: 503 })
      return ack.promise
    })
    const view = renderEditor()
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => view.result.current.setContentDirty(original))
    await act(async () => { expect(await view.result.current.saveNote()).toBe(false) })
    view.unmount()
    const reopened = renderEditor()
    try {
      await waitFor(() => expect(writes).toHaveLength(2))
      await waitFor(() => expect(Object.values(reopened.result.current.offlineDraftQueue)[0]?.syncState).toBe('error'))
      if (boundary === 'leave') act(() => {
        reopened.result.current.setContentDirty('Temporary authored edit')
        reopened.result.current.setContentDirty(original)
      })
      let saving!: Promise<boolean>
      act(() => { saving = boundary === 'leave'
        ? reopened.result.current.confirmDiscardIfDirty(undefined, { trigger: 'leave' })
        : reopened.result.current.retrySave() })
      await waitFor(() => expect(writes).toHaveLength(3))
      if (boundary === 'navigation') await act(async () => { await reopened.result.current.loadDetail('other') })
      if (boundary === 'owner') {
        mocks.userId = 8
        reopened.rerender({ scope: authority(), online: true })
      }
      await act(async () => {
        ack.resolve(note('retry-clean', { ...writes[0].body, title: 'Canonical retry heading', version: 1 }))
        expect(await saving).toBe(boundary === 'retry' || boundary === 'leave')
      })
      expect(modal).not.toHaveBeenCalled()
      if (boundary === 'retry' || boundary === 'leave') {
        expect(reopened.result.current.title).toBe('Canonical retry heading')
        expect(reopened.result.current.saveIndicator).toBe('saved')
      }
      if (boundary === 'navigation') expect(reopened.result.current.selectedId).toBe('other')
      if (boundary === 'owner') expect(reopened.result.current.selectedId).not.toBe('retry-clean')
    } finally {
      modal.mockRestore()
      reopened.unmount()
    }
  })

  it('reads back an unchanged queued create with its automatic title, canonical tags, recent entry and saved checklist tasks', async () => {
    const stored = note('queued-clean', { title: 'Canonical checklist heading', content: '- [ ] Check source', version: 1, metadata: { keywords: ['captured'] } })
    const task = { id: 'task-one', note_id: 'queued-clean', text: 'Check source', status: 'open', metadata: {}, projection_status: 'live', version: 1,
      projection: { note_id: 'queued-clean', note_version: 1, line_number: 1, start_offset: 0, end_offset: 18, raw_line: '- [ ] Check source', has_child_content: false, projection_status: 'live' } }
    mocks.noteTasks.mockResolvedValue({ tasks: [task], reconciliation: { status: 'clean', note_id: 'queued-clean', note_version: 1, parsed_count: 1 } })
    mocks.request.mockResolvedValue(stored)
    const view = renderEditor(false)
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => view.result.current.setContentDirty('- [ ] Check source'))
    await act(async () => { await view.result.current.saveNote() })
    view.rerender({ scope: authority(), online: true })
    await waitFor(() => expect(view.result.current.selectedId).toBe('queued-clean'))
    await waitFor(() => expect(view.result.current.title).toBe('Canonical checklist heading'))
    expect(view.result.current.originalMetadata?.keywords).toEqual(['captured'])
    await waitFor(() => expect(view.result.current.noteTasks).toEqual([task]))
    expect(view.result.current.recentNotes).toEqual([expect.objectContaining({ id: 'queued-clean', title: 'Canonical checklist heading' })])
    expect(view.result.current.content).toBe('- [ ] Check source')
    expect(view.result.current.saveIndicator).toBe('saved')
    view.unmount()
  })

  it('chains a leave flush from the acknowledged independent history head before React renders', async () => {
    const history = { origin: 'knowledge_qa', question: 'Original question' }
    const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
    const firstAck = deferred<ReturnType<typeof note>>()
    const writes: BgRequestInit[] = []
    const initial = note('one', { content: `Body\n\n${marker}`, knowledge_provenance_state: 'absent', knowledge_provenance_version: 0 })
    mocks.request.mockImplementation(async request => {
      if (request.method !== 'PUT') return initial
      writes.push(request)
      if (writes.length === 1) return firstAck.promise
      return note('one', { ...request.body, version: 4, knowledge_provenance_state: 'active', knowledge_provenance_version: 1, knowledge_provenance: history })
    })
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => view.result.current.setContentDirty('First submitted edit'))
    let saving!: Promise<boolean>
    act(() => { saving = view.result.current.saveNote({ trigger: 'leave', showSuccessMessage: false }) })
    await waitFor(() => expect(writes).toHaveLength(1))
    act(() => view.result.current.setContentDirty('Newer edit to flush'))
    let leaving!: Promise<boolean>
    act(() => { leaving = view.result.current.confirmDiscardIfDirty(undefined, { trigger: 'leave' }) })
    await act(async () => {
      firstAck.resolve(note('one', { ...writes[0].body, version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 1, knowledge_provenance: history }))
      expect(await saving).toBe(true)
      expect(await leaving).toBe(true)
    })
    expect(writes).toHaveLength(2)
    expect(writes[1].headers['expected-version']).toBe('3')
    expect(writes[1].body.content).toMatch(/^Newer edit to flush(?:\n|$)/)
    expect(writes[1].body).not.toHaveProperty('expected_provenance_version')
    expect(writes[1].body).not.toHaveProperty('knowledge_provenance')
    view.unmount()
  })

  it('retains the queued sourced request identity across an offline retry and remount', async () => {
    const history = { origin: 'knowledge_qa', question: 'Offline question' }
    const original = `Offline body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
    let stored: ReturnType<typeof note> | null = null
    const writes: BgRequestInit[] = []
    mocks.request.mockImplementation(async request => {
      if (request.method === 'POST') {
        writes.push(request)
        stored = note('offline-created', { ...request.body, version: 1, knowledge_provenance_state: 'active', knowledge_provenance_version: 1 })
        if (writes.length === 1) throw new Error('Lost response')
      }
      return stored || {}
    })
    const view = renderEditor(false)
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => { view.result.current.setContentDirty(original) })
    await act(async () => { await view.result.current.saveNote() })
    view.rerender({ scope: authority(), online: true })
    await waitFor(() => expect(writes.length).toBeGreaterThanOrEqual(1))
    await waitFor(() => expect(view.result.current.saving).not.toBe(true))
    view.unmount()
    const reopened = renderEditor(true)
    await waitFor(() => expect(writes.length).toBeGreaterThanOrEqual(2))
    expect(writes[0].headers['Idempotency-Key']).toBeTruthy()
    expect(writes[1].headers['Idempotency-Key']).toBe(writes[0].headers['Idempotency-Key'])
    expect(writes[1].body).toEqual(writes[0].body)
    expect(writes[0].body).toMatchObject({ knowledge_provenance: history, expected_provenance_version: 0 })
    reopened.unmount()
  })

  it('refreshes the exact owned history head after enrollment conflict and deliberately retries the preserved draft', async () => {
    const history = { origin: 'knowledge_qa', question: 'Q' }
    let current = note('one', { content: `Body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`, knowledge_provenance_state: 'absent', knowledge_provenance_version: 0 })
    const writes: BgRequestInit[] = []
    mocks.request.mockImplementation(async request => {
      if (request.method === 'PUT') {
        writes.push(request)
        current = { ...current, knowledge_provenance_state: 'active', knowledge_provenance: history, knowledge_provenance_version: 1 }
        if (writes.length === 1) throw Object.assign(new Error('provenance version conflict'), { status: 409 })
        current = { ...current, ...request.body, version: 3 }
      }
      return current
    })
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Authored draft') })
    await act(async () => { expect(await view.result.current.saveNote({ showSuccessMessage: false })).toBe(false) })
    expect(view.result.current.content).toBe('Authored draft')
    expect(view.result.current.originalMetadata).toMatchObject({ knowledge_provenance_version: 1 })
    expect(view.result.current.saveIndicator).toBe('conflict')
    await act(async () => { expect(await view.result.current.keepMyVersion()).toBe(true) })
    expect(writes[0].body.expected_provenance_version).toBe(0)
    expect(writes[1].body).not.toHaveProperty('expected_provenance_version')
    expect(writes[1].headers['expected-version']).toBe('2')
    expect(writes[1].headers['Idempotency-Key']).not.toBe(writes[0].headers['Idempotency-Key'])
  })
  it.each(['retry', 'remount'])('retries the pending history restore instead of turning it into a content save (%s)', async via => {
    const removed = note('one', { knowledge_provenance_state: 'deleted', knowledge_provenance_version: 4, knowledge_provenance_hash: `sha256:${'a'.repeat(64)}` })
    let stored = removed
    const writes: BgRequestInit[] = []
    mocks.request.mockImplementation(async request => {
      if (request.method !== 'GET') {
        writes.push(request)
        if (writes.length === 1) throw Object.assign(new Error('Restore response was lost'), { status: 503 })
        stored = { ...removed, version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 5 }
        return stored
      }
      return stored
    })
    const view = renderEditor()
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    await act(async () => { await view.result.current.loadDetail('one') })
    await act(async () => { expect(await view.result.current.saveNote({ restoreProvenance: true })).toBe(false) })
    let current = view
    if (via === 'remount') {
      view.unmount()
      current = renderEditor()
      await waitFor(() => expect(writes).toHaveLength(2))
      await act(async () => { await current.result.current.loadDetail('one') })
    } else await act(async () => { expect(await view.result.current.saveNote({ trigger: 'retry' })).toBe(true) })
    expect(writes[1].path).toBe('/api/v1/notes/one/provenance/restore')
    expect(writes[1].body).toEqual(writes[0].body)
    expect(writes[1].headers['Idempotency-Key']).toBe(writes[0].headers['Idempotency-Key'])
    await waitFor(() => expect(current.result.current.saveIndicator).toBe('saved'))
    current.unmount()
  })

  it('does not install an explicit restore acknowledgment after the account changes', async () => {
    const gate = deferred<ReturnType<typeof note>>()
    const removed = note('one', { knowledge_provenance_state: 'deleted', knowledge_provenance_version: 4, knowledge_provenance_hash: `sha256:${'a'.repeat(64)}` })
    mocks.request.mockImplementation(request => request.path.endsWith('/provenance/restore') ? gate.promise : Promise.resolve(removed))
    const view = renderEditor()
    await act(async () => { await view.result.current.loadDetail('one') })
    let pending!: Promise<boolean>
    act(() => { pending = view.result.current.saveNote({ restoreProvenance: true }) })
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/notes/one/provenance/restore' })))
    mocks.userId = 8
    view.rerender({ scope: authority(), online: true })
    mocks.request.mockResolvedValue(note('two'))
    await act(async () => { await view.result.current.loadDetail('two') })
    await act(async () => { gate.resolve({ ...removed, version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 5 }); expect(await pending).toBe(false) })
    expect(view.result.current.content).toBe("Body two")
    expect(view.result.current.originalMetadata?.knowledge_provenance_state).not.toBe("active")
  })

  it('does not count acknowledged history heads as authored edits during delayed refresh', async () => {
    const gate = deferred<void>()
    const history = { origin: 'knowledge_qa', question: 'Q' }
    const saved = note('created', { knowledge_provenance_state: 'active', knowledge_provenance_version: 1, knowledge_provenance: history })
    mocks.request.mockResolvedValue(saved)
    const view = renderEditor(true, () => gate.promise)
    act(() => { view.result.current.setContentDirty(`Draft\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`) })
    let saving!: Promise<boolean>
    act(() => { saving = view.result.current.saveNote() })
    await waitFor(() => expect(view.result.current.selectedId).toBe('created'))
    await act(async () => { gate.resolve(); await saving })
    expect(view.result.current.isDirty).toBe(false)
  })

  it.each([false, true])('acknowledges the offline queue without changing a switched editor (return to A=%s)', async returnToA => {
    const gate = deferred<ReturnType<typeof note>>()
    mocks.request.mockImplementation(request => request.method === 'PUT' ? gate.promise : Promise.resolve(note(request.path.endsWith('/two') ? 'two' : 'one')))
    const view = renderEditor(false)
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Queued A') })
    await act(async () => { await view.result.current.saveNote() })
    view.rerender({ scope: authority(), online: true })
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ method: 'PUT' })))
    await act(async () => { await view.result.current.loadDetail('two') })
    if (returnToA) await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Current authored draft') })
    const selected = view.result.current.selectedId
    const metadata = view.result.current.originalMetadata
    await act(async () => { gate.resolve(note('one', { version: 3, knowledge_provenance_state: 'active', knowledge_provenance_version: 1 })) })
    await waitFor(() => expect(view.result.current.offlineDraftQueue['note:one']).toBeUndefined())
    expect(view.result.current.selectedId).toBe(selected)
    expect(view.result.current.content).toBe('Current authored draft')
    expect(view.result.current.originalMetadata).toEqual(metadata)
    expect(view.result.current.selectedVersion).toBe(2)
    expect(view.result.current.isDirty).toBe(true)
  })

  it('releases a definitively rejected request so corrected Notes input gets a new identity', async () => {
    const view = renderEditor()
    mocks.request.mockImplementation(async request => {
      if (request.method === 'PUT' && request.body.content === 'Invalid body') throw Object.assign(new Error('validation rejected'), { status: 422 })
      return note('one', request.method === 'PUT' ? { ...request.body, version: 3 } : {})
    })
    await act(async () => { await view.result.current.loadDetail('one') })
    act(() => { view.result.current.setContentDirty('Invalid body') })
    await act(async () => { expect(await view.result.current.saveNote()).toBe(false) })
    act(() => { view.result.current.setContentDirty('Corrected body') })
    await act(async () => { expect(await view.result.current.saveNote()).toBe(true) })
    const writes = mocks.request.mock.calls.map(([request]) => request).filter(request => request.method === 'PUT')
    expect(writes[1].body.content).toBe('Corrected body')
    expect(writes[1].headers['Idempotency-Key']).not.toBe(writes[0].headers['Idempotency-Key'])
  })

  it.each([
    { status: 409, detail: { error_code: 'notes_provenance_encryption_unsupported' }, message: 'Policy unavailable' },
    { status: 429, detail: 'Rate limit exceeded for notes.create', message: 'Rate limit exceeded for notes.create' },
    { status: 409, detail: { error_code: 'notes_organization_sync_not_ready' }, message: 'Notes organization Sync is not ready for writes.' },
  ].flatMap(rejection => [false, true].map(offline => ({ ...rejection, offline }))))('preserves a lost-ack create through normalized $status $message (offline=$offline)', async ({ status, detail, message, offline }) => {
    const history = { origin: 'knowledge_qa', question: 'Original question' }
    const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
    const writes: Array<{ body: Record<string, unknown>; headers: Record<string, string> }> = []
    mocks.request.mockImplementation(async request => {
      if (request.method === 'POST') {
        writes.push(request)
        if (writes.length === 1) throw new Error('Lost acknowledgment')
        if (writes.length === 2) throw Object.assign(new Error(message), { status, details: { detail } })
      }
      return note('receipt-note', { ...writes[0]?.body, version: 1 })
    })
    const view = renderEditor(!offline)
    await waitFor(() => expect(view.result.current.offlineDraftQueueHydrated).toBe(true))
    act(() => { view.result.current.setContentDirty(`Original sourced body\n\n${marker}`) })
    await act(async () => { await view.result.current.saveNote() })
    if (offline) view.rerender({ scope: authority(), online: true })
    await waitFor(() => expect(writes).toHaveLength(1))
    const retry = async () => {
      if (offline) {
        view.rerender({ scope: authority(), online: false })
        view.rerender({ scope: authority(), online: true })
      } else await act(async () => { await view.result.current.saveNote() })
    }
    await retry()
    await waitFor(() => expect(writes).toHaveLength(2))
    await retry()
    await waitFor(() => expect(writes).toHaveLength(3))
    expect(writes[2].headers['Idempotency-Key']).toBe(writes[0].headers['Idempotency-Key'])
    expect(writes[2].body).toEqual(writes[0].body)
    expect(writes[0].body.knowledge_provenance).toEqual(history)
    await waitFor(() => expect(view.result.current.selectedId).toBe('receipt-note'))
  })

})
