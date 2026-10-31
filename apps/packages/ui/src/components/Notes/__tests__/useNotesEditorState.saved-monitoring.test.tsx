import React from 'react'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  discovery: vi.fn(),
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
  listNoteTasks: vi.fn(async () => ({ tasks: [], reconciliation: null })),
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
const renderEditor = (isOnline = true) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  const deps: UseNotesEditorStateDeps = {
    authorityScope: authority(),
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
    refetch: vi.fn(async () => undefined),
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
  afterEach(cleanup)

  it('announces saved after a versioned server load, retaining its saved timestamp', async () => {
    const view = renderEditor()
    await act(async () => {
      await view.result.current.loadDetail('one')
    })
    expect(view.result.current.saveIndicator).toBe('saved')
    expect(view.result.current.saveIndicatorText).toBe('All changes saved')
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
    act(() => view.result.current.setContentDirty('Typed during load'))
    await act(async () => {
      pending.resolve(note())
      await loading
    })
    expect(view.result.current.content).toBe('Typed during load')
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
    expect(view.result.current.saveIndicator).toBe('error')
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
        expect(mocks.messages.success).not.toHaveBeenCalled()
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
})
