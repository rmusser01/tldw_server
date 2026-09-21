import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Storage } from '@plasmohq/storage'
import { useConnectionStore } from '@/store/connection'
import { bgRequest } from '@/services/background-proxy'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { useMediaSelection } from '../hooks/useMediaSelection'
import type { MediaResultItem } from '@/components/Media/types'

const mocks = vi.hoisted(() => ({ undo: vi.fn() }))
vi.mock('@/hooks/useUndoNotification', () => ({ useUndoNotification: () => ({ showUndoNotification: mocks.undo }) }))
vi.mock('@/services/background-proxy', () => ({ bgRequest: vi.fn() }))
vi.mock('@/services/tldw/TldwApiClient', () => ({ tldwClient: {} }))
vi.mock('@/services/settings/registry', async original => ({
  ...await original<typeof import('@/services/settings/registry')>(), setSetting: vi.fn(), clearSetting: vi.fn()
}))
const items = [1, 2].map(id => ({ id, kind: 'media', title: `Private ${id}`, keywords: ['private'], meta: { type: 'video' } } as MediaResultItem))
const wrapper = ({ children }: React.PropsWithChildren) => <MemoryRouter>{children}</MemoryRouter>
const mount = (ownerScope: string | null = 'account-a') => {
  const feedback = { error: vi.fn(), warning: vi.fn(), success: vi.fn() }
  const refetch = vi.fn().mockResolvedValue({ data: items })
  const view = renderHook(({ owner }) => useMediaSelection({
    ownerScope: owner, t: key => key, message: feedback, displayResults: items, selected: null,
    setSelected: vi.fn(), setSelectedContent: vi.fn(), setSelectedDetail: vi.fn(), setLastFetchedId: vi.fn(), refetch
  }), { wrapper, initialProps: { owner: ownerScope } })
  return { ...view, feedback, refetch }
}
const select = (view: ReturnType<typeof mount>) => act(() => {
  view.result.current.setBulkSelectionMode(true)
  view.result.current.setBulkSelectedIds(['1', '2'])
  view.result.current.setBulkKeywordsDraft('confidential')
})
beforeEach(() => {
  // The package config runs the actual extension hook, whose default storage
  // requires browser.storage. The WebUI config uses its localStorage adapter.
  const values: Record<string, unknown> = {}
  type Changes = Record<string, { oldValue: unknown; newValue: unknown }>
  const listeners = new Set<(changes: Changes, area: string) => void>()
  vi.stubGlobal('browser', { storage: {
    sync: {
      get: async (keys: string[]) => Object.fromEntries(keys.filter(key => key in values).map(key => [key, values[key]])),
      set: async (next: Record<string, unknown>) => {
        const changes = Object.fromEntries(Object.entries(next).map(([key, value]) => [key, { oldValue: values[key], newValue: value }]))
        Object.assign(values, next)
        listeners.forEach(listener => listener(changes, 'sync'))
      }
    },
    onChanged: {
      addListener: (listener: (changes: Changes, area: string) => void) => listeners.add(listener),
      removeListener: (listener: (changes: Changes, area: string) => void) => listeners.delete(listener)
    }
  } })
  localStorage.clear(); vi.clearAllMocks()
  Reflect.deleteProperty(tldwClient, "getCurrentUserStorageQuota")
  Reflect.deleteProperty(tldwClient, "getCurrentUserProfile")
  vi.mocked(bgRequest).mockResolvedValue({ version: 1 })
  useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: 'http://fixture.invalid' } }))
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('media selection authority', () => {
  it('loads and parses the current owner quota through the guarded transport', async () => {
    Object.assign(tldwClient, { getCurrentUserStorageQuota: vi.fn() })
    vi.mocked(bgRequest).mockResolvedValue({ storage_used_mb: '15', storage_quota_mb: 100, usage_percentage: '15', warning: 'Near quota' })
    const view = mount()
    await waitFor(() => expect(view.result.current.libraryStorageUsage.loading).toBe(false))
    expect(view.result.current.libraryStorageUsage).toMatchObject({ totalMb: 15, quotaMb: 100, usagePercentage: 15, warning: 'Near quota', error: null })
    expect(vi.mocked(bgRequest).mock.calls[0][0]).toMatchObject({ path: '/api/v1/users/storage', method: 'GET', abortSignal: expect.any(AbortSignal) })
  })

  it('parses profile quota fallback while the original owner remains current', async () => {
    Object.assign(tldwClient, { getCurrentUserStorageQuota: vi.fn(), getCurrentUserProfile: vi.fn() })
    vi.mocked(bgRequest).mockRejectedValueOnce(new Error('Quota route unavailable')).mockResolvedValueOnce({ quotas: { storage_used_mb: 12, storage_quota_mb: '80', usage_percentage: 15 } })
    const view = mount()
    await waitFor(() => expect(view.result.current.libraryStorageUsage.loading).toBe(false))
    expect(view.result.current.libraryStorageUsage).toMatchObject({ totalMb: 12, quotaMb: 80, usagePercentage: 15, error: null })
    expect(vi.mocked(bgRequest).mock.calls.map(([request]) => request.path)).toEqual(['/api/v1/users/storage', '/api/v1/users/me/profile?sections=quotas'])
  })

  it('does not dispatch quota fallback when the first read outlives its owner', async () => {
    Object.assign(tldwClient, { getCurrentUserStorageQuota: vi.fn(), getCurrentUserProfile: vi.fn() })
    let reject!: (error: Error) => void
    vi.mocked(bgRequest).mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail }))
    const view = mount()
    await waitFor(() => expect(reject).toBeTypeOf('function'))
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    await act(async () => { reject(new Error('Retired quota')) })
    expect(bgRequest).toHaveBeenCalledTimes(1)
    expect(view.result.current.libraryStorageUsage).toMatchObject({ loading: false, totalMb: null, error: null })
  })

  it.each(['delete', 'keywords'])('retires sequential bulk %s after the first held request', async action => {
    const view = mount(); select(view)
    let resolve!: (value: unknown) => void
    vi.mocked(bgRequest).mockImplementationOnce(() => new Promise(done => { resolve = done }))
    let pending!: Promise<void>
    act(() => { pending = action === 'delete' ? view.result.current.handleBulkDelete() : view.result.current.handleBulkAddKeywords() })
    await waitFor(() => expect(resolve).toBeTypeOf('function'))
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    await act(async () => { resolve({}); await pending })
    expect(bgRequest).toHaveBeenCalledTimes(1)
    expect(vi.mocked(bgRequest).mock.calls[0][0].abortSignal?.aborted).toBe(true)
    expect(view.refetch).not.toHaveBeenCalled()
    expect(view.feedback.success).not.toHaveBeenCalled()
    expect(view.feedback.warning).not.toHaveBeenCalled()
  })

  it('does not send a note delete after an account switch during its version lookup', async () => {
    const view = mount()
    let resolve!: (value: unknown) => void
    vi.mocked(bgRequest).mockImplementationOnce(() => new Promise(done => { resolve = done }))
    let pending!: Promise<void>
    act(() => { pending = view.result.current.handleDeleteItem({ ...items[0], kind: 'note' }, null) })
    await waitFor(() => expect(resolve).toBeTypeOf('function'))
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    await act(async () => { resolve({ version: 1 }); await pending })
    expect(bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.undo).not.toHaveBeenCalled()
  })

  it('allows sequential mutations while the selected owner remains current', async () => {
    const view = mount(); select(view)
    await act(async () => view.result.current.handleBulkAddKeywords())
    expect(vi.mocked(bgRequest).mock.calls.map(([request]) => request.path)).toEqual(['/api/v1/media/1', '/api/v1/media/2'])
    expect(vi.mocked(bgRequest).mock.calls.every(([request]) => request.abortSignal && !request.abortSignal.aborted)).toBe(true)
    expect(view.refetch).toHaveBeenCalledTimes(1)
  })

  it('aborts a held restore on unmount without applying its response', async () => {
    const view = mount()
    await act(async () => view.result.current.handleDeleteItem(items[0], null))
    const undo = mocks.undo.mock.calls[0][0].onUndo
    let resolve!: (value: unknown) => void
    vi.mocked(bgRequest).mockImplementationOnce(() => new Promise(done => { resolve = done }))
    const pending = undo().catch((error: Error) => error)
    await waitFor(() => expect(resolve).toBeTypeOf('function'))
    view.refetch.mockClear()
    view.unmount()
    await act(async () => { resolve({}); await pending })
    expect(vi.mocked(bgRequest).mock.calls.at(-1)?.[0].abortSignal?.aborted).toBe(true)
    expect(view.refetch).not.toHaveBeenCalled()
  })

  it('does not execute retained undo against the replacement account', async () => {
    const view = mount()
    await act(async () => view.result.current.handleDeleteItem(items[0], null))
    const undo = mocks.undo.mock.calls[0][0].onUndo
    vi.mocked(bgRequest).mockClear()
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    await act(async () => undo())
    expect(bgRequest).not.toHaveBeenCalled()
  })

  it('isolates collection names and favorites by owner and preserves same-owner reload', async () => {
    const storage = new Storage()
    const legacyCollections = [{ id: 'legacy', name: 'Legacy private project', itemIds: ['2'] }]
    await storage.set('media:collections:v1', legacyCollections)
    await storage.set('media:favorites', ['2'])
    const a = mount(); select(a)
    await act(async () => {})
    expect(a.result.current.mediaCollections).toEqual([])
    expect(a.result.current.favorites).toEqual([])
    act(() => a.result.current.setCollectionDraftName('A private project'))
    await act(async () => {
      a.result.current.handleAddSelectionToCollection()
      a.result.current.toggleFavorite('1')
    })
    await waitFor(() => expect(a.result.current.mediaCollections[0]?.name).toBe('A private project'))
    a.unmount()
    const b = mount('account-b')
    await act(async () => {})
    expect(b.result.current.mediaCollections).toEqual([])
    expect(b.result.current.favorites).toEqual([])
    b.unmount()
    const reloaded = mount()
    await waitFor(() => expect(reloaded.result.current.mediaCollections[0]?.name).toBe('A private project'))
    expect(reloaded.result.current.favorites).toEqual(['1'])
    expect(await storage.get('media:collections:v1')).toEqual(legacyCollections)
    expect(await storage.get('media:favorites')).toEqual(['2'])
  })

  it('hides the previous key during an in-place owner change and rejects old setters', async () => {
    const view = mount()
    await act(async () => {})
    await act(async () => { view.result.current.toggleFavorite('1') })
    expect(view.result.current.favorites).toEqual(['1'])
    const staleToggle = view.result.current.toggleFavorite
    act(() => view.rerender({ owner: 'account-b' }))
    expect(view.result.current.favorites).toEqual([])
    await act(async () => { staleToggle('2') })
    expect(view.result.current.favorites).toEqual([])
    view.unmount()
    const reloaded = mount()
    await waitFor(() => expect(reloaded.result.current.favorites).toEqual(['1']))
  })

  it('does not persist collections or favorites without a verified owner', async () => {
    const view = mount(null); select(view)
    act(() => view.result.current.setCollectionDraftName('Unknown private project'))
    await act(async () => { view.result.current.handleAddSelectionToCollection(); view.result.current.toggleFavorite('1') })
    expect(view.result.current.mediaCollections).toEqual([])
    expect(view.result.current.favorites).toEqual([])
    expect(JSON.stringify(localStorage)).not.toContain('Unknown private project')
  })
})
