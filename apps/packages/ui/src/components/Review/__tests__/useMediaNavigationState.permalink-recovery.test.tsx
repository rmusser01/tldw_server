import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { bgRequest } from '@/services/background-proxy'
import { getSetting } from '@/services/settings/registry'
import { useConnectionStore } from '@/store/connection'
import { useMediaNavigationState } from '../hooks/useMediaNavigationState'

vi.mock('@/services/background-proxy', () => ({ bgRequest: vi.fn() }))
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  clearSetting: vi.fn(), getSetting: vi.fn().mockResolvedValue(null), setSetting: vi.fn()
}))
vi.mock('@/components/Review/ViewMediaPage', () => ({ MEDIA_STALE_CHECK_INTERVAL_MS: 60_000 }))

const media = (id: number, text: string) => ({
  media_id: id,
  source: { url: 'rowan.txt', title: 'Rowan', type: 'document', duration: null },
  processing: {}, content: { text, metadata: {}, word_count: 3 }, keywords: [],
  timestamps: {}, versions: [], has_original_file: false, original_file_url: null
})
const mount = (entry = '/media?id=7') => renderHook(() => ({
  nav: useMediaNavigationState({
    t: (key, options) => options?.defaultValue || key,
    message: { error: vi.fn(), warning: vi.fn(), success: vi.fn() },
    displayResults: [], refetch: vi.fn().mockResolvedValue({ data: [] })
  }),
  navigate: useNavigate(), location: useLocation()
}), { wrapper: ({ children }) => <MemoryRouter initialEntries={[entry]}>{children}</MemoryRouter> })

describe('failed permalink recovery', () => {
  beforeEach(() => {
    vi.mocked(bgRequest).mockReset()
    vi.mocked(getSetting).mockResolvedValue(null)
    useConnectionStore.setState(({ state }) => ({ state: { ...state, isConnected: true, serverUrl: 'http://localhost:8000' } }))
  })

  it('keeps a missing target recoverable without inventing a selected record', async () => {
    vi.mocked(bgRequest).mockRejectedValueOnce({ status: 404 }).mockResolvedValue(media(7, 'Recovered Rowan body'))
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '7' }))
    expect(result.current.location.search).toBe('?id=7')
    expect(result.current.nav.selected).toBeNull()
    expect(result.current.nav.selectedContent).toBe('')
    act(() => result.current.nav.handleRetryDetailFetch())
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Recovered Rowan body'))
    expect(result.current.nav.detailFetchError).toBeNull()
    expect(result.current.location.search).toBe('?id=7')
  })

  it('recovers a failed last-viewed source on a page without a permalink', async () => {
    vi.mocked(getSetting).mockResolvedValue('7')
    vi.mocked(bgRequest).mockRejectedValueOnce({ status: 404 }).mockResolvedValue(media(7, 'Recovered saved source'))
    const { result } = mount('/media')
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '7' }))
    expect(result.current.location.search).toBe('')
    act(() => result.current.nav.handleRetryDetailFetch())
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Recovered saved source'))
    expect(result.current.nav.detailLoading).toBe(false)
    expect(result.current.nav.detailFetchError).toBeNull()
    expect(result.current.location.search).toBe('?id=7')
  })

  it('retries the failed new URL rather than the previously displayed source', async () => {
    let failed = false
    vi.mocked(bgRequest).mockImplementation(({ path }) => {
      if (path === '/api/v1/media/7') return Promise.resolve(media(7, 'Prior Rowan body'))
      if (!failed) { failed = true; return Promise.reject({ status: 404 }) }
      return Promise.resolve(media(9, 'Recovered new source'))
    })
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Prior Rowan body'))
    act(() => result.current.navigate('/media?id=9'))
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '9' }))
    expect(result.current.location.search).toBe('?id=9')
    act(() => result.current.nav.handleRetryDetailFetch())
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Recovered new source'))
    expect(result.current.nav.selected?.id).toBe(9)
    expect(result.current.nav.detailFetchError).toBeNull()
  })

  it('ignores an old permalink failure after a newer target succeeds', async () => {
    let rejectOld!: (error: unknown) => void
    const old = new Promise((_resolve, reject) => { rejectOld = reject })
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/7'
      ? old : Promise.resolve(media(9, 'Current source body')))
    const { result } = mount()
    act(() => result.current.navigate('/media?id=9'))
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Current source body'))
    await act(async () => { rejectOld({ status: 404 }); await old.catch(() => {}) })
    expect(result.current.nav.detailFetchError).toBeNull()
    expect(result.current.nav.selectedContent).toBe('Current source body')
    expect(result.current.location.search).toBe('?id=9')
  })

  it('rejects a late retry result after the account boundary retires', async () => {
    let resolveRetry!: (value: unknown) => void
    const retry = new Promise(resolve => { resolveRetry = resolve })
    vi.mocked(bgRequest).mockRejectedValueOnce({ status: 404 }).mockReturnValueOnce(retry)
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '7' }))
    act(() => result.current.nav.handleRetryDetailFetch())
    await waitFor(() => expect(result.current.nav.detailLoading).toBe(true))
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    await act(async () => { resolveRetry(media(7, 'Late private body')); await retry })
    expect(result.current.nav.selected).toBeNull()
    expect(result.current.nav.selectedContent).toBe('')
    expect(result.current.nav.detailFetchError).toBeNull()
  })

  it('preserves a manual choice when the old permalink completes in the same turn', async () => {
    let resolveOld!: (value: unknown) => void
    const old = new Promise(resolve => { resolveOld = resolve })
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/7'
      ? old : Promise.resolve(media(9, 'Manually selected source')))
    const { result } = mount()
    await act(async () => {
      result.current.nav.setSelected({ kind: 'media', id: 9, title: 'New choice', snippet: '', keywords: [], meta: {}, raw: {} })
      resolveOld(media(7, 'Obsolete permalink body'))
      await old
    })
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Manually selected source'))
    expect(result.current.nav.selected?.id).toBe(9)
    expect(result.current.location.search).toBe('?id=9')
    expect(result.current.nav.detailFetchError).toBeNull()
  })

  it.each(['clear', 'restore'] as const)('preserves a functional %s while an older permalink is pending', async action => {
    let resolveOld!: (value: unknown) => void
    const old = new Promise(resolve => { resolveOld = resolve })
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/9'
      ? old : Promise.resolve(media(Number(String(path).split('/').at(-1)), 'Current source body')))
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.selected?.id).toBe(7))
    act(() => result.current.navigate('/media?id=9'))
    await waitFor(() => expect(result.current.nav.detailLoading).toBe(true))
    await act(async () => {
      result.current.nav.setSelected(previous => action === 'clear' ? null : {
        ...previous!, id: 11, title: 'Restored source'
      })
      resolveOld(media(9, 'Obsolete pending source'))
      await old
    })
    await waitFor(() => expect(result.current.nav.selected?.id ?? null).toBe(action === 'clear' ? null : 11))
    expect(result.current.nav.selectedContent).not.toBe('Obsolete pending source')
    expect(result.current.nav.detailLoading).toBe(false)
  })

  it('preserves a reselected note whose ID collides with the pending Media URL', async () => {
    let resolveMedia!: (value: unknown) => void
    const pending = new Promise(resolve => { resolveMedia = resolve })
    vi.mocked(bgRequest).mockReturnValue(pending)
    const { result } = mount('/media')
    const note = { kind: 'note' as const, id: 7, title: 'Note', snippet: '', keywords: [], meta: {}, raw: { content: 'Selected note body' } }
    act(() => result.current.nav.setSelected(note))
    await waitFor(() => expect(result.current.nav.selected?.kind).toBe('note'))
    act(() => result.current.navigate('/media?id=7'))
    await waitFor(() => expect(result.current.nav.detailLoading).toBe(true))
    await act(async () => {
      result.current.nav.setSelected(note)
      resolveMedia(media(7, 'Coincident Media body'))
      await pending
    })
    await waitFor(() => expect(result.current.nav.selected?.kind).toBe('note'))
    expect(result.current.nav.selectedContent).not.toBe('Coincident Media body')
    expect(result.current.location.search).toBe('')
  })

  it('keeps a pending new source through a metadata-only functional update', async () => {
    let resolveNext!: (value: unknown) => void
    const pending = new Promise(resolve => { resolveNext = resolve })
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/9'
      ? pending : Promise.resolve(media(7, 'Prior source body')))
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.selected?.id).toBe(7))
    act(() => result.current.navigate('/media?id=9'))
    await waitFor(() => expect(result.current.nav.detailLoading).toBe(true))
    act(() => result.current.nav.setSelected(current => ({ ...current!, keywords: ['confirmed'] })))
    expect(result.current.nav.selected?.keywords).toEqual(['confirmed'])
    await act(async () => { resolveNext(media(9, 'Requested new source')); await pending })
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Requested new source'))
    expect(result.current.nav.selected?.id).toBe(9)
    expect(result.current.location.search).toBe('?id=9')
  })

  it('keeps a newly accepted source through an older metadata callback before render', async () => {
    let resolveNext!: (value: unknown) => void
    const pending = new Promise(resolve => { resolveNext = resolve })
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/9'
      ? pending : Promise.resolve(media(7, 'Prior source body')))
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.selected?.id).toBe(7))
    act(() => result.current.navigate('/media?id=9'))
    await waitFor(() => expect(result.current.nav.detailLoading).toBe(true))
    await act(async () => {
      resolveNext(media(9, 'Accepted new source'))
      await pending
      result.current.nav.setSelected(current => current?.id === 7 ? { ...current, keywords: ['confirmed'] } : current)
    })
    await waitFor(() => expect(result.current.nav.selected?.id).toBe(9))
    expect(result.current.nav.selectedContent).toBe('Accepted new source')
    expect(result.current.location.search).toBe('?id=9')
  })

  it('retries failed Media even when a selected note has the same ID', async () => {
    const requested: string[] = []
    vi.mocked(bgRequest).mockImplementation(({ path }) => {
      requested.push(String(path))
      return requested.length === 1 ? Promise.reject({ status: 404 }) : Promise.resolve(media(7, 'Recovered Media body'))
    })
    const { result } = mount('/media')
    act(() => result.current.nav.setSelected({ kind: 'note', id: 7, title: 'Note', snippet: '', keywords: [], meta: {}, raw: { content: 'Note body' } }))
    await waitFor(() => expect(result.current.nav.selected?.kind).toBe('note'))
    act(() => result.current.navigate('/media?id=7'))
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '7' }))
    act(() => result.current.nav.handleRetryDetailFetch())
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Recovered Media body'))
    expect(result.current.nav.selected?.kind).toBe('media')
    expect(requested).toEqual(['/api/v1/media/7', '/api/v1/media/7'])
  })

  it('coalesces a repeated Retry before its new request has rendered', async () => {
    const requested: string[] = []
    vi.mocked(bgRequest).mockImplementation(({ path }) => {
      requested.push(String(path))
      return requested.length === 1 ? Promise.reject({ status: 404 }) : Promise.resolve(media(7, 'One recovered source'))
    })
    const { result } = mount()
    await waitFor(() => expect(result.current.nav.detailFetchError).toMatchObject({ mediaId: '7' }))
    act(() => { result.current.nav.handleRetryDetailFetch(); result.current.nav.handleRetryDetailFetch() })
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('One recovered source'))
    expect(requested).toEqual(['/api/v1/media/7', '/api/v1/media/7'])
  })

})
