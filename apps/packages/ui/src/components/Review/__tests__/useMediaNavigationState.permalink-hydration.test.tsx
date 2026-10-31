import React from 'react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useMediaNavigationState } from '../hooks/useMediaNavigationState'
import { bgRequest } from '@/services/background-proxy'

vi.mock('@/services/background-proxy', () => ({
  bgRequest: vi.fn()
}))

vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  clearSetting: vi.fn(),
  getSetting: vi.fn().mockResolvedValue(null),
  setSetting: vi.fn()
}))

vi.mock('@/components/Review/ViewMediaPage', () => ({
  MEDIA_STALE_CHECK_INTERVAL_MS: 60_000
}))

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

const createDeps = (displayResults: any[]) => ({
  t: (key: string) => key,
  message: {
    error: vi.fn(),
    warning: vi.fn(),
    success: vi.fn()
  },
  displayResults,
  refetch: vi.fn().mockResolvedValue({ data: displayResults })
})

describe('useMediaNavigationState permalink hydration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('preserves a new URL target while the previous selection is still displayed', async () => {
    const next = createDeferred<any>()
    vi.mocked(bgRequest).mockImplementation(({ path }) => path === '/api/v1/media/9'
      ? next.promise : Promise.resolve({ id: 7, title: 'Old selection', content: { text: 'Old body' } }))
    const { result } = renderHook(() => ({
      nav: useMediaNavigationState(createDeps([])),
      navigate: useNavigate(),
      location: useLocation()
    }), { wrapper: ({ children }) => <MemoryRouter initialEntries={['/media?id=7']}>{children}</MemoryRouter> })
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('Old body'))
    await act(async () => {
      result.current.navigate('/media?id=9')
    })
    await waitFor(() => expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/media/9' })))
    expect(result.current.location.search).toBe('?id=9')
    await act(async () => {
      next.resolve({ id: 9, title: 'New selection', content: { text: 'New body' } })
      await next.promise
    })
    await waitFor(() => expect(result.current.nav.selectedContent).toBe('New body'))
    expect(result.current.nav.selected?.id).toBe(9)
    expect(result.current.location.search).toBe('?id=9')
  })

  it("clears selected private details and rejects their late replacement on logout", async () => {
    vi.mocked(bgRequest).mockResolvedValue({ media_id: 7, content: { text: "Prior private body" } })
    const { result } = renderHook(() => useMediaNavigationState(createDeps([])), {
      wrapper: ({ children }) => <MemoryRouter initialEntries={["/media?id=7"]}>{children}</MemoryRouter>
    })
    await waitFor(() => expect(result.current.selectedContent).toBe("Prior private body"))
    const pending = createDeferred<{ media_id: number; content: { text: string } }>()
    vi.mocked(bgRequest).mockReturnValueOnce(pending.promise)
    let reload!: Promise<boolean>
    act(() => { reload = result.current.loadSelectedDetails(result.current.selected!) })
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
    expect(result.current.selected).toBeNull()
    expect(result.current.selectedDetail).toBeNull()
    expect(result.current.selectedContent).toBe("")
    await act(async () => {
      pending.resolve({ media_id: 7, content: { text: "Late private body" } })
      await reload
    })
    expect(result.current.selectedContent).toBe("")
  })

  it('reuses the pending detail request when search results rerender', async () => {
    const detailRequest = createDeferred<any>()
    vi.mocked(bgRequest)
      .mockReturnValueOnce(detailRequest.promise)
      .mockResolvedValueOnce({})

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <MemoryRouter initialEntries={['/media?id=7']}>{children}</MemoryRouter>
    )

    const { result, rerender } = renderHook(
      ({ displayResults }) =>
        useMediaNavigationState(createDeps(displayResults)),
      {
        wrapper,
        initialProps: { displayResults: [] as any[] }
      }
    )

    await waitFor(() => {
      expect(bgRequest).toHaveBeenCalledTimes(1)
    })

    rerender({
      displayResults: [
        {
          kind: 'media',
          id: '7',
          title: 'Permalink document',
          raw: { id: '7' }
        }
      ]
    })

    await act(async () => {
      detailRequest.resolve({
        media_id: 7,
        source: { title: 'Permalink document' },
        content: { text: 'Resolved media body' }
      })
      await detailRequest.promise
    })

    await waitFor(() => {
      expect(result.current.selectedContent).toBe('Resolved media body')
    })
    expect(bgRequest).toHaveBeenCalledTimes(1)
  })
})
