import React from 'react'
import { MemoryRouter } from 'react-router-dom'
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
