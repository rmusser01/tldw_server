import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useMediaSelection } from '../hooks/useMediaSelection'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import type { MediaResultItem } from '@/components/Media/types'

vi.mock('@plasmohq/storage/hook', () => ({
  useStorage: (_key: string, initialValue: unknown) => [
    initialValue,
    vi.fn(),
    { isLoading: false }
  ]
}))

vi.mock('@/hooks/useUndoNotification', () => ({
  useUndoNotification: () => ({ showUndoNotification: vi.fn() })
}))

vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  setSetting: vi.fn()
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: vi.fn()
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getReadingProgress: vi.fn()
  }
}))

const createMediaItem = (id: string): MediaResultItem => ({
  id,
  kind: 'media',
  title: `Document ${id}`,
  content: '',
  meta: { type: 'pdf' },
  raw: { id, type: 'pdf' }
} as MediaResultItem)

const createDeps = (displayResults: MediaResultItem[]) => ({
  t: (key: string) => key,
  message: {
    error: vi.fn(),
    warning: vi.fn(),
    success: vi.fn()
  },
  displayResults,
  selected: null,
  setSelected: vi.fn(),
  setSelectedContent: vi.fn(),
  setSelectedDetail: vi.fn(),
  setLastFetchedId: vi.fn(),
  refetch: vi.fn()
})

describe('useMediaSelection reading progress', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked((tldwClient as any).getReadingProgress).mockResolvedValue({
      has_progress: true,
      percent_complete: 25
    })
  })

  it('does not refetch progress when visible reading-progress ids are unchanged', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <MemoryRouter>{children}</MemoryRouter>
    )
    const initialItems = [createMediaItem('101')]

    const { rerender } = renderHook(
      ({ displayResults }) => useMediaSelection(createDeps(displayResults)),
      {
        wrapper,
        initialProps: { displayResults: initialItems }
      }
    )

    await waitFor(() => {
      expect((tldwClient as any).getReadingProgress).toHaveBeenCalledTimes(1)
    })

    rerender({ displayResults: [createMediaItem('101')] })

    await waitFor(() => {
      expect((tldwClient as any).getReadingProgress).toHaveBeenCalledTimes(1)
    })
  })
})
