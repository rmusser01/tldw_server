import React from 'react'
import { act, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useContentViewerModals } from '../hooks/useContentViewerModals'
import { bgRequest } from '@/services/background-proxy'
import type { BgRequestInit } from '@/services/background-proxy'
import type { UseContentViewerModalsDeps } from '../hooks/useContentViewerModals'

vi.mock('@/services/background-proxy', () => ({ bgRequest: vi.fn() }))
afterEach(() => vi.restoreAllMocks())

const mediaDeps = (id: string): UseContentViewerModalsDeps => ({
  selectedMedia: {
    id,
    kind: 'media',
    meta: { type: 'video' },
    raw: { has_original_file: true }
  } as unknown as UseContentViewerModalsDeps['selectedMedia'],
  selectedMediaId: id,
  mediaDetail: {},
  content: '',
  isNote: false,
  editingKeywords: [],
  selectedAnalysis: null,
  collapsedSections: {},
  setCollapsedSections: vi.fn(),
  contentBodyRef: { current: null },
  t: (key) => key
})

it('unloads a detached preview player before its ref is cleared', async () => {
  vi.mocked(bgRequest).mockResolvedValue(new ArrayBuffer(4))
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
  const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  const load = vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  const deps = mediaDeps('1')
  function Preview() {
    const modals = useContentViewerModals(deps)
    return modals.embeddedMediaUrl ? (
      <video data-testid="preview" src={modals.embeddedMediaUrl} ref={modals.setMediaPlayerRef} />
    ) : null
  }
  const view = render(<Preview />)
  const player = await screen.findByTestId('preview')
  pause.mockClear()
  load.mockClear()
  view.unmount()
  expect(pause).toHaveBeenCalledOnce()
  expect(player).not.toHaveAttribute('src')
  expect(load).toHaveBeenCalledOnce()
})

it('bounds preview requests and aborts retired downloads without creating late URLs', async () => {
  const requests: { init: BgRequestInit; resolve: (value: ArrayBuffer) => void }[] = []
  vi.mocked(bgRequest).mockImplementation(
    (init) => new Promise((resolve) => requests.push({ init, resolve }))
  )
  const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
  const hook = renderHook(
    ({ id }) =>
      useContentViewerModals({
        selectedMedia: {
          id,
          kind: 'media',
          meta: { type: 'video' },
          raw: { has_original_file: true }
        } as unknown as UseContentViewerModalsDeps['selectedMedia'],
        selectedMediaId: id,
        mediaDetail: {},
        content: '',
        isNote: false,
        editingKeywords: [],
        selectedAnalysis: null,
        collapsedSections: {},
        setCollapsedSections: vi.fn(),
        contentBodyRef: { current: null },
        t: (key) => key
      }),
    { initialProps: { id: '1' } }
  )
  await waitFor(() => expect(requests).toHaveLength(1))
  expect(requests[0].init.maxResponseBytes).toBeLessThanOrEqual(64 * 1024 * 1024)
  hook.rerender({ id: '2' })
  expect(requests[0].init.abortSignal.aborted).toBe(true)
  hook.unmount()
  expect(requests[1].init.abortSignal.aborted).toBe(true)
  await act(async () => requests.forEach((request) => request.resolve(new ArrayBuffer(4))))
  expect(createUrl).not.toHaveBeenCalled()
})
