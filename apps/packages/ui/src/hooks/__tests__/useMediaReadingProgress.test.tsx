import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useMediaReadingProgress } from '../useMediaReadingProgress'

const mocks = vi.hoisted(() => ({
  getReadingProgress: vi.fn(),
  updateReadingProgress: vi.fn(),
  deleteReadingProgress: vi.fn()
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getReadingProgress: mocks.getReadingProgress,
    updateReadingProgress: mocks.updateReadingProgress,
    deleteReadingProgress: mocks.deleteReadingProgress
  }
}))

const createScrollContainer = (
  scrollHeight: number,
  clientHeight: number,
  initialScrollTop = 0
): HTMLDivElement => {
  const element = document.createElement('div') as HTMLDivElement

  let top = initialScrollTop
  Object.defineProperty(element, 'scrollHeight', {
    configurable: true,
    get: () => scrollHeight
  })
  Object.defineProperty(element, 'clientHeight', {
    configurable: true,
    get: () => clientHeight
  })
  Object.defineProperty(element, 'scrollTop', {
    configurable: true,
    get: () => top,
    set: (value: number) => {
      top = Number(value)
    }
  })

  return element
}

describe('useMediaReadingProgress', () => {
  let originalRequestAnimationFrame: typeof window.requestAnimationFrame | undefined

  beforeEach(() => {
    originalRequestAnimationFrame = window.requestAnimationFrame
    const raf = (cb: FrameRequestCallback) => {
      cb(0)
      return 1
    }
    ;(window as any).requestAnimationFrame = raf
    ;(globalThis as any).requestAnimationFrame = raf

    mocks.getReadingProgress.mockReset()
    mocks.updateReadingProgress.mockReset()
    mocks.deleteReadingProgress.mockReset()

    mocks.updateReadingProgress.mockResolvedValue({
      media_id: 1,
      current_page: 1,
      total_pages: 1,
      zoom_level: 1,
      view_mode: 'continuous',
      percent_complete: 0,
      last_read_at: '2026-02-18T00:00:00Z'
    })
  })

  afterEach(() => {
    if (originalRequestAnimationFrame) {
      ;(window as any).requestAnimationFrame = originalRequestAnimationFrame
      ;(globalThis as any).requestAnimationFrame = originalRequestAnimationFrame
    } else {
      delete (window as any).requestAnimationFrame
      delete (globalThis as any).requestAnimationFrame
    }
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('restores scroll position from percent_complete on load', async () => {
    const container = createScrollContainer(2000, 1000, 0)
    const scrollRef = { current: container }

    mocks.getReadingProgress.mockResolvedValue({
      media_id: 12,
      current_page: 2,
      total_pages: 4,
      percent_complete: 50,
      zoom_level: 1,
      view_mode: 'continuous',
      last_read_at: '2026-02-18T00:00:00Z'
    })

    renderHook(() =>
      useMediaReadingProgress({
        mediaId: 12,
        mediaKind: 'media',
        mediaDetail: {},
        contentLength: 10000,
        scrollContainerRef: scrollRef
      })
    )

    await waitFor(() => {
      expect(mocks.getReadingProgress).toHaveBeenCalledWith('12')
    })

    await waitFor(() => {
      expect(container.scrollTop).toBe(500)
    })
  })

  it('falls back to CFI-based scroll restore when percent_complete is absent', async () => {
    const container = createScrollContainer(2000, 1000, 0)
    const scrollRef = { current: container }

    mocks.getReadingProgress.mockResolvedValue({
      media_id: 15,
      current_page: 1,
      total_pages: 3,
      cfi: 'scroll:25',
      last_read_at: '2026-02-18T00:00:00Z'
    })

    renderHook(() =>
      useMediaReadingProgress({
        mediaId: 15,
        mediaKind: 'media',
        mediaDetail: {},
        contentLength: 10000,
        scrollContainerRef: scrollRef
      })
    )

    await waitFor(() => {
      expect(container.scrollTop).toBe(250)
    })
  })

  it('debounces scroll writes and avoids duplicate writes when unchanged', async () => {
    vi.useFakeTimers()

    const container = createScrollContainer(2000, 1000, 0)
    const scrollRef = { current: container }

    mocks.getReadingProgress.mockResolvedValue({
      media_id: 21,
      has_progress: false
    })

    const { unmount } = renderHook(() =>
      useMediaReadingProgress({
        mediaId: 21,
        mediaKind: 'media',
        mediaDetail: {},
        contentLength: 12000,
        scrollContainerRef: scrollRef,
        debounceMs: 900
      })
    )

    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.getReadingProgress).toHaveBeenCalledWith('21')

    act(() => {
      container.scrollTop = 400
      container.dispatchEvent(new Event('scroll'))
    })

    act(() => {
      vi.advanceTimersByTime(899)
    })
    expect(mocks.updateReadingProgress).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
    })

    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)

    expect(mocks.updateReadingProgress).toHaveBeenCalledWith(
      '21',
      expect.objectContaining({
        zoom_level: 100,
        percentage: 40,
        cfi: 'scroll:40'
      })
    )

    act(() => {
      container.dispatchEvent(new Event('scroll'))
    })

    await act(async () => {
      vi.advanceTimersByTime(900)
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)

    unmount()

    await act(async () => {
      await Promise.resolve()
    })

    expect(mocks.updateReadingProgress).toHaveBeenCalledTimes(1)
  })

  it('flushes pending progress when switching media ids and restores the next selection', async () => {
    const container = createScrollContainer(3000, 1000, 0)
    const scrollRef = { current: container }

    mocks.getReadingProgress.mockImplementation(async (mediaId: string | number) => {
      if (String(mediaId) === '2') {
        return {
          media_id: 2,
          current_page: 1,
          total_pages: 4,
          percent_complete: 25,
          last_read_at: '2026-02-18T00:00:00Z'
        }
      }

      return {
        media_id: Number(mediaId),
        has_progress: false
      }
    })

    const { rerender, result } = renderHook(
      ({ mediaId }: { mediaId: number }) =>
        useMediaReadingProgress({
          mediaId,
          mediaKind: 'media',
          mediaDetail: {},
          contentLength: 15000,
          scrollContainerRef: scrollRef,
          debounceMs: 900
        }),
      {
        initialProps: { mediaId: 1 }
      }
    )

    await waitFor(() => {
      expect(mocks.getReadingProgress).toHaveBeenCalledWith('1')
    })

    act(() => {
      container.scrollTop = 1000
      container.dispatchEvent(new Event('scroll'))
    })

    rerender({ mediaId: 2 })

    await waitFor(() => {
      expect(mocks.updateReadingProgress).toHaveBeenCalledWith(
        '1',
        expect.objectContaining({
          zoom_level: 100,
          percentage: 50,
          cfi: 'scroll:50'
        })
      )
    })

    await waitFor(() => {
      expect(mocks.getReadingProgress).toHaveBeenCalledWith('2')
    })

    await waitFor(() => {
      expect(container.scrollTop).toBe(500)
    })

    await act(async () => {
      container.scrollTop = 1500
      await result.current.saveProgress()
    })
    expect(mocks.updateReadingProgress).toHaveBeenLastCalledWith(
      '2',
      expect.objectContaining({
        zoom_level: 100,
        percentage: 75,
        cfi: 'scroll:75'
      })
    )
  })
})
