import React from 'react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { PdfPage, ReleasingPage } from '../PdfViewer/PdfPage'

// A react-pdf <Page> stand-in with the callback order of the real one: the
// page proxy loads on mount, and the canvas paints later (when the test says
// so) unless the <Page> unmounts first, which cancels the render.
const pdf = vi.hoisted(() => ({
  proxies: new Map<number, { cleanup: ReturnType<typeof vi.fn> }>(),
  mounts: 0,
  pendingPaints: [] as Array<() => void>
}))
const proxy = (pageNumber: number) => {
  if (!pdf.proxies.has(pageNumber)) {
    // pdf.js returns false (and defers the release) while a render is running.
    pdf.proxies.set(pageNumber, { cleanup: vi.fn(() => false) })
  }
  return pdf.proxies.get(pageNumber)!
}
const paintAll = () =>
  act(() => {
    for (const paint of pdf.pendingPaints.splice(0)) paint()
  })

vi.mock('../PageNoteButton', () => ({
  PageNoteButton: ({ pageNumber }: { pageNumber: number }) => <button>Note page {pageNumber}</button>
}))
vi.mock('react-pdf', async () => {
  const React = await import('react')
  return {
    Page: ({
      pageNumber,
      onLoadSuccess,
      onRenderSuccess
    }: {
      pageNumber: number
      onLoadSuccess?: (page: unknown) => void
      onRenderSuccess?: (page: unknown) => void
    }) => {
      React.useEffect(() => {
        pdf.mounts++
        const page = proxy(pageNumber)
        onLoadSuccess?.(page)
        let mounted = true
        pdf.pendingPaints.push(() => {
          if (mounted) onRenderSuccess?.(page)
        })
        return () => {
          mounted = false
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- react-pdf reads its callbacks at call time
      }, [pageNumber])
      return <canvas aria-label={`Page ${pageNumber}`} />
    }
  }
})

const observers: IntersectionObserverCallback[] = []
const setThumbnailVisible = (isIntersecting: boolean) =>
  act(() => {
    for (const callback of observers) {
      callback([{ isIntersecting } as IntersectionObserverEntry], {} as IntersectionObserver)
    }
  })

beforeEach(() => {
  pdf.proxies.clear()
  pdf.mounts = 0
  pdf.pendingPaints.length = 0
  observers.length = 0
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      constructor(callback: IntersectionObserverCallback) {
        observers.push(callback)
      }
      observe() {}
      disconnect() {}
    }
  )
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('releases the page render data once its canvas is painted', () => {
  render(<PdfPage pageNumber={1} scale={1} />)
  expect(proxy(1).cleanup).not.toHaveBeenCalled()

  paintAll()

  expect(proxy(1).cleanup).toHaveBeenCalledTimes(1)
  expect(screen.getByLabelText('Page 1')).toBeInTheDocument()
})

it('still clears the page spinner and shows the page note once the page is painted', () => {
  render(<PdfPage pageNumber={1} scale={1} />)
  expect(screen.queryByText('Note page 1')).toBeNull()

  paintAll()

  expect(screen.getByText('Note page 1')).toBeInTheDocument()
})

it('passes the loaded and painted page on to the caller callbacks', () => {
  const onLoadSuccess = vi.fn()
  const onRenderSuccess = vi.fn()
  render(<ReleasingPage pageNumber={1} onLoadSuccess={onLoadSuccess} onRenderSuccess={onRenderSuccess} />)
  expect(onLoadSuccess).toHaveBeenCalledWith(proxy(1))

  paintAll()

  expect(onRenderSuccess).toHaveBeenCalledWith(proxy(1))
})

it('releases the page render data when the page unmounts before its render finishes', () => {
  const view = render(<PdfPage pageNumber={1} scale={1} />)

  view.unmount()
  paintAll() // the cancelled render never reports success

  expect(proxy(1).cleanup).toHaveBeenCalledTimes(1)
})

it('releases a thumbnail that scrolls out of view and renders it again when it returns', () => {
  render(<PdfPage pageNumber={3} scale={0.25} lazy />)
  setThumbnailVisible(true)
  expect(screen.getByLabelText('Page 3')).toBeInTheDocument()

  setThumbnailVisible(false)
  expect(screen.queryByLabelText('Page 3')).toBeNull()
  expect(proxy(3).cleanup).toHaveBeenCalledTimes(1)

  setThumbnailVisible(true)
  expect(screen.getByLabelText('Page 3')).toBeInTheDocument()
  expect(pdf.mounts).toBe(2)
  paintAll()
  expect(proxy(3).cleanup).toHaveBeenCalledTimes(2)
})

it('releases the previous page when the single-page fallback switches to another page', () => {
  const view = render(<PdfPage pageNumber={1} scale={1} />)

  view.rerender(<PdfPage pageNumber={2} scale={1} />)

  expect(proxy(1).cleanup).toHaveBeenCalledTimes(1)
  expect(proxy(2).cleanup).not.toHaveBeenCalled()
  expect(screen.getByLabelText('Page 2')).toBeInTheDocument()
})

// pdf.js itself defers cleanup() while another render of the proxy runs; that
// is not exercised here (the proxy is mocked).
it('releases a page proxy shared by two consumers when one unmounts and again when the other paints', () => {
  const view = render(
    <>
      <PdfPage key="viewer" pageNumber={1} scale={1} />
      <PdfPage key="thumbnail" pageNumber={1} scale={0.25} hidePageNote />
    </>
  )

  view.rerender(<PdfPage key="thumbnail" pageNumber={1} scale={0.25} hidePageNote />)

  expect(proxy(1).cleanup).toHaveBeenCalledTimes(1)
  expect(screen.getByLabelText('Page 1')).toBeInTheDocument()
  paintAll()
  expect(proxy(1).cleanup).toHaveBeenCalledTimes(2)
})

it('paints and releases under StrictMode without a render or release loop', async () => {
  render(
    <React.StrictMode>
      <PdfPage pageNumber={1} scale={1} />
    </React.StrictMode>
  )
  paintAll()
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })

  expect(screen.getByLabelText('Page 1')).toBeInTheDocument()
  const releases = proxy(1).cleanup.mock.calls.length
  expect(releases).toBeGreaterThanOrEqual(1)
  expect(releases).toBeLessThanOrEqual(2)
  expect(pdf.pendingPaints).toHaveLength(0)
})
