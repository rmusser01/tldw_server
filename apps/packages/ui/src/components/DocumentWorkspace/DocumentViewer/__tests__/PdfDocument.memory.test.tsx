import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { PdfDocument } from '../PdfViewer/PdfDocument'

const metrics = vi.hoisted(() => ({ gate: null as Promise<void> | null }))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (_: string, fallback: string) => fallback })
}))
vi.mock('@/hooks/document-workspace/useTextSelection', () => ({
  useTextSelection: () => ({ selection: null })
}))
vi.mock('../TextSelectionPopover', () => ({ TextSelectionPopover: () => null }))
vi.mock('../PageNoteButton', () => ({ PageNoteButton: () => null }))
vi.mock('react-pdf', async () => {
  const React = await import('react')
  const pdf = {
    numPages: 1000,
    getPage: async (n: number) => {
      await metrics.gate
      return {
        getViewport: ({ scale }: { scale: number }) => ({
          height: (n === 500 ? 1800 : 600) * scale,
          width: (n === 500 ? 1200 : 400) * scale
        })
      }
    }
  }
  return {
    pdfjs: { version: 'test', GlobalWorkerOptions: {} },
    Document: ({
      onLoadSuccess,
      onItemClick,
      children
    }: React.PropsWithChildren<{
      onLoadSuccess: (loaded: typeof pdf) => void
      onItemClick?: (item: { pageNumber: number }) => void
    }>) => {
      const initialItemClick = React.useRef(onItemClick)
      React.useEffect(() => onLoadSuccess(pdf), [onLoadSuccess])
      return (
        <div>
          <a
            href="#page500"
            onClick={(event) => {
              event.preventDefault()
              initialItemClick.current?.({ pageNumber: 500 })
            }}
          >
            Internal PDF link
          </a>
          <a
            href="#page1"
            onClick={(event) => {
              event.preventDefault()
              initialItemClick.current?.({ pageNumber: 1 })
            }}
          >
            Current page PDF link
          </a>
          {children}
        </div>
      )
    },
    Page: ({
      pageNumber,
      onRenderTextLayerSuccess
    }: {
      pageNumber: number
      onRenderTextLayerSuccess?: () => void
    }) => {
      React.useEffect(() => {
        onRenderTextLayerSuccess?.()
      }, [pageNumber, onRenderTextLayerSuccess])
      return <canvas aria-label={`Page ${pageNumber}`} />
    }
  }
})

beforeEach(() => {
  metrics.gate = null
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      observe() {}
      disconnect() {}
    }
  )
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(600)
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(800)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(600)
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(620_000)
  HTMLElement.prototype.scrollIntoView = function () {}
  HTMLElement.prototype.scrollTo = function (options: ScrollToOptions | number) {
    const requestedTop = typeof options === 'number' ? options : (options.top ?? 0)
    const top = Math.max(0, Math.min(requestedTop, this.scrollHeight - this.clientHeight))
    if (this.scrollTop === top) return
    this.scrollTop = top
    fireEvent.scroll(this)
  }
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('bounds continuous canvases and renders a distant mixed-size page on navigation', async () => {
  const props = {
    url: '/large.pdf',
    documentId: 1,
    currentPage: 1,
    zoomLevel: 100,
    viewMode: 'continuous' as const,
    onLoadSuccess: vi.fn(),
    onLoadError: vi.fn(),
    onPageChange: vi.fn()
  }
  const view = render(<PdfDocument {...props} />)
  await screen.findByLabelText('Page 1')
  expect(view.container.querySelectorAll('canvas').length).toBeLessThan(12)
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
  view.rerender(<PdfDocument {...props} currentPage={500} />)
  await screen.findByLabelText('Page 500')
  expect(view.container.querySelectorAll('canvas').length).toBeLessThan(12)
  expect(screen.queryByLabelText('Page 1')).toBeNull()
  const spacer = screen.getByLabelText('Page 500').parentElement?.parentElement?.parentElement
  expect(spacer?.style.width).toBe('1200px')
})

it('preserves the requested page while initial PDF dimensions are still loading', async () => {
  let ready!: () => void
  metrics.gate = new Promise<void>((resolve) => {
    ready = resolve
  })
  const onPageChange = vi.fn()
  const view = render(
    <PdfDocument
      url="/large.pdf"
      documentId={1}
      currentPage={500}
      zoomLevel={100}
      viewMode="continuous"
      onLoadSuccess={vi.fn()}
      onLoadError={vi.fn()}
      onPageChange={onPageChange}
    />
  )
  await screen.findByLabelText('Page 500')
  const container = view.container.firstElementChild as HTMLElement
  container.scrollTop = 403_192
  fireEvent.scroll(container)
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
  })
  expect(onPageChange).not.toHaveBeenCalled()
  await act(async () => {
    ready()
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
})

it('retains document width after narrower single-page measurements and mode switches', async () => {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function () {
    return new DOMRect(
      0,
      0,
      this.dataset.pageNumber === '500' ? 1200 : 400,
      this.dataset.pageNumber === '500' ? 1800 : 600
    )
  })
  const props = {
    url: '/large.pdf',
    documentId: 1,
    currentPage: 1,
    zoomLevel: 100,
    viewMode: 'single' as const,
    onLoadSuccess: vi.fn(),
    onLoadError: vi.fn(),
    onPageChange: vi.fn()
  }
  const view = render(<PdfDocument {...props} />)
  await screen.findByLabelText('Page 1')
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
  view.rerender(<PdfDocument {...props} currentPage={500} />)
  await screen.findByLabelText('Page 500')
  view.rerender(<PdfDocument {...props} currentPage={501} />)
  await screen.findByLabelText('Page 501')
  view.rerender(<PdfDocument {...props} currentPage={501} viewMode="continuous" />)
  const canvas = await screen.findByLabelText('Page 501')
  expect(canvas.parentElement?.parentElement?.parentElement?.style.width).toBe('1200px')
})

it('routes an internal PDF link to a destination that has no mounted canvas', async () => {
  const onPageChange = vi.fn()
  render(
    <PdfDocument
      url="/large.pdf"
      documentId={1}
      currentPage={1}
      zoomLevel={100}
      viewMode="continuous"
      onLoadSuccess={vi.fn()}
      onLoadError={vi.fn()}
      onPageChange={onPageChange}
    />
  )
  await screen.findByLabelText('Page 1')
  fireEvent.click(screen.getByText('Internal PDF link'))
  expect(onPageChange).toHaveBeenCalledWith(500)
})

it('uses current navigation after React-PDF retains its initial link handler across modes', async () => {
  const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView')
  const props = {
    url: '/large.pdf',
    documentId: 1,
    currentPage: 1,
    zoomLevel: 100,
    viewMode: 'continuous' as const,
    onLoadSuccess: vi.fn(),
    onLoadError: vi.fn(),
    onPageChange: vi.fn()
  }
  const view = render(<PdfDocument {...props} />)
  await screen.findByLabelText('Page 1')
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
  view.rerender(<PdfDocument {...props} viewMode="single" />)
  fireEvent.click(screen.getByText('Current page PDF link'))
  expect(scroll).toHaveBeenCalled()
})

it('scrolls back to the page for a same-page internal link', async () => {
  const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView')
  render(
    <PdfDocument
      url="/large.pdf"
      documentId={1}
      currentPage={1}
      zoomLevel={100}
      viewMode="single"
      onLoadSuccess={vi.fn()}
      onLoadError={vi.fn()}
      onPageChange={vi.fn()}
    />
  )
  await screen.findByLabelText('Page 1')
  fireEvent.click(screen.getByText('Current page PDF link'))
  expect(scroll).toHaveBeenCalled()
})

it.each([998, 999, 1000])(
  'preserves navigation to page %s when low zoom clamps scrolling before its start',
  async (page) => {
    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(166_332)
    const props = {
      url: '/large.pdf',
      documentId: 1,
      currentPage: 1,
      zoomLevel: 25,
      viewMode: 'continuous' as const,
      onLoadSuccess: vi.fn(),
      onLoadError: vi.fn(),
      onPageChange: vi.fn()
    }
    const view = render(<PdfDocument {...props} />)
    await screen.findByLabelText('Page 1')
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    view.rerender(<PdfDocument {...props} currentPage={page} />)
    await screen.findByLabelText(`Page ${page}`)
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50))
    })
    expect(props.onPageChange).not.toHaveBeenCalled()
  }
)

it('announces completion of newly mounted text layers for active search highlighting', async () => {
  const ready = vi.fn()
  document.addEventListener('pdf-text-layer-rendered', ready)
  try {
    render(
      <PdfDocument
        url="/large.pdf"
        documentId={1}
        currentPage={1}
        zoomLevel={100}
        viewMode="continuous"
        onLoadSuccess={vi.fn()}
        onLoadError={vi.fn()}
        onPageChange={vi.fn()}
      />
    )
    await screen.findByLabelText('Page 1')
    expect(ready).toHaveBeenCalled()
  } finally {
    document.removeEventListener('pdf-text-layer-rendered', ready)
  }
})

it('only mounts thumbnail canvases near the viewport and releases them when hidden', async () => {
  const callbacks: IntersectionObserverCallback[] = []
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      constructor(callback: IntersectionObserverCallback) {
        callbacks.push(callback)
      }
      observe() {}
      disconnect() {}
    }
  )
  const view = render(
    <PdfDocument
      url="/large.pdf"
      documentId={1}
      currentPage={1}
      zoomLevel={100}
      viewMode="thumbnails"
      onLoadSuccess={vi.fn()}
      onLoadError={vi.fn()}
      onPageChange={vi.fn()}
    />
  )
  await waitFor(() => expect(view.container.querySelectorAll('button').length).toBe(1000))
  expect(view.container.querySelectorAll('canvas').length).toBeLessThan(12)
  act(() =>
    callbacks[0](
      [{ isIntersecting: true } as IntersectionObserverEntry],
      {} as IntersectionObserver
    )
  )
  expect(view.container.querySelectorAll('canvas').length).toBe(1)
  act(() =>
    callbacks[0](
      [{ isIntersecting: false } as IntersectionObserverEntry],
      {} as IntersectionObserver
    )
  )
  expect(view.container.querySelectorAll('canvas').length).toBe(0)
})
