import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import NotesGraphModal from '../NotesGraphModal'

const {
  mockBgRequest,
  mockCytoscapeFactory,
  mockCyInstance,
  cyHandlers,
  resetCyState
} = vi.hoisted(() => {
  const handlers: Record<string, (event: any) => void> = {}
  let zoomLevel = 1

  const instance: Record<string, any> = {}
  instance.on = vi.fn((event: string, selectorOrHandler: any, maybeHandler?: any) => {
    if (typeof selectorOrHandler === 'string') {
      handlers[`${event}:${selectorOrHandler}`] = maybeHandler
    } else {
      handlers[event] = selectorOrHandler
    }
    return instance
  })
  instance.fit = vi.fn()
  instance.destroy = vi.fn()
  instance.zoom = vi.fn((next?: number) => {
    if (typeof next === 'number') {
      zoomLevel = next
      return instance
    }
    return zoomLevel
  })

  const factory: any = vi.fn(() => instance)
  factory.use = vi.fn()

  return {
    mockBgRequest: vi.fn(),
    mockCytoscapeFactory: factory,
    mockCyInstance: instance,
    cyHandlers: handlers,
    resetCyState: () => {
      zoomLevel = 1
      Object.keys(handlers).forEach((key) => delete handlers[key])
    }
  }
})

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === 'string') return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mockBgRequest
}))

vi.mock('cytoscape', () => ({
  default: mockCytoscapeFactory
}))

vi.mock('cytoscape-dagre', () => ({
  default: {}
}))

const graphPayload = {
  elements: {
    nodes: [
      { data: { id: 'note:note-a', type: 'note', label: 'Current note' } },
      { data: { id: 'note:note-b', type: 'note', label: 'Linked note' } },
      { data: { id: 'tag:research', type: 'tag', label: 'research' } }
    ],
    edges: [
      { data: { id: 'e1', source: 'note:note-a', target: 'note:note-b', type: 'manual' } },
      { data: { id: 'e2', source: 'note:note-a', target: 'tag:research', type: 'tag_membership' } }
    ]
  },
  truncated: false
}

const renderModal = (props?: Partial<React.ComponentProps<typeof NotesGraphModal>>) => {
  const onClose = vi.fn()
  const onOpenNote = vi.fn()
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  render(
    <QueryClientProvider client={queryClient}>
      <NotesGraphModal
        open
        noteId="note-a"
        onClose={onClose}
        onOpenNote={onOpenNote}
        {...props}
      />
    </QueryClientProvider>
  )

  return { onClose, onOpenNote }
}

describe('NotesGraphModal stage 2 graph view', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetCyState()
    mockBgRequest.mockResolvedValue(graphPayload)
  })

  it('fetches graph with controls and clamps max nodes for radius 2', async () => {
    renderModal()

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalled()
    })

    const firstPath = String(mockBgRequest.mock.calls[0]?.[0]?.path || '')
    expect(firstPath).toContain('/api/v1/notes/graph?')
    expect(firstPath).toContain('center_note_id=note-a')
    expect(firstPath).toContain('radius=1')
    expect(firstPath).toContain('max_nodes=120')
    expect(decodeURIComponent(firstPath)).toContain('edge_types=manual,wikilink,backlink,tag_membership,source_membership')

    fireEvent.change(screen.getByTestId('notes-graph-max-nodes-control'), {
      target: { value: '280' }
    })
    fireEvent.change(screen.getByTestId('notes-graph-radius-control'), {
      target: { value: '2' }
    })
    fireEvent.click(screen.getByTestId('notes-graph-refresh'))

    await waitFor(() => {
      const lastPath = String(mockBgRequest.mock.calls.at(-1)?.[0]?.path || '')
      expect(lastPath).toContain('radius=2')
      expect(lastPath).toContain('max_nodes=200')
    })
  })

  it('renders loading -> canvas lifecycle and supports zoom controls', async () => {
    let resolveGraph: ((payload: any) => void) | null = null
    const deferred = new Promise((resolve) => {
      resolveGraph = resolve as (payload: any) => void
    })
    mockBgRequest.mockImplementationOnce(async () => deferred)

    renderModal()

    expect(screen.getByTestId('notes-graph-loading')).toBeInTheDocument()

    resolveGraph?.(graphPayload)

    await waitFor(() => {
      expect(screen.getByTestId('notes-graph-canvas')).toBeInTheDocument()
      expect(mockCytoscapeFactory).toHaveBeenCalled()
    })

    const fitCountBefore = mockCyInstance.fit.mock.calls.length
    fireEvent.click(screen.getByTestId('notes-graph-zoom-in'))
    fireEvent.click(screen.getByTestId('notes-graph-zoom-out'))
    fireEvent.click(screen.getByTestId('notes-graph-fit'))

    expect(mockCyInstance.zoom).toHaveBeenCalled()
    expect(mockCyInstance.fit.mock.calls.length).toBeGreaterThan(fitCountBefore)
  })

  it('opens selected note when a note node is tapped and confirmed', async () => {
    const { onClose, onOpenNote } = renderModal()

    await waitFor(() => {
      expect(screen.getByTestId('notes-graph-canvas')).toBeInTheDocument()
    })

    await act(async () => {
      cyHandlers['tap:node']?.({
        target: {
          data: (key: string) => {
            if (key === 'id') return 'note:note-b'
            if (key === 'type') return 'note'
            return undefined
          }
        }
      })
    })

    const openButton = screen.getByRole('button', { name: 'Open selected note' })
    expect(openButton).not.toBeDisabled()

    fireEvent.click(openButton)

    expect(onOpenNote).toHaveBeenCalledWith('note-b')
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on Escape key', async () => {
    const { onClose } = renderModal()

    await waitFor(() => {
      expect(screen.getByTestId('notes-graph-canvas')).toBeInTheDocument()
    })

    const dialog = screen.getByRole('dialog')
    fireEvent.keyDown(dialog, { key: 'Escape' })

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled()
    })
  })

  it('renders error state when graph request fails', async () => {
    mockBgRequest.mockRejectedValueOnce(new Error('graph unavailable'))

    renderModal()

    expect(await screen.findByTestId('notes-graph-error')).toHaveTextContent(
      'Failed to load graph view.'
    )
  })
})
