import React from 'react'
import { Button, Modal, Typography } from 'antd'
import { ZoomIn as ZoomInIcon, ZoomOut as ZoomOutIcon, Maximize2 as FitIcon } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import cytoscape, { Core } from 'cytoscape'
import dagre from 'cytoscape-dagre'
import { bgRequest } from '@/services/background-proxy'
import { useTranslation } from 'react-i18next'

cytoscape.use(dagre)

const EDGE_TYPES = 'manual,wikilink,backlink,tag_membership,source_membership'

const normalizeNoteId = (rawId: string): string => {
  if (rawId.startsWith('note:')) return rawId.slice(5)
  return rawId
}

type NotesGraphModalProps = {
  open: boolean
  noteId: string | number | null
  refreshToken?: number
  onClose: () => void
  onOpenNote: (noteId: string) => void
}

type CytoscapeResponse = {
  elements?: {
    nodes?: Array<{ data?: Record<string, any> }>
    edges?: Array<{ data?: Record<string, any> }>
  }
  truncated?: boolean
  truncated_by?: string[]
}

const NotesGraphModal: React.FC<NotesGraphModalProps> = ({
  open,
  noteId,
  refreshToken = 0,
  onClose,
  onOpenNote
}) => {
  const { t } = useTranslation(['option', 'common'])
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const cyRef = React.useRef<Core | null>(null)
  const [radius, setRadius] = React.useState<1 | 2>(1)
  const [maxNodesInput, setMaxNodesInput] = React.useState(120)
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null)
  const [refreshTick, setRefreshTick] = React.useState(0)

  const maxNodeCap = radius === 2 ? 200 : 300
  const maxNodes = Math.min(Math.max(20, Number(maxNodesInput) || 120), maxNodeCap)
  const maxEdges = Math.min(radius === 2 ? 800 : 1200, maxNodes * 4)

  React.useEffect(() => {
    setMaxNodesInput((current) => Math.min(current, maxNodeCap))
  }, [maxNodeCap])

  React.useEffect(() => {
    if (!open) {
      setSelectedNodeId(null)
    }
  }, [open])

  const { data, isLoading, isError } = useQuery({
    queryKey: ['notes-graph-modal', noteId, radius, maxNodes, maxEdges, refreshTick, refreshToken],
    enabled: open && noteId != null,
    queryFn: async (): Promise<CytoscapeResponse> => {
      const params = new URLSearchParams()
      params.set('center_note_id', String(noteId))
      params.set('radius', String(radius))
      params.set('max_nodes', String(maxNodes))
      params.set('max_edges', String(maxEdges))
      params.set('format', 'cytoscape')
      params.set('edge_types', EDGE_TYPES)
      return bgRequest<any>({
        path: `/api/v1/notes/graph?${params.toString()}` as any,
        method: 'GET' as any
      })
    }
  })

  const graphElements = React.useMemo(() => {
    const nodes = Array.isArray(data?.elements?.nodes) ? data?.elements?.nodes : []
    const edges = Array.isArray(data?.elements?.edges) ? data?.elements?.edges : []
    return [...nodes, ...edges]
  }, [data])

  const openSelectedNote = React.useCallback(() => {
    if (!selectedNodeId) return
    onOpenNote(selectedNodeId)
    onClose()
  }, [onClose, onOpenNote, selectedNodeId])

  React.useEffect(() => {
    if (!open || !containerRef.current || graphElements.length === 0) return

    cyRef.current?.destroy()

    const cy = cytoscape({
      container: containerRef.current,
      elements: graphElements as any,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'font-size': '10px',
            color: '#111827',
            'text-wrap': 'ellipsis',
            'text-max-width': '120px',
            width: 34,
            height: 34,
            'background-color': '#6b7280'
          } as any
        },
        {
          selector: 'node[type="note"]',
          style: {
            'background-color': '#2563eb',
            color: '#ffffff'
          } as any
        },
        {
          selector: 'node[type="tag"]',
          style: {
            'background-color': '#0f766e',
            shape: 'round-rectangle'
          } as any
        },
        {
          selector: 'node[type="source"]',
          style: {
            'background-color': '#7c3aed',
            shape: 'diamond',
            color: '#ffffff'
          } as any
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#22c55e'
          } as any
        },
        {
          selector: 'edge',
          style: {
            width: 1.5,
            'line-color': '#94a3b8',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#94a3b8',
            label: 'data(type)',
            'font-size': '8px',
            color: '#64748b'
          } as any
        }
      ],
      layout: {
        name: 'dagre',
        rankDir: 'LR',
        nodeSep: 45,
        rankSep: 120,
        animate: false
      } as any,
      minZoom: 0.2,
      maxZoom: 2,
      wheelSensitivity: 0.2,
      boxSelectionEnabled: false,
      autounselectify: false
    })

    cy.on('tap', 'node', (event) => {
      const node = event.target
      const rawId = String(node.data('id') || '')
      const type = String(node.data('type') || '')
      if (type === 'note' || rawId.startsWith('note:')) {
        setSelectedNodeId(normalizeNoteId(rawId))
      } else {
        setSelectedNodeId(null)
      }
    })

    cy.fit(undefined, 40)
    cyRef.current = cy

    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [graphElements, open])

  const handleZoomIn = React.useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    const next = Math.min(2, cy.zoom() * 1.2)
    cy.zoom(next)
  }, [])

  const handleZoomOut = React.useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    const next = Math.max(0.2, cy.zoom() / 1.2)
    cy.zoom(next)
  }, [])

  const handleFit = React.useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.fit(undefined, 40)
  }, [])

  return (
    <Modal
      open={open}
      onCancel={onClose}
    onOk={openSelectedNote}
    okText={t('option:notesSearch.graphOpenSelected', { defaultValue: 'Open selected note' })}
    okButtonProps={{ disabled: !selectedNodeId }}
    width={1024}
    title={t('option:notesSearch.graphViewTitle', { defaultValue: 'Notes graph view' })}
    aria-label={t('option:notesSearch.graphViewTitle', { defaultValue: 'Notes graph view' })}
    keyboard
    destroyOnHidden
  >
      <div className="flex flex-wrap items-end gap-3 mb-3">
        <label className="flex flex-col text-xs text-text-muted gap-1">
          {t('option:notesSearch.graphRadiusLabel', { defaultValue: 'Radius' })}
          <select
            value={radius}
            onChange={(event) => setRadius(Number(event.target.value) === 2 ? 2 : 1)}
            className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
            data-testid="notes-graph-radius-control"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label className="flex flex-col text-xs text-text-muted gap-1">
          {t('option:notesSearch.graphMaxNodesLabel', { defaultValue: 'Max nodes' })}
          <input
            type="number"
            min={20}
            max={maxNodeCap}
            value={maxNodesInput}
            onChange={(event) => setMaxNodesInput(Number(event.target.value))}
            className="w-24 rounded border border-border bg-surface px-2 py-1 text-sm text-text"
            data-testid="notes-graph-max-nodes-control"
          />
        </label>
        <Button
          size="small"
          onClick={() => setRefreshTick((current) => current + 1)}
          data-testid="notes-graph-refresh"
        >
          {t('common:refresh', { defaultValue: 'Refresh' })}
        </Button>
        <Button
          size="small"
          icon={(<ZoomInIcon className="w-4 h-4" />) as any}
          onClick={handleZoomIn}
          aria-label={t('option:notesSearch.graphZoomIn', { defaultValue: 'Zoom in' })}
          data-testid="notes-graph-zoom-in"
        />
        <Button
          size="small"
          icon={(<ZoomOutIcon className="w-4 h-4" />) as any}
          onClick={handleZoomOut}
          aria-label={t('option:notesSearch.graphZoomOut', { defaultValue: 'Zoom out' })}
          data-testid="notes-graph-zoom-out"
        />
        <Button
          size="small"
          icon={(<FitIcon className="w-4 h-4" />) as any}
          onClick={handleFit}
          aria-label={t('option:notesSearch.graphFit', { defaultValue: 'Fit graph to view' })}
          data-testid="notes-graph-fit"
        />
      </div>

      <Typography.Text type="secondary" className="block text-xs mb-2">
        {t('option:notesSearch.graphHints', {
          defaultValue: 'Drag to pan. Scroll to zoom. Select a note node, then open it.'
        })}
      </Typography.Text>

      {(data?.truncated ?? false) && (
        <Typography.Text
          type="warning"
          className="block text-xs mb-2"
          data-testid="notes-graph-truncated-warning"
        >
          {t('option:notesSearch.graphTruncatedWarning', {
            defaultValue: 'Graph was truncated by server limits. Increase focus or reduce radius.'
          })}
        </Typography.Text>
      )}

      {isLoading ? (
        <div
          className="h-[520px] rounded border border-border bg-surface2 flex items-center justify-center text-sm text-text-muted"
          data-testid="notes-graph-loading"
        >
          {t('option:notesSearch.graphLoading', { defaultValue: 'Loading graph...' })}
        </div>
      ) : isError ? (
        <div
          className="h-[520px] rounded border border-border bg-surface2 flex items-center justify-center text-sm text-danger"
          data-testid="notes-graph-error"
        >
          {t('option:notesSearch.graphError', { defaultValue: 'Failed to load graph view.' })}
        </div>
      ) : (
        <div
          className="h-[520px] rounded border border-border bg-surface2"
          ref={containerRef}
          data-testid="notes-graph-canvas"
        />
      )}
    </Modal>
  )
}

export default NotesGraphModal
