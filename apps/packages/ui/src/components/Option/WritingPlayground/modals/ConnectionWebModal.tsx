import { useEffect, useRef } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button, Modal } from "antd"
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react"
import cytoscape, {
  type Core,
  type ElementDefinition,
  type LayoutOptions,
  type StylesheetJson,
} from "cytoscape"
import dagre from "cytoscape-dagre"
import { EmptyState } from "@/components/ui/feedback/EmptyState"
import { LoadingState } from "@/components/ui/feedback/LoadingState"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  listManuscriptCharacters,
  listManuscriptRelationships,
  listManuscriptWorldInfo,
  type ManuscriptCharacterListResponse,
  type ManuscriptRelationshipListResponse,
  type ManuscriptWorldInfoListResponse,
} from "@/services/writing-playground"

cytoscape.use(dagre)

type ConnectionWebModalProps = {
  open: boolean
  onClose: () => void
}

const NODE_COLORS: Record<string, string> = {
  character: "#2563eb",
  location: "#ea580c",
  faction: "#16a34a",
  item: "#7c3aed",
}

export function ConnectionWebModal({ open, onClose }: ConnectionWebModalProps) {
  const { activeProjectId } = useWritingPlaygroundStore()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<Core | null>(null)

  const { data: charsData, isLoading: charsLoading } = useQuery<ManuscriptCharacterListResponse>({
    queryKey: ["manuscript-characters", activeProjectId],
    queryFn: () => listManuscriptCharacters(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const { data: relsData, isLoading: relsLoading } = useQuery<ManuscriptRelationshipListResponse>({
    queryKey: ["manuscript-relationships", activeProjectId],
    queryFn: () => listManuscriptRelationships(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const { data: worldData, isLoading: worldLoading } = useQuery<ManuscriptWorldInfoListResponse>({
    queryKey: ["manuscript-world-info", activeProjectId],
    queryFn: () => listManuscriptWorldInfo(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const isLoading = charsLoading || relsLoading || worldLoading

  const characters = charsData?.characters || []
  const relationships = relsData?.relationships || []
  const worldItems = worldData?.items || []

  useEffect(() => {
    if (!open || !containerRef.current || isLoading) return
    if (characters.length === 0 && worldItems.length === 0) return

    cyRef.current?.destroy()

    const nodes: ElementDefinition[] = []
    const edges: ElementDefinition[] = []

    for (const ch of characters) {
      nodes.push({
        data: {
          id: `char-${ch.id}`,
          label: ch.name || ch.id,
          type: "character",
        },
      })
    }

    for (const wi of worldItems) {
      nodes.push({
        data: {
          id: `world-${wi.id}`,
          label: wi.name || wi.id,
          type: wi.kind || "item",
        },
      })
    }

    for (const rel of relationships) {
      const sourceCharacterId = rel.from_character_id || rel.character_a_id
      const targetCharacterId = rel.to_character_id || rel.character_b_id
      if (!sourceCharacterId || !targetCharacterId) {
        continue
      }
      edges.push({
        data: {
          id: `rel-${rel.id}`,
          source: `char-${sourceCharacterId}`,
          target: `char-${targetCharacterId}`,
          type: rel.relationship_type || "related",
        },
      })
    }

    const styles: StylesheetJson = [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": "10px",
          color: "#111827",
          "text-wrap": "ellipsis",
          "text-max-width": "120px",
          width: 34,
          height: 34,
          "background-color": "#6b7280",
        },
      },
      {
        selector: 'node[type="character"]',
        style: { "background-color": NODE_COLORS.character },
      },
      {
        selector: 'node[type="location"]',
        style: {
          "background-color": NODE_COLORS.location,
          shape: "round-rectangle",
        },
      },
      {
        selector: 'node[type="faction"]',
        style: {
          "background-color": NODE_COLORS.faction,
          shape: "diamond",
        },
      },
      {
        selector: 'node[type="item"]',
        style: {
          "background-color": NODE_COLORS.item,
          shape: "round-triangle",
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.5,
          "line-color": "#94a3b8",
          "curve-style": "bezier",
          "target-arrow-shape": "triangle",
          "target-arrow-color": "#94a3b8",
          label: "data(type)",
          "font-size": "8px",
          color: "#64748b",
        },
      },
    ]

    const layout: LayoutOptions & {
      rankDir: "LR"
      nodeSep: number
      rankSep: number
      animate: false
    } = {
      name: "dagre",
      rankDir: "LR",
      nodeSep: 50,
      rankSep: 100,
      animate: false,
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...nodes, ...edges],
      style: styles,
      layout,
      minZoom: 0.2,
      maxZoom: 2,
      wheelSensitivity: 0.2,
      boxSelectionEnabled: false,
    })

    cy.fit(undefined, 40)
    cyRef.current = cy

    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [open, isLoading, characters, relationships, worldItems])

  const handleZoomIn = () => {
    const cy = cyRef.current
    if (!cy) return
    cy.zoom(Math.min(2, cy.zoom() * 1.2))
  }

  const handleZoomOut = () => {
    const cy = cyRef.current
    if (!cy) return
    cy.zoom(Math.max(0.2, cy.zoom() / 1.2))
  }

  const handleFit = () => {
    cyRef.current?.fit(undefined, 40)
  }

  const hasData = characters.length > 0 || worldItems.length > 0

  return (
    <Modal title="Connection Web" open={open} onCancel={onClose} footer={null} width={900}>
      {!activeProjectId ? (
        <EmptyState
          variant="inline"
          size="md"
          title="Select a project first"
          className="py-10"
        />
      ) : isLoading ? (
        <div className="h-[520px] flex items-center justify-center">
          <LoadingState mode="spinner" size="md" />
        </div>
      ) : !hasData ? (
        <EmptyState
          variant="inline"
          size="md"
          title="Add characters or world info to visualize connections"
          className="py-10"
        />
      ) : (
        <>
          <div className="flex items-center gap-2 mb-2">
            <Button
              size="small"
              icon={<ZoomIn className="w-4 h-4" />}
              onClick={handleZoomIn}
              aria-label="Zoom in"
            />
            <Button
              size="small"
              icon={<ZoomOut className="w-4 h-4" />}
              onClick={handleZoomOut}
              aria-label="Zoom out"
            />
            <Button
              size="small"
              icon={<Maximize2 className="w-4 h-4" />}
              onClick={handleFit}
              aria-label="Fit to view"
            />
            <span className="text-xs text-text-muted ml-2">
              Drag to pan. Scroll to zoom.
            </span>
          </div>
          <div
            className="h-[520px] rounded border border-border bg-surface2"
            ref={containerRef}
          />
        </>
      )}
    </Modal>
  )
}

export default ConnectionWebModal
