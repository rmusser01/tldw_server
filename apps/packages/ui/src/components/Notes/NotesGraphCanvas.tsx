import type {
  NotesGraphEdgeType,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { getComputedTokens } from "@/themes/runtime-tokens"
import cytoscape, { type Core, type ElementDefinition } from "cytoscape"
import dagre from "cytoscape-dagre"
import React from "react"

import type { ProvisionalNotesGraphOverlay } from "./hooks/useNotesGraphSuggestions"
import type { NotesGraphLayout } from "./hooks/useNotesGraphWorkspace"
import {
  getNotesGraphEdgeLabel,
  normalizeGraphNoteId
} from "./notes-manager-utils"

cytoscape.use(dagre)

export type NotesGraphCanvasHandle = {
  zoomIn: () => void
  zoomOut: () => void
  fit: () => void
  focusNode: (nodeId: string) => void
}

type NotesGraphCanvasProps = {
  graph: NotesGraphResponse
  layout: NotesGraphLayout
  focusNoteId: string | null
  selectedNodeId: string | null
  visibleEdgeTypes: ReadonlySet<NotesGraphEdgeType>
  provisionalOverlays: ProvisionalNotesGraphOverlay[]
  showProvisional: boolean
  onSelectNode: (nodeId: string) => void
}

const layoutOptions = (layout: NotesGraphLayout) =>
  layout === "dagre"
    ? {
        name: "dagre",
        rankDir: "LR",
        nodeSep: 45,
        rankSep: 120,
        animate: false
      }
    : { name: layout, animate: false, padding: 40 }

const NotesGraphCanvas = React.forwardRef<
  NotesGraphCanvasHandle,
  NotesGraphCanvasProps
>(
  (
    {
      graph,
      layout,
      focusNoteId,
      selectedNodeId,
      visibleEdgeTypes,
      provisionalOverlays,
      showProvisional,
      onSelectNode
    },
    ref
  ) => {
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const cyRef = React.useRef<Core | null>(null)
    const focusNoteIdRef = React.useRef(focusNoteId)
    const selectedNodeIdRef = React.useRef(selectedNodeId)
    focusNoteIdRef.current = focusNoteId
    selectedNodeIdRef.current = selectedNodeId

    const syncNodeState = React.useCallback((cy: Core) => {
      if (typeof cy.$id !== "function") return
      cy.nodes().removeClass("graph-focus-node graph-label-visible").unselect()
      if (focusNoteIdRef.current) {
        cy.$id(`note:${normalizeGraphNoteId(focusNoteIdRef.current)}`).addClass(
          "graph-focus-node graph-label-visible"
        )
      }
      if (selectedNodeIdRef.current) cy.$id(selectedNodeIdRef.current).select()
    }, [])

    React.useImperativeHandle(
      ref,
      () => ({
        zoomIn: () => {
          const cy = cyRef.current
          if (cy) cy.zoom(Math.min(2, cy.zoom() * 1.2))
        },
        zoomOut: () => {
          const cy = cyRef.current
          if (cy) cy.zoom(Math.max(0.2, cy.zoom() / 1.2))
        },
        fit: () => cyRef.current?.fit(undefined, 40),
        focusNode: (nodeId) => {
          const cy = cyRef.current
          if (!cy) return
          const node = cy.getElementById(nodeId)
          if (node.nonempty()) {
            node.select()
            cy.center(node)
          }
        }
      }),
      []
    )

    React.useEffect(() => {
      if (!containerRef.current) return

      const tokens = getComputedTokens()
      const elements: ElementDefinition[] = graph.nodes.map((node) => ({
        data: { ...node }
      }))

      graph.edges.forEach((edge) => {
        if (!visibleEdgeTypes.has(edge.type)) return
        elements.push({
          data: {
            ...edge,
            directed: String(edge.directed),
            displayLabel:
              edge.label?.trim() || getNotesGraphEdgeLabel(edge.type)
          }
        })
      })

      if (showProvisional) {
        provisionalOverlays.forEach((overlay) => {
          if (overlay.node) {
            elements.push({
              data: { ...overlay.node },
              classes: "provisional"
            })
          }
          elements.push({
            data: {
              ...overlay.edge,
              directed: "false",
              displayLabel: "Suggestion"
            },
            classes: "provisional"
          })
        })
      }

      cyRef.current?.destroy()
      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: "node",
            style: {
              label: "",
              width: 34,
              height: 34,
              "background-color": tokens.muted,
              "border-color": tokens.borderStrong,
              "border-width": 1,
              color: tokens.text,
              "font-size": "10px",
              "text-wrap": "ellipsis",
              "text-max-width": "120px",
              "text-background-color": tokens.surface,
              "text-background-opacity": 0.92,
              "text-background-padding": "3px"
            }
          },
          {
            selector: 'node[type="note"]',
            style: { "background-color": tokens.primary }
          },
          {
            selector: 'node[type="tag"]',
            style: {
              "background-color": tokens.accent,
              shape: "round-rectangle"
            }
          },
          {
            selector: 'node[type="source"]',
            style: { "background-color": tokens.surface2, shape: "diamond" }
          },
          {
            selector:
              "node.graph-label-visible, node.graph-label-hovered, node:selected",
            style: { label: "data(label)" }
          },
          {
            selector: "node.graph-focus-node",
            style: { "border-width": 3, "border-style": "double" }
          },
          {
            selector: "node:selected",
            style: { "border-width": 4, "border-color": tokens.focus }
          },
          {
            selector: "edge",
            style: {
              width: 1.5,
              "line-color": tokens.borderStrong,
              "curve-style": "bezier",
              "target-arrow-shape": "none",
              "target-arrow-color": tokens.borderStrong,
              label: "data(displayLabel)",
              "font-size": "8px",
              color: tokens.textMuted,
              "text-background-color": tokens.surface,
              "text-background-opacity": 0.9
            }
          },
          {
            selector: 'edge[directed="true"]',
            style: { "target-arrow-shape": "triangle" }
          },
          {
            selector: "node.provisional",
            style: {
              "background-color": tokens.surface2,
              "border-color": tokens.warn,
              "border-style": "dashed",
              "border-width": 2
            }
          },
          {
            selector: "edge.provisional",
            style: {
              "line-color": tokens.warn,
              "line-style": "dashed",
              "target-arrow-shape": "none"
            }
          }
        ] as cytoscape.StylesheetJson,
        layout: layoutOptions(layout) as cytoscape.LayoutOptions,
        minZoom: 0.2,
        maxZoom: 2,
        wheelSensitivity: 0.2,
        boxSelectionEnabled: false,
        autounselectify: false
      })

      cy.on("tap", "node", (event) => onSelectNode(event.target.id()))
      cy.on("mouseover", "node", (event) =>
        event.target.addClass("graph-label-hovered")
      )
      cy.on("mouseout", "node", (event) =>
        event.target.removeClass("graph-label-hovered")
      )
      syncNodeState(cy)
      cy.fit(undefined, 40)
      cyRef.current = cy

      return () => {
        cy.destroy()
        if (cyRef.current === cy) cyRef.current = null
      }
    }, [
      graph,
      layout,
      onSelectNode,
      provisionalOverlays,
      showProvisional,
      syncNodeState,
      visibleEdgeTypes
    ])

    React.useEffect(() => {
      const cy = cyRef.current
      if (cy) syncNodeState(cy)
    }, [focusNoteId, selectedNodeId, syncNodeState])

    return (
      <div
        ref={containerRef}
        className="h-full min-h-[420px] w-full bg-bg sm:min-h-[520px]"
        data-testid="notes-graph-canvas"
        role="img"
        aria-label="Notes graph canvas"
      />
    )
  }
)

NotesGraphCanvas.displayName = "NotesGraphCanvas"

export default NotesGraphCanvas
