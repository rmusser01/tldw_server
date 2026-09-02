import type {
  NotesGraphEdgeType,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { getComputedTokens } from "@/themes/runtime-tokens"
import cytoscape, { type Core, type ElementDefinition } from "cytoscape"
import dagre from "cytoscape-dagre"
import React from "react"
import { useTranslation } from "react-i18next"

import type { ProvisionalNotesGraphOverlay } from "./hooks/useNotesGraphSuggestions"
import type { NotesGraphLayout } from "./hooks/useNotesGraphWorkspace"
import {
  getNotesGraphEdgeLabel,
  groupNotesGraphEdgesByPair,
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
  onSelectEdge?: (edgeId: string) => void
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
      onSelectNode,
      onSelectEdge
    },
    ref
  ) => {
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const cyRef = React.useRef<Core | null>(null)
    const focusNoteIdRef = React.useRef(focusNoteId)
    const selectedNodeIdRef = React.useRef(selectedNodeId)
    const onSelectNodeRef = React.useRef(onSelectNode)
    const onSelectEdgeRef = React.useRef(onSelectEdge)
    const { t } = useTranslation("option")
    focusNoteIdRef.current = focusNoteId
    selectedNodeIdRef.current = selectedNodeId
    onSelectNodeRef.current = onSelectNode
    onSelectEdgeRef.current = onSelectEdge

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

      groupNotesGraphEdgesByPair(
        graph.edges.filter((edge) => visibleEdgeTypes.has(edge.type))
      ).forEach((group) => {
        const edgeTypes = group.edges.map((edge) => edge.type)
        const edgeIds = group.edges.map((edge) => edge.id)
        const manual = group.edges.find((edge) => edge.type === "manual")
        const semantic = group.edges.find((edge) => edge.type === "semantic")
        const representative = manual ?? group.edges[0]
        const displayTypes = manual
          ? group.edges.filter((edge) => edge.type !== "semantic")
          : group.edges
        const displayLabel =
          displayTypes.length === 1 && semantic && !manual
            ? `${semantic.evidence?.similarity.toFixed(3) ?? semantic.weight?.toFixed(3) ?? ""} ${t(
                "notesSearch.graphPassageSimilarityShort",
                { defaultValue: "passage similarity" }
              )}`
            : displayTypes
                .map(
                  (edge) =>
                    edge.label?.trim() || getNotesGraphEdgeLabel(edge.type)
                )
                .join(" · ")
        elements.push({
          data: {
            ...representative,
            id: group.id,
            source: representative.source,
            target: representative.target,
            directed: String(representative.directed),
            displayLabel,
            edgeIds,
            edgeTypes,
            primaryEdgeId: representative.id
          },
          classes: semantic && !manual ? "semantic" : undefined
        })
      })

      if (showProvisional) {
        provisionalOverlays.forEach((overlay) => {
          if (overlay.node) {
            elements.push({
              data: { ...overlay.node },
              classes: "provisional",
              selectable: false
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
            selector: "edge.semantic",
            style: {
              "line-color": tokens.accent,
              "line-style": "dotted",
              width: 2
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

      cy.on("tap", "node", (event) => {
        if (event.target.hasClass("provisional")) return
        onSelectNodeRef.current(event.target.id())
      })
      cy.on("tap", "edge", (event) => {
        if (event.target.hasClass("provisional")) return
        const edgeId = event.target.data("primaryEdgeId")
        if (typeof edgeId === "string") onSelectEdgeRef.current?.(edgeId)
      })
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
      <div className="relative h-full min-w-0">
        <div
          ref={containerRef}
          className="h-full w-full bg-bg"
          data-testid="notes-graph-canvas"
          role="img"
          aria-label={t("notesSearch.graphCanvasAria", {
            defaultValue: "Notes graph canvas"
          })}
        />
        <ul
          data-testid="notes-graph-edge-legend"
          aria-label={t("notesSearch.graphEdgeLegend", {
            defaultValue: "Relationship legend"
          })}
          className="absolute bottom-2 left-2 flex max-w-[calc(100%-1rem)] flex-wrap gap-x-3 gap-y-1 border border-border bg-elevated px-2 py-1 text-xs text-text shadow-sm motion-reduce:transition-none">
          <li className="flex items-center gap-1.5">
            <span
              className="w-5 border-t-2 border-text-muted"
              aria-hidden="true"
            />
            {t("notesSearch.graphLegendAuthoritative", {
              defaultValue: "Authoritative"
            })}
          </li>
          <li className="flex items-center gap-1.5">
            <span
              className="w-5 border-t-2 border-dotted border-accent"
              aria-hidden="true"
            />
            {t("notesSearch.graphSimilarContent", {
              defaultValue: "Similar content"
            })}
          </li>
          <li className="flex items-center gap-1.5">
            <span
              className="w-5 border-t-2 border-dashed border-warn"
              aria-hidden="true"
            />
            {t("notesSearch.graphSuggestions", { defaultValue: "Suggestions" })}
          </li>
        </ul>
      </div>
    )
  }
)

NotesGraphCanvas.displayName = "NotesGraphCanvas"

export default NotesGraphCanvas
