import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphCanvas, {
  type NotesGraphCanvasHandle
} from "../NotesGraphCanvas"
import NotesGraphToolbar from "../NotesGraphToolbar"

type CytoscapeTestEvent = {
  target: {
    id: () => string
    data: (key: string) => unknown
    hasClass: (name: string) => boolean
  }
}

type CytoscapeTestElement = {
  data: Record<string, unknown>
  selectable?: boolean
}

type CytoscapeTestStyle = {
  selector: string
  style: Record<string, unknown>
}

type CytoscapeTestConfig = {
  elements: CytoscapeTestElement[]
  style: CytoscapeTestStyle[]
  layout: Record<string, unknown>
}

const { cyHandlers, mockCytoscapeFactory, mockCyInstance, resetCyState } =
  vi.hoisted(() => {
    const handlers: Record<string, (event: CytoscapeTestEvent) => void> = {}
    let zoomLevel = 1
    let panPosition = { x: 0, y: 0 }
    const instance = {
      on: vi.fn(),
      fit: vi.fn(),
      destroy: vi.fn(),
      zoom: vi.fn(),
      pan: vi.fn()
    }
    instance.on.mockImplementation(
      (
        event: string,
        selectorOrHandler: string | ((event: CytoscapeTestEvent) => void),
        maybeHandler?: (event: CytoscapeTestEvent) => void
      ) => {
        handlers[
          typeof selectorOrHandler === "string"
            ? `${event}:${selectorOrHandler}`
            : event
        ] =
          typeof selectorOrHandler === "string"
            ? (maybeHandler as (event: CytoscapeTestEvent) => void)
            : selectorOrHandler
        return instance
      }
    )
    instance.zoom.mockImplementation((next?: number) => {
      if (typeof next === "number") {
        zoomLevel = next
        return instance
      }
      return zoomLevel
    })
    instance.pan.mockImplementation((next?: { x: number; y: number }) => {
      if (next) {
        panPosition = next
        return instance
      }
      return panPosition
    })

    const factory = Object.assign(
      vi.fn((_config: CytoscapeTestConfig) => instance),
      { use: vi.fn() }
    )

    return {
      cyHandlers: handlers,
      mockCytoscapeFactory: factory,
      mockCyInstance: instance,
      resetCyState: () => {
        zoomLevel = 1
        panPosition = { x: 0, y: 0 }
        Object.keys(handlers).forEach((key) => delete handlers[key])
      }
    }
  })

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: string | { defaultValue?: string; [key: string]: unknown }
    ) => (typeof options === "string" ? options : options?.defaultValue ?? key)
  })
}))

vi.mock("cytoscape", () => ({
  default: mockCytoscapeFactory
}))

vi.mock("cytoscape-dagre", () => ({
  default: {}
}))

vi.mock("@/themes/runtime-tokens", () => ({
  getComputedTokens: () => ({
    bg: "#101010",
    surface: "#202020",
    surface2: "#303030",
    elevated: "#404040",
    primary: "#505050",
    primaryStrong: "#606060",
    accent: "#707070",
    success: "#808080",
    warn: "#909090",
    danger: "#a0a0a0",
    muted: "#b0b0b0",
    border: "#c0c0c0",
    borderStrong: "#d0d0d0",
    text: "#e0e0e0",
    textMuted: "#e1e1e1",
    textSubtle: "#e2e2e2",
    focus: "#f0f0f0"
  })
}))

const graph = {
  nodes: [
    {
      id: "note:a",
      type: "note" as const,
      label: "Current note",
      created_at: null,
      deleted: false,
      degree: 2,
      tag_count: 1,
      primary_source_id: null
    },
    {
      id: "note:b",
      type: "note" as const,
      label: "Linked note",
      created_at: null,
      deleted: false,
      degree: 1,
      tag_count: 0,
      primary_source_id: null
    },
    {
      id: "tag:research",
      type: "tag" as const,
      label: "Research",
      created_at: null,
      deleted: null,
      degree: 1,
      tag_count: 2,
      primary_source_id: null
    }
  ],
  edges: [
    {
      id: "edge:manual",
      source: "note:a",
      target: "note:b",
      type: "manual" as const,
      directed: false,
      weight: 1,
      label: null
    },
    {
      id: "edge:wikilink",
      source: "note:b",
      target: "note:a",
      type: "wikilink" as const,
      directed: true,
      weight: 1,
      label: null
    }
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 2,
  all_notes_note_cap: 100,
  all_notes_eligible: true
}

const overlays = [
  {
    edge: {
      id: "suggestion-edge:s1",
      suggestionId: "s1",
      source: "note:a",
      target: "suggestion-node:s1",
      type: "provisional_suggestion" as const,
      directed: false as const
    },
    node: {
      id: "suggestion-node:s1",
      suggestionId: "s1",
      type: "provisional_note" as const,
      label: "Suggested note" as const
    }
  }
]

describe("NotesGraphCanvas graph view", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetCyState()
  })

  it("preserves the Cytoscape lifecycle, readable labels, zoom, fit, and note selection", async () => {
    const onSelectNode = vi.fn()
    const ref = React.createRef<NotesGraphCanvasHandle>()
    const { unmount } = render(
      <NotesGraphCanvas
        ref={ref}
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId={null}
        visibleEdgeTypes={new Set(["manual", "wikilink"])}
        provisionalOverlays={[]}
        showProvisional
        onSelectNode={onSelectNode}
      />
    )

    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalled())
    const config = mockCytoscapeFactory.mock.calls[0]?.[0]
    const edgeElements = config.elements.filter(
      (element) => element.data.source && element.data.target
    )
    expect(edgeElements.map((edge) => edge.data.displayLabel)).toEqual([
      "Manual link",
      "Note link"
    ])
    expect(config.layout).toMatchObject({ name: "dagre", animate: false })

    const baseNodeStyle = config.style.find(
      (entry) => entry.selector === "node"
    )
    const visibleLabelStyle = config.style.find((entry) =>
      String(entry.selector).includes("graph-label-visible")
    )
    const baseEdgeStyle = config.style.find(
      (entry) => entry.selector === "edge"
    )
    const directedEdgeStyle = config.style.find(
      (entry) => entry.selector === 'edge[directed="true"]'
    )
    expect(baseNodeStyle.style.label).toBe("")
    expect(visibleLabelStyle.style.label).toBe("data(label)")
    expect(baseEdgeStyle.style["target-arrow-shape"]).toBe("none")
    expect(directedEdgeStyle.style["target-arrow-shape"]).toBe("triangle")

    await act(async () => {
      cyHandlers["tap:node"]?.({
        target: {
          id: () => "note:b",
          data: (key: string) => (key === "type" ? "note" : undefined),
          hasClass: () => false
        }
      })
    })
    expect(onSelectNode).toHaveBeenCalledWith("note:b")

    const fitCount = mockCyInstance.fit.mock.calls.length
    act(() => {
      ref.current?.zoomIn()
      ref.current?.zoomOut()
      ref.current?.fit()
    })
    expect(mockCyInstance.zoom).toHaveBeenCalled()
    expect(mockCyInstance.fit.mock.calls.length).toBeGreaterThan(fitCount)

    unmount()
    expect(mockCyInstance.destroy).toHaveBeenCalled()
  })

  it("filters authoritative edges and renders provisional nodes and edges with non-color styling", async () => {
    render(
      <NotesGraphCanvas
        graph={graph}
        layout="grid"
        focusNoteId="a"
        selectedNodeId="suggestion-node:s1"
        visibleEdgeTypes={new Set(["manual"])}
        provisionalOverlays={overlays}
        showProvisional
        onSelectNode={vi.fn()}
      />
    )

    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalled())
    const config = mockCytoscapeFactory.mock.calls[0]?.[0]
    const elementIds = config.elements.map((element) => element.data.id)
    expect(elementIds).toContain("edge:manual")
    expect(elementIds).not.toContain("edge:wikilink")
    expect(elementIds).toContain("suggestion-edge:s1")
    expect(elementIds).toContain("suggestion-node:s1")

    const provisionalEdgeStyle = config.style.find(
      (entry) => entry.selector === "edge.provisional"
    )
    const provisionalNodeStyle = config.style.find(
      (entry) => entry.selector === "node.provisional"
    )
    expect(provisionalEdgeStyle.style["line-style"]).toBe("dashed")
    expect(provisionalEdgeStyle.style["target-arrow-shape"]).toBe("none")
    expect(provisionalNodeStyle.style["border-style"]).toBe("dashed")
    expect(provisionalEdgeStyle.style["line-color"]).toBe("#909090")
    expect(
      config.style.find((entry) => entry.selector === 'node[type="note"]')
        ?.style?.["background-color"]
    ).toBe("#505050")
  })

  it("keeps provisional nodes inert while authoritative nodes remain selectable", async () => {
    const onSelectNode = vi.fn()
    render(
      <NotesGraphCanvas
        graph={graph}
        layout="grid"
        focusNoteId="a"
        selectedNodeId="note:a"
        visibleEdgeTypes={new Set(["manual"])}
        provisionalOverlays={overlays}
        showProvisional
        onSelectNode={onSelectNode}
      />
    )

    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalled())
    act(() => {
      cyHandlers["tap:node"]?.({
        target: {
          id: () => "suggestion-node:s1",
          data: () => undefined,
          hasClass: (name) => name === "provisional"
        }
      })
    })
    expect(onSelectNode).not.toHaveBeenCalled()
    expect(
      mockCytoscapeFactory.mock.calls[0]?.[0].elements.find(
        (element) => element.data.id === "suggestion-node:s1"
      )
    ).toMatchObject({ selectable: false })

    act(() => {
      cyHandlers["tap:node"]?.({
        target: {
          id: () => "note:b",
          data: () => undefined,
          hasClass: () => false
        }
      })
    })
    expect(onSelectNode).toHaveBeenCalledTimes(1)
    expect(onSelectNode).toHaveBeenCalledWith("note:b")
  })

  it("re-runs a session layout without animated decorative motion", async () => {
    const { rerender } = render(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId={null}
        visibleEdgeTypes={new Set(["manual", "wikilink"])}
        provisionalOverlays={[]}
        showProvisional={false}
        onSelectNode={vi.fn()}
      />
    )
    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalledTimes(1))

    rerender(
      <NotesGraphCanvas
        graph={graph}
        layout="circle"
        focusNoteId="a"
        selectedNodeId={null}
        visibleEdgeTypes={new Set(["manual", "wikilink"])}
        provisionalOverlays={[]}
        showProvisional={false}
        onSelectNode={vi.fn()}
      />
    )

    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalledTimes(2))
    expect(mockCyInstance.destroy).toHaveBeenCalled()
    expect(mockCytoscapeFactory.mock.calls[1]?.[0]?.layout).toMatchObject({
      name: "circle",
      animate: false
    })
  })

  it("keeps the current Cytoscape canvas when sidebar selection changes", async () => {
    const onSelectNode = vi.fn()
    const visibleEdgeTypes = new Set(["manual", "wikilink"] as const)
    const provisionalOverlays: [] = []
    const { rerender } = render(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId="note:a"
        visibleEdgeTypes={visibleEdgeTypes}
        provisionalOverlays={provisionalOverlays}
        showProvisional={false}
        onSelectNode={onSelectNode}
      />
    )
    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalledTimes(1))

    rerender(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId="note:b"
        visibleEdgeTypes={visibleEdgeTypes}
        provisionalOverlays={provisionalOverlays}
        showProvisional={false}
        onSelectNode={onSelectNode}
      />
    )

    expect(mockCytoscapeFactory).toHaveBeenCalledTimes(1)
  })

  it("keeps zoom and pan while a fresh parent callback reaches the existing canvas", async () => {
    const firstCallback = vi.fn()
    const latestCallback = vi.fn()
    const visibleEdgeTypes = new Set(["manual", "wikilink"] as const)
    const provisionalOverlays: [] = []
    const { rerender } = render(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId="note:a"
        visibleEdgeTypes={visibleEdgeTypes}
        provisionalOverlays={provisionalOverlays}
        showProvisional={false}
        onSelectNode={firstCallback}
      />
    )
    await waitFor(() => expect(mockCytoscapeFactory).toHaveBeenCalledTimes(1))
    mockCyInstance.zoom(1.6)
    mockCyInstance.pan({ x: 24, y: 12 })

    rerender(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId="note:a"
        visibleEdgeTypes={visibleEdgeTypes}
        provisionalOverlays={provisionalOverlays}
        showProvisional={false}
        onSelectNode={latestCallback}
      />
    )

    expect(mockCytoscapeFactory).toHaveBeenCalledTimes(1)
    expect(mockCyInstance.zoom()).toBe(1.6)
    expect(mockCyInstance.pan()).toEqual({ x: 24, y: 12 })
    act(() => {
      cyHandlers["tap:node"]?.({
        target: {
          id: () => "note:b",
          data: () => undefined,
          hasClass: () => false
        }
      })
    })
    expect(firstCallback).not.toHaveBeenCalled()
    expect(latestCallback).toHaveBeenCalledWith("note:b")
  })

  it("fills its responsive parent without defining a second minimum height", () => {
    render(
      <NotesGraphCanvas
        graph={graph}
        layout="dagre"
        focusNoteId="a"
        selectedNodeId={null}
        visibleEdgeTypes={new Set(["manual"])}
        provisionalOverlays={[]}
        showProvisional={false}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByTestId("notes-graph-canvas")).toHaveClass(
      "h-full",
      "w-full"
    )
    expect(screen.getByTestId("notes-graph-canvas").className).not.toContain(
      "min-h-"
    )
  })
})

describe("NotesGraphToolbar controls", () => {
  const toolbarProps = {
    viewMode: "canvas" as const,
    suggestionsAuthorized: true,
    search: "load",
    searchResults: [
      { id: "note:a", label: "Loaded alpha" },
      { id: "note:b", label: "Loaded beta" }
    ],
    radius: 1 as const,
    maxNodes: 120,
    maxNodeCap: 300,
    layout: "dagre" as const,
    scope: "focused" as const,
    allNotes: { activeNoteCount: 8, effectiveNoteCap: 7, eligible: false },
    visibleEdgeTypes: new Set([
      "manual",
      "wikilink",
      "backlink",
      "tag_membership",
      "source_membership"
    ] as const),
    showProvisional: true,
    canExpand: true,
    isRefreshing: false,
    onSearchChange: vi.fn(),
    onViewModeChange: vi.fn(),
    onSelectSearchResult: vi.fn(),
    onRadiusChange: vi.fn(),
    onMaxNodesChange: vi.fn(),
    onLayoutChange: vi.fn(),
    onShowFocused: vi.fn(),
    onShowAllNotes: vi.fn(),
    onToggleEdgeType: vi.fn(),
    onToggleProvisional: vi.fn(),
    onFocusCurrent: vi.fn(),
    onExpand: vi.fn(),
    onRefresh: vi.fn(),
    onZoomIn: vi.fn(),
    onZoomOut: vi.fn(),
    onFit: vi.fn()
  }

  beforeEach(() => vi.clearAllMocks())

  it("keeps icon controls stable and exposes labels for search, focus, expansion, zoom, refresh, and fit", () => {
    render(<NotesGraphToolbar {...toolbarProps} />)

    expect(
      screen.getByRole("searchbox", { name: "Search loaded nodes" })
    ).toHaveClass("h-9")
    const iconLabels = [
      "Focus current note",
      "Expand graph",
      "Refresh graph",
      "Zoom in",
      "Zoom out",
      "Fit graph to view"
    ]
    iconLabels.forEach((label) => {
      expect(screen.getByRole("button", { name: label })).toHaveClass(
        "min-h-[44px]",
        "min-w-[44px]"
      )
      expect(screen.getByRole("button", { name: label })).toHaveAttribute(
        "title",
        label
      )
    })
    expect(screen.getByText("Loaded alpha")).toBeInTheDocument()
    expect(screen.queryByText("Unloaded library note")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Focus current note" }))
    fireEvent.click(screen.getByRole("button", { name: "Expand graph" }))
    fireEvent.click(screen.getByRole("button", { name: "Refresh graph" }))
    fireEvent.click(screen.getByRole("button", { name: "Fit graph to view" }))
    expect(toolbarProps.onFocusCurrent).toHaveBeenCalled()
    expect(toolbarProps.onExpand).toHaveBeenCalled()
    expect(toolbarProps.onRefresh).toHaveBeenCalled()
    expect(toolbarProps.onFit).toHaveBeenCalled()
  })

  it("uses bounded radius, node-limit, layout, and edge-filter controls", () => {
    render(<NotesGraphToolbar {...toolbarProps} />)

    fireEvent.change(screen.getByRole("combobox", { name: "Graph radius" }), {
      target: { value: "2" }
    })
    fireEvent.change(
      screen.getByRole("spinbutton", { name: "Maximum graph nodes" }),
      {
        target: { value: "280" }
      }
    )
    fireEvent.change(screen.getByRole("combobox", { name: "Graph layout" }), {
      target: { value: "grid" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Edge visibility" }))
    expect(
      screen.getByRole("button", { name: "Edge visibility" })
    ).not.toHaveAttribute("aria-haspopup")
    expect(
      screen.getByRole("button", { name: "Edge visibility" })
    ).toHaveAttribute("aria-expanded", "true")
    expect(
      screen.getByRole("button", { name: "Edge visibility" })
    ).toHaveAttribute("aria-controls", "notes-graph-edge-menu")
    expect(document.getElementById("notes-graph-edge-menu")).toBeInTheDocument()
    expect(
      screen.getByRole("group", { name: "Edge visibility filters" })
    ).toHaveAttribute("id", "notes-graph-edge-menu")
    fireEvent.click(screen.getByRole("checkbox", { name: "Manual links" }))
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "option:notesSearch.graphSuggestions"
      })
    )

    expect(toolbarProps.onRadiusChange).toHaveBeenCalledWith(2)
    expect(toolbarProps.onMaxNodesChange).toHaveBeenCalledWith(280)
    expect(toolbarProps.onLayoutChange).toHaveBeenCalledWith("grid")
    expect(toolbarProps.onToggleEdgeType).toHaveBeenCalledWith("manual")
    expect(toolbarProps.onToggleProvisional).toHaveBeenCalled()
  })

  it("uses the semantic foreground token for the selected scope", () => {
    render(<NotesGraphToolbar {...toolbarProps} />)

    expect(screen.getByRole("button", { name: "Focused" })).toHaveClass(
      "text-primary-foreground"
    )
    expect(screen.getByRole("button", { name: "Focused" })).not.toHaveClass(
      "text-white"
    )
  })
})
