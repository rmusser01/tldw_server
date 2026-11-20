/**
 * NL-11 review (#3101): list picks keep the current scope while a graph
 * loads, say when a note is outside the view, reveal notes hidden by
 * "Linked notes only", and the controls say what the graph shows.
 */
import type {
  NotesGraphNode,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

const { mockCanvas, mockFocusNode, mockUseNotesGraphWorkspace } = vi.hoisted(() => ({
  mockCanvas: vi.fn(),
  mockFocusNode: vi.fn(),
  mockUseNotesGraphWorkspace: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      const fallback =
        typeof options === "string" ? options : options?.defaultValue ?? key
      if (typeof options !== "object") return fallback
      return Object.entries(options).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        fallback
      )
    }
  })
}))

vi.mock("../hooks/useNotesGraphWorkspace", () => ({
  useNotesGraphWorkspace: mockUseNotesGraphWorkspace
}))

vi.mock("../hooks/useNotesGraphSuggestions", () => ({
  useNotesGraphSuggestions: () => ({ provisionalBySuggestionId: {} })
}))

vi.mock("../NotesGraphInspector", () => ({
  default: () => <div data-testid="notes-graph-inspector" />
}))

vi.mock("../NotesGraphCanvas", async () => {
  const ReactModule = await import("react")
  return {
    default: ReactModule.forwardRef((props: Record<string, unknown>, ref) => {
      mockCanvas(props)
      ReactModule.useImperativeHandle(ref, () => ({
        zoomIn: vi.fn(),
        zoomOut: vi.fn(),
        fit: vi.fn(),
        focusNode: mockFocusNode
      }))
      return <div data-testid="notes-graph-canvas" />
    })
  }
})

const noteNode = (id: string, label = id): NotesGraphNode => ({
  id,
  type: "note",
  label,
  created_at: null,
  deleted: false,
  degree: 0,
  tag_count: 0,
  primary_source_id: null
})

const baseGraph: NotesGraphResponse = {
  nodes: [noteNode("note:a", "Alpha note"), noteNode("note:b", "Beta note"), noteNode("note:c", "Lonely note")],
  edges: [
    {
      id: "wikilink:a:b",
      source: "note:a",
      target: "note:b",
      type: "wikilink",
      directed: true,
      weight: null,
      label: null
    }
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 3,
  all_notes_note_cap: 500,
  all_notes_eligible: true,
  manual_link_authorized: false
}

const workspaceState = (overrides: Record<string, unknown> = {}) => ({
  graph: baseGraph,
  graphQuery: { isFetching: false },
  focusNoteId: "a",
  scope: "focused" as const,
  tagFilter: null,
  filterByTag: vi.fn(),
  layout: "dagre" as const,
  setLayout: vi.fn(),
  search: "",
  setSearch: vi.fn(),
  searchResults: [],
  visibleEdgeTypes: new Set(["manual", "wikilink", "backlink", "tag_membership", "source_membership"]),
  toggleEdgeType: vi.fn(),
  semantic: {
    enabled: false,
    setEnabled: vi.fn(),
    threshold: 0.75,
    setThreshold: vi.fn(),
    topK: 10,
    maxTopK: 50,
    setTopK: vi.fn(),
    focusRequired: false,
    reset: vi.fn()
  },
  semanticIndex: undefined,
  allNotes: { activeNoteCount: 3, effectiveNoteCap: 500, eligible: true },
  canExpand: false,
  expand: vi.fn(),
  focus: vi.fn(),
  showAllNotes: vi.fn(() => true),
  refresh: vi.fn(),
  isPermissionDenied: false,
  createManualLink: vi.fn(),
  manualLinkPendingEdgeIds: new Set<string>(),
  queryIdentity: "q",
  isOffline: false,
  isLoading: false,
  error: null,
  ...overrides
})

type WorkspaceProps = React.ComponentProps<typeof NotesGraphWorkspace>

const renderWorkspace = (props: Partial<WorkspaceProps> = {}) => {
  const client = new QueryClient()
  const view = (next: Partial<WorkspaceProps>) => (
    <QueryClientProvider client={client}>
      <NotesGraphWorkspace
        authorityScope="authority-a"
        isOnline
        initialFocusNoteId="a"
        selectedNoteId="a"
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
        {...props}
        {...next}
      />
    </QueryClientProvider>
  )
  const rendered = render(view({}))
  return { ...rendered, rerenderWith: (next: Partial<WorkspaceProps>) => rendered.rerender(view(next)) }
}

const lastCanvasProps = () =>
  mockCanvas.mock.calls.at(-1)?.[0] as {
    graph: NotesGraphResponse
    layout: string
    onSelectNode: (nodeId: string) => void
  }

describe("NotesGraphWorkspace NL-11 review", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => cleanup())

  it("keeps All notes while it loads and pans to the picked note once it arrives", () => {
    const loading = workspaceState({ scope: "all", graph: null, isLoading: true })
    mockUseNotesGraphWorkspace.mockReturnValue(loading)
    const { rerenderWith } = renderWorkspace()

    rerenderWith({ selectedNoteId: "b" })
    expect(loading.focus).not.toHaveBeenCalled()

    const loaded = workspaceState({ scope: "all" })
    mockUseNotesGraphWorkspace.mockReturnValue(loaded)
    rerenderWith({ selectedNoteId: "b" })

    expect(loaded.focus).not.toHaveBeenCalled()
    expect(mockFocusNode).toHaveBeenCalledWith("note:b")
  })

  it("keeps the tag and says when the picked note is not in the filtered view", () => {
    const state = workspaceState({ scope: "all", tagFilter: "reading" })
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const { rerenderWith } = renderWorkspace({ tagOptions: ["reading"] })

    rerenderWith({ selectedNoteId: "outside", tagOptions: ["reading"] })

    expect(state.focus).not.toHaveBeenCalled()
    expect(state.filterByTag).not.toHaveBeenCalled()
    const notice = screen.getByTestId("notes-graph-pick-outside-view")
    expect(notice).toHaveTextContent("The note you picked isn't in this view.")
    fireEvent.click(within(notice).getByRole("button", { name: "Show it in the graph" }))
    expect(state.focus).toHaveBeenCalledWith("outside")
  })

  it("re-centres on a note picked again after the first pick was cancelled", async () => {
    const state = workspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const onSelectNote = vi.fn(async () => false)
    const { rerenderWith } = renderWorkspace({ onSelectNote })

    await act(async () => {
      lastCanvasProps().onSelectNode("note:b")
    })
    rerenderWith({ selectedNoteId: "b" })

    expect(state.focus).toHaveBeenCalledWith("b")
  })

  it("reveals and centres a search result hidden by Linked notes only", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(
      workspaceState({
        search: "lonely",
        searchResults: [{ id: "note:c", label: "Lonely note" }]
      })
    )
    // Centring only works on a node the canvas already draws.
    const centredOnDrawnNode: boolean[] = []
    mockFocusNode.mockImplementation((nodeId: string) => {
      centredOnDrawnNode.push(lastCanvasProps().graph.nodes.some((node) => node.id === nodeId))
    })
    renderWorkspace()
    fireEvent.click(screen.getByRole("checkbox", { name: "Linked notes only" }))
    expect(lastCanvasProps().graph.nodes.map((node) => node.id)).not.toContain("note:c")

    fireEvent.click(screen.getByRole("button", { name: "Lonely note" }))

    expect(lastCanvasProps().graph.nodes.map((node) => node.id)).toContain("note:c")
    expect(mockFocusNode).toHaveBeenLastCalledWith("note:c")
    expect(centredOnDrawnNode.at(-1)).toBe(true)
  })

  it("chooses Rings for a large graph without a notice, and shows it in Graph settings", () => {
    const nodes = Array.from({ length: 201 }, (_, index) => noteNode(`note:n${index}`))
    mockUseNotesGraphWorkspace.mockReturnValue(
      workspaceState({ graph: { ...baseGraph, nodes, edges: [] }, scope: "all" })
    )
    renderWorkspace()

    expect(lastCanvasProps().layout).toBe("concentric")
    expect(screen.queryByTestId("notes-graph-layout-fallback")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Graph settings" }))
    expect(screen.getByRole("combobox", { name: "Graph layout" })).toHaveValue("concentric")
  })

  it("explains Rings when the reader chose Tree for a large graph", () => {
    const nodes = Array.from({ length: 201 }, (_, index) => noteNode(`note:n${index}`))
    const state = workspaceState({ graph: { ...baseGraph, nodes, edges: [] }, scope: "all" })
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "Graph settings" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Graph layout" }), {
      target: { value: "dagre" }
    })

    expect(state.setLayout).toHaveBeenCalledWith("dagre")
    expect(screen.getByTestId("notes-graph-layout-fallback")).toHaveTextContent(
      "Tree is slow for a graph this large, so it uses the Rings layout."
    )
  })

  it("does not press All notes for a tag-filtered graph over the cap", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(
      workspaceState({
        scope: "all",
        tagFilter: "reading",
        allNotes: { activeNoteCount: 2000, effectiveNoteCap: 500, eligible: false }
      })
    )
    renderWorkspace({ tagOptions: ["reading"] })

    const allNotes = screen.getByRole("button", { name: "All notes" })
    expect(allNotes).toHaveAttribute("aria-pressed", "false")
    expect(allNotes).toBeDisabled()
    expect(screen.getByRole("button", { name: "Focused" })).toHaveAttribute("aria-pressed", "false")
  })

  it("disables distance and item limit in All notes, and says why", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState({ scope: "all" }))
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "Graph settings" }))

    expect(screen.getByRole("combobox", { name: "Distance from focus note" })).toBeDisabled()
    expect(screen.getByRole("spinbutton", { name: "Most items shown in the graph" })).toBeDisabled()
    expect(screen.getByTestId("notes-graph-settings-library-hint")).toHaveTextContent(
      "All notes and tag filters show every matching note"
    )
  })
})
