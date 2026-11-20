/**
 * NL-11 (decision 15, #3101): in Graph mode a note picked in the notes list
 * re-centres the graph and opens a preview, without leaving Graph. The graph
 * also gets filters (tag, linked notes only) and stays fast when it is large.
 */
import type {
  NotesGraphNode,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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

describe("NotesGraphWorkspace list picks and filters (NL-11)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => cleanup())

  it("re-centres the graph on a note picked in the notes list, staying in Graph", () => {
    const state = workspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const { rerenderWith } = renderWorkspace()

    rerenderWith({ selectedNoteId: "b" })

    expect(state.focus).toHaveBeenCalledWith("b")
    expect(screen.getByTestId("notes-graph-canvas")).toBeInTheDocument()
  })

  it("does not re-centre when the note was picked in the graph itself", () => {
    const state = workspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const onSelectNote = vi.fn()
    const { rerenderWith } = renderWorkspace({ onSelectNote })

    act(() => lastCanvasProps().onSelectNode("note:b"))
    expect(onSelectNote).toHaveBeenCalledWith("b")
    rerenderWith({ selectedNoteId: "b" })

    expect(state.focus).not.toHaveBeenCalled()
  })

  it("in All notes, centres the canvas on a loaded note instead of refocusing", () => {
    const state = workspaceState({ scope: "all" })
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const { rerenderWith } = renderWorkspace()

    rerenderWith({ selectedNoteId: "b" })

    expect(mockFocusNode).toHaveBeenCalledWith("note:b")
    expect(state.focus).not.toHaveBeenCalled()
  })

  it("previews the selected note and opens it in the editor on request", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState())
    const onOpenInEditor = vi.fn()
    renderWorkspace({
      selectedNoteId: "b",
      notePreview: {
        noteId: "b",
        title: "Beta note",
        content: "# Beta note\n\nFirst **bold** line about [[Alpha note]]."
      },
      onOpenInEditor
    })

    const preview = screen.getByRole("region", { name: "Note preview" })
    expect(within(preview).getByText("Beta note")).toBeInTheDocument()
    expect(
      within(preview).getByText("First bold line about Alpha note.")
    ).toBeInTheDocument()
    fireEvent.click(within(preview).getByRole("button", { name: "Open in editor" }))
    expect(onOpenInEditor).toHaveBeenCalledWith("b")
  })

  it("does not preview another note's content while the picked note loads", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState())
    renderWorkspace({
      selectedNoteId: "b",
      notePreview: { noteId: "a", title: "Alpha note", content: "Alpha body" }
    })

    expect(screen.queryByText("Alpha body")).not.toBeInTheDocument()
  })

  it("filters the graph by tag", async () => {
    const state = workspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace({ tagOptions: ["reading", "work"] })

    fireEvent.change(screen.getByRole("combobox", { name: "Tag" }), {
      target: { value: "reading" }
    })

    // The select settles before it refetches (NL-11 review).
    await waitFor(() => expect(state.filterByTag).toHaveBeenCalledWith("reading"))
  })

  it("hides notes without links when 'Linked notes only' is on", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState())
    renderWorkspace()
    expect(lastCanvasProps().graph.nodes.map((node) => node.id)).toContain("note:c")

    fireEvent.click(screen.getByRole("checkbox", { name: "Linked notes only" }))

    expect(lastCanvasProps().graph.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b"
    ])
  })

  it("points to the tag filter when the library is over the All-notes cap", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(
      workspaceState({
        allNotes: { activeNoteCount: 2000, effectiveNoteCap: 500, eligible: false }
      })
    )
    renderWorkspace()

    expect(screen.getByTestId("notes-graph-all-disabled-reason")).toHaveTextContent(
      "All notes is available for up to 500 active notes. This library has 2000. Pick a tag to see part of it."
    )
  })

  it("lays out a large graph as Rings by itself, without a notice (NL-11 review)", () => {
    const nodes = Array.from({ length: 201 }, (_, index) => noteNode(`note:n${index}`))
    mockUseNotesGraphWorkspace.mockReturnValue(
      workspaceState({ graph: { ...baseGraph, nodes, edges: [] }, scope: "all" })
    )
    renderWorkspace()

    expect(lastCanvasProps().layout).toBe("concentric")
    expect(screen.queryByTestId("notes-graph-layout-fallback")).not.toBeInTheDocument()
  })
})
